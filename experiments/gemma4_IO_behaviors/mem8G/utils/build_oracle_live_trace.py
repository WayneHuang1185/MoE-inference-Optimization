#!/usr/bin/env python3
"""Convert router-label NPZ files into live oracle-window JSONL traces."""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def decode_meta(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            raw = raw.item()
        elif raw.size == 1:
            raw = raw.reshape(()).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    return {}


def normalize_router_topk(router_topk: np.ndarray, seq_len: int) -> np.ndarray:
    arr = np.asarray(router_topk)
    if arr.ndim != 3:
        raise ValueError(f"router_topk must have 3 dimensions, got shape={arr.shape}")
    if arr.shape[0] == seq_len:
        return arr
    if arr.shape[1] == seq_len:
        return np.transpose(arr, (1, 0, 2))
    raise ValueError(f"cannot align router_topk shape={arr.shape} to loss_mask length={seq_len}")


def token_ids_from_npz(data: Any, seq_len: int) -> list[int | None]:
    for key in ("input_ids", "token_ids", "tokens"):
        if key in data:
            arr = np.asarray(data[key]).reshape(-1)
            if len(arr) >= seq_len:
                return [int(x) for x in arr[:seq_len]]
    return [None] * seq_len


def iter_oracle_rows(npz_path: Path, *, include_prefill: bool, include_decode: bool) -> Iterable[dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as data:
        if "router_topk" not in data:
            raise KeyError(f"{npz_path} missing router_topk")
        if "loss_mask" not in data:
            raise KeyError(f"{npz_path} missing loss_mask")
        meta = decode_meta(data["meta_json"] if "meta_json" in data else None)
        sample_id = str(meta.get("sample_id") or npz_path.stem)
        loss_mask = np.asarray(data["loss_mask"]).reshape(-1).astype(bool)
        router_topk = normalize_router_topk(np.asarray(data["router_topk"]), len(loss_mask))
        token_ids = token_ids_from_npz(data, len(loss_mask))

        prefill_index = 0
        decode_index = 0
        for pos in range(len(loss_mask)):
            phase = "decode" if loss_mask[pos] else "prefill"
            if phase == "prefill":
                token_index = prefill_index
                prefill_index += 1
                if not include_prefill:
                    continue
            else:
                token_index = decode_index
                decode_index += 1
                if not include_decode:
                    continue
            experts = {
                (int(layer), int(expert))
                for layer in range(router_topk.shape[1])
                for expert in router_topk[pos, layer].tolist()
                if int(expert) >= 0
            }
            row = {
                "sample_id": sample_id,
                "phase": phase,
                "token_index": token_index,
                "source_position": int(pos),
                "token_id": token_ids[int(pos)],
                "experts": [
                    {"layer": layer, "expert": expert}
                    for layer, expert in sorted(experts)
                ],
            }
            if phase == "decode":
                row["decode_index"] = token_index
            else:
                row["prefill_index"] = token_index
            yield row


def list_npz_files(npz_dir: Path, manifest: Path | None, limit: int) -> list[Path]:
    if manifest is not None:
        import csv

        files: list[Path] = []
        with manifest.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status", "ok") != "ok":
                    continue
                raw = row.get("npz_path")
                if not raw:
                    continue
                path = Path(raw)
                if not path.is_absolute():
                    path = Path.cwd() / path
                files.append(path)
    else:
        files = sorted(npz_dir.glob("*.npz"))
    if limit > 0:
        files = files[:limit]
    return files


def write_trace(files: list[Path], output: Path, *, include_prefill: bool, include_decode: bool) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = 0
    rows = 0
    with output.open("w", encoding="utf-8") as f:
        for path in files:
            sample_rows = list(iter_oracle_rows(path, include_prefill=include_prefill, include_decode=include_decode))
            if sample_rows:
                samples += 1
            for row in sample_rows:
                f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                rows += 1
    return {
        "files": len(files),
        "samples": samples,
        "rows": rows,
        "include_prefill": include_prefill,
        "include_decode": include_decode,
        "output": str(output),
    }


def counter_prefetch_candidates(token_maps: list[set[tuple[int, int]]], threshold: int) -> list[tuple[int, int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for token_map in token_maps:
        counts.update(token_map)
    return [
        (layer, expert, count)
        for (layer, expert), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        if count >= threshold
    ]


def run_synthetic_tests() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sample_a.npz"
        router_topk = np.array(
            [
                [[[1, 1, 2], [3, 4, 4]]],
                [[[5, 6, 6], [7, 8, 7]]],
                [[[9, 9, 9], [10, 11, 10]]],
            ],
            dtype=np.int16,
        ).reshape(3, 2, 3)
        np.savez(
            path,
            router_topk=router_topk,
            loss_mask=np.array([0, 1, 1], dtype=np.int8),
            input_ids=np.array([100, 101, 102], dtype=np.int32),
            meta_json=json.dumps({"sample_id": "sample_a"}),
        )
        decode_rows = list(iter_oracle_rows(path, include_prefill=False, include_decode=True))
        assert [r["decode_index"] for r in decode_rows] == [0, 1]
        assert [r["source_position"] for r in decode_rows] == [1, 2]
        assert decode_rows[0]["token_id"] == 101
        assert decode_rows[0]["experts"] == [
            {"layer": 0, "expert": 5},
            {"layer": 0, "expert": 6},
            {"layer": 1, "expert": 7},
            {"layer": 1, "expert": 8},
        ]
        prefill_rows = list(iter_oracle_rows(path, include_prefill=True, include_decode=False))
        assert len(prefill_rows) == 1
        assert prefill_rows[0]["phase"] == "prefill"
        assert prefill_rows[0]["prefill_index"] == 0
        assert prefill_rows[0]["source_position"] == 0

    token_maps = [{(0, 1), (1, 2)} for _ in range(5)] + [{(0, 1), (2, 3)} for _ in range(5)]
    candidates = counter_prefetch_candidates(token_maps, threshold=5)
    assert candidates == [(0, 1, 10), (1, 2, 5), (2, 3, 5)]
    print("synthetic oracle live trace tests ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", default="experiments/gemma4_IO_behaviors/mem8G/statistics/truth_prefill_logits_5_short_n16_20260623_040619/router_label_npz/npz")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=False)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-prefill", action="store_true", help="also emit loss_mask == 0 prefill token expert maps")
    parser.add_argument("--no-decode", action="store_true", help="do not emit loss_mask == 1 decode token expert maps")
    parser.add_argument("--run-synthetic-tests", action="store_true")
    args = parser.parse_args()

    if args.run_synthetic_tests:
        run_synthetic_tests()
        return
    if not args.output:
        raise SystemExit("--output is required unless --run-synthetic-tests is set")
    files = list_npz_files(Path(args.npz_dir), Path(args.manifest) if args.manifest else None, args.limit)
    if not files:
        raise SystemExit("no NPZ files found")
    summary = write_trace(files, Path(args.output), include_prefill=args.include_prefill, include_decode=not args.no_decode)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
