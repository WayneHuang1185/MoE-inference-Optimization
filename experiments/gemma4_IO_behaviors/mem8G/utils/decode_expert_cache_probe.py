#!/usr/bin/env python3
"""Probe Gemma4 expert page-cache residency during 8G decode.

This sidecar intentionally keeps llama.cpp unchanged. It drives llama-server via
HTTP, samples GGUF expert tensor residency with mincore(), and optionally records
perf page-fault samples only after prefill has completed.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import html
import json
import mmap
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
MB = 1024 * 1024
EXPERTS = 128
LAYERS = 30
HEX_RE = re.compile(r"0x[0-9a-fA-F]+")

LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
LIBC.mincore.restype = ctypes.c_int


@dataclass(frozen=True)
class TensorRange:
    name: str
    layer: int | None
    family: str
    tensor_type: str
    n_bytes: int
    file_start: int
    file_end: int


@dataclass(frozen=True)
class ExpertRange:
    layer: int
    expert: int
    role: str
    file_start: int
    file_end: int


@dataclass
class TensorWatch:
    tensor: TensorRange
    previous: bytearray | None = None
    total_refaulted_pages: int = 0
    total_evicted_pages: int = 0


@dataclass(frozen=True)
class Mapping:
    start: int
    end: int
    file_offset: int
    path: str


class PerfRecorder:
    def __init__(self, *, pid: int, out_dir: Path) -> None:
        self.pid = pid
        self.out_dir = out_dir
        self.data_path = out_dir / "decode_page_faults.perf.data"
        self.text_path = out_dir / "decode_page_faults.perf.script.txt"
        self.proc: subprocess.Popen[str] | None = None
        self.error: str | None = None

    def start(self) -> bool:
        cmd = [
            "perf",
            "record",
            "-q",
            "-e",
            "page-faults",
            "-d",
            "-p",
            str(self.pid),
            "-o",
            str(self.data_path),
            "--",
            "sleep",
            "3600",
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError as exc:
            self.error = f"perf is not available: {exc}"
            return False
        time.sleep(0.25)
        if self.proc.poll() is not None:
            _out, err = self.proc.communicate()
            self.error = err.strip() or "perf exited before decode"
            return False
        return True

    def stop(self) -> bool:
        if self.proc is None:
            return False
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                _out, err = self.proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                _out, err = self.proc.communicate(timeout=10)
            if self.proc.returncode not in (0, -signal.SIGINT):
                self.error = (err or "").strip() or f"perf exited with {self.proc.returncode}"
                return False
        try:
            with self.text_path.open("w", encoding="utf-8") as f:
                subprocess.run(["perf", "script", "-i", str(self.data_path)], stdout=f, stderr=subprocess.PIPE, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            self.error = f"perf script failed: {exc}"
            return False
        return True


def page_floor(value: int) -> int:
    return value - (value % PAGE_SIZE)


def page_ceil(value: int) -> int:
    rem = value % PAGE_SIZE
    return value if rem == 0 else value + PAGE_SIZE - rem


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def slot_erase(base_url: str, slot_id: int) -> dict[str, Any]:
    try:
        return request_json(f"{base_url}/slots/{slot_id}?action=erase", {})
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode("utf-8", errors="replace"), "status": exc.code}


def wait_slot_idle(base_url: str, slot_id: int, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            slots = request_json(f"{base_url}/slots", timeout=5)
            if isinstance(slots, list):
                for slot in slots:
                    if slot.get("id") == slot_id:
                        if slot.get("is_processing") is False:
                            return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def load_prompts(prompt_dir: Path, limit: int) -> list[tuple[str, str]]:
    paths = sorted(prompt_dir.glob("*.txt"))
    if limit > 0:
        paths = paths[:limit]
    return [(path.name, path.read_text(encoding="utf-8")) for path in paths]


def load_tensor_ranges(path: Path) -> list[TensorRange]:
    out: list[TensorRange] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            layer_s = row.get("layer", "")
            out.append(
                TensorRange(
                    name=row["name"],
                    layer=int(layer_s) if layer_s != "" else None,
                    family=row.get("family", ""),
                    tensor_type=row.get("type", ""),
                    n_bytes=int(row["n_bytes"]),
                    file_start=int(row["file_start"]),
                    file_end=int(row["file_end"]),
                )
            )
    return out


def build_expert_ranges(tensors: list[TensorRange]) -> dict[tuple[int, int], list[ExpertRange]]:
    ranges: dict[tuple[int, int], list[ExpertRange]] = {}
    for tensor in tensors:
        if tensor.layer is None:
            continue
        if tensor.name.endswith(".ffn_down_exps.weight"):
            role = "down"
        elif tensor.name.endswith(".ffn_gate_up_exps.weight"):
            role = "gate_up"
        else:
            continue
        if tensor.n_bytes % EXPERTS != 0:
            raise ValueError(f"expert tensor is not evenly divisible by {EXPERTS}: {tensor.name}")
        expert_bytes = tensor.n_bytes // EXPERTS
        for expert in range(EXPERTS):
            start = tensor.file_start + expert * expert_bytes
            ranges.setdefault((tensor.layer, expert), []).append(
                ExpertRange(
                    layer=tensor.layer,
                    expert=expert,
                    role=role,
                    file_start=start,
                    file_end=start + expert_bytes,
                )
            )
    missing = [(layer, expert) for layer in range(LAYERS) for expert in range(EXPERTS) if (layer, expert) not in ranges]
    if missing:
        raise ValueError(f"missing expert ranges, first missing={missing[0]}, count={len(missing)}")
    return ranges


def mincore_range_states(base_addr: int, start: int, end: int, page_stride: int) -> bytearray:
    aligned_start = page_floor(start)
    aligned_end = page_ceil(end)
    pages = (aligned_end - aligned_start) // PAGE_SIZE
    vec = ctypes.create_string_buffer(pages)
    if LIBC.mincore(ctypes.c_void_p(base_addr + aligned_start), ctypes.c_size_t(pages * PAGE_SIZE), vec) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    states = bytearray()
    for i, byte in enumerate(vec.raw):
        if i % page_stride == 0:
            states.append(1 if byte & 1 else 0)
    return states


def mincore_range(base_addr: int, start: int, end: int, page_stride: int) -> tuple[int, int]:
    states = mincore_range_states(base_addr, start, end, page_stride)
    return states.count(1), len(states)


def sample_tensor_watch(base_addr: int, watch: TensorWatch, page_stride: int) -> dict[str, Any]:
    states = mincore_range_states(base_addr, watch.tensor.file_start, watch.tensor.file_end, page_stride)
    present = states.count(1)
    sampled = len(states)
    refaulted = 0
    evicted = 0
    if watch.previous is not None:
        for before, after in zip(watch.previous, states):
            if before == 0 and after == 1:
                refaulted += 1
            elif before == 1 and after == 0:
                evicted += 1
    watch.previous = states
    watch.total_refaulted_pages += refaulted
    watch.total_evicted_pages += evicted
    group = "moe" if is_moe_tensor(watch.tensor.name, watch.tensor.family) else "non_moe"
    return {
        "tensor": watch.tensor.name,
        "layer": "" if watch.tensor.layer is None else watch.tensor.layer,
        "family": watch.tensor.family,
        "type": watch.tensor.tensor_type,
        "group": group,
        "sampled_pages": sampled,
        "present_pages": present,
        "refaulted_pages": refaulted,
        "evicted_pages": evicted,
        "present_ratio": present / sampled if sampled else 0.0,
        "refaulted_mb_est": refaulted * page_stride * PAGE_SIZE / MB,
        "evicted_mb_est": evicted * page_stride * PAGE_SIZE / MB,
    }
    sampled = 0
    present = 0
    for i, byte in enumerate(vec.raw):
        if i % page_stride != 0:
            continue
        sampled += 1
        if byte & 1:
            present += 1
    return present, sampled


def sample_matrix(
    *,
    base_addr: int,
    expert_ranges: dict[tuple[int, int], list[ExpertRange]],
    page_stride: int,
    threshold: float,
) -> tuple[list[list[bool]], list[dict[str, Any]]]:
    matrix = [[False for _ in range(EXPERTS)] for _ in range(LAYERS)]
    rows: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            present = 0
            sampled = 0
            for r in expert_ranges[(layer, expert)]:
                p, s = mincore_range(base_addr, r.file_start, r.file_end, page_stride)
                present += p
                sampled += s
            ratio = present / sampled if sampled else 0.0
            resident = ratio >= threshold
            matrix[layer][expert] = resident
            rows.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "present_pages_sampled": present,
                    "sampled_pages": sampled,
                    "resident_ratio": ratio,
                    "resident": int(resident),
                }
            )
    return matrix, rows


def load_maps(pid: int, model_name: str) -> list[Mapping]:
    mappings: list[Mapping] = []
    for line in Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        addr_range, perms, offset_hex, _dev, _inode, pathname = parts
        if "r" not in perms or model_name not in pathname:
            continue
        start_s, end_s = addr_range.split("-")
        mappings.append(Mapping(start=int(start_s, 16), end=int(end_s, 16), file_offset=int(offset_hex, 16), path=pathname))
    mappings.sort(key=lambda m: m.start)
    return mappings


def address_to_file_offset(addr: int, mappings: list[Mapping]) -> int | None:
    for m in mappings:
        if m.start <= addr < m.end:
            return m.file_offset + (addr - m.start)
    return None


def find_tensor(offset: int, tensors: list[TensorRange], starts: list[int]) -> TensorRange | None:
    idx = bisect.bisect_right(starts, offset) - 1
    if idx < 0:
        return None
    tensor = tensors[idx]
    if tensor.file_start <= offset < tensor.file_end:
        return tensor
    return None


def is_moe_tensor(name: str, family: str) -> bool:
    return family == "moe_ffn" or "ffn_down_exps" in name or "ffn_gate_up_exps" in name or "ffn_gate_inp" in name


def write_perf_unavailable(out_dir: Path, reason: str) -> None:
    (out_dir / "PERF_UNAVAILABLE.md").write_text(
        "# Perf Page-Fault Attribution Unavailable\n\n"
        f"Reason: `{reason}`\n\n"
        "Run the Docker container with perf permissions, for example "
        "`--cap-add PERFMON --cap-add SYS_ADMIN --security-opt seccomp=unconfined`.\n",
        encoding="utf-8",
    )


def attribute_perf_faults(out_dir: Path, perf_text: Path, maps: list[Mapping], tensors: list[TensorRange]) -> None:
    tensors_sorted = sorted(tensors, key=lambda t: t.file_start)
    starts = [t.file_start for t in tensors_sorted]
    by_tensor: dict[str, dict[str, Any]] = {}
    by_group = {
        "moe": {"group": "moe", "faults": 0, "estimated_fault_mb": 0.0},
        "non_moe": {"group": "non_moe", "faults": 0, "estimated_fault_mb": 0.0},
        "unattributed": {"group": "unattributed", "faults": 0, "estimated_fault_mb": 0.0},
    }
    hex_lines = 0
    for line in perf_text.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = HEX_RE.findall(line)
        if not tokens:
            continue
        hex_lines += 1
        matched: TensorRange | None = None
        for token in tokens:
            off = address_to_file_offset(int(token, 16), maps)
            if off is None:
                continue
            matched = find_tensor(off, tensors_sorted, starts)
            if matched is not None:
                break
        if matched is None:
            by_group["unattributed"]["faults"] += 1
            by_group["unattributed"]["estimated_fault_mb"] += PAGE_SIZE / MB
            continue
        group = "moe" if is_moe_tensor(matched.name, matched.family) else "non_moe"
        by_group[group]["faults"] += 1
        by_group[group]["estimated_fault_mb"] += PAGE_SIZE / MB
        item = by_tensor.setdefault(
            matched.name,
            {
                "tensor": matched.name,
                "layer": "" if matched.layer is None else matched.layer,
                "family": matched.family,
                "type": matched.tensor_type,
                "group": group,
                "faults": 0,
                "estimated_fault_mb": 0.0,
            },
        )
        item["faults"] += 1
        item["estimated_fault_mb"] += PAGE_SIZE / MB

    with (out_dir / "decode_faults_by_tensor.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["tensor", "layer", "family", "type", "group", "faults", "estimated_fault_mb"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(by_tensor.values(), key=lambda r: int(r["faults"]), reverse=True))

    total_attr = by_group["moe"]["faults"] + by_group["non_moe"]["faults"]
    rows = []
    for key in ("moe", "non_moe", "unattributed"):
        row = dict(by_group[key])
        row["attributed_fault_ratio"] = row["faults"] / total_attr if key != "unattributed" and total_attr else 0.0
        rows.append(row)
    with (out_dir / "decode_moe_vs_non_moe_faults.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["group", "faults", "estimated_fault_mb", "attributed_fault_ratio"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "decode_perf_summary.json").write_text(
        json.dumps({"hex_lines": hex_lines, "attributed_faults": total_attr, "groups": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_refault_outputs(out_dir: Path, tensor_samples: list[dict[str, Any]]) -> None:
    fields = [
        "sample_index",
        "sample_label",
        "prompt_index",
        "prompt_name",
        "tensor",
        "layer",
        "family",
        "type",
        "group",
        "sampled_pages",
        "present_pages",
        "refaulted_pages",
        "evicted_pages",
        "present_ratio",
        "refaulted_mb_est",
        "evicted_mb_est",
    ]
    with (out_dir / "decode_refaults_by_tensor_sample.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tensor_samples)

    by_tensor: dict[tuple[str, str], dict[str, Any]] = {}
    by_group: dict[str, dict[str, Any]] = {}
    for row in tensor_samples:
        key = (str(row["tensor"]), str(row["group"]))
        item = by_tensor.setdefault(
            key,
            {
                "tensor": row["tensor"],
                "layer": row["layer"],
                "family": row["family"],
                "type": row["type"],
                "group": row["group"],
                "refaulted_pages": 0,
                "evicted_pages": 0,
                "refaulted_mb_est": 0.0,
                "evicted_mb_est": 0.0,
            },
        )
        group = by_group.setdefault(
            str(row["group"]),
            {
                "group": row["group"],
                "refaulted_pages": 0,
                "evicted_pages": 0,
                "refaulted_mb_est": 0.0,
                "evicted_mb_est": 0.0,
            },
        )
        for target in (item, group):
            target["refaulted_pages"] += int(row["refaulted_pages"])
            target["evicted_pages"] += int(row["evicted_pages"])
            target["refaulted_mb_est"] += float(row["refaulted_mb_est"])
            target["evicted_mb_est"] += float(row["evicted_mb_est"])

    tensor_fields = [
        "tensor",
        "layer",
        "family",
        "type",
        "group",
        "refaulted_pages",
        "evicted_pages",
        "refaulted_mb_est",
        "evicted_mb_est",
    ]
    with (out_dir / "decode_refaults_by_tensor.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tensor_fields)
        writer.writeheader()
        writer.writerows(sorted(by_tensor.values(), key=lambda r: float(r["refaulted_mb_est"]), reverse=True))

    total_refaulted = sum(int(row["refaulted_pages"]) for row in by_group.values())
    group_rows = []
    for group in ("moe", "non_moe"):
        row = by_group.get(
            group,
            {
                "group": group,
                "refaulted_pages": 0,
                "evicted_pages": 0,
                "refaulted_mb_est": 0.0,
                "evicted_mb_est": 0.0,
            },
        )
        row = dict(row)
        row["refaulted_ratio"] = int(row["refaulted_pages"]) / total_refaulted if total_refaulted else 0.0
        group_rows.append(row)
    with (out_dir / "decode_moe_vs_non_moe_refaults.csv").open("w", encoding="utf-8", newline="") as f:
        fields2 = ["group", "refaulted_pages", "evicted_pages", "refaulted_mb_est", "evicted_mb_est", "refaulted_ratio"]
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(group_rows)

    (out_dir / "decode_refault_summary.json").write_text(
        json.dumps({"total_refaulted_pages": total_refaulted, "groups": group_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def matrix_lines(matrix: list[list[bool]]) -> list[str]:
    lines = []
    for layer, row in enumerate(matrix):
        lines.append(f"L{layer:02d} " + "".join("1" if x else "." for x in row))
    return lines


def write_matrix_outputs(out_dir: Path, samples: list[dict[str, Any]], threshold: float) -> None:
    csv_rows = []
    for sample in samples:
        for row in sample["rows"]:
            csv_rows.append(
                {
                    "sample_index": sample["sample_index"],
                    "sample_label": sample["sample_label"],
                    "prompt_index": sample["prompt_index"],
                    "prompt_name": sample["prompt_name"],
                    **row,
                }
            )
    fields = [
        "sample_index",
        "sample_label",
        "prompt_index",
        "prompt_name",
        "layer",
        "expert",
        "present_pages_sampled",
        "sampled_pages",
        "resident_ratio",
        "resident",
    ]
    with (out_dir / "expert_cache_matrix_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    json_samples = [
        {
            key: sample[key]
            for key in ("sample_index", "sample_label", "prompt_index", "prompt_name", "ts_s", "elapsed_s", "resident_experts")
        }
        | {"matrix": sample["matrix"]}
        for sample in samples
    ]
    (out_dir / "expert_cache_matrices.json").write_text(json.dumps(json_samples, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Decode Expert Page-Cache Matrices",
        "",
        f"`resident=true` means combined down + gate_up expert pages are at least `{threshold:.2f}` resident.",
        "",
    ]
    for sample in samples:
        lines.extend(
            [
                f"## {sample['sample_label']}",
                "",
                f"- prompt: `{sample['prompt_name']}`",
                f"- resident experts: `{sample['resident_experts']}` / `{LAYERS * EXPERTS}`",
                "",
                "```text",
                *matrix_lines(sample["matrix"]),
                "```",
                "",
            ]
        )
    refault_path = out_dir / "decode_moe_vs_non_moe_refaults.csv"
    if refault_path.exists():
        with refault_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        lines.extend(
            [
                "## MOE vs non-MOE Refault Attribution",
                "",
                "This is a `mincore` page-cache transition attribution: sampled pages that changed from nonresident to resident during decode. It is not a kernel perf page-fault event trace.",
                "",
                "| group | refaulted pages sampled | refaulted MB est | ratio |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['group']} | {int(float(row['refaulted_pages']))} | "
                f"{float(row['refaulted_mb_est']):.3f} | {float(row['refaulted_ratio']):.6f} |"
            )
        lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_svg(path: Path, samples: list[dict[str, Any]]) -> None:
    cell = 4
    gap = 36
    left = 58
    top = 36
    panel_w = EXPERTS * cell
    panel_h = LAYERS * cell
    width = left + panel_w + 20
    height = top + len(samples) * (panel_h + gap) + 20
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#374151}.label{font-size:13px;font-weight:700}</style>",
    ]
    for si, sample in enumerate(samples):
        y0 = top + si * (panel_h + gap)
        parts.append(f"<text class='label' x='12' y='{y0 - 10}'>{html.escape(sample['sample_label'])} resident={sample['resident_experts']}</text>")
        for layer in range(LAYERS):
            if layer % 5 == 0:
                parts.append(f"<text x='12' y='{y0 + layer * cell + cell}'>{layer}</text>")
            for expert in range(EXPERTS):
                fill = "#2563eb" if sample["matrix"][layer][expert] else "#e5e7eb"
                parts.append(f"<rect x='{left + expert * cell}' y='{y0 + layer * cell}' width='{cell}' height='{cell}' fill='{fill}'/>")
        for expert in range(0, EXPERTS, 16):
            parts.append(f"<text x='{left + expert * cell}' y='{y0 + panel_h + 14}'>{expert}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def run_prompt(
    *,
    prompt_index: int,
    prompt_name: str,
    prompt: str,
    args: argparse.Namespace,
    base_addr: int,
    expert_ranges: dict[tuple[int, int], list[ExpertRange]],
    tensors: list[TensorRange],
    maps: list[Mapping],
) -> None:
    out_dir = Path(args.output_dir) / f"prompt_{prompt_index:03d}_{Path(prompt_name).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(Path(args.output_dir) / "prompt_runs.jsonl", {"prompt_index": prompt_index, "prompt_name": prompt_name, "out_dir": str(out_dir)})

    erase_before = slot_erase(args.base_url, args.slot_id)
    idle_before = wait_slot_idle(args.base_url, args.slot_id)
    time.sleep(args.sleep_after_erase)

    perf = PerfRecorder(pid=args.pid, out_dir=out_dir) if args.perf else None
    perf_started = False
    samples: list[dict[str, Any]] = []
    tensor_watches = [TensorWatch(tensor=t) for t in tensors if t.n_bytes > 0]
    tensor_samples: list[dict[str, Any]] = []
    t0 = time.time()

    def take_sample(label: str) -> None:
        matrix, rows = sample_matrix(
            base_addr=base_addr,
            expert_ranges=expert_ranges,
            page_stride=args.page_stride,
            threshold=args.resident_threshold,
        )
        samples.append(
            {
                "sample_index": len(samples),
                "sample_label": label,
                "prompt_index": prompt_index,
                "prompt_name": prompt_name,
                "ts_s": time.time(),
                "elapsed_s": time.time() - t0,
                "resident_experts": sum(1 for row in matrix for value in row if value),
                "matrix": matrix,
                "rows": rows,
            }
        )
        sample_meta = samples[-1]
        for watch in tensor_watches:
            tensor_row = sample_tensor_watch(base_addr, watch, args.page_stride)
            tensor_samples.append(
                {
                    "sample_index": sample_meta["sample_index"],
                    "sample_label": sample_meta["sample_label"],
                    "prompt_index": prompt_index,
                    "prompt_name": prompt_name,
                    **tensor_row,
                }
            )

    def start_decode_observation(boundary: str) -> None:
        nonlocal perf_started
        if samples:
            return
        if perf is not None:
            perf_started = perf.start()
            if not perf_started:
                write_perf_unavailable(out_dir, perf.error or "perf failed to start")
        take_sample(f"t0_before_decode_{boundary}")

    payload = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "stream": True,
        "return_progress": True,
        "cache_prompt": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed + prompt_index,
        "id_slot": args.slot_id,
    }
    req = urllib.request.Request(
        f"{args.base_url}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token_events = 0
    final_event: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=None) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            progress = event.get("prompt_progress")
            if not samples and isinstance(progress, dict):
                total = progress.get("total")
                processed = progress.get("processed")
                if isinstance(total, int) and isinstance(processed, int) and total > 0 and processed >= total:
                    start_decode_observation("prompt_progress")
            has_token = bool(event.get("content")) or bool(event.get("tokens"))
            if has_token:
                if not samples:
                    start_decode_observation("first_token_fallback")
                token_events += 1
                if token_events <= args.n_predict:
                    take_sample(f"token_{token_events}")
            if event.get("stop", False) or "timings" in event:
                final_event = event

    if perf is not None and perf_started:
        if perf.stop():
            attribute_perf_faults(out_dir, perf.text_path, maps, tensors)
        else:
            write_perf_unavailable(out_dir, perf.error or "perf failed after decode")

    erase_after = slot_erase(args.base_url, args.slot_id)
    idle_after = wait_slot_idle(args.base_url, args.slot_id)
    time.sleep(args.sleep_after_erase)

    write_refault_outputs(out_dir, tensor_samples)
    write_matrix_outputs(out_dir, samples, args.resident_threshold)
    timings = final_event.get("timings", {}) if isinstance(final_event, dict) else {}
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "prompt_index": prompt_index,
                "prompt_name": prompt_name,
                "n_predict": args.n_predict,
                "token_events": token_events,
                "cache_n": timings.get("cache_n") if isinstance(timings, dict) else None,
                "predicted_n": timings.get("predicted_n") if isinstance(timings, dict) else None,
                "erase_before": erase_before,
                "idle_before": idle_before,
                "erase_after": erase_after,
                "idle_after": idle_after,
                "perf_requested": bool(args.perf),
                "perf_started": perf_started,
                "timings": timings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prompt={prompt_name} token_events={token_events} samples={len(samples)} idle_after={idle_after}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="gemma4-26B.gguf")
    parser.add_argument("--tensor-ranges", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-predict", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resident-threshold", type=float, default=0.95)
    parser.add_argument("--page-stride", type=int, default=16)
    parser.add_argument("--sleep-after-erase", type=float, default=1.0)
    parser.add_argument("--perf", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_predict < 1:
        raise SystemExit("--n-predict must be >= 1")
    if args.page_stride < 1:
        raise SystemExit("--page-stride must be >= 1")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_tensor_ranges(Path(args.tensor_ranges))
    expert_ranges = build_expert_ranges(tensors)
    maps = load_maps(args.pid, args.model_name)
    (out_dir / "proc_maps_model.txt").write_text(
        "".join(f"{m.start:x}-{m.end:x} {m.file_offset:x} {m.path}\n" for m in maps),
        encoding="utf-8",
    )
    prompts = load_prompts(Path(args.prompt_dir), args.limit)
    if not prompts:
        raise SystemExit(f"no prompts found under {args.prompt_dir}")

    with Path(args.model).open("rb") as model_f:
        model_map = mmap.mmap(model_f.fileno(), 0, access=mmap.ACCESS_COPY)
        base_addr = ctypes.addressof(ctypes.c_char.from_buffer(model_map))
        try:
            for idx, (name, prompt) in enumerate(prompts):
                run_prompt(
                    prompt_index=idx,
                    prompt_name=name,
                    prompt=prompt,
                    args=args,
                    base_addr=base_addr,
                    expert_ranges=expert_ranges,
                    tensors=tensors,
                    maps=maps,
                )
        finally:
            model_map.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
