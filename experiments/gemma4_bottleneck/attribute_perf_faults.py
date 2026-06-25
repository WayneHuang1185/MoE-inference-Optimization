#!/usr/bin/env python3
"""Attribute perf page-fault addresses to GGUF tensor ranges.

This script expects:
  1. A /proc/<pid>/maps snapshot while llama-server is running.
  2. A tensor range CSV from gguf_tensor_ranges.py.
  3. A perf trace/script text file that contains fault addresses as hex values.

It is intentionally tolerant about perf text format. It extracts all hex
addresses from each line and attributes the first address that falls inside a
mapping for the model file.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


HEX_RE = re.compile(r"0x[0-9a-fA-F]+")


def load_maps(path: Path, model_name: str):
    mappings = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6:
            continue
        addr_range, perms, offset_hex, _dev, _inode, pathname = parts
        if model_name not in pathname:
            continue
        start_s, end_s = addr_range.split("-")
        mappings.append({
            "start": int(start_s, 16),
            "end": int(end_s, 16),
            "file_offset": int(offset_hex, 16),
            "path": pathname,
        })
    mappings.sort(key=lambda m: m["start"])
    return mappings


def load_tensors(path: Path):
    rows = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        row["file_start"] = int(row["file_start"])
        row["file_end"] = int(row["file_end"])
        row["n_bytes"] = int(row["n_bytes"])
        rows.append(row)
    rows.sort(key=lambda r: r["file_start"])
    starts = [r["file_start"] for r in rows]
    return rows, starts


def address_to_file_offset(addr: int, mappings):
    for m in mappings:
        if m["start"] <= addr < m["end"]:
            return m["file_offset"] + (addr - m["start"])
    return None


def find_tensor(file_offset: int, tensors, starts):
    idx = bisect.bisect_right(starts, file_offset) - 1
    if idx < 0:
        return None
    row = tensors[idx]
    if row["file_start"] <= file_offset < row["file_end"]:
        return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", required=True)
    parser.add_argument("--tensor-ranges", required=True)
    parser.add_argument("--perf-text", required=True)
    parser.add_argument("--model-name", default="gemma4-26B.gguf")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mappings = load_maps(Path(args.maps), args.model_name)
    if not mappings:
        raise SystemExit(f"no mappings found for model name: {args.model_name}")
    tensors, starts = load_tensors(Path(args.tensor_ranges))

    counts: Counter[str] = Counter()
    bytes_by_tensor: defaultdict[str, int] = defaultdict(int)
    meta = {}
    unattributed = 0
    matched_fault_lines = 0

    for line in Path(args.perf_text).read_text(encoding="utf-8", errors="replace").splitlines():
        attributed = False
        for token in HEX_RE.findall(line):
            addr = int(token, 16)
            file_offset = address_to_file_offset(addr, mappings)
            if file_offset is None:
                continue
            tensor = find_tensor(file_offset, tensors, starts)
            if tensor is None:
                continue
            name = tensor["name"]
            counts[name] += 1
            bytes_by_tensor[name] += 4096
            meta[name] = tensor
            attributed = True
            matched_fault_lines += 1
            break
        if not attributed and HEX_RE.search(line):
            unattributed += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tensor", "layer", "family", "type", "faults",
                "estimated_fault_mb", "tensor_mb", "file_start", "file_end",
            ],
        )
        writer.writeheader()
        for name, faults in counts.most_common():
            row = meta[name]
            writer.writerow({
                "tensor": name,
                "layer": row["layer"],
                "family": row["family"],
                "type": row["type"],
                "faults": faults,
                "estimated_fault_mb": bytes_by_tensor[name] / (1024 ** 2),
                "tensor_mb": int(row["n_bytes"]) / (1024 ** 2),
                "file_start": row["file_start"],
                "file_end": row["file_end"],
            })
    print(f"wrote {args.output}")
    print(f"matched_fault_lines={matched_fault_lines}")
    print(f"unattributed_hex_lines={unattributed}")


if __name__ == "__main__":
    main()
