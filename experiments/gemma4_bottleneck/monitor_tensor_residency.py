#!/usr/bin/env python3
"""Monitor GGUF tensor page residency through /proc/<pid>/pagemap.

For GGUF mmap weights, Linux usually evicts clean file-backed pages instead of
writing them to swap. In pagemap terms those pages become non-present, not
swapped. This monitor records both states:

- present: currently resident in RAM for the target process.
- nonresident: not present; a later access will fault it back from the model
  file or page cache.
- swapped: pagemap's swapped bit is set. This is uncommon for clean model mmap
  pages, but the column is kept so the same output is useful when private dirty
  mappings are involved.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import mmap
import os
import re
import signal
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
PRESENT_BIT = 1 << 63
SWAPPED_BIT = 1 << 62
STATE_NONRESIDENT = 0
STATE_PRESENT = 1
STATE_SWAPPED = 2
LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
LIBC.mincore.restype = ctypes.c_int


@dataclass(frozen=True)
class Mapping:
    start: int
    end: int
    file_offset: int
    path: str

    @property
    def file_end(self) -> int:
        return self.file_offset + (self.end - self.start)


@dataclass(frozen=True)
class TensorRange:
    name: str
    layer: str
    family: str
    tensor_type: str
    n_bytes: int
    file_start: int
    file_end: int


@dataclass(frozen=True)
class VirtualRun:
    vpage_start: int
    pages: int


@dataclass
class TensorWatch:
    tensor: TensorRange
    runs: list[VirtualRun]
    pages: int
    previous: bytearray | None = None
    total_evicted_pages: int = 0
    total_refaulted_pages: int = 0
    total_swapout_pages: int = 0
    total_swapin_pages: int = 0
    min_present_pages: int | None = None
    max_present_pages: int = 0
    max_swapped_pages: int = 0
    max_nonresident_pages: int = 0


@dataclass
class SampleRow:
    sample: int
    ts_s: float
    elapsed_s: float
    tensor: str
    layer: str
    family: str
    tensor_type: str
    pages: int
    present_pages: int
    swapped_pages: int
    nonresident_pages: int
    evicted_pages: int
    refaulted_pages: int
    swapout_pages: int
    swapin_pages: int


stop_requested = False


def request_stop(_signum: int, _frame) -> None:
    global stop_requested
    stop_requested = True


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
        mappings.append(
            Mapping(
                start=int(start_s, 16),
                end=int(end_s, 16),
                file_offset=int(offset_hex, 16),
                path=pathname,
            )
        )
    mappings.sort(key=lambda item: item.file_offset)
    return mappings


def load_tensors(path: Path, include_regex: str | None) -> list[TensorRange]:
    pattern = re.compile(include_regex) if include_regex else None
    tensors: list[TensorRange] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            if pattern and not pattern.search(name):
                continue
            tensors.append(
                TensorRange(
                    name=name,
                    layer=row.get("layer", ""),
                    family=row.get("family", ""),
                    tensor_type=row.get("type", ""),
                    n_bytes=int(row["n_bytes"]),
                    file_start=int(row["file_start"]),
                    file_end=int(row["file_end"]),
                )
            )
    tensors.sort(key=lambda item: item.file_start)
    return tensors


def build_watches(tensors: list[TensorRange], mappings: list[Mapping], stride: int) -> list[TensorWatch]:
    starts = [m.file_offset for m in mappings]
    watches: list[TensorWatch] = []
    for tensor in tensors:
        runs: list[VirtualRun] = []
        idx = max(0, bisect.bisect_right(starts, tensor.file_start) - 1)
        while idx < len(mappings):
            mapping = mappings[idx]
            if mapping.file_offset >= tensor.file_end:
                break
            overlap_start = max(tensor.file_start, mapping.file_offset)
            overlap_end = min(tensor.file_end, mapping.file_end)
            if overlap_start < overlap_end:
                vaddr_start = mapping.start + (overlap_start - mapping.file_offset)
                vaddr_end = mapping.start + (overlap_end - mapping.file_offset)
                first_page = vaddr_start // PAGE_SIZE
                last_page = (vaddr_end - 1) // PAGE_SIZE
                page_count = last_page - first_page + 1
                if stride <= 1:
                    runs.append(VirtualRun(first_page, page_count))
                else:
                    for page in range(first_page, first_page + page_count, stride):
                        runs.append(VirtualRun(page, 1))
            idx += 1
        pages = sum(run.pages for run in runs)
        if pages > 0:
            watches.append(TensorWatch(tensor=tensor, runs=runs, pages=pages))
    return watches


def build_file_watches(tensors: list[TensorRange], stride: int) -> list[TensorWatch]:
    watches: list[TensorWatch] = []
    for tensor in tensors:
        first_page = tensor.file_start // PAGE_SIZE
        last_page = (tensor.file_end - 1) // PAGE_SIZE
        page_count = last_page - first_page + 1
        runs: list[VirtualRun] = []
        if stride <= 1:
            runs.append(VirtualRun(first_page, page_count))
        else:
            for page in range(first_page, first_page + page_count, stride):
                runs.append(VirtualRun(page, 1))
        watches.append(TensorWatch(tensor=tensor, runs=runs, pages=sum(run.pages for run in runs)))
    return watches


def state_from_entry(entry: int) -> int:
    if entry & PRESENT_BIT:
        return STATE_PRESENT
    if entry & SWAPPED_BIT:
        return STATE_SWAPPED
    return STATE_NONRESIDENT


def read_states(pagemap, runs: list[VirtualRun]) -> bytearray:
    states = bytearray()
    for run in runs:
        pagemap.seek(run.vpage_start * 8)
        data = pagemap.read(run.pages * 8)
        if len(data) != run.pages * 8:
            raise RuntimeError("short read from pagemap")
        states.extend(state_from_entry(entry[0]) for entry in struct.iter_unpack("<Q", data))
    return states


def read_file_states(base_addr: int, runs: list[VirtualRun]) -> bytearray:
    states = bytearray()
    for run in runs:
        vec = ctypes.create_string_buffer(run.pages)
        addr = base_addr + run.vpage_start * PAGE_SIZE
        length = run.pages * PAGE_SIZE
        if LIBC.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(length), vec) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        states.extend(STATE_PRESENT if byte & 1 else STATE_NONRESIDENT for byte in vec.raw)
    return states


def sample_watch(watch: TensorWatch, states: bytearray, sample: int, ts_s: float, elapsed_s: float) -> SampleRow:
    present = states.count(STATE_PRESENT)
    swapped = states.count(STATE_SWAPPED)
    nonresident = len(states) - present - swapped

    evicted = 0
    refaulted = 0
    swapout = 0
    swapin = 0
    if watch.previous is not None:
        for before, after in zip(watch.previous, states):
            if before == STATE_PRESENT and after != STATE_PRESENT:
                evicted += 1
            if before != STATE_PRESENT and after == STATE_PRESENT:
                refaulted += 1
            if before != STATE_SWAPPED and after == STATE_SWAPPED:
                swapout += 1
            if before == STATE_SWAPPED and after != STATE_SWAPPED:
                swapin += 1

    watch.previous = states
    watch.total_evicted_pages += evicted
    watch.total_refaulted_pages += refaulted
    watch.total_swapout_pages += swapout
    watch.total_swapin_pages += swapin
    watch.min_present_pages = present if watch.min_present_pages is None else min(watch.min_present_pages, present)
    watch.max_present_pages = max(watch.max_present_pages, present)
    watch.max_swapped_pages = max(watch.max_swapped_pages, swapped)
    watch.max_nonresident_pages = max(watch.max_nonresident_pages, nonresident)

    return SampleRow(
        sample=sample,
        ts_s=ts_s,
        elapsed_s=elapsed_s,
        tensor=watch.tensor.name,
        layer=watch.tensor.layer,
        family=watch.tensor.family,
        tensor_type=watch.tensor.tensor_type,
        pages=watch.pages,
        present_pages=present,
        swapped_pages=swapped,
        nonresident_pages=nonresident,
        evicted_pages=evicted,
        refaulted_pages=refaulted,
        swapout_pages=swapout,
        swapin_pages=swapin,
    )


def mb(pages: int, stride: int) -> float:
    return pages * stride * PAGE_SIZE / (1024 * 1024)


def row_to_csv(row: SampleRow, stride: int) -> dict[str, object]:
    return {
        **row.__dict__,
        "present_mb": mb(row.present_pages, stride),
        "swapped_mb": mb(row.swapped_pages, stride),
        "nonresident_mb": mb(row.nonresident_pages, stride),
        "evicted_mb": mb(row.evicted_pages, stride),
        "refaulted_mb": mb(row.refaulted_pages, stride),
        "swapout_mb": mb(row.swapout_pages, stride),
        "swapin_mb": mb(row.swapin_pages, stride),
    }


def write_summaries(out_dir: Path, watches: list[TensorWatch], stride: int) -> None:
    tensor_fields = [
        "tensor",
        "layer",
        "family",
        "type",
        "tensor_mb",
        "sampled_pages",
        "sampled_mb",
        "min_present_mb",
        "max_present_mb",
        "max_nonresident_mb",
        "max_swapped_mb",
        "total_evicted_mb",
        "total_refaulted_mb",
        "total_swapout_mb",
        "total_swapin_mb",
        "total_evicted_pages",
        "total_refaulted_pages",
        "total_swapout_pages",
        "total_swapin_pages",
    ]
    with (out_dir / "tensor_residency_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tensor_fields)
        writer.writeheader()
        for watch in sorted(watches, key=lambda item: item.total_refaulted_pages + item.total_evicted_pages, reverse=True):
            writer.writerow(
                {
                    "tensor": watch.tensor.name,
                    "layer": watch.tensor.layer,
                    "family": watch.tensor.family,
                    "type": watch.tensor.tensor_type,
                    "tensor_mb": watch.tensor.n_bytes / (1024 * 1024),
                    "sampled_pages": watch.pages,
                    "sampled_mb": mb(watch.pages, stride),
                    "min_present_mb": mb(watch.min_present_pages or 0, stride),
                    "max_present_mb": mb(watch.max_present_pages, stride),
                    "max_nonresident_mb": mb(watch.max_nonresident_pages, stride),
                    "max_swapped_mb": mb(watch.max_swapped_pages, stride),
                    "total_evicted_mb": mb(watch.total_evicted_pages, stride),
                    "total_refaulted_mb": mb(watch.total_refaulted_pages, stride),
                    "total_swapout_mb": mb(watch.total_swapout_pages, stride),
                    "total_swapin_mb": mb(watch.total_swapin_pages, stride),
                    "total_evicted_pages": watch.total_evicted_pages,
                    "total_refaulted_pages": watch.total_refaulted_pages,
                    "total_swapout_pages": watch.total_swapout_pages,
                    "total_swapin_pages": watch.total_swapin_pages,
                }
            )

    groups: dict[tuple[str, str, str], dict[str, float | int | str]] = {}
    for watch in watches:
        key = (watch.tensor.family, watch.tensor.layer, watch.tensor.tensor_type)
        item = groups.setdefault(
            key,
            {
                "family": watch.tensor.family,
                "layer": watch.tensor.layer,
                "type": watch.tensor.tensor_type,
                "tensors": 0,
                "tensor_mb": 0.0,
                "sampled_mb": 0.0,
                "total_evicted_mb": 0.0,
                "total_refaulted_mb": 0.0,
                "total_swapout_mb": 0.0,
                "total_swapin_mb": 0.0,
            },
        )
        item["tensors"] = int(item["tensors"]) + 1
        item["tensor_mb"] = float(item["tensor_mb"]) + watch.tensor.n_bytes / (1024 * 1024)
        item["sampled_mb"] = float(item["sampled_mb"]) + mb(watch.pages, stride)
        item["total_evicted_mb"] = float(item["total_evicted_mb"]) + mb(watch.total_evicted_pages, stride)
        item["total_refaulted_mb"] = float(item["total_refaulted_mb"]) + mb(watch.total_refaulted_pages, stride)
        item["total_swapout_mb"] = float(item["total_swapout_mb"]) + mb(watch.total_swapout_pages, stride)
        item["total_swapin_mb"] = float(item["total_swapin_mb"]) + mb(watch.total_swapin_pages, stride)

    group_fields = [
        "family",
        "layer",
        "type",
        "tensors",
        "tensor_mb",
        "sampled_mb",
        "total_evicted_mb",
        "total_refaulted_mb",
        "total_swapout_mb",
        "total_swapin_mb",
    ]
    with (out_dir / "family_layer_residency_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_fields)
        writer.writeheader()
        for item in sorted(groups.values(), key=lambda row: float(row["total_refaulted_mb"]) + float(row["total_evicted_mb"]), reverse=True):
            writer.writerow(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor mapped GGUF tensor page residency.")
    parser.add_argument("--pid", type=int, help="target llama-server PID for pagemap mode")
    parser.add_argument("--model", help="GGUF model path for file-cache mincore mode")
    parser.add_argument("--tensor-ranges", required=True, help="CSV from gguf_tensor_ranges.py")
    parser.add_argument("--model-name", default="gemma4-26B.gguf", help="substring used to find model mappings")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    parser.add_argument("--duration", type=float, help="optional maximum runtime in seconds")
    parser.add_argument("--max-samples", type=int, help="optional maximum number of samples")
    parser.add_argument("--page-stride", type=int, default=1, help="sample every Nth page; MB columns are scaled by N")
    parser.add_argument("--include-regex", help="only monitor tensor names matching this regex")
    parser.add_argument("--flush-every", type=int, default=1, help="flush output every N samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be > 0")
    if args.page_stride < 1:
        raise SystemExit("--page-stride must be >= 1")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_tensors(Path(args.tensor_ranges), args.include_regex)
    mappings: list[Mapping] = []
    if args.model:
        watches = build_file_watches(tensors, args.page_stride)
        mode = "file_mincore"
    else:
        if args.pid is None:
            raise SystemExit("either --model or --pid is required")
        mappings = load_maps(args.pid, args.model_name)
        if not mappings:
            raise SystemExit(f"no readable mappings for model name {args.model_name!r} in /proc/{args.pid}/maps")
        watches = build_watches(tensors, mappings, args.page_stride)
        mode = "process_pagemap"
    if not watches:
        raise SystemExit("no tensor pages overlapped the target model mappings")

    with (out_dir / "monitor_meta.txt").open("w", encoding="utf-8") as f:
        f.write(f"mode={mode}\n")
        f.write(f"page_size={PAGE_SIZE}\n")
        f.write(f"page_stride={args.page_stride}\n")
        if args.model:
            f.write(f"model={args.model}\n")
        if args.pid is not None:
            f.write(f"pid={args.pid}\n")

    if mappings:
        with (out_dir / "proc_maps_model.txt").open("w", encoding="utf-8") as f:
            for mapping in mappings:
                f.write(f"{mapping.start:x}-{mapping.end:x} {mapping.file_offset:x} {mapping.path}\n")

    fields = [
        "sample",
        "ts_s",
        "elapsed_s",
        "tensor",
        "layer",
        "family",
        "tensor_type",
        "pages",
        "present_pages",
        "swapped_pages",
        "nonresident_pages",
        "evicted_pages",
        "refaulted_pages",
        "swapout_pages",
        "swapin_pages",
        "present_mb",
        "swapped_mb",
        "nonresident_mb",
        "evicted_mb",
        "refaulted_mb",
        "swapout_mb",
        "swapin_mb",
    ]
    totals_fields = [
        "sample",
        "ts_s",
        "elapsed_s",
        "present_mb",
        "swapped_mb",
        "nonresident_mb",
        "evicted_mb",
        "refaulted_mb",
        "swapout_mb",
        "swapin_mb",
    ]

    start = time.time()
    sample_id = 0
    try:
        with (out_dir / "tensor_residency_samples.csv").open("w", encoding="utf-8", newline="") as samples_f, (
            out_dir / "tensor_residency_totals.csv"
        ).open("w", encoding="utf-8", newline="") as totals_f:
            sample_writer = csv.DictWriter(samples_f, fieldnames=fields)
            total_writer = csv.DictWriter(totals_f, fieldnames=totals_fields)
            sample_writer.writeheader()
            total_writer.writeheader()

            if args.model:
                model_f = open(args.model, "r+b")
                model_map = mmap.mmap(model_f.fileno(), 0, access=mmap.ACCESS_COPY)
                reader = lambda watch: read_file_states(
                    ctypes.addressof(ctypes.c_char.from_buffer(model_map)), watch.runs
                )
            else:
                pagemap = open(f"/proc/{args.pid}/pagemap", "rb", buffering=0)
                model_f = None
                model_map = None
                reader = lambda watch: read_states(pagemap, watch.runs)

            try:
                while not stop_requested:
                    ts_s = time.time()
                    elapsed_s = ts_s - start
                    rows: list[SampleRow] = []
                    for watch in watches:
                        states = reader(watch)
                        rows.append(sample_watch(watch, states, sample_id, ts_s, elapsed_s))

                    for row in rows:
                        sample_writer.writerow(row_to_csv(row, args.page_stride))

                    totals = defaultdict(int)
                    for row in rows:
                        totals["present_pages"] += row.present_pages
                        totals["swapped_pages"] += row.swapped_pages
                        totals["nonresident_pages"] += row.nonresident_pages
                        totals["evicted_pages"] += row.evicted_pages
                        totals["refaulted_pages"] += row.refaulted_pages
                        totals["swapout_pages"] += row.swapout_pages
                        totals["swapin_pages"] += row.swapin_pages
                    total_writer.writerow(
                        {
                            "sample": sample_id,
                            "ts_s": ts_s,
                            "elapsed_s": elapsed_s,
                            "present_mb": mb(totals["present_pages"], args.page_stride),
                            "swapped_mb": mb(totals["swapped_pages"], args.page_stride),
                            "nonresident_mb": mb(totals["nonresident_pages"], args.page_stride),
                            "evicted_mb": mb(totals["evicted_pages"], args.page_stride),
                            "refaulted_mb": mb(totals["refaulted_pages"], args.page_stride),
                            "swapout_mb": mb(totals["swapout_pages"], args.page_stride),
                            "swapin_mb": mb(totals["swapin_pages"], args.page_stride),
                        }
                    )

                    if args.flush_every > 0 and sample_id % args.flush_every == 0:
                        samples_f.flush()
                        totals_f.flush()

                    sample_id += 1
                    if args.max_samples is not None and sample_id >= args.max_samples:
                        break
                    if args.duration is not None and time.time() - start >= args.duration:
                        break
                    time.sleep(args.interval)
            finally:
                if model_map is not None:
                    model_map.close()
                if model_f is not None:
                    model_f.close()
                if not args.model:
                    pagemap.close()
    except FileNotFoundError:
        print(f"process exited before monitor completed: pid={args.pid}", file=sys.stderr)
    finally:
        write_summaries(out_dir, watches, args.page_stride)
        print(f"wrote tensor residency monitor outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
