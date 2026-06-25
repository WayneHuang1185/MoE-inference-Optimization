#!/usr/bin/env python3
"""Monitor prompt1000 Global RPP pipeline progress.

This is read-only. It inspects files under dataset/prompt1000 plus current
process/container state so long-running stages can be checked without ad-hoc
shell commands.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": repr(exc)}


def csv_status_counts(path: Path, status_col: str = "status") -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = row.get(status_col, "") or "unknown"
                counts[key] = counts.get(key, 0) + 1
    except Exception as exc:
        counts[f"read_error:{type(exc).__name__}"] = 1
    return counts


def tail(path: Path, n: int) -> list[str]:
    if not path.exists() or n <= 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception as exc:
        return [f"<tail failed: {exc!r}>"]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def proc_cmdline(pid: str) -> str:
    raw = read_text(Path("/proc") / pid / "cmdline")
    return raw.replace("\x00", " ").strip()


def proc_status(pid: str) -> dict[str, str]:
    out: dict[str, str] = {}
    text = read_text(Path("/proc") / pid / "status")
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def proc_stat(pid: str) -> dict[str, Any]:
    text = read_text(Path("/proc") / pid / "stat")
    if not text:
        return {}
    try:
        after = text.rsplit(") ", 1)[1].split()
        return {
            "state": after[0],
            "utime": int(after[11]),
            "stime": int(after[12]),
            "starttime": int(after[19]),
        }
    except Exception:
        return {}


def clock_ticks() -> int:
    try:
        return int(__import__("os").sysconf("SC_CLK_TCK"))
    except Exception:
        return 100


def host_uptime_s() -> float:
    text = read_text(Path("/proc/uptime")).split()
    try:
        return float(text[0])
    except Exception:
        return 0.0


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def process_rows(patterns: list[str], limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ticks = clock_ticks()
    uptime = host_uptime_s()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        cmd = proc_cmdline(p.name)
        if not cmd:
            continue
        if not any(pat in cmd for pat in patterns):
            continue
        st = proc_stat(p.name)
        status = proc_status(p.name)
        elapsed = uptime - (float(st.get("starttime", 0)) / ticks)
        cpu_time = (float(st.get("utime", 0)) + float(st.get("stime", 0))) / ticks
        pcpu = (cpu_time / elapsed * 100.0) if elapsed > 0 else 0.0
        rss_kb = int((status.get("VmRSS", "0 kB").split() or ["0"])[0])
        rows.append({
            "pid": int(p.name),
            "state": st.get("state", "?"),
            "elapsed": format_elapsed(elapsed),
            "pcpu": pcpu,
            "rss_mb": rss_kb / 1024.0,
            "cmd": cmd,
        })
    rows.sort(key=lambda r: r["pcpu"], reverse=True)
    return rows[:limit]


def system_summary() -> str:
    load = read_text(Path("/proc/loadavg")).strip()
    meminfo = {}
    for line in read_text(Path("/proc/meminfo")).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        parts = v.split()
        if parts:
            meminfo[k] = int(parts[0])
    total = meminfo.get("MemTotal", 0) / 1024 / 1024
    avail = meminfo.get("MemAvailable", 0) / 1024 / 1024
    free = meminfo.get("MemFree", 0) / 1024 / 1024
    swap_total = meminfo.get("SwapTotal", 0) / 1024 / 1024
    swap_free = meminfo.get("SwapFree", 0) / 1024 / 1024
    return (
        f"loadavg: {load}\n"
        f"mem GiB: total={total:.1f} free={free:.1f} available={avail:.1f}\n"
        f"swap GiB: total={swap_total:.1f} free={swap_free:.1f}"
    )


def pct(done: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{done / total * 100:.1f}%"


def print_section(name: str) -> None:
    print(f"\n## {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("dataset/prompt1000"))
    p.add_argument("--tail", type=int, default=20)
    p.add_argument("--watch", type=int, default=0,
                   help="refresh interval seconds; 0 prints once")
    return p.parse_args()


def render(args: argparse.Namespace) -> None:
    root = args.root
    db = root / "prompt_database.jsonl"
    metadata = read_json(root / "metadata.json")
    total_prompts = int(metadata.get("records") or count_jsonl(db))

    gen_dir = root / "generations"
    gen_manifest = gen_dir / "generations_manifest.jsonl"
    generated = count_jsonl(gen_manifest)

    label_dir = root / "router_label_npz"
    dump_pack_manifest = label_dir / "dump_pack_manifest.csv"
    label_summary = read_json(label_dir / "dataset_summary.json")
    npz_count = len(list((label_dir / "npz").glob("*.npz"))) if (label_dir / "npz").exists() else 0

    print(f"prompt1000 monitor @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"root: {root}")

    print_section("Database")
    print(f"prompts: {total_prompts}")
    if metadata.get("database_sha256"):
        print(f"sha256: {metadata['database_sha256']}")

    print_section("Generation")
    print(f"manifest rows: {generated} / {total_prompts} ({pct(generated, total_prompts)})")
    print(f"completions files: {len(list((gen_dir / 'completions').glob('*.json'))) if (gen_dir / 'completions').exists() else 0}")
    print(f"status counts: {csv_status_counts(gen_manifest) if gen_manifest.suffix == '.csv' else 'jsonl manifest'}")
    if gen_manifest.exists():
        last = tail(gen_manifest, min(args.tail, 5))
        print("last generation rows:")
        for line in last:
            print(f"  {line[:240]}")

    print_section("Router Labels")
    status_counts = csv_status_counts(dump_pack_manifest)
    print(f"dump-pack rows: {sum(status_counts.values()) if status_counts else 0}")
    print(f"dump-pack status: {status_counts}")
    print(f"npz files: {npz_count}")
    if label_summary:
        print(f"summary: {json.dumps(label_summary, ensure_ascii=False, sort_keys=True)}")

    print_section("Processes")
    rows = process_rows(["rpp_prompt1000_pipeline.py", "llama-server", "dump_moe_routing", "podman run"], limit=20)
    if not rows:
        print("<none>")
    else:
        print("PID STATE ELAPSED %CPU RSS_MB CMD")
        for row in rows:
            print(
                f"{row['pid']} {row['state']} {row['elapsed']} "
                f"{row['pcpu']:.1f} {row['rss_mb']:.1f} {row['cmd'][:180]}"
            )

    print_section("Containers")
    container_rows = process_rows(["/usr/bin/tini --", "conmon", "pasta --config-net"], limit=20)
    if not container_rows:
        print("<container runtime listing unavailable from this monitor container>")
    else:
        for row in container_rows:
            print(f"{row['pid']} {row['state']} {row['elapsed']} {row['cmd'][:220]}")

    print_section("System")
    print(system_summary())

    if args.tail > 0:
        print_section("Recent Logs")
        for path in [
            gen_dir / "server_generate.log",
            label_dir / "server_dump_pack_labels.log",
            label_dir / "REPORT.md",
        ]:
            if path.exists():
                print(f"\n### {path}")
                for line in tail(path, args.tail):
                    print(line)


def main() -> int:
    args = parse_args()
    while True:
        render(args)
        if args.watch <= 0:
            return 0
        print("\n" + "=" * 80 + "\n", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
