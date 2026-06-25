#!/usr/bin/env python3
import os
import glob
import time
import json
import argparse
import urllib.request

CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


def parse_stat_file(path):
    """
    Parse /proc/<pid>/task/<tid>/stat.

    Important fields:
      field 10 = minflt
      field 12 = majflt
      field 14 = utime
      field 15 = stime
      field 42 = delayacct_blkio_ticks
    """
    s = open(path, "r").read()
    rparen = s.rfind(")")
    rest = s[rparen + 2:].split()

    minflt = int(rest[7])
    majflt = int(rest[9])
    utime = int(rest[11])
    stime = int(rest[12])

    # field 42, 1-based, after removing pid+comm:
    # rest index = 42 - 3 = 39
    blkio_ticks = int(rest[39]) if len(rest) > 39 else 0

    return {
        "minflt": minflt,
        "majflt": majflt,
        "utime": utime,
        "stime": stime,
        "blkio_ticks": blkio_ticks,
    }


def read_task_stats(pid):
    total = {
        "minflt": 0,
        "majflt": 0,
        "utime": 0,
        "stime": 0,
        "blkio_ticks": 0,
        "threads": 0,
    }

    for stat_path in glob.glob(f"/proc/{pid}/task/*/stat"):
        try:
            st = parse_stat_file(stat_path)
        except FileNotFoundError:
            continue

        for k in ["minflt", "majflt", "utime", "stime", "blkio_ticks"]:
            total[k] += st[k]
        total["threads"] += 1

    return total


def read_proc_io(pid):
    path = f"/proc/{pid}/io"
    result = {}

    try:
        with open(path, "r") as f:
            for line in f:
                key, value = line.strip().split(":")
                result[key.strip()] = int(value.strip())
    except FileNotFoundError:
        pass

    return {
        "read_bytes": result.get("read_bytes", 0),
        "write_bytes": result.get("write_bytes", 0),
        "rchar": result.get("rchar", 0),
        "wchar": result.get("wchar", 0),
        "syscr": result.get("syscr", 0),
        "syscw": result.get("syscw", 0),
    }


def snapshot(pid):
    st = read_task_stats(pid)
    io = read_proc_io(pid)

    return {
        **st,
        **io,
    }


def diff(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after.keys()}


def send_request(url, prompt, n_predict):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=None) as resp:
        body = resp.read()
    t1 = time.perf_counter()

    return t1 - t0, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--url", default="http://127.0.0.1:8080/completion")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--n-predict", type=int, default=32)
    args = parser.parse_args()

    before = snapshot(args.pid)

    wall_s, body = send_request(args.url, args.prompt, args.n_predict)

    after = snapshot(args.pid)
    d = diff(after, before)

    cpu_ticks = d["utime"] + d["stime"]
    cpu_s = cpu_ticks / CLK_TCK
    user_s = d["utime"] / CLK_TCK
    sys_s = d["stime"] / CLK_TCK
    io_delay_s = d["blkio_ticks"] / CLK_TCK

    accounted = cpu_s + io_delay_s
    compute_share = cpu_s / accounted if accounted > 0 else 0.0
    io_share = io_delay_s / accounted if accounted > 0 else 0.0

    print("===== Request Timing =====")
    print(f"wall_time_s              = {wall_s:.6f}")

    print("\n===== CPU / Computation =====")
    print(f"user_cpu_s               = {user_s:.6f}")
    print(f"system_cpu_s             = {sys_s:.6f}")
    print(f"total_cpu_s              = {cpu_s:.6f}")
    print(f"cpu_parallelism          = {cpu_s / wall_s:.3f}x")

    print("\n===== I/O / Page Fault =====")
    print(f"block_io_delay_s         = {io_delay_s:.6f}")
    print(f"minor_faults_delta       = {d['minflt']}")
    print(f"major_faults_delta       = {d['majflt']}")
    print(f"read_bytes_delta         = {d['read_bytes']}")
    print(f"read_MB_delta            = {d['read_bytes'] / (1024 ** 2):.3f}")
    print(f"read_syscalls_delta      = {d['syscr']}")

    print("\n===== Resource-time Share =====")
    print(f"compute_share_cpu_vs_io  = {compute_share * 100:.2f}%")
    print(f"io_share_cpu_vs_io       = {io_share * 100:.2f}%")

    print("\n===== Notes =====")
    print("CPU time is summed across all llama-server threads.")
    print("Therefore total_cpu_s can be larger than wall_time_s.")
    print("block_io_delay_s depends on kernel task delay accounting support.")
    print("major_faults_delta is the key signal for mmap demand paging.")


if __name__ == "__main__":
    main()
