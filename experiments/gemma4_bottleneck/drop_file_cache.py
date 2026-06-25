#!/usr/bin/env python3
"""Ask Linux to evict one file from the host page cache.

This uses posix_fadvise(POSIX_FADV_DONTNEED), so it does not require root.
It only drops clean cached pages that are not actively in use.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


POSIX_FADV_DONTNEED = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop a file's clean pages from the Linux page cache.")
    parser.add_argument("path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
    libc.posix_fadvise.restype = ctypes.c_int

    fd = os.open(path, os.O_RDONLY)
    try:
        rc = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        if rc != 0:
            raise OSError(rc, os.strerror(rc))
    finally:
        os.close(fd)

    print(f"requested page-cache eviction for {path}")


if __name__ == "__main__":
    main()
