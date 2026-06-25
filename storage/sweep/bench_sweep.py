#!/usr/bin/env python3
"""
storage/sweep/bench_sweep.py
----------------------------
File-size sweep benchmark for shared filesystems.

Tests sequential read and write throughput at block sizes from 64 KB to 1 GB,
using configurable worker concurrency and multiple passes per size.

Interpreting results in the context of LeRobot datasets
--------------------------------------------------------
LeRobot v3 datasets consist of:

  • Parquet shards  — episode metadata (states, actions, timestamps).
    Typical shard size: 1–20 MB (configurable via --shard-size when pushing).
    Access pattern: sequential within a shard; random across shards.

  • MP4 video shards — camera observations, compressed H.264/H.265.
    Typical file size: 5–100 MB per episode chunk.
    Access pattern: random seek to an I-frame, then sequential decode.

  • episodes/data/  — one Parquet row group per episode, ≈ 10–100 KB each.
    These are SMALL files — many thousands of them.

Rule of thumb for shard-size selection:
  • If throughput is flat from 1 MB onwards → shard size barely matters;
    pick 50–100 MB for convenience.
  • If throughput rises steeply up to X MB then plateaus → set shards ≥ X.
  • If small-file (<1 MB) throughput is very low → avoid tiny shards; the
    filesystem has high metadata/open overhead (common on NFS, Lustre with
    small stripe counts).

Usage:
    python3 bench_sweep.py [--storage-path /mnt/scratch] [--workers 1] [--passes 3]
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Block sizes to sweep (bytes)
BLOCK_SIZES = [
    64 * 1024,           #  64 KB  — small Parquet row groups / episode metadata
    256 * 1024,          # 256 KB
    1 * 1024 * 1024,     #   1 MB  — small Parquet shard
    4 * 1024 * 1024,     #   4 MB
    16 * 1024 * 1024,    #  16 MB  — typical Parquet shard
    64 * 1024 * 1024,    #  64 MB  — large Parquet / small MP4 shard
    256 * 1024 * 1024,   # 256 MB  — large MP4 shard / checkpoint fragment
    1024 * 1024 * 1024,  #   1 GB  — large checkpoint shard
]

BLOCK_LABELS = [
    "64 KB", "256 KB", "1 MB", "4 MB", "16 MB", "64 MB", "256 MB", "1 GB"
]


def _human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}/s"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB/s"


def _write_file(path: Path, size: int) -> float:
    """Write `size` bytes to `path` using O_DIRECT-like large writes. Returns elapsed seconds."""
    data = os.urandom(min(size, 4 * 1024 * 1024))  # 4 MB chunk, reuse
    written = 0
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        while written < size:
            chunk = data[: min(len(data), size - written)]
            f.write(chunk)
            written += len(chunk)
        f.flush()
        os.fsync(f.fileno())
    return time.perf_counter() - t0


def _read_file(path: Path, size: int) -> float:
    """Read `size` bytes from `path` sequentially. Returns elapsed seconds."""
    read = 0
    chunk_size = min(size, 4 * 1024 * 1024)
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        while read < size:
            buf = f.read(chunk_size)
            if not buf:
                break
            read += len(buf)
    return time.perf_counter() - t0


def run_single(storage_path: Path, size: int, label: str, passes: int, workers: int) -> dict:
    """Run write then read sweep for a single block size. Returns result dict."""
    tmp_dir = storage_path / "sweep_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    write_bws, read_bws = [], []

    for p in range(passes):
        # ---- WRITE ----
        paths = [tmp_dir / f"sweep_{label.replace(' ', '')}_{w}.bin" for w in range(workers)]
        t0 = time.perf_counter()
        if workers == 1:
            _write_file(paths[0], size)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_write_file, p, size) for p in paths]
                for f in as_completed(futs):
                    f.result()
        elapsed_w = time.perf_counter() - t0
        write_bws.append(size * workers / elapsed_w)

        # Drop OS page cache for the written files (best-effort; no sudo needed for /proc)
        try:
            with open("/proc/sys/vm/drop_caches", "w") as dc:
                dc.write("3\n")
        except OSError:
            pass  # non-root: cache may be warm; note in output

        # ---- READ ----
        t0 = time.perf_counter()
        if workers == 1:
            _read_file(paths[0], size)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_read_file, p, size) for p in paths]
                for f in as_completed(futs):
                    f.result()
        elapsed_r = time.perf_counter() - t0
        read_bws.append(size * workers / elapsed_r)

        for p in paths:
            p.unlink(missing_ok=True)

    return {
        "block_size_bytes": size,
        "block_size_label": label,
        "workers": workers,
        "passes": passes,
        "write_bw_median_mbs": round(statistics.median(write_bws) / 1e6, 1),
        "write_bw_min_mbs": round(min(write_bws) / 1e6, 1),
        "read_bw_median_mbs": round(statistics.median(read_bws) / 1e6, 1),
        "read_bw_min_mbs": round(min(read_bws) / 1e6, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="File-size sweep storage benchmark")
    parser.add_argument("--storage-path", default=os.environ.get("STORAGE_PATH", "/tmp"),
                        help="Directory on the filesystem to test (default: $STORAGE_PATH or /tmp)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (simulates DataLoader num_workers)")
    parser.add_argument("--passes", type=int, default=3,
                        help="Passes per block size (median is reported)")
    parser.add_argument("--max-size-mb", type=int, default=1024,
                        help="Skip block sizes above this many MB (default: 1024)")
    parser.add_argument("--output", default=None,
                        help="Write JSON results to this path")
    args = parser.parse_args()

    storage_path = Path(args.storage_path)
    if not storage_path.exists():
        print(f"ERROR: storage path does not exist: {storage_path}", file=sys.stderr)
        sys.exit(1)

    max_bytes = args.max_size_mb * 1024 * 1024
    sizes = [(s, l) for s, l in zip(BLOCK_SIZES, BLOCK_LABELS) if s <= max_bytes]

    print(f"\n{'='*60}")
    print(f"  File-size sweep — {storage_path}")
    print(f"  workers={args.workers}  passes={args.passes}")
    print(f"{'='*60}")
    print(f"\n  LeRobot context:")
    print(f"    64 KB–1 MB  → episode metadata / Parquet row groups")
    print(f"    1 MB–16 MB  → small Parquet shards (default LeRobot config)")
    print(f"    16 MB–64 MB → typical Parquet shards / video frame chunks")
    print(f"    64 MB–1 GB  → large video shards / model checkpoint fragments")
    print()
    print(f"  {'Size':<10}  {'Write MB/s':>12}  {'Read MB/s':>12}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}")

    results = []
    for size, label in sizes:
        r = run_single(storage_path, size, label, args.passes, args.workers)
        results.append(r)
        print(f"  {label:<10}  {r['write_bw_median_mbs']:>12.1f}  {r['read_bw_median_mbs']:>12.1f}")

    print()

    print()

    output = {
        "storage_path": str(storage_path),
        "workers": args.workers,
        "passes": args.passes,
        "sweep": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results written to {args.output}")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
