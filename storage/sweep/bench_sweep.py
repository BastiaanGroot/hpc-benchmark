#!/usr/bin/env python3
"""
storage/sweep/bench_sweep.py
----------------------------
File-size sweep benchmark using fio (Flexible I/O Tester).

Runs sequential read and write jobs at block sizes from 64 KB to 1 GB,
with configurable worker concurrency and multiple passes per size.
Results are parsed from fio's JSON output and written as a summary JSON.

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

Requirements:
    fio >= 3.0  (install with: apt install fio  /  yum install fio)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BLOCK_SIZES = [
    ("64k",   "64 KB"),
    ("256k",  "256 KB"),
    ("1m",    "1 MB"),
    ("4m",    "4 MB"),
    ("16m",   "16 MB"),
    ("64m",   "64 MB"),
    ("256m",  "256 MB"),
    ("1g",    "1 GB"),
]

# File size written per worker per job. Large enough to avoid cache effects.
# fio will cap at the block size if file_size < bs, so we use 2× block size minimum.
FILE_SIZE_MULTIPLIER = 4  # file = block_size * this (ensures multiple I/Os per job)


def check_fio() -> str:
    fio = shutil.which("fio")
    if not fio:
        print("ERROR: fio not found. Install with: apt install fio  /  yum install fio",
              file=sys.stderr)
        sys.exit(1)
    out = subprocess.check_output([fio, "--version"], text=True).strip()
    print(f"  fio: {out}")
    return fio


def run_fio_job(fio: str, tmp_dir: Path, bs: str, label: str,
                workers: int, passes: int, rw: str) -> dict:
    """
    Run a single fio job (read or write) and return parsed bandwidth stats.

    rw: "write" or "read"
    Returns dict with median_mbs, min_mbs from across passes.
    """
    # Compute file size: at least 4× block size, minimum 64 MB to avoid trivial jobs
    bs_bytes = _parse_size(bs)
    file_size_bytes = max(bs_bytes * FILE_SIZE_MULTIPLIER, 64 * 1024 * 1024)
    file_size = f"{file_size_bytes // (1024 * 1024)}m" if file_size_bytes >= 1024 * 1024 else f"{file_size_bytes // 1024}k"

    bws = []
    for _ in range(passes):
        cmd = [
            fio,
            "--name", f"sweep_{rw}_{bs}",
            "--directory", str(tmp_dir),
            "--rw", rw,
            "--bs", bs,
            "--numjobs", str(workers),
            "--size", file_size,
            "--group_reporting",
            "--output-format", "json",
            "--ioengine", "psync",
            "--direct", "1",       # bypass page cache — no drop_caches needed
            "--iodepth", "1",
            "--fallocate", "none",
            "--end_fsync", "1" if rw == "write" else "0",
            "--unlink", "1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            # direct I/O not supported on this filesystem (e.g. NFS with rsize mismatch)
            # retry without --direct and its value
            filtered = []
            skip_next = False
            for c in cmd:
                if skip_next:
                    skip_next = False
                    continue
                if c == "--direct":
                    skip_next = True
                    continue
                filtered.append(c)
            result = subprocess.run(filtered, capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)
        job = data["jobs"][0]
        key = "write" if rw == "write" else "read"
        bw_bytes = job[key]["bw_bytes"]   # bytes/sec
        bws.append(bw_bytes)

    bws_mbs = [b / 1e6 for b in bws]
    return {
        "median_mbs": round(sorted(bws_mbs)[len(bws_mbs) // 2], 1),
        "min_mbs": round(min(bws_mbs), 1),
    }


def _parse_size(s: str) -> int:
    """Parse fio size string like '4m', '256k', '1g' to bytes."""
    s = s.lower()
    if s.endswith("g"):
        return int(s[:-1]) * 1024 * 1024 * 1024
    if s.endswith("m"):
        return int(s[:-1]) * 1024 * 1024
    if s.endswith("k"):
        return int(s[:-1]) * 1024
    return int(s)


def main():
    parser = argparse.ArgumentParser(description="fio file-size sweep benchmark")
    parser.add_argument("--storage-path", default=os.environ.get("STORAGE_PATH", "/tmp"),
                        help="Directory on the filesystem to test (default: $STORAGE_PATH or /tmp)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel fio jobs (simulates DataLoader num_workers)")
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

    fio = check_fio()

    tmp_dir = storage_path / "sweep_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = args.max_size_mb * 1024 * 1024
    sizes = [(bs, label) for bs, label in BLOCK_SIZES if _parse_size(bs) <= max_bytes]

    print(f"\n{'='*62}")
    print(f"  File-size sweep — {storage_path}")
    print(f"  workers={args.workers}  passes={args.passes}  direct_io=1")
    print(f"{'='*62}")
    print(f"\n  LeRobot shard-size context:")
    print(f"    64 KB – 1 MB   episode metadata / Parquet row groups")
    print(f"    1 MB  – 16 MB  small-to-default Parquet shards")
    print(f"    16 MB – 64 MB  large Parquet shards / MP4 episode chunks")
    print(f"    64 MB – 1 GB   long video shards / checkpoint fragments")
    print()
    print(f"  {'Size':<10}  {'Write MB/s':>12}  {'Read MB/s':>12}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}")

    results = []
    for bs, label in sizes:
        w = run_fio_job(fio, tmp_dir, bs, label, args.workers, args.passes, "write")
        r = run_fio_job(fio, tmp_dir, bs, label, args.workers, args.passes, "read")
        row = {
            "block_size_label": label,
            "block_size": bs,
            "workers": args.workers,
            "passes": args.passes,
            "write_bw_median_mbs": w["median_mbs"],
            "write_bw_min_mbs": w["min_mbs"],
            "read_bw_median_mbs": r["median_mbs"],
            "read_bw_min_mbs": r["min_mbs"],
        }
        results.append(row)
        print(f"  {label:<10}  {w['median_mbs']:>12.1f}  {r['median_mbs']:>12.1f}")

    print()

    # Clean up temp dir
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

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
