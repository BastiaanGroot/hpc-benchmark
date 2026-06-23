#!/usr/bin/env python3
"""
LeRobot I/O Benchmark — measures storage throughput for LeRobot v3 data loading.

Simulates the read pattern of a PyTorch DataLoader loading episodes from a
LeRobot v3 dataset:
  1. Read episode metadata from meta/episodes Parquet
  2. For each batch: read a row-group slice from a data Parquet shard
  3. For each video camera: seek + decode N consecutive frames from an MP4 shard

Metrics reported:
  - samples/s       (frames delivered per second)
  - MB/s read       (raw bytes read per second)
  - avg latency/batch (ms)

Usage:
  python bench_lerobot.py <dataset_root> [options]
"""
import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_shards(root: Path, subdir: str) -> list[Path]:
    base = root / subdir
    if not base.exists():
        return []
    return sorted(base.rglob("*.parquet")) + sorted(base.rglob("*.mp4"))


def _discover_cameras(root: Path) -> list[Path]:
    videos_dir = root / "videos"
    if not videos_dir.exists():
        return []
    # <camera>/chunk-NNN/file-NNN.mp4
    return sorted({p.parent.parent for p in videos_dir.rglob("*.mp4")})


def _read_parquet_batch(
    parquet_paths: list[Path], batch_size: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Read a random row-group from a random Parquet shard.
    Returns (rows_read, bytes_read)."""
    path = rng.choice(parquet_paths)
    pf = pq.ParquetFile(path)
    n_rg = pf.num_row_groups
    rg = int(rng.integers(0, n_rg))
    table = pf.read_row_group(rg)
    rows = len(table)
    # Approximate bytes: sum of buffer sizes
    raw_bytes = sum(
        buf.size
        for col in table.columns
        for chunk in col.chunks
        for buf in chunk.buffers()
        if buf is not None
    )
    return rows, raw_bytes


def _decode_video_frames(
    mp4_paths: list[Path], n_frames: int, rng: np.random.Generator
) -> tuple[int, int]:
    """Seek to a random offset and decode n_frames from a random MP4 shard.
    Returns (frames_decoded, bytes_read)."""
    path = rng.choice(mp4_paths)
    file_bytes = path.stat().st_size

    container = av.open(str(path))
    stream = container.streams.video[0]
    total_frames = stream.frames or 0

    frames_decoded = 0
    if total_frames > n_frames:
        # Seek to a random start
        target_frame = int(rng.integers(0, total_frames - n_frames))
        seek_pts = int(target_frame * stream.duration / (total_frames or 1))
        container.seek(seek_pts, stream=stream)

    for frame in container.decode(stream):
        # Force decode to numpy (simulates what DataLoader does)
        _ = frame.to_ndarray(format="rgb24")
        frames_decoded += 1
        if frames_decoded >= n_frames:
            break
    container.close()
    return frames_decoded, file_bytes  # bytes_read approximated as full file size / segments


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    root: Path,
    batch_size: int,
    n_batches: int,
    n_workers: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    # Discover shards
    parquet_paths = sorted((root / "data").rglob("*.parquet")) if (root / "data").exists() else []
    camera_dirs = _discover_cameras(root)
    all_mp4: dict[str, list[Path]] = {}
    for cam_dir in camera_dirs:
        mp4s = sorted(cam_dir.rglob("*.mp4"))
        if mp4s:
            all_mp4[cam_dir.name] = mp4s

    if not parquet_paths:
        raise FileNotFoundError(f"No Parquet shards found under {root}/data/")
    if not all_mp4:
        print("Warning: no MP4 shards found; running Parquet-only benchmark.")

    print(f"Dataset root:   {root}")
    print(f"Parquet shards: {len(parquet_paths)}")
    print(f"Video cameras:  {list(all_mp4.keys())}")
    print(f"Batch size:     {batch_size} frames")
    print(f"Batches:        {n_batches}")
    print(f"Workers:        {n_workers}")
    print()

    total_frames = 0
    total_bytes = 0
    latencies = []

    def _load_one_batch(_):
        t0 = time.perf_counter()
        frames, parquet_bytes = _read_parquet_batch(parquet_paths, batch_size, rng)
        video_bytes = 0
        for cam, mp4s in all_mp4.items():
            _f, vb = _decode_video_frames(mp4s, batch_size, rng)
            video_bytes += vb
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return frames, parquet_bytes + video_bytes, elapsed_ms

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for frames, nbytes, lat_ms in pool.map(_load_one_batch, range(n_batches)):
            total_frames += frames
            total_bytes += nbytes
            latencies.append(lat_ms)
    elapsed = time.perf_counter() - t_wall

    return {
        "total_frames": total_frames,
        "total_bytes_mb": total_bytes / 1e6,
        "elapsed_s": elapsed,
        "samples_per_sec": total_frames / elapsed,
        "throughput_mb_s": total_bytes / 1e6 / elapsed,
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description="LeRobot v3 I/O benchmark")
    parser.add_argument("root", help="Dataset root directory (output of generate_dataset.py)")
    parser.add_argument("--batch-size", type=int, default=48,
                        help="Frames per batch (default 48)")
    parser.add_argument("--n-batches", type=int, default=200,
                        help="Number of batches to load (default 200)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel DataLoader workers (default 8)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", help="Write results to this JSON file")
    args = parser.parse_args()

    results = run_benchmark(
        root=Path(args.root),
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        n_workers=args.workers,
        seed=args.seed,
    )

    print("=== Results ===")
    print(f"  Samples/s:          {results['samples_per_sec']:.1f}")
    print(f"  Throughput:         {results['throughput_mb_s']:.1f} MB/s")
    print(f"  Avg batch latency:  {results['avg_latency_ms']:.1f} ms")
    print(f"  p50 latency:        {results['p50_latency_ms']:.1f} ms")
    print(f"  p99 latency:        {results['p99_latency_ms']:.1f} ms")
    print(f"  Total read:         {results['total_bytes_mb']:.0f} MB in {results['elapsed_s']:.1f}s")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
