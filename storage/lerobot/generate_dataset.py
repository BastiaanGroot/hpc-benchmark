#!/usr/bin/env python3
"""
Generate a synthetic LeRobot v3-style dataset on the target filesystem.

Structure mirrors the real LeRobot v3 layout:
  <root>/
    meta/
      info.json
      stats.json
      episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet      (state + action columns)
    videos/<camera>/chunk-000/file-000.mp4

This is used by bench_lerobot.py to measure realistic I/O throughput.
"""
import argparse
import json
import os
import struct
import time
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Defaults that approximate a real LeRobot manipulation dataset
# ---------------------------------------------------------------------------
DEFAULT_CAMERAS = ["observation.images.top", "observation.images.wrist"]
DEFAULT_FPS = 30
DEFAULT_EPISODES_PER_FILE = 100   # episodes packed per .parquet / .mp4 shard
DEFAULT_STEPS_PER_EPISODE = 300   # ~10 s per episode at 30 fps
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_CHUNKS = 1                # number of chunk-NNN dirs


def make_parquet_shard(
    path: Path,
    n_episodes: int,
    steps_per_episode: int,
    state_dim: int = 14,
    action_dim: int = 14,
    rng: np.random.Generator = None,
) -> int:
    """Write one Parquet shard; returns bytes written."""
    if rng is None:
        rng = np.random.default_rng()

    total_steps = n_episodes * steps_per_episode
    arrays = {
        "episode_index": pa.array(
            np.repeat(np.arange(n_episodes), steps_per_episode), type=pa.int64()
        ),
        "frame_index": pa.array(
            np.tile(np.arange(steps_per_episode), n_episodes), type=pa.int64()
        ),
        "timestamp": pa.array(
            np.tile(
                np.arange(steps_per_episode, dtype=np.float32) / DEFAULT_FPS,
                n_episodes,
            ),
            type=pa.float32(),
        ),
    }
    for i in range(state_dim):
        arrays[f"observation.state_{i}"] = pa.array(
            rng.standard_normal(total_steps).astype(np.float32), type=pa.float32()
        )
    for i in range(action_dim):
        arrays[f"action_{i}"] = pa.array(
            rng.standard_normal(total_steps).astype(np.float32), type=pa.float32()
        )

    table = pa.table(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy", row_group_size=steps_per_episode)
    return path.stat().st_size


def make_video_shard(
    path: Path,
    n_episodes: int,
    steps_per_episode: int,
    height: int = DEFAULT_VIDEO_HEIGHT,
    width: int = DEFAULT_VIDEO_WIDTH,
    fps: int = DEFAULT_FPS,
    rng: np.random.Generator = None,
) -> int:
    """Write one MP4 shard containing n_episodes worth of frames; returns bytes written."""
    if rng is None:
        rng = np.random.default_rng()

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "ultrafast"}

    total_frames = n_episodes * steps_per_episode
    for _ in range(total_frames):
        frame_data = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame = frame.reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path.stat().st_size


def write_metadata(root: Path, n_episodes: int, steps_per_episode: int, cameras: list):
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "fps": DEFAULT_FPS,
        "total_episodes": n_episodes,
        "total_frames": n_episodes * steps_per_episode,
        "cameras": cameras,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "stats.json").write_text(json.dumps({"mean": 0.0, "std": 1.0}))

    ep_dir = meta / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_table = pa.table(
        {
            "episode_index": pa.array(np.arange(n_episodes), type=pa.int64()),
            "length": pa.array(
                np.full(n_episodes, steps_per_episode), type=pa.int64()
            ),
        }
    )
    pq.write_table(ep_table, ep_dir / "file-000.parquet")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LeRobot v3 dataset")
    parser.add_argument("root", help="Output directory (filesystem under test)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps-per-episode", type=int, default=DEFAULT_STEPS_PER_EPISODE)
    parser.add_argument("--episodes-per-file", type=int, default=DEFAULT_EPISODES_PER_FILE)
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    rng = np.random.default_rng(args.seed)

    n_files = (args.episodes + args.episodes_per_file - 1) // args.episodes_per_file
    total_bytes = 0
    t0 = time.perf_counter()

    print(f"Generating {args.episodes} episodes across {n_files} shard(s)...")
    print(f"Cameras: {args.cameras}")

    for file_idx in range(n_files):
        ep_start = file_idx * args.episodes_per_file
        ep_count = min(args.episodes_per_file, args.episodes - ep_start)
        chunk = "chunk-000"
        fname = f"file-{file_idx:03d}"

        # Parquet shard
        parquet_path = root / "data" / chunk / f"{fname}.parquet"
        nb = make_parquet_shard(
            parquet_path, ep_count, args.steps_per_episode, rng=rng
        )
        total_bytes += nb
        print(f"  [{file_idx+1}/{n_files}] parquet shard: {nb / 1e6:.1f} MB")

        # Video shards (one per camera)
        for cam in args.cameras:
            cam_safe = cam.replace(".", "_")
            video_path = root / "videos" / cam_safe / chunk / f"{fname}.mp4"
            nb = make_video_shard(
                video_path,
                ep_count,
                args.steps_per_episode,
                args.height,
                args.width,
                rng=rng,
            )
            total_bytes += nb
            print(f"  [{file_idx+1}/{n_files}] video shard ({cam_safe}): {nb / 1e6:.1f} MB")

    write_metadata(root, args.episodes, args.steps_per_episode, args.cameras)

    elapsed = time.perf_counter() - t0
    print(f"\nDone. Total: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s "
          f"({total_bytes / elapsed / 1e6:.1f} MB/s write)")


if __name__ == "__main__":
    main()
