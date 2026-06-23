#!/usr/bin/env python3
"""
Download physical-intelligence/libero to the target filesystem.

~35 GB | 1,693 episodes | 273K frames | dual camera 256×256 @ 10 fps
LeRobot v2.1 format (per-episode Parquet + MP4)

This is the canonical dataset for π₀.₅ fine-tuning and is shared between
the storage I/O benchmark and the training benchmark.

Usage:
  python download_dataset.py --dest /mnt/scratch/lerobot-datasets
"""
import argparse
import os
import time
from pathlib import Path


def download(dest: Path, hf_token: str | None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    repo_id = "physical-intelligence/libero"
    out_dir = dest / "physical-intelligence--libero"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} → {out_dir}")
    t0 = time.perf_counter()

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        token=hf_token,
        ignore_patterns=["*.git*", "*.lock"],
    )

    elapsed = time.perf_counter() - t0
    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"\nDone. {size/1e9:.1f} GB in {elapsed:.0f}s ({size/elapsed/1e6:.1f} MB/s write)")
    print(f"Dataset ready at: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True,
                        help="Parent directory on the filesystem under test")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()
    download(Path(args.dest), args.hf_token)


if __name__ == "__main__":
    main()
