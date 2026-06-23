#!/usr/bin/env python3
"""
π₀.₅ FSDP Checkpoint Save/Restore Benchmark

Measures the wall-clock time and throughput for:
  1. SAVE: writing a sharded FSDP checkpoint to the target filesystem
  2. LOAD: restoring that checkpoint from the filesystem

π₀.₅ (PaliGemma 3B backbone + action head) produces roughly:
  - ~12 GB total checkpoint size
  - N shards when saved with FSDP (one shard per rank by default)

This is important because:
  - Slow saves stall training (checkpoint frequency directly affects wasted work
    in the event of a node failure)
  - Slow restores hurt cluster restart time after preemption

Metrics reported per rank and aggregated:
  - Write GB/s   (parallel write from all ranks simultaneously)
  - Read GB/s    (parallel read back to all ranks)
  - Total checkpoint size (GB)

Usage (called via torchrun by bench_checkpoint.slurm):
  torchrun [...] bench_checkpoint.py \
      --checkpoint-dir /path/on/filesystem/under/test \
      [--model-size-gb 12.0] \
      [--n-rounds 3] \
      [--json-out /path/to/results.json]
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.checkpoint import (
    save as dcp_save,
    load as dcp_load,
    FileSystemWriter,
    FileSystemReader,
)


# ---------------------------------------------------------------------------
# Synthetic model that approximates π₀.₅ parameter count
# PaliGemma 3B + action head ≈ 3.3B params in BF16 ≈ 6.6 GB per full copy
# With optimizer states (Adam): ~3× = ~20 GB; checkpoint = model only ≈ 12 GB
# ---------------------------------------------------------------------------

class SyntheticPi05(nn.Module):
    """Approximate π₀.₅ parameter count without actual architecture."""

    def __init__(self, target_params: int = 3_300_000_000):
        super().__init__()
        # Single large weight tensor to approximate total param count
        # Split into blocks to avoid single 6GB allocation
        block_size = 100_000_000  # 100M params per block
        n_blocks = target_params // block_size
        self.blocks = nn.ModuleList(
            [nn.Linear(10_000, 10_000, bias=False) for _ in range(n_blocks)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _reduce_sum(val: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return val
    t = torch.tensor(val, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# ---------------------------------------------------------------------------
# Save / load with DCP (DistributedCheckpointing — the FSDP-native approach)
# ---------------------------------------------------------------------------

def _save_checkpoint(model: FSDP, ckpt_dir: Path) -> tuple[float, float]:
    """Save sharded checkpoint. Returns (elapsed_s, bytes_written)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _barrier()
    t0 = time.perf_counter()
    dcp_save({"model": model}, storage_writer=FileSystemWriter(str(ckpt_dir)))
    torch.cuda.synchronize()
    _barrier()
    elapsed = time.perf_counter() - t0

    # Sum shard sizes across all ranks
    shard_bytes = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
    total_bytes = _reduce_sum(float(shard_bytes))
    return elapsed, total_bytes


def _load_checkpoint(model: FSDP, ckpt_dir: Path) -> tuple[float, float]:
    """Load sharded checkpoint. Returns (elapsed_s, bytes_read)."""
    total_bytes = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
    total_bytes = _reduce_sum(float(total_bytes))

    _barrier()
    t0 = time.perf_counter()
    dcp_load({"model": model}, storage_reader=FileSystemReader(str(ckpt_dir)))
    torch.cuda.synchronize()
    _barrier()
    elapsed = time.perf_counter() - t0
    return elapsed, total_bytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="π₀.₅ FSDP checkpoint benchmark")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory on the filesystem under test")
    parser.add_argument("--model-size-gb", type=float, default=None,
                        help="Override synthetic model size in GB (default: auto)")
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="Number of save+load rounds to average (default 3)")
    parser.add_argument("--json-out", help="Write JSON results to this path")
    args = parser.parse_args()

    rank, local_rank, world_size = _setup_distributed()
    is_main = rank == 0

    # Scale parameter count so total checkpoint ≈ 12 GB (π₀.₅ realistic size)
    # BF16: 2 bytes/param → 12 GB = 6e9 params across all shards
    # With world_size shards each rank holds 6e9 / world_size params
    if args.model_size_gb:
        total_params = int(args.model_size_gb * 1e9 / 2)
    else:
        total_params = 6_000_000_000  # 12 GB total in BF16
    params_per_rank = total_params // world_size

    if is_main:
        print(f"=== π₀.₅ Checkpoint Benchmark ===")
        print(f"Checkpoint dir:  {args.checkpoint_dir}")
        print(f"World size:      {world_size}")
        print(f"Total params:    {total_params / 1e9:.1f}B  "
              f"(≈ {total_params * 2 / 1e9:.1f} GB in BF16)")
        print(f"Rounds:          {args.n_rounds}")
        print()

    # Build and wrap synthetic model with FSDP
    device = torch.device(f"cuda:{local_rank}")
    model = SyntheticPi05(target_params=params_per_rank).to(device=device, dtype=torch.bfloat16)
    if world_size > 1:
        model = FSDP(model, device_id=local_rank)

    ckpt_dir = Path(args.checkpoint_dir) / f"pi05_bench_ckpt_ws{world_size}"

    save_results, load_results = [], []

    for rnd in range(args.n_rounds):
        # --- SAVE ---
        save_elapsed, total_bytes = _save_checkpoint(model, ckpt_dir)
        save_gb_s = total_bytes / 1e9 / save_elapsed if save_elapsed > 0 else 0.0
        save_results.append(save_gb_s)

        # --- LOAD ---
        load_elapsed, _ = _load_checkpoint(model, ckpt_dir)
        load_gb_s = total_bytes / 1e9 / load_elapsed if load_elapsed > 0 else 0.0
        load_results.append(load_gb_s)

        if is_main:
            print(f"  Round {rnd+1}: save {save_gb_s:.2f} GB/s  "
                  f"({total_bytes/1e9:.1f} GB in {save_elapsed:.1f}s)  |  "
                  f"load {load_gb_s:.2f} GB/s  ({load_elapsed:.1f}s)")

    avg_save = sum(save_results) / len(save_results)
    avg_load = sum(load_results) / len(load_results)
    ckpt_size_gb = total_bytes / 1e9

    results = {
        "world_size": world_size,
        "checkpoint_size_gb": round(ckpt_size_gb, 2),
        "save_gb_s": round(avg_save, 2),
        "load_gb_s": round(avg_load, 2),
        "save_rounds": save_results,
        "load_rounds": load_results,
    }

    if is_main:
        print(f"\n=== Results (avg over {args.n_rounds} rounds) ===")
        print(f"  Checkpoint size: {ckpt_size_gb:.1f} GB")
        print(f"  Save:            {avg_save:.2f} GB/s")
        print(f"  Load:            {avg_load:.2f} GB/s")

        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(results, indent=2))
            print(f"\nResults written to {args.json_out}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
