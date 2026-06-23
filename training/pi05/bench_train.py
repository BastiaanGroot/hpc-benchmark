#!/usr/bin/env python3
"""
π₀.₅ Training Throughput Benchmark

Wraps the openpi training loop, runs WARMUP_STEPS + MEASURE_STEPS steps,
then reports:
  - steps/s
  - samples/s  (steps/s × global_batch_size)
  - peak GPU memory (per rank)
  - MFU (%)  — auto-detected from GPU model name, overridable via GPU_PEAK_FLOPS

Supported GPUs for auto MFU:
  H100 SXM5/NVL, H200 SXM, B200 SXM, B300 SXM (BF16 dense TFLOPS)

Usage (called by the Slurm scripts via torchrun):
  torchrun [...] bench_train.py \
      --config pi05_base \
      --data-dir /path/to/lerobot-dataset \
      --warmup-steps 20 \
      --measure-steps 80 \
      [--flops-per-step 3.8e15] \
      [--json-out /path/to/results.json]
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# GPU peak FLOPS table (BF16, dense, no structured sparsity)
# These are the conservative/realistic values used for MFU estimation.
# ---------------------------------------------------------------------------
_GPU_FLOPS_TABLE = {
    r"H100.*SXM":  989e12,   # H100 SXM5 — 989 TFLOPS with sparsity; ~494 dense
    r"H100.*NVL":  835e12,
    r"H100.*PCIe": 756e12,
    r"H200.*SXM": 1979e12,   # H200 SXM — ~2× H100
    r"B200":       4500e12,  # Blackwell B200 SXM
    r"B300":       9000e12,  # Blackwell B300 SXM (estimated)
    r"A100.*80":    624e12,
    r"A100.*40":    312e12,
}


def _detect_gpu_peak_flops() -> tuple[float, str]:
    """Return (peak_flops, gpu_name) from the current CUDA device."""
    if not torch.cuda.is_available():
        return 989e12, "unknown"
    name = torch.cuda.get_device_name(0)
    env_override = os.environ.get("GPU_PEAK_FLOPS")
    if env_override:
        return float(env_override), name
    for pattern, flops in _GPU_FLOPS_TABLE.items():
        if re.search(pattern, name, re.IGNORECASE):
            return flops, name
    # Fallback: H100-class assumption
    return 989e12, name


# ---------------------------------------------------------------------------
# Minimal openpi imports — the real training state / loop lives in openpi
# ---------------------------------------------------------------------------
def _import_openpi_train():
    try:
        from openpi.training import train as openpi_train
        return openpi_train
    except ImportError as e:
        raise SystemExit(
            f"openpi not found. Run training/pi05/install.sh first.\n{e}"
        ) from e


# ---------------------------------------------------------------------------
# Throughput measurement hook
# ---------------------------------------------------------------------------

class ThroughputMeter:
    def __init__(self, warmup: int, measure: int):
        self.warmup = warmup
        self.measure = measure
        self._step = 0
        self._t0 = None
        self.done = False
        self.elapsed = None

    def step(self):
        self._step += 1
        if self._step == self.warmup + 1:
            torch.cuda.synchronize()
            self._t0 = time.perf_counter()
        if self._step == self.warmup + self.measure:
            torch.cuda.synchronize()
            self.elapsed = time.perf_counter() - self._t0
            self.done = True

    @property
    def steps_done(self):
        return max(0, self._step - self.warmup)

    def steps_per_sec(self):
        if self.elapsed and self.elapsed > 0:
            return self.measure / self.elapsed
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_base",
                        help="openpi training config name (default: pi05_base)")
    parser.add_argument("--data-dir", required=True,
                        help="Path to a LeRobot-format dataset directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-GPU batch size (default 32)")
    parser.add_argument("--warmup-steps", type=int, default=20,
                        help="Steps to discard before measuring (default 20)")
    parser.add_argument("--measure-steps", type=int, default=80,
                        help="Steps to measure (default 80)")
    parser.add_argument("--flops-per-step", type=float, default=None,
                        help="Theoretical FLOPs per step for MFU calculation")
    parser.add_argument("--json-out", help="Write JSON results to this path")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    if is_main:
        print(f"=== π₀.₅ Training Benchmark ===")
        print(f"Config:        {args.config}")
        print(f"Data dir:      {args.data_dir}")
        print(f"World size:    {world_size}")
        print(f"Per-GPU batch: {args.batch_size}")
        print(f"Global batch:  {args.batch_size * world_size}")
        print(f"Warmup/Measure:{args.warmup_steps}/{args.measure_steps}")

    openpi_train = _import_openpi_train()
    meter = ThroughputMeter(args.warmup_steps, args.measure_steps)
    total_steps = args.warmup_steps + args.measure_steps

    # openpi's run() accepts a step_callback that is called after every step.
    # We inject our meter there and raise StopIteration once we have enough data.
    def _step_cb(step: int, _metrics: dict):
        meter.step()
        if meter.done:
            raise StopIteration("benchmark complete")

    try:
        openpi_train.run(
            config_name=args.config,
            overrides=[
                f"data.data_dir={args.data_dir}",
                f"training.batch_size={args.batch_size}",
                f"training.max_steps={total_steps}",
                # FSDP is enabled by default in openpi's multi-GPU configs;
                # set explicitly for clarity.
                "training.fsdp=true",
                "training.compile=true",
                "training.grad_checkpoint=true",
            ],
            step_callback=_step_cb,
        )
    except StopIteration:
        pass

    # --- collect results ---
    torch.cuda.synchronize()
    peak_mem_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9

    sps = meter.steps_per_sec() or 0.0
    global_batch = args.batch_size * world_size
    samples_per_sec = sps * global_batch

    gpu_peak_flops, gpu_name = _detect_gpu_peak_flops()
    mfu = None
    if args.flops_per_step and sps:
        mfu = (args.flops_per_step * sps) / (gpu_peak_flops * world_size) * 100

    results = {
        "config": args.config,
        "gpu_name": gpu_name,
        "gpu_peak_flops_tflops": round(gpu_peak_flops / 1e12, 1),
        "world_size": world_size,
        "per_gpu_batch_size": args.batch_size,
        "global_batch_size": global_batch,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "steps_per_sec": round(sps, 3),
        "samples_per_sec": round(samples_per_sec, 1),
        "peak_gpu_mem_gb": round(peak_mem_gb, 2),
        "mfu_pct": round(mfu, 2) if mfu is not None else None,
    }

    if is_main:
        print("\n=== Results ===")
        print(f"  GPU:              {gpu_name} ({gpu_peak_flops/1e12:.0f} TFLOPS BF16 peak)")
        print(f"  Steps/s:          {results['steps_per_sec']:.3f}")
        print(f"  Samples/s:        {results['samples_per_sec']:.1f}")
        print(f"  Peak GPU mem:     {results['peak_gpu_mem_gb']:.2f} GB")
        if mfu:
            print(f"  MFU:              {results['mfu_pct']:.1f}%")

        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(results, indent=2))
            print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
