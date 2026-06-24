#!/usr/bin/env python3
"""
π₀.₅ Training Throughput Benchmark

Directly instantiates PI0Pytorch with synthetic batches (matching LIBERO format)
and runs DDP forward+backward+optimizer steps. Reports:
  - steps/s, samples/s (steps/s × global_batch_size)
  - peak GPU memory per rank
  - MFU (%) — auto-detected from GPU model name

Usage (via torchrun):
  torchrun --standalone --nnodes=1 --nproc_per_node=8 bench_train.py \
      --warmup-steps 20 --measure-steps 80 \
      [--batch-size 32] [--json-out results.json]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

# cuDNN SDPA frontend may lack execution plans on very new GPUs (e.g. B300 sm_103);
# fall back to math/flash-attn paths which work via CUDA kernels.
torch.backends.cuda.enable_cudnn_sdp(False)

# ---------------------------------------------------------------------------
# GPU peak FLOPS table (BF16, dense)
# ---------------------------------------------------------------------------
_GPU_FLOPS_TABLE = {
    r"H100.*SXM":  989e12,
    r"H100.*NVL":  835e12,
    r"H100.*PCIe": 756e12,
    r"H200.*SXM": 1979e12,
    r"B200":       4500e12,
    r"B300":       9000e12,
    r"A100.*80":    624e12,
    r"A100.*40":    312e12,
}


def _detect_gpu_peak_flops() -> tuple[float, str]:
    if not torch.cuda.is_available():
        return 989e12, "unknown"
    name = torch.cuda.get_device_name(0)
    env_override = os.environ.get("GPU_PEAK_FLOPS")
    if env_override:
        return float(env_override), name
    for pattern, flops in _GPU_FLOPS_TABLE.items():
        if re.search(pattern, name, re.IGNORECASE):
            return flops, name
    return 989e12, name


def _setup_openpi(openpi_dir: Path):
    src = str(openpi_dir / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _make_synthetic_batch(batch_size: int, device: torch.device):
    """Return (observation, actions) matching LIBERO/pi05 shapes."""
    from openpi.models.model import Observation  # noqa: PLC0415

    B = batch_size
    # Three cameras matching openpi's IMAGE_KEYS, 256×256 RGB in [-1, 1] float32
    images = {
        "base_0_rgb":        torch.rand(B, 3, 256, 256, dtype=torch.float32, device=device) * 2 - 1,
        "left_wrist_0_rgb":  torch.rand(B, 3, 256, 256, dtype=torch.float32, device=device) * 2 - 1,
        "right_wrist_0_rgb": torch.rand(B, 3, 256, 256, dtype=torch.float32, device=device) * 2 - 1,
    }
    image_masks = {k: torch.ones(B, dtype=torch.bool, device=device) for k in images}
    state = torch.zeros(B, 32, dtype=torch.float32, device=device)
    tokenized_prompt = torch.zeros(B, 200, dtype=torch.int64, device=device)
    tokenized_prompt_mask = torch.ones(B, 200, dtype=torch.bool, device=device)
    obs = Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    actions = torch.zeros(B, 10, 32, dtype=torch.float32, device=device)
    return obs, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-steps",  type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=80)
    parser.add_argument("--batch-size",    type=int, default=32,
                        help="Per-GPU batch size")
    parser.add_argument("--data-dir",      type=str, default=None,
                        help="(Unused — benchmark uses synthetic data)")
    parser.add_argument("--config",        type=str, default="pi05_base",
                        help="(Unused — config is fixed for LIBERO/pi05)")
    parser.add_argument("--json-out",      type=str, default=None)
    parser.add_argument("--flops-per-step", type=float, default=None)
    args = parser.parse_args()

    # --- distributed setup ---------------------------------------------------
    is_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank       = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    if is_main:
        print(f"=== π₀.₅ Training Benchmark ===", flush=True)
        print(f"Config:        pi05_libero (synthetic data)", flush=True)
        print(f"World size:    {world_size}", flush=True)
        print(f"Per-GPU batch: {args.batch_size}", flush=True)
        print(f"Global batch:  {args.batch_size * world_size}", flush=True)
        print(f"Warmup/Measure:{args.warmup_steps}/{args.measure_steps}", flush=True)

    # --- openpi imports -------------------------------------------------------
    openpi_dir = Path(os.environ.get("OPENPI_DIR", Path.home() / "openpi"))
    _setup_openpi(openpi_dir)

    try:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch   # noqa: PLC0415
        from openpi.models.pi0_config import Pi0Config              # noqa: PLC0415
    except ImportError as e:
        sys.exit(f"openpi not found. Run training/pi05/install.sh first.\n{e}")

    # --- model ---------------------------------------------------------------
    model_cfg = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    # Match the real training script: move to device, leave mixed-precision as-is
    # (PaliGemma/Gemma weights are bfloat16 via precision= arg; projection layers stay float32)
    model = PI0Pytorch(model_cfg).to(device=device)
    total_params = sum(p.numel() for p in model.parameters())

    if is_dist:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if is_main:
        print(f"Model params:  {total_params/1e9:.1f}B", flush=True)

    # --- optimizer -----------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- warmup --------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)
    if is_main:
        print("Warming up...", flush=True)

    for _ in range(args.warmup_steps):
        obs, actions = _make_synthetic_batch(args.batch_size, device)
        loss = model(obs, actions).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if is_dist:
        dist.barrier()
    torch.cuda.synchronize(device)

    # --- measure -------------------------------------------------------------
    if is_main:
        print("Measuring...", flush=True)

    t0 = time.perf_counter()
    for _ in range(args.measure_steps):
        obs, actions = _make_synthetic_batch(args.batch_size, device)
        loss = model(obs, actions).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if is_dist:
        dist.barrier()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    # --- metrics -------------------------------------------------------------
    steps_per_sec  = args.measure_steps / elapsed
    samples_per_sec = steps_per_sec * args.batch_size * world_size
    peak_mem_gb    = torch.cuda.max_memory_allocated(device) / 1e9

    gpu_peak_flops, gpu_name = _detect_gpu_peak_flops()
    mfu = None
    if args.flops_per_step:
        flops_per_sec = args.flops_per_step * steps_per_sec
        mfu = flops_per_sec / (gpu_peak_flops * world_size) * 100

    if is_main:
        print("\n=== Results ===", flush=True)
        print(f"GPU:           {gpu_name}", flush=True)
        print(f"Elapsed:       {elapsed:.1f}s over {args.measure_steps} steps", flush=True)
        print(f"Throughput:    {steps_per_sec:.2f} steps/s  |  {samples_per_sec:.1f} samples/s", flush=True)
        print(f"Peak GPU mem:  {peak_mem_gb:.1f} GB / rank", flush=True)
        if mfu is not None:
            print(f"MFU:           {mfu:.1f}%", flush=True)

        result = {
            "schema": ["pi05_train", "1node" if world_size <= 8 else "2node"],
            "gpu": gpu_name,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "elapsed_s": elapsed,
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": samples_per_sec,
            "peak_mem_gb": peak_mem_gb,
            "mfu_pct": mfu,
        }
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(result, indent=2))
            print(f"\nResults written to {args.json_out}", flush=True)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
