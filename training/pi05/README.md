# π₀.₅ Training Benchmark

Measures end-to-end training throughput for π₀.₅ fine-tuning with FSDP.
Uses [openpi](https://github.com/Physical-Intelligence/openpi) pinned to commit
[`15a9616`](https://github.com/Physical-Intelligence/openpi/commit/15a9616a00943ada6c20a0f158e3adb39df2ccac)
(Jun 16 2026).

## Why this benchmark

- Covers the full stack: filesystem read → PyTorch DataLoader → PaliGemma backbone → flow-matching action head → FSDP all-gather/reduce-scatter
- Complements the storage benchmarks: if `samples/s` here is lower than the storage benchmark suggests it should be, the bottleneck is compute or NCCL, not storage
- Scaling efficiency (1-node → 2-node ratio) is a direct measure of interconnect quality

## Metrics

| Metric | Description |
|--------|-------------|
| `steps/s` | Training steps per second |
| `samples/s` | Frames processed per second (`steps/s × global_batch_size`) |
| `peak GPU mem (GB)` | Per-GPU peak memory allocation |
| `MFU (%)` | Model FLOP Utilization — set `GPU_PEAK_FLOPS` env var for your GPU |

## Quick start

```bash
# 1. Install openpi (once per cluster)
bash training/pi05/install.sh

# 2. Generate a LeRobot-format dataset on your storage (reuses storage benchmark data)
export STORAGE_PATH=/mnt/scratch        # any mounted filesystem
bash storage/lerobot/run.slurm   # or set DATASET_DIR to an existing dataset

# 3. Export your HuggingFace token (needed to download lerobot/pi05_base weights)
export HF_TOKEN=hf_...

# 4. Run benchmark
export OPENPI_DIR=${HOME}/openpi

sbatch training/pi05/1node_8gpu.slurm    # 1 × 8 GPU
sbatch training/pi05/2node_16gpu.slurm   # 2 × 8 GPU — measure scaling efficiency
```

## Pinned dependency

```
openpi @ git+https://github.com/Physical-Intelligence/openpi.git@15a9616a00943ada6c20a0f158e3adb39df2ccac
```

To update to a newer commit, change `OPENPI_COMMIT` in `install.sh` and re-run it.

## Tuning notes

- `--batch-size 32` (per GPU) is the default from the LeRobot π₀.₅ docs; reduce to 16 if OOM
- `GPU_PEAK_FLOPS`: H100 SXM5 ≈ `989e12`, A100 SXM4 ≈ `312e12` (BF16 with sparsity)
- NCCL multi-node: set `NCCL_IB_HCA` to your InfiniBand HCA name if auto-detect fails
