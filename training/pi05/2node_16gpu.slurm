#!/bin/bash
#SBATCH --job-name=pi05-bench-2n16g
#SBATCH --output=results/pi05_2n16g_%j.out
#SBATCH --error=results/pi05_2n16g_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --exclusive

# π₀.₅ Training Benchmark — 2 nodes × 8 GPUs = 16 GPUs (FSDP)
#
# Same as 1node_8gpu.slurm but scales across 2 nodes.
# The scaling ratio (samples/s × 2 vs 1 node) is a key cluster quality metric.
#
# Required env:
#   OPENPI_DIR      : path to openpi checkout (from install.sh)
#   STORAGE_PATH    : filesystem under test
#
# Optional env:
#   DATASET_DIR, BATCH_SIZE, HF_TOKEN  (same as 1node script)

set -euo pipefail
BENCH_DIR="${BENCH_DIR:-${HOME}/git/hpc-benchmark}"

OPENPI_DIR="${OPENPI_DIR:-${HOME}/openpi}"
STORAGE_PATH="${STORAGE_PATH:?Set STORAGE_PATH to the filesystem mount under test}"
HF_DATASET="${HF_DATASET:-physical-intelligence/libero}"
DATASET_DIR="${DATASET_DIR:-${STORAGE_PATH}/lerobot-datasets/${HF_DATASET/\//--}}"
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Warning: ${DATASET_DIR} not found. Falling back to synthetic data."
  DATASET_DIR="${STORAGE_PATH}/lerobot-bench-data"
fi
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULTS_DIR="${RESULTS_DIR:-results}"

source "${OPENPI_DIR}/.venv/bin/activate"

[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

# Slurm provides SLURM_NODELIST; pick the first node as master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29500

mkdir -p "${RESULTS_DIR}"

echo "=== π₀.₅ Benchmark: 2 nodes × 8 GPUs ==="
echo "openpi commit: 15a9616a00943ada6c20a0f158e3adb39df2ccac"
echo "Dataset:       ${DATASET_DIR}"
echo "Batch/GPU:     ${BATCH_SIZE}  (global: $((BATCH_SIZE * 16)))"
echo "Nodes:         $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Started:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------
# NCCL / InfiniBand tuning
# Most of these are safe defaults for HDR/NDR IB. Override via env if needed.
# ---------------------------------------------------------------------------
# IB is enabled; set NCCL_IB_DISABLE=1 to fall back to TCP (for debugging only)
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

# GPUDirect RDMA: level 5 = use GDR when possible (requires nv_peer_mem / gdrcopy)
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"

# Enable DMA-BUF for GDR on Blackwell / newer H100 systems (no-op if unsupported)
export NCCL_IB_GDR_DMABUF_ENABLE="${NCCL_IB_GDR_DMABUF_ENABLE:-1}"

# Set to your IB HCA name if auto-detect picks the wrong one, e.g. "mlx5_0:1"
# export NCCL_IB_HCA=mlx5_0:1

# Prevent NCCL from using the management Ethernet interface for data traffic
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,^docker,^virbr}"

# Tree algorithm is faster for large all-reduces (>64MB); ring is better for small
export NCCL_ALGO="${NCCL_ALGO:-Tree}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

srun torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --node-rank="${SLURM_NODEID}" \
  "${BENCH_DIR}/training/pi05/bench_train.py" \
    --config pi05_base \
    --data-dir "${DATASET_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --warmup-steps 20 \
    --measure-steps 80 \
    --json-out "${RESULTS_DIR}/pi05_2n16g_${SLURM_JOB_ID}.json"

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
