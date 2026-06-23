#!/bin/bash
#SBATCH --job-name=pi05-ckpt-bench
#SBATCH --output=results/pi05_ckpt_%j.out
#SBATCH --error=results/pi05_ckpt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --exclusive

# π₀.₅ FSDP Checkpoint Save/Restore Benchmark
#
# Measures how fast the target filesystem can absorb and serve a sharded
# FSDP checkpoint (~12 GB for π₀.₅). Run with --nodes=1 and --nodes=2
# to compare single-node vs multi-node checkpoint throughput.
#
# Required env:
#   STORAGE_PATH   : path to a mounted filesystem (any POSIX-compatible FS)
#   OPENPI_DIR     : openpi venv with torch installed
#
# Optional env:
#   NODES          : 1 (8 GPUs) or 2 (16 GPUs), default 1
#   N_ROUNDS       : save/load rounds to average, default 3

set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-${HOME}/openpi}"
STORAGE_PATH="${STORAGE_PATH:?Set STORAGE_PATH to the filesystem mount under test}"
CKPT_DIR="${STORAGE_PATH}/pi05-ckpt-bench"
N_ROUNDS="${N_ROUNDS:-3}"
RESULTS_DIR="${RESULTS_DIR:-results}"
N_GPUS=$((SLURM_NNODES * 8))

source "${OPENPI_DIR}/.venv/bin/activate"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29501

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_IB_GDR_DMABUF_ENABLE="${NCCL_IB_GDR_DMABUF_ENABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,^docker,^virbr}"

mkdir -p "${RESULTS_DIR}"

echo "=== π₀.₅ Checkpoint Benchmark ==="
echo "Nodes/GPUs:    ${SLURM_NNODES} × 8 = ${N_GPUS}"
echo "Storage:       ${STORAGE_PATH}"
echo "Ckpt dir:      ${CKPT_DIR}"
echo "Rounds:        ${N_ROUNDS}"
echo "Started:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"

srun torchrun \
  --nnodes="${SLURM_NNODES}" \
  --nproc-per-node=8 \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --node-rank="${SLURM_NODEID}" \
  "$(dirname "$0")/bench_checkpoint.py" \
    --checkpoint-dir "${CKPT_DIR}" \
    --n-rounds "${N_ROUNDS}" \
    --json-out "${RESULTS_DIR}/pi05_ckpt_${SLURM_NNODES}n_${SLURM_JOB_ID}.json"

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
