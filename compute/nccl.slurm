#!/bin/bash
#SBATCH --job-name=nccl-benchmark
#SBATCH --output=results/nccl_%j.out
#SBATCH --error=results/nccl_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --exclusive

# NVIDIA HPC-Benchmarks 26.02 — NCCL AllReduce
# Measures inter-node GPU communication bandwidth using Docker + nvidia runtime.
#
# Usage:
#   sbatch compute/nccl.slurm
#
# Required env:
#   NGC_IMAGE   : NGC container image (default: nvcr.io/nvidia/hpc-benchmarks:26.02)

set -euo pipefail

NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:26.02}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MIN_BYTES="${MIN_BYTES:-1M}"
MAX_BYTES="${MAX_BYTES:-8G}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_PORT=29500

# InfiniBand / NCCL tuning
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_IB_GDR_DMABUF_ENABLE="${NCCL_IB_GDR_DMABUF_ENABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,^docker,^virbr}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# export NCCL_IB_HCA=mlx5_0:1  # set if auto-detect picks wrong HCA

mkdir -p "${RESULTS_DIR}"

echo "=== NCCL AllReduce Benchmark ==="
echo "Nodes:      ${SLURM_NNODES}"
echo "Msg range:  ${MIN_BYTES} – ${MAX_BYTES}"
echo "Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "Started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"

docker pull "${NGC_IMAGE}"

# Run nccl-tests via mpirun inside the container across both nodes
# (Using host network so NCCL can see IB interfaces directly)
docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --env NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
  --env NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}" \
  --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  --env NCCL_DEBUG="${NCCL_DEBUG}" \
  --env MASTER_ADDR="${MASTER_ADDR}" \
  "${NGC_IMAGE}" \
  all_reduce_perf \
    -b "${MIN_BYTES}" \
    -e "${MAX_BYTES}" \
    -f 2 \
    -g "${SLURM_GPUS_ON_NODE:-8}" \
    -c 1

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
