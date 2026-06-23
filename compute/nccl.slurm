#!/bin/bash
#SBATCH --job-name=nccl-benchmark
#SBATCH --output=results/nccl_%j.out
#SBATCH --error=results/nccl_%j.err
#SBATCH --nodes=${NODES:-2}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE:-8}
#SBATCH --gpus-per-node=${GPUS_PER_NODE:-8}
#SBATCH --time=00:30:00
#SBATCH --exclusive

# NVIDIA HPC-Benchmarks 26.02 — NCCL AllReduce / AlltoAll
# Measures inter-node GPU communication bandwidth.
# Critical for distributed training scalability.
#
# Usage:
#   sbatch --nodes=8 compute/nccl.slurm

set -euo pipefail

NGC_CONTAINER="${NGC_CONTAINER:-/shared/containers/hpc-benchmarks:26.02.sqsh}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# Message sizes: 1MB → 8GB (covers typical gradient all-reduce sizes)
MIN_BYTES="${MIN_BYTES:-1M}"
MAX_BYTES="${MAX_BYTES:-8G}"

mkdir -p "${RESULTS_DIR}"

echo "=== NCCL AllReduce Benchmark ==="
echo "Nodes:        ${SLURM_NNODES}"
echo "Msg range:    ${MIN_BYTES} – ${MAX_BYTES}"
echo "Started:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# InfiniBand / NCCL tuning — same defaults as training/pi05/2node_16gpu.slurm
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-5}"
export NCCL_IB_GDR_DMABUF_ENABLE="${NCCL_IB_GDR_DMABUF_ENABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,^docker,^virbr}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# export NCCL_IB_HCA=mlx5_0:1  # uncomment and set if auto-detect picks wrong HCA

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR

srun \
  --mpi=pmix \
  --container-image="${NGC_CONTAINER}" \
  --container-env="NCCL_DEBUG=WARN,MASTER_ADDR=${MASTER_ADDR}" \
  all_reduce_perf \
    -b "${MIN_BYTES}" \
    -e "${MAX_BYTES}" \
    -f 2 \
    -g 1 \
    -c 1

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
