#!/bin/bash
#SBATCH --job-name=hpl-benchmark
#SBATCH --output=results/hpl_%j.out
#SBATCH --error=results/hpl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --exclusive

# NVIDIA HPC-Benchmarks 26.02 — HPL (Linpack)
# Measures peak FP64 GEMM performance (TFLOPS) across all GPUs on a node.
# Uses Docker with nvidia runtime (no Enroot required).
#
# Usage:
#   sbatch compute/hpl.slurm
#
# Required env:
#   NGC_IMAGE   : NGC container image (default: nvcr.io/nvidia/hpc-benchmarks:26.02)
#   HPL_DAT     : path to HPL.dat (default: compute/configs/hpl.dat)

set -euo pipefail

NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:26.02}"
HPL_DAT="${HPL_DAT:-$(cd "$(dirname "$0")" && pwd)/configs/hpl.dat}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SCRATCH="/tmp/hpl-${SLURM_JOB_ID}"

mkdir -p "${RESULTS_DIR}" "${SCRATCH}"
cp "${HPL_DAT}" "${SCRATCH}/HPL.dat"

echo "=== HPL Benchmark ==="
echo "Node:       $(hostname)"
echo "GPUs:       ${SLURM_GPUS_ON_NODE:-8}"
echo "Image:      ${NGC_IMAGE}"
echo "HPL.dat:    ${HPL_DAT}"
echo "Started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Pull image if not already cached
docker pull "${NGC_IMAGE}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --volume="${SCRATCH}:/workspace/hpl" \
  --workdir="/workspace/hpl" \
  "${NGC_IMAGE}" \
  hpl.sh \
    --dat /workspace/hpl/HPL.dat \
    --cpu-cores-per-rank 4 \
    --cpu-affinity broad

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
