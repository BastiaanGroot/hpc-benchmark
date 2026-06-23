#!/bin/bash
#SBATCH --job-name=hpl-benchmark
#SBATCH --output=results/hpl_%j.out
#SBATCH --error=results/hpl_%j.err
#SBATCH --nodes=${NODES:-1}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE:-8}
#SBATCH --gpus-per-node=${GPUS_PER_NODE:-8}
#SBATCH --time=02:00:00
#SBATCH --exclusive

# NVIDIA HPC-Benchmarks 26.02 — HPL (Linpack)
# Measures peak FP64 GEMM performance (FLOPS) across nodes.
#
# Usage:
#   sbatch --nodes=4 compute/hpl.slurm
#
# Required env:
#   NGC_CONTAINER  : path or URI to the HPC-benchmarks container image
#                    e.g. /shared/containers/hpc-benchmarks:26.02.sqsh
#   HPL_DAT        : path to HPL.dat config (defaults to compute/configs/hpl.dat)

set -euo pipefail

NGC_CONTAINER="${NGC_CONTAINER:-/shared/containers/hpc-benchmarks:26.02.sqsh}"
HPL_DAT="${HPL_DAT:-$(dirname "$0")/configs/hpl.dat}"
RESULTS_DIR="${RESULTS_DIR:-results}"

mkdir -p "${RESULTS_DIR}"

echo "=== HPL Benchmark ==="
echo "Nodes:        ${SLURM_NNODES}"
echo "GPUs/node:    ${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE:-8}}"
echo "Container:    ${NGC_CONTAINER}"
echo "HPL.dat:      ${HPL_DAT}"
echo "Started:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Copy HPL.dat to a shared scratch location visible to all ranks
SCRATCH="${SCRATCH:-/tmp/hpl-${SLURM_JOB_ID}}"
mkdir -p "${SCRATCH}"
cp "${HPL_DAT}" "${SCRATCH}/HPL.dat"

srun \
  --mpi=pmix \
  --container-image="${NGC_CONTAINER}" \
  --container-mounts="${SCRATCH}:/workspace/hpl" \
  --container-workdir="/workspace/hpl" \
  hpl.sh \
    --dat /workspace/hpl/HPL.dat \
    --cpu-cores-per-rank 4 \
    --cpu-affinity broad

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
