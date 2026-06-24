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
BENCH_DIR="${BENCH_DIR:-${HOME}/git/hpc-benchmark}"

NGC_IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:26.02}"
HPL_DAT="${HPL_DAT:-${BENCH_DIR}/compute/configs/hpl.dat}"
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

# Run HPL via mpirun inside the container.
#
# Key notes for NVIDIA HPC-Benchmarks 26.02 on Blackwell (sm_103) with Docker:
#
# 1. hpl.sh must be invoked via mpirun — it does NOT self-launch MPI; it expects
#    to be running as an MPI rank (detects rank via PMI/MPI env vars).
#
# 2. NVSHMEM version: xhpl links against the workspace libnvshmem_host.so.3.5.19.
#    Overriding NVSHMEM_PATH to the CUDA-toolkit NVSHMEM (3.4.5) causes a device/
#    host library version mismatch. Use the default workspace NVSHMEM (no override).
#
# 3. The nvidia_entrypoint prints "driverInitFileInfo result=11" at container start;
#    these are non-fatal warnings from CUDA forward-compatibility probing.
#
docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --volume="${SCRATCH}/HPL.dat:/workspace/hpl-linux-x86_64/HPL.dat" \
  --entrypoint=/usr/local/mpi/bin/mpirun \
  "${NGC_IMAGE}" \
    -np 8 --allow-run-as-root --bind-to none \
    bash -c 'source /workspace/hpc-benchmarks-gpu-env.sh && cd /workspace/hpl-linux-x86_64 && ./xhpl'

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
