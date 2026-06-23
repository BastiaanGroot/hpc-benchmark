#!/bin/bash
#SBATCH --job-name=mlperf-storage
#SBATCH --output=results/mlperf_storage_%j.out
#SBATCH --error=results/mlperf_storage_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

# MLPerf Storage v2.0 — industry-standard ML storage benchmark
#
# Complements the LIBERO I/O benchmark:
#   - LIBERO measures your actual training data access pattern
#   - MLPerf measures raw storage throughput with a synthetic workload,
#     isolating the filesystem from video decoding / Parquet overhead,
#     and gives a vendor-comparable number
#
# Usage:
#   sbatch storage/mlperf/run.slurm
#
# Required env:
#   STORAGE_PATH         : path to a mounted filesystem
#   MLPERF_STORAGE_DIR   : path to mlcommons/storage checkout (see install.sh)
#
# Optional env:
#   MODEL          : retinanet (default) | resnet50 | dlrmv2 | unet3d | cosmoflow
#   NUM_ACCELERATORS : simulated GPU count (default 8)
#   BATCH_SIZE     : per-accelerator batch size (default 48)

set -euo pipefail

MLPERF_STORAGE_DIR="${MLPERF_STORAGE_DIR:-${HOME}/mlperf-storage}"
STORAGE_PATH="${STORAGE_PATH:?Set STORAGE_PATH to the mounted filesystem under test}"
MODEL="${MODEL:-retinanet}"
NUM_ACCELERATORS="${NUM_ACCELERATORS:-8}"
BATCH_SIZE="${BATCH_SIZE:-48}"
RESULTS_DIR="${RESULTS_DIR:-results}"

source "${MLPERF_STORAGE_DIR}/.venv/bin/activate"

mkdir -p "${RESULTS_DIR}"

echo "=== MLPerf Storage v2.0 ==="
echo "Model:            ${MODEL}"
echo "Accelerators:     ${NUM_ACCELERATORS}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Storage path:     ${STORAGE_PATH}"
echo "Node:             $(hostname)"
echo "Started:          $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo ""
echo "--- Generating synthetic dataset ---"
mlpstorage training datagen \
  --file "${STORAGE_PATH}/mlperf-storage-data" \
  --model "${MODEL}" \
  --num-parallel "${SLURM_CPUS_PER_TASK}"

echo ""
echo "--- Running benchmark ---"
mlpstorage training run \
  --file "${STORAGE_PATH}/mlperf-storage-data" \
  --model "${MODEL}" \
  --num-accelerators "${NUM_ACCELERATORS}" \
  --batch-size "${BATCH_SIZE}" \
  --results-dir "${RESULTS_DIR}/mlperf_${MODEL}_${SLURM_JOB_ID}"

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
