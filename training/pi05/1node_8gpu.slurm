#!/bin/bash
#SBATCH --job-name=pi05-bench-1n8g
#SBATCH --output=results/pi05_1n8g_%j.out
#SBATCH --error=results/pi05_1n8g_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --exclusive

# π₀.₅ Training Benchmark — 1 node × 8 GPUs (FSDP)
#
# Measures training throughput (steps/s, samples/s) for π₀.₅ fine-tuning
# using full FSDP across 8 GPUs on a single node.
#
# Required env:
#   OPENPI_DIR      : path to openpi checkout (from install.sh)
#   STORAGE_PATH    : filesystem under test; dataset will be read from here
#
# Optional env:
#   DATASET_DIR     : full path to a LeRobot dataset (default: STORAGE_PATH/lerobot-bench-data)
#   BATCH_SIZE      : per-GPU batch size (default 32)
#   HF_TOKEN        : HuggingFace token for downloading lerobot/pi05_base weights

set -euo pipefail
BENCH_DIR="${BENCH_DIR:-${HOME}/git/hpc-benchmark}"

OPENPI_DIR="${OPENPI_DIR:-${HOME}/openpi}"
STORAGE_PATH="${STORAGE_PATH:?Set STORAGE_PATH to the filesystem mount under test}"
# Default to the real LIBERO dataset downloaded by storage/lerobot/run.slurm.
# Falls back to synthetic data if the real dataset hasn't been downloaded yet.
HF_DATASET="${HF_DATASET:-physical-intelligence/libero}"
DATASET_DIR="${DATASET_DIR:-${STORAGE_PATH}/lerobot-datasets/${HF_DATASET/\//--}}"
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Warning: ${DATASET_DIR} not found. Falling back to synthetic data."
  DATASET_DIR="${STORAGE_PATH}/lerobot-bench-data"
fi
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULTS_DIR="${RESULTS_DIR:-results}"

source "${OPENPI_DIR}/.venv/bin/activate"

# Export HF token if provided so openpi can fetch pi05_base weights
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

MASTER_ADDR=$(hostname)
MASTER_PORT=29500

mkdir -p "${RESULTS_DIR}"

echo "=== π₀.₅ Benchmark: 1 node × 8 GPUs ==="
echo "openpi commit: 15a9616a00943ada6c20a0f158e3adb39df2ccac"
echo "Dataset:       ${DATASET_DIR}"
echo "Batch/GPU:     ${BATCH_SIZE}  (global: $((BATCH_SIZE * 8)))"
echo "Node:          $(hostname)"
echo "Started:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${BENCH_DIR}/training/pi05/bench_train.py" \
    --config pi05_base \
    --data-dir "${DATASET_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --warmup-steps 20 \
    --measure-steps 80 \
    --json-out "${RESULTS_DIR}/pi05_1n8g_${SLURM_JOB_ID}.json"

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
