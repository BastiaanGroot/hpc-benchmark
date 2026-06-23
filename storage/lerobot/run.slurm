#!/bin/bash
#SBATCH --job-name=lerobot-io-bench
#SBATCH --output=results/lerobot_io_%j.out
#SBATCH --error=results/lerobot_io_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00

# LeRobot I/O Benchmark — LIBERO Parquet + MP4 read throughput
#
# Downloads physical-intelligence/libero (~35 GB) on first run, then measures
# parallel read throughput simulating a π₀.₅ training DataLoader.
#
# Usage:
#   sbatch storage/lerobot/run.slurm
#
# Required env:
#   STORAGE_PATH  : path to a mounted filesystem (any POSIX-compatible FS)
#   HF_TOKEN      : HuggingFace API token (needed to download LIBERO)
#
# Optional env:
#   DATASET_DIR   : explicit dataset path (skip download if already present)
#   WORKERS       : DataLoader worker threads (default = cpus-per-task)
#   N_BATCHES     : number of batches to measure (default 500)

set -euo pipefail
BENCH_DIR="${BENCH_DIR:-${HOME}/git/hpc-benchmark}"

STORAGE_PATH="${STORAGE_PATH:?Set STORAGE_PATH to the filesystem mount under test}"
DATASET_DIR="${DATASET_DIR:-${STORAGE_PATH}/lerobot-datasets/physical-intelligence--libero}"
RESULTS_DIR="${RESULTS_DIR:-results}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK}}"
N_BATCHES="${N_BATCHES:-500}"
VENV="${LEROBOT_BENCH_VENV:-${HOME}/.venv/lerobot-bench}"

mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  python3 -m venv "${VENV}"
  source "${VENV}/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet -r "${BENCH_DIR}/storage/lerobot/requirements.txt"
  pip install --quiet huggingface_hub
else
  source "${VENV}/bin/activate"
fi

echo "=== LeRobot I/O Benchmark (LIBERO) ==="
echo "Storage path:   ${STORAGE_PATH}"
echo "Dataset dir:    ${DATASET_DIR}"
echo "Workers:        ${WORKERS}"
echo "Batches:        ${N_BATCHES}"
echo "Node:           $(hostname)"
echo "Started:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo ""
  echo "--- Downloading physical-intelligence/libero (~35 GB) ---"
  python3 "${BENCH_DIR}/storage/lerobot/download_dataset.py" \
    --dest "${STORAGE_PATH}/lerobot-datasets" \
    --dataset "physical-intelligence/libero" \
    ${HF_TOKEN:+--hf-token "${HF_TOKEN}"}
fi

# ⚠ Cold-read: on 512+ GB RAM nodes the 35 GB dataset may warm into page cache.
# For a true cold-read, drop caches before running (requires root/sudo):
#   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

echo ""
echo "--- Running I/O benchmark ---"
python3 "${BENCH_DIR}/storage/lerobot/bench_lerobot.py" \
  "${DATASET_DIR}" \
  --workers "${WORKERS}" \
  --n-batches "${N_BATCHES}" \
  --json-out "${RESULTS_DIR}/lerobot_${SLURM_JOB_ID}.json"

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
