#!/bin/bash
#SBATCH --job-name=storage-sweep
#SBATCH --output=results/storage_sweep_%j.out
#SBATCH --error=results/storage_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

# File-size sweep benchmark — sequential read/write across block sizes from 64 KB to 1 GB.
#
# Tests the filesystem at sizes that span the full LeRobot data layout:
#   64 KB – 1 MB  : Parquet row groups and episode metadata
#   1 MB  – 64 MB : Parquet shards (configurable with --shard-size when pushing datasets)
#   64 MB – 1 GB  : video shards and checkpoint fragments
#
# If LIBERO dataset is present at $STORAGE_PATH/lerobot-datasets/physical-intelligence/libero,
# a real-data read pass is also performed and appended to the JSON output.
#
# Usage:
#   sbatch storage/sweep/run.slurm
#
# Optional env overrides:
#   STORAGE_PATH   : filesystem to test (default: /tmp)
#   WORKERS        : parallel reader threads (default: 1; try 4 or 8 for DataLoader simulation)
#   PASSES         : measurement passes per block size (default: 3; median is reported)
#   MAX_SIZE_MB    : skip sizes above this MB (default: 1024)

set -euo pipefail
BENCH_DIR="${BENCH_DIR:-${HOME}/git/hpc-benchmark}"
RESULTS_DIR="${RESULTS_DIR:-results}"
STORAGE_PATH="${STORAGE_PATH:-/tmp}"
WORKERS="${WORKERS:-1}"
PASSES="${PASSES:-3}"
MAX_SIZE_MB="${MAX_SIZE_MB:-1024}"

mkdir -p "${RESULTS_DIR}"

OUTPUT_JSON="${RESULTS_DIR}/storage_sweep_${SLURM_JOB_ID}.json"

echo "=== Storage Sweep Benchmark ==="
echo "Node:         $(hostname)"
echo "Storage path: ${STORAGE_PATH}"
echo "Workers:      ${WORKERS}"
echo "Passes:       ${PASSES}"
echo "Max size:     ${MAX_SIZE_MB} MB"
echo "Started:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

python3 "${BENCH_DIR}/storage/sweep/bench_sweep.py" \
    --storage-path "${STORAGE_PATH}" \
    --workers      "${WORKERS}" \
    --passes       "${PASSES}" \
    --max-size-mb  "${MAX_SIZE_MB}" \
    --output       "${OUTPUT_JSON}"

echo ""
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Results:  ${OUTPUT_JSON}"
