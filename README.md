# HPC Benchmark Suite

Estimates performance of Slurm-orchestrated cloud clusters for π₀.₅ robotics training workloads.

## Benchmarks

| # | Benchmark | What it measures | Key metric |
|---|-----------|-----------------|------------|
| 1 | HPL (Linpack) | Peak GPU FP64 FLOPS | TFLOPS |
| 2 | NCCL AllReduce | Inter-node GPU communication | GB/s |
| 3 | LIBERO I/O | Real training data read throughput | samples/s, MB/s |
| 4 | MLPerf Storage | Synthetic ML storage throughput (vendor-comparable) | samples/s |
| 5 | π₀.₅ training — 1 node | End-to-end training throughput, GPU utilization | steps/s, MFU % |
| 6 | π₀.₅ training — 2 nodes | Multi-node scaling efficiency | steps/s, scale ratio |
| 7 | π₀.₅ checkpoint | FSDP checkpoint save/restore | GB/s |

Run 3 before 5–6 (LIBERO download is reused by the training benchmarks).

## Structure

```
hpc-benchmark/
├── compute/
│   ├── hpl.slurm               # HPL — peak FP64 FLOPS
│   ├── nccl.slurm              # NCCL AllReduce — GPU interconnect
│   └── configs/hpl.dat         # HPL problem size (tune N, NB, P×Q)
├── storage/
│   ├── lerobot/
│   │   ├── download_dataset.py # Download physical-intelligence/libero to target filesystem
│   │   ├── bench_lerobot.py    # Measure Parquet + MP4 read throughput
│   │   ├── generate_dataset.py # Synthetic fallback (air-gapped clusters)
│   │   └── run.slurm
│   └── mlperf/
│       ├── install.sh          # Install mlcommons/storage
│       └── run.slurm           # Synthetic ML storage benchmark (vendor-comparable)
├── training/
│   └── pi05/
│       ├── install.sh          # Install openpi @ 15a9616 (Jun 16 2026)
│       ├── bench_train.py      # Training throughput + MFU (auto-detects GPU)
│       ├── bench_checkpoint.py # FSDP checkpoint save/restore
│       ├── 1node_8gpu.slurm
│       ├── 2node_16gpu.slurm
│       └── bench_checkpoint.slurm
└── scripts/
    └── report_to_clearml.py    # Aggregate all JSON results → ClearML
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Slurm | 22.x or newer |
| Enroot / Singularity | For NGC container (HPL + NCCL only) |
| NGC API key | `docker login nvcr.io` |
| Python ≥ 3.10 | For storage + training benchmarks |
| HuggingFace token | For downloading LIBERO dataset |
| Mounted filesystem | Any POSIX-compatible mount; set `STORAGE_PATH` |

## Running the full suite

```bash
export NGC_CONTAINER=/shared/containers/hpc-benchmarks:26.02.sqsh
export STORAGE_PATH=/mnt/scratch        # any mounted filesystem
export HF_TOKEN=hf_...

# 1. Pull NGC container (once per cluster)
enroot import --output "${NGC_CONTAINER}" docker://nvcr.io#nvidia/hpc-benchmarks:26.02

# 2. Install openpi (once per cluster)
bash training/pi05/install.sh

# 3. Run benchmarks
sbatch --nodes=4 compute/hpl.slurm              # ~2 h
sbatch --nodes=8 compute/nccl.slurm             # ~30 min
sbatch             storage/lerobot/run.slurm    # ~1 h  (downloads LIBERO on first run)
sbatch             storage/mlperf/run.slurm     # ~4 h  (optional, vendor-comparable)
sbatch training/pi05/1node_8gpu.slurm           # ~1 h
sbatch training/pi05/2node_16gpu.slurm          # ~1 h
sbatch training/pi05/bench_checkpoint.slurm     # ~30 min

# 4. Report to ClearML
python scripts/report_to_clearml.py \
    --results-dir results/ \
    --cluster-name "your-cluster-name"
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NGC_CONTAINER` | `/shared/containers/hpc-benchmarks:26.02.sqsh` | NGC image path |
| `STORAGE_PATH` | *(required)* | Filesystem mount under test |
| `HF_TOKEN` | *(required for download)* | HuggingFace API token |
| `MLPERF_STORAGE_DIR` | `~/mlperf-storage` | mlcommons/storage checkout (from `storage/mlperf/install.sh`) |
| `OPENPI_DIR` | `~/openpi` | openpi checkout (from `training/pi05/install.sh`) |
| `DATASET_DIR` | `STORAGE_PATH/lerobot-datasets/physical-intelligence--libero` | Override dataset path |
| `NCCL_IB_HCA` | *(auto)* | Set to e.g. `mlx5_0:1` if IB auto-detect picks wrong HCA |
| `GPU_PEAK_FLOPS` | *(auto: H100/H200/B200/B300)* | Override GPU BF16 TFLOPS for MFU |
