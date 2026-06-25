# Storage Benchmarks

Two complementary benchmarks covering different perspectives.

| Benchmark | What it measures | When to use |
|-----------|-----------------|-------------|
| `sweep/` | Sequential read/write throughput across block sizes 64 KB → 1 GB | Always — determines optimal LeRobot shard size and raw filesystem ceiling |
| `mlperf/` | Synthetic ML workload (vendor-comparable, isolates FS from decode) | Vendor evaluation, procurement, isolating FS from decoding overhead |

---

## 1. File-size Sweep (`sweep/`)

Measures sequential read and write throughput at eight block sizes spanning the full range
of file sizes that appear in a typical LeRobot deployment:

| Block size | LeRobot equivalent |
|------------|-------------------|
| 64 KB | Episode metadata, Parquet row groups |
| 256 KB – 1 MB | Small Parquet shards (fine-grained indexing) |
| 4 MB – 16 MB | Default Parquet shard size (`--shard-size` when pushing) |
| 64 MB | Large Parquet shards / short MP4 episode chunks |
| 256 MB – 1 GB | Long video shards, model checkpoint fragments |

**Interpreting the sweep:**
- Throughput flat from 1 MB onwards → shard size barely matters; pick 50–100 MB for convenience.
- Throughput rises steeply up to _X_ MB then plateaus → set `--shard-size` ≥ _X_.
- Very low small-file (<1 MB) throughput → NFS/Lustre metadata overhead; avoid tiny shards and
  minimize the number of distinct files per episode.

Requires `fio >= 3.0`. The Slurm script installs it automatically via `apt`/`yum` if not present.

```bash
export STORAGE_PATH=/mnt/scratch    # filesystem to test
sbatch storage/sweep/run.slurm
```

Optionally increase worker count to simulate PyTorch DataLoader concurrency:

```bash
WORKERS=8 sbatch storage/sweep/run.slurm
```

fio uses `--direct=1` (O_DIRECT) to bypass the OS page cache, so results reflect true
storage throughput without needing root access or `drop_caches`.

---

## 2. MLPerf Storage v2.0 (`mlperf/`)

Industry-standard benchmark using synthetic data sized to match real ML datasets
(ResNet-50, RetinaNet, DLRMv2, Llama-3 checkpointing). Simulates GPU "think time"
to isolate pure filesystem throughput.

Use this when you need a vendor-comparable number or want to separate filesystem
throughput from video decoding / Parquet overhead.

```bash
export STORAGE_PATH=/mnt/scratch
export MLPERF_STORAGE_DIR=${HOME}/mlperf-storage

bash storage/mlperf/install.sh     # once per cluster
sbatch storage/mlperf/run.slurm
```

Set `MODEL` to `retinanet` (default), `resnet50`, `dlrmv2`, `unet3d`, or `cosmoflow`.
Set `NUM_ACCELERATORS` to match the number of GPUs you want to simulate (default 8).
