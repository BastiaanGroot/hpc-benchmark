# Storage Benchmarks

Two complementary benchmarks covering different perspectives.

| Benchmark | What it measures | When to use |
|-----------|-----------------|-------------|
| `lerobot/` | Real LIBERO data read throughput (your actual training I/O) | Always — tells you if the filesystem can keep π₀.₅ fed |
| `mlperf/` | Synthetic ML workload (vendor-comparable, isolates FS from decode) | Vendor evaluation, procurement, isolating FS from decoding overhead |

---

## 1. LIBERO I/O (`lerobot/`)

Uses [`physical-intelligence/libero`](https://huggingface.co/datasets/physical-intelligence/libero)
— 35 GB, 1,693 episodes, dual camera, 256×256 @ 10 fps —
the actual dataset used to fine-tune π₀.₅-LIBERO.

Access pattern: random Parquet row-group reads (states/actions) + random-seek frame
decode from MP4 shards, matching a π₀.₅ PyTorch DataLoader.

```bash
export STORAGE_PATH=/mnt/scratch   # any mounted filesystem
export HF_TOKEN=hf_...

sbatch storage/lerobot/run.slurm
```

On first run downloads LIBERO to `STORAGE_PATH/lerobot-datasets/`. The training
benchmarks reuse the same data automatically.

> **Cold-read note:** on nodes with ≥ 512 GB RAM the 35 GB dataset may warm into
> page cache. Drop caches before running for a true cold-read measurement:
> `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`

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
