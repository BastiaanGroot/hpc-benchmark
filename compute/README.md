# Compute Benchmarks — NVIDIA HPC-Benchmarks 26.02

Uses the [NVIDIA HPC-Benchmarks NGC container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks?version=26.02).

## Pull the container

```bash
# Enroot (.sqsh) — recommended on Slurm clusters
enroot import --output /shared/containers/hpc-benchmarks:26.02.sqsh \
  docker://nvcr.io#nvidia/hpc-benchmarks:26.02

# Or Singularity/Apptainer (.sif)
singularity pull hpc-benchmarks_26.02.sif \
  docker://nvcr.io/nvidia/hpc-benchmarks:26.02
```

Set `NGC_CONTAINER` to the resulting image path before submitting.

## Benchmarks

| Script | What it measures | Key metric |
|--------|-----------------|------------|
| `hpl.slurm` | Peak FP64 GEMM (Linpack) | TFLOPS |
| `nccl.slurm` | Inter-node GPU AllReduce bandwidth | GB/s |

## Quick start

```bash
export NGC_CONTAINER=/shared/containers/hpc-benchmarks:26.02.sqsh

sbatch --nodes=4 compute/hpl.slurm     # peak FLOPS
sbatch --nodes=8 compute/nccl.slurm    # interconnect bandwidth
```

## HPL tuning (`configs/hpl.dat`)

- **N**: `sqrt(0.8 × total_gpu_mem_bytes / 8)` — fill ~80% GPU memory
- **NB**: 336 for H100/H200; try 192–512 for Blackwell
- **P × Q**: equals total MPI ranks; keep P ≤ Q
