#!/usr/bin/env python3
"""
Report benchmark results to ClearML using agile-clearml.

Reads all JSON result files from a results directory and creates a ClearML
task per benchmark type with scalar metrics, hyperparameters, and a summary table.

Usage:
  python scripts/report_to_clearml.py \
      --results-dir results/ \
      --project "HPC Benchmarks" \
      --cluster-name "nebius-b300"

Requirements:
  pip install agile-clearml \
      --index-url https://artifactory.agile-robots.com/artifactory/api/pypi/MU-AR_PYPI_STAGING/simple

Credentials (env vars):
  CLEARML_OTC_API_ACCESS_KEY
  CLEARML_OTC_API_SECRET_KEY
"""
import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark schema: filename prefix → (task_name, metric_keys, param_keys)
# ---------------------------------------------------------------------------
_BENCHMARK_SCHEMAS = {
    "hpl_": (
        "HPL (Linpack)",
        {"tflops": "TFLOPS"},
        ["nodes"],
    ),
    "nccl_": (
        "NCCL AllReduce",
        {"busbw_gb_s": "Bus BW GB/s", "algbw_gb_s": "Algo BW GB/s"},
        ["nodes", "msg_size_bytes"],
    ),
    "mlperf_": (
        "MLPerf Storage",
        {"samples_per_sec": "Samples/s", "throughput_mb_s": "Throughput MB/s"},
        ["model", "num_accelerators", "batch_size"],
    ),
    "lerobot_": (
        "LeRobot I/O (LIBERO)",
        {
            "samples_per_sec": "Samples/s",
            "throughput_mb_s": "Throughput MB/s",
            "avg_latency_ms": "Avg Latency ms",
            "p99_latency_ms": "p99 Latency ms",
        },
        ["workers"],
    ),
    "pi05_1n": (
        "π₀.₅ Training (1 node × 8 GPU)",
        {
            "steps_per_sec": "Steps/s",
            "samples_per_sec": "Samples/s",
            "peak_gpu_mem_gb": "Peak GPU Mem GB",
            "mfu_pct": "MFU %",
        },
        ["world_size", "per_gpu_batch_size", "global_batch_size", "gpu_name"],
    ),
    "pi05_2n": (
        "π₀.₅ Training (2 nodes × 8 GPU)",
        {
            "steps_per_sec": "Steps/s",
            "samples_per_sec": "Samples/s",
            "peak_gpu_mem_gb": "Peak GPU Mem GB",
            "mfu_pct": "MFU %",
        },
        ["world_size", "per_gpu_batch_size", "global_batch_size", "gpu_name"],
    ),
    "pi05_ckpt": (
        "π₀.₅ Checkpoint",
        {
            "save_gb_s": "Save GB/s",
            "load_gb_s": "Load GB/s",
            "checkpoint_size_gb": "Checkpoint Size GB",
        },
        ["world_size"],
    ),
}


def _detect_schema(filename: str):
    for prefix, schema in _BENCHMARK_SCHEMAS.items():
        if filename.startswith(prefix):
            return schema
    return None


def _load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        schema = _detect_schema(path.stem)
        if schema is None:
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  Warning: could not parse {path.name}: {e}")
            continue
        rows.append({"file": path.name, "schema": schema, "data": data})
    return rows


def _report(rows: list[dict], project: str, cluster_name: str):
    try:
        from agile_clearml import ClearMLClient, ClearMLConfig
    except ImportError:
        raise SystemExit(
            "agile-clearml not installed.\n"
            "pip install agile-clearml "
            "--index-url https://artifactory.agile-robots.com/artifactory/api/pypi/MU-AR_PYPI_STAGING/simple"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    by_bench: dict[str, list] = {}
    for row in rows:
        by_bench.setdefault(row["schema"][0], []).append(row)

    created = []
    for bench_name, bench_rows in by_bench.items():
        cfg = ClearMLConfig(
            project_name=project,
            task_name=f"{bench_name} — {cluster_name} — {timestamp}",
        )
        client = ClearMLClient().initialize_task(cfg)
        task = client.task

        task.set_parameter("cluster_name", cluster_name)
        task.set_parameter("host", socket.gethostname())

        _, metric_keys, param_keys = bench_rows[0]["schema"]
        logger = task.get_logger()

        for idx, row in enumerate(bench_rows):
            data = row["data"]
            series = row["file"]
            for json_key, display_name in metric_keys.items():
                val = data.get(json_key)
                if val is not None:
                    logger.report_scalar(
                        title=display_name,
                        series=series,
                        value=float(val),
                        iteration=idx,
                    )
            if idx == 0:
                for pk in param_keys:
                    if pk in data:
                        task.set_parameter(pk, data[pk])

        if len(bench_rows) > 1:
            cols = ["file"] + list(metric_keys.keys())
            table = [cols] + [
                [r["file"]] + [r["data"].get(k, "—") for k in metric_keys]
                for r in bench_rows
            ]
            logger.report_table(
                title=f"{bench_name} Summary",
                series="comparison",
                iteration=0,
                table_plot=table,
            )

        task.close()
        created.append(task.id)
        print(f"  Created task: {bench_name}  (id={task.id})")

    return created


def _print_summary(rows: list[dict], cluster_name: str):
    print(f"\n{'='*70}")
    print(f"  Results — {cluster_name}")
    print(f"{'='*70}")
    for row in rows:
        bench_name, metric_keys, _ = row["schema"]
        data = row["data"]
        metrics_str = "  ".join(
            f"{label}: {data.get(jk, '—')}"
            for jk, label in metric_keys.items()
            if data.get(jk) is not None
        )
        print(f"  [{bench_name}]  {row['file']}")
        print(f"    {metrics_str}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--project", default="HPC Benchmarks")
    parser.add_argument("--cluster-name", default=socket.gethostname())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    rows = _load_results(results_dir)
    if not rows:
        raise SystemExit("No recognised JSON result files found.")

    print(f"Found {len(rows)} result file(s).")
    _print_summary(rows, args.cluster_name)

    if args.dry_run:
        print("(dry-run — not uploading to ClearML)")
        return

    print(f"Uploading to ClearML project '{args.project}' ...")
    _report(rows, args.project, args.cluster_name)


if __name__ == "__main__":
    main()
