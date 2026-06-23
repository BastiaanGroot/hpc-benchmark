#!/usr/bin/env python3
"""
Report benchmark results to ClearML.

Reads all JSON result files from a results directory and creates a single
ClearML task per benchmark run with:
  - Scalar metrics (steps/s, samples/s, GB/s, MFU, etc.)
  - Hyperparameters (world_size, batch_size, gpu, config, …)
  - A comparison table as a ClearML Table artifact

Usage:
  python scripts/report_to_clearml.py \
      --results-dir results/ \
      --project "HPC Benchmarks" \
      --cluster-name "aws-p5-us-east-1"

Requirements:
  pip install clearml

Configure ClearML credentials once with:
  clearml-init
or set CLEARML_API_HOST / CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY env vars.
"""
import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Result file → benchmark type mapping
# ---------------------------------------------------------------------------

# Maps filename prefix → (clearml_task_name, metric_keys, param_keys)
_BENCHMARK_SCHEMAS = {
    "hpl_": (
        "HPL (Linpack)",
        {"tflops": "TFLOPS"},
        ["nodes"],
    ),
    "hpcg_": (
        "HPCG",
        {"gflops": "GFLOPS", "bandwidth_gb_s": "Bandwidth GB/s"},
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
        "LeRobot I/O",
        {
            "samples_per_sec": "Samples/s",
            "throughput_mb_s": "Throughput MB/s",
            "avg_latency_ms": "Avg Latency ms",
            "p99_latency_ms": "p99 Latency ms",
        },
        ["workers"],
    ),
    "pi05_1n": (
        "π₀.₅ Training (1 node)",
        {
            "steps_per_sec": "Steps/s",
            "samples_per_sec": "Samples/s",
            "peak_gpu_mem_gb": "Peak GPU Mem GB",
            "mfu_pct": "MFU %",
        },
        ["world_size", "per_gpu_batch_size", "global_batch_size", "gpu_name"],
    ),
    "pi05_2n": (
        "π₀.₅ Training (2 nodes)",
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


def _report_to_clearml(rows: list[dict], project: str, cluster_name: str):
    try:
        from clearml import Task
    except ImportError:
        raise SystemExit(
            "clearml not installed. Run: pip install clearml\n"
            "Then configure credentials with: clearml-init"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Group rows by benchmark type
    by_bench: dict[str, list] = {}
    for row in rows:
        task_name = row["schema"][0]
        by_bench.setdefault(task_name, []).append(row)

    created_tasks = []

    for bench_name, bench_rows in by_bench.items():
        task = Task.init(
            project_name=project,
            task_name=f"{bench_name} — {cluster_name} — {timestamp}",
            task_type=Task.TaskTypes.custom,
            auto_connect_frameworks=False,
            reuse_last_task_id=False,
        )
        task.set_parameter("cluster_name", cluster_name)
        task.set_parameter("benchmark", bench_name)
        task.set_parameter("host", socket.gethostname())
        task.set_parameter("reported_at", timestamp)

        logger = task.get_logger()
        _, metric_keys, param_keys = bench_rows[0]["schema"]

        for idx, row in enumerate(bench_rows):
            data = row["data"]
            series = row["file"]

            # Log scalar metrics
            for json_key, display_name in metric_keys.items():
                val = data.get(json_key)
                if val is not None:
                    logger.report_scalar(
                        title=display_name,
                        series=series,
                        value=float(val),
                        iteration=idx,
                    )

            # Log hyperparameters from first result file
            if idx == 0:
                for pk in param_keys:
                    if pk in data:
                        task.set_parameter(pk, data[pk])

        # Upload a comparison table if multiple files
        if len(bench_rows) > 1:
            cols = ["file"] + list(metric_keys.keys())
            table = [cols]
            for row in bench_rows:
                table.append(
                    [row["file"]]
                    + [row["data"].get(k, "—") for k in metric_keys.keys()]
                )
            logger.report_table(
                title=f"{bench_name} Summary",
                series="comparison",
                iteration=0,
                table_plot=table,
            )

        task.close()
        created_tasks.append(task.id)
        print(f"  Created task: {bench_name}  (id={task.id})")

    return created_tasks


def _print_summary(rows: list[dict], cluster_name: str):
    """Print a human-readable comparison table to stdout."""
    print(f"\n{'='*70}")
    print(f"  Benchmark Results — {cluster_name}")
    print(f"{'='*70}")
    for row in rows:
        bench_name = row["schema"][0]
        metric_keys = row["schema"][1]
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
    parser = argparse.ArgumentParser(description="Report benchmark results to ClearML")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing JSON result files (default: results/)")
    parser.add_argument("--project", default="HPC Benchmarks",
                        help="ClearML project name (default: 'HPC Benchmarks')")
    parser.add_argument("--cluster-name", default=socket.gethostname(),
                        help="Human-readable cluster identifier (e.g. aws-p5-us-east-1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary only, do not upload to ClearML")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    print(f"Loading results from {results_dir} ...")
    rows = _load_results(results_dir)

    if not rows:
        raise SystemExit("No recognised JSON result files found.")

    print(f"Found {len(rows)} result file(s).")
    _print_summary(rows, args.cluster_name)

    if args.dry_run:
        print("(dry-run mode — not uploading to ClearML)")
        return

    print(f"Uploading to ClearML project '{args.project}' ...")
    task_ids = _report_to_clearml(rows, args.project, args.cluster_name)
    print(f"\nDone. Created {len(task_ids)} ClearML task(s).")


if __name__ == "__main__":
    main()
