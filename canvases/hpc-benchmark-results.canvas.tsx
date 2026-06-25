import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const CLUSTER = "Nebius · 8× NVIDIA B300 SXM6 AC";
const RUN_DATE = "2026-06-24";

// B300 official specs (NVIDIA datasheet)
const B300_PEAK_FP64_TFLOPS = 1.3;      // per GPU
const B300_PEAK_MEM_BW_GBS = 8000;      // per GPU
const B300_NVLINK_BW_GBS = 1800;        // per GPU bidirectional

export default function BenchmarkResults() {
  const hplEfficiency = ((1018 / (B300_PEAK_FP64_TFLOPS * 1000)) * 100).toFixed(0);
  const streamEfficiency = ((7089 / B300_PEAK_MEM_BW_GBS) * 100).toFixed(1);
  const nclEfficiencyPct = ((835 / B300_NVLINK_BW_GBS) * 100).toFixed(0);
  const vramUsedPct = ((68.8 / 288) * 100).toFixed(0);

  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 960, margin: "0 auto" }}>
      {/* Header */}
      <Stack gap={4}>
        <Row gap={12} align="center">
          <H1>HPC Benchmark Results</H1>
          <Pill active>{RUN_DATE}</Pill>
        </Row>
        <Text tone="secondary">{CLUSTER} · NVIDIA HPC-Benchmarks 26.02 · openpi commit 15a9616</Text>
      </Stack>

      {/* Top-level KPI strip */}
      <Grid columns={5} gap={16}>
        <Stat value="8,145" label="HPL GFLOPS (total)" tone="success" />
        <Stat value="7.09 TB/s" label="STREAM memory BW" tone="success" />
        <Stat value="835 GB/s" label="NCCL bus BW" tone="info" />
        <Stat value="7,064 MB/s" label="Peak storage read (1 GB)" tone="success" />
        <Stat value="262" label="π₀.₅ samples / sec" />
      </Grid>

      <Divider />

      {/* === EVALUATION === */}
      <H2>Evaluation</H2>

      <Grid columns={2} gap={16}>
        {/* HPL assessment */}
        <Card>
          <CardHeader trailing={<Pill size="sm">Expected</Pill>}>HPL FP64</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Grid columns={2} gap={12}>
                <Stat value={`${hplEfficiency}%`} label="of peak FP64" />
                <Stat value="1.02 TF" label="vs 34 TF on H100" tone="warning" />
              </Grid>
              <Callout tone="warning" title="B300 is not a FP64 GPU">
                The B300 has only 1.3 TFLOPS peak native FP64 — ~26× less than H100 SXM5 (34 TFLOPS).
                Blackwell is designed for AI workloads (FP8: 4,500 TFLOPS; FP4: 13,500 TFLOPS), not HPC Linpack.
                Our 78% efficiency is actually excellent for this hardware.
                <Text as="span" weight="semibold"> Use HPL-MxP (tensor-core HPL) for a meaningful HPC number on B300.</Text>
              </Callout>
              <BarChart
                categories={["B300 (this run)", "H100 SXM5 (ref)", "B300 FP8 peak (ref)"]}
                series={[{ name: "TFLOPS FP64/GPU", data: [1.02, 34, 0] }]}
                height={120}
                beginAtZero
                valueSuffix=" TF"
              />
            </Stack>
          </CardBody>
        </Card>

        {/* STREAM assessment */}
        <Card>
          <CardHeader trailing={<Pill active size="sm">Excellent</Pill>}>STREAM Memory Bandwidth</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Grid columns={2} gap={12}>
                <Stat value={`${streamEfficiency}%`} label="of 8 TB/s peak" tone="success" />
                <Stat value="2.1×" label="vs H100 (3.35 TB/s)" tone="success" />
              </Grid>
              <Callout tone="success" title="Near-optimal">
                88.6% HBM efficiency is excellent. The B300's 7,680-bit HBM3e bus is ~2.1× wider than H100's.
                High bandwidth directly benefits LLM/VLA inference and large-batch training.
              </Callout>
              <BarChart
                categories={["B300 (this run)", "B300 peak", "H100 SXM5 (ref)"]}
                series={[{ name: "Memory BW (TB/s)", data: [7.09, 8.0, 3.35] }]}
                height={120}
                beginAtZero
                valueSuffix=" TB/s"
                referenceLines={[{ value: 8.0, label: "B300 peak", tone: "info" }]}
              />
            </Stack>
          </CardBody>
        </Card>

        {/* NCCL assessment */}
        <Card>
          <CardHeader trailing={<Pill size="sm">Healthy</Pill>}>NCCL AllReduce</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Grid columns={2} gap={12}>
                <Stat value="835 GB/s" label="bus BW achieved" tone="success" />
                <Stat value={`~${nclEfficiencyPct}%`} label="of NVLink 5 (1.8 TB/s)" />
              </Grid>
              <Callout tone="info" title="NVLink 5 — good collective efficiency">
                835 GB/s all-reduce bus bandwidth is healthy for 8 GPUs. Collective operations
                typically achieve 40–55% of peak unidirectional NVLink due to the reduce+scatter
                pattern. Note: TCP fallback was used (IB fabric not configured); NVLink handles
                intra-node traffic fine for single-node runs.
              </Callout>
            </Stack>
          </CardBody>
        </Card>

        {/* Storage sweep */}
        <Card>
          <CardHeader trailing={<Pill active size="sm">Scales to 7 GB/s</Pill>}>File-size Sweep (fio)</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Grid columns={3} gap={12}>
                <Stat value="7,064 MB/s" label="Peak read (1 GB)" tone="success" />
                <Stat value="5,187 MB/s" label="Peak write (1 GB)" tone="success" />
                <Stat value="215 MB/s" label="Small-shard read (1 MB)" />
              </Grid>
              <Callout tone="success" title="Shared filestore scales continuously — no plateau">
                Unlike typical NFS, this filestore keeps improving with block size all the way to 1 GB
                (7 GB/s read, 5.2 GB/s write). There is no sharp drop-off, so larger shards are always better.
                For LeRobot datasets, prefer <Text as="span" weight="semibold">64 MB+ episode shards</Text> to
                reach multi-GB/s throughput. Storage is ~880× faster than the training data demand (~8 MB/s).
              </Callout>
            </Stack>
          </CardBody>
        </Card>

        {/* Training assessment */}
        <Card>
          <CardHeader trailing={<Pill size="sm">Baseline — headroom available</Pill>}>π₀.₅ Training</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Grid columns={2} gap={12}>
                <Stat value={`${vramUsedPct}%`} label="VRAM used (68.8 / 288 GB)" tone="warning" />
                <Stat value="262" label="samples/sec (global)" />
              </Grid>
              <Callout tone="warning" title="VRAM only 24% utilized">
                Batch size is 32/GPU using 68.8 GB of 288 GB. There is significant room to increase
                batch size (up to ~4×) before hitting the memory wall — which would likely improve
                GPU utilization and throughput. MFU was not calculated; suggested next step for
                this config.
              </Callout>
            </Stack>
          </CardBody>
        </Card>

        {/* Checkpoint assessment */}
        <Card>
          <CardHeader trailing={<Pill active size="sm">Good</Pill>}>Checkpoint I/O</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Grid columns={2} gap={12}>
                <Stat value="10.5 GB/s" label="Save (warm rounds)" tone="success" />
                <Stat value="8.4 GB/s" label="Load (consistent)" tone="success" />
              </Grid>
              <Callout tone="info" title="Cold-start penalty on first save">
                Round 1 save is 1.28 GB/s (cold NFS/page-cache). Rounds 2–3 jump to ~10.5 GB/s
                once the path is warm. Load is consistently ~8.4 GB/s. At 11.2 GB, a warm save
                takes ~1.1 s and a load ~1.3 s — negligible for training runs.
              </Callout>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      {/* Raw results */}
      <H2>Raw Results</H2>

      <Grid columns={2} gap={16}>
        {/* HPL */}
        <Card>
          <CardHeader trailing={<Pill active size="sm">PASSED</Pill>}>HPL FP64 — Linpack</CardHeader>
          <CardBody>
            <Stack gap={16}>
              <Grid columns={2} gap={12}>
                <Stat value="8,145" label="GFLOPS total" tone="success" />
                <Stat value="1,018" label="GFLOPS / GPU" />
              </Grid>
              <Table
                headers={["Param", "Value"]}
                rows={[
                  ["Container", "nvcr.io/nvidia/hpc-benchmarks:26.02"],
                  ["Problem size N", "200,000"],
                  ["NB (block size)", "512"],
                  ["Process grid P×Q", "2 × 4"],
                  ["Elapsed", "651 s (10m 51s)"],
                  ["Residual check", "0.000261 — PASSED"],
                ]}
                framed
              />
            </Stack>
          </CardBody>
        </Card>

        {/* STREAM */}
        <Card>
          <CardHeader>STREAM — GPU Memory Bandwidth</CardHeader>
          <CardBody>
            <Stack gap={16}>
              <Grid columns={2} gap={12}>
                <Stat value="7,089" label="GB/s Triad" tone="success" />
                <Stat value="88.6%" label="of 8 TB/s peak" tone="success" />
              </Grid>
              <BarChart
                categories={["Copy", "Scale", "Add", "Triad"]}
                series={[{ name: "Bandwidth (GB/s)", data: [7061, 6996, 7111, 7089] }]}
                height={140}
                beginAtZero
                valueSuffix=" GB/s"
                referenceLines={[{ value: 8000, label: "Peak (8 TB/s)", tone: "neutral" }]}
              />
              <Text tone="secondary" size="small">
                Source: NVIDIA HPC-Benchmarks 26.02 STREAM · {RUN_DATE}. Single GPU measurement.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      {/* NCCL */}
      <Card>
        <CardHeader>NCCL AllReduce — Inter-GPU Bandwidth</CardHeader>
        <CardBody>
          <Grid columns={3} gap={12}>
            <Stat value="835 GB/s" label="Peak bus bandwidth" tone="success" />
            <Stat value="NVLink 5" label="Interconnect (1.8 TB/s)" />
            <Stat value="8 GPUs" label="Collective size" />
          </Grid>
          <Text tone="secondary" size="small" style={{ marginTop: 8 }}>
            All-reduce over 8× B300 within a single node. Source: nccl-tests · {RUN_DATE}.
          </Text>
        </CardBody>
      </Card>

      <Grid columns={2} gap={16}>
        {/* Storage sweep */}
        <Card>
          <CardHeader>File-size Sweep (fio) — /mnt/data Shared Filestore</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <BarChart
                categories={["64 KB", "256 KB", "1 MB", "4 MB", "16 MB", "64 MB", "256 MB", "1 GB"]}
                series={[
                  { name: "Write MB/s", data: [13.7, 54.4, 215.1, 798.9, 1973.8, 3728.3, 4549.8, 5187.2] },
                  { name: "Read MB/s",  data: [25.2, 88.4, 306.4, 1082.4, 2917.8, 5263.4, 6316.1, 7064.1] },
                ]}
                height={180}
                beginAtZero
                valueSuffix=" MB/s"
              />
              <Text tone="secondary" size="small">
                Sequential I/O · /mnt/data · workers=1 · passes=3. Source: fio-3.36 · job 3487 · 2026-06-25.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        {/* π₀.₅ Training */}
        <Card>
          <CardHeader>π₀.₅ Training</CardHeader>
          <CardBody>
            <Grid columns={2} gap={12}>
              <Stat value="262" label="samples / sec" />
              <Stat value="1.02" label="steps / sec" />
            </Grid>
            <Text tone="secondary" size="small" style={{ marginTop: 8 }}>
              pi05_libero config · batch 32/GPU · 8 GPUs · 68.8 GB VRAM/GPU · openpi 15a9616. Source: {RUN_DATE}.
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      {/* Summary table */}
      <H2>Summary</H2>
      <Table
        headers={["Benchmark", "Result", "vs Reference / Peak", "Verdict"]}
        rows={[
          ["HPL FP64 (8× GPU)", "8,145 GFLOPS total", "78% of B300 FP64 peak; 30× below H100 (expected)", "Expected — B300 is AI-optimized, not FP64 HPC"],
          ["STREAM (1× GPU)", "7,089 GB/s Triad", "88.6% of 8 TB/s peak; 2.1× vs H100 (3.35 TB/s)", "Excellent — near-optimal HBM utilization"],
          ["NCCL AllReduce", "835 GB/s bus BW", "~46% of NVLink 5 unidirectional BW", "Healthy — typical for 8-GPU collective"],
          ["File-size Sweep (fio)", "7,064 MB/s read · 5,187 MB/s write (1 GB)", "~880× over training demand; scales to 7 GB/s", "Prefer 64 MB+ LeRobot shards"],
          ["π₀.₅ Training", "262 samples/sec", "VRAM only 24% used (68.8 / 288 GB)", "Baseline; larger batch could improve throughput"],
          ["π₀.₅ Checkpoint", "Save 10.5 GB/s (warm) · Load 8.4 GB/s", "~1.1 s save, ~1.3 s load for 11.2 GB", "Good — cold-start penalty on round 1 only"],
        ]}
        rowTone={["warning", "success", "success", "success", "info", "success"]}
        columnAlign={["left", "left", "left", "left"]}
        striped
        framed
      />

      <Callout tone="neutral" title="Hardware context">
        B300 peak: FP64 1.3 TFLOPS · FP8 4,500 TFLOPS · FP4 13,500 TFLOPS · Mem BW 8 TB/s · NVLink 5 1.8 TB/s/GPU.
        CUDA 13.1 forward-compat on kernel 580.126.09. Networking: NVLink (intra-node) + TCP (no IB fabric).
      </Callout>

      <Divider />

      {/* === BENCHMARKS EXPLAINED === */}
      <H2>What each benchmark measures</H2>
      <Table
        headers={["Benchmark", "What it measures", "Why it matters"]}
        rows={[
          [
            "HPL FP64 (Linpack)",
            "Peak double-precision (FP64) floating-point throughput. Solves a dense linear system across all GPUs.",
            "Industry standard for HPC cluster ranking (TOP500). Stresses FP64 GEMM and intra-node memory bandwidth.",
          ],
          [
            "STREAM",
            "Sustained GPU memory bandwidth across four array kernels: Copy, Scale, Add, Triad.",
            "Reveals whether HBM is being used efficiently. Bottleneck for memory-bound workloads like large-batch LLM inference.",
          ],
          [
            "NCCL AllReduce",
            "Collective communication throughput: every GPU sends data, all GPUs receive the sum.",
            "Core operation in DDP/FSDP gradient synchronization. Sets the ceiling for multi-GPU training scaling efficiency.",
          ],
          [
            "File-size Sweep (fio)",
            "Sequential read and write throughput at block sizes from 64 KB to 1 GB on the shared filestore.",
            "Reveals the raw filesystem ceiling and the optimal LeRobot shard size. Look for where throughput plateaus — set shards at or above that size.",
          ],
          [
            "π₀.₅ Training",
            "End-to-end training throughput of the π₀.₅ vision-language-action model using synthetic data on 8 GPUs.",
            "Directly reflects how fast you can train your actual workload. Combines compute, memory bandwidth, and DDP communication.",
          ],
          [
            "π₀.₅ Checkpoint",
            "Speed of saving and restoring a full model checkpoint (weights + optimizer state) to/from shared storage.",
            "Long checkpoints add overhead to fault-tolerant training. Fast save/load reduces downtime when resuming after a preemption.",
          ],
        ]}
        columnAlign={["left", "left", "left"]}
        striped
        framed
      />
    </Stack>
  );
}
