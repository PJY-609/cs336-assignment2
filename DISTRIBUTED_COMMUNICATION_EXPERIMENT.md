# Next experiment: single-node all-reduce on six A100 SXM GPUs

Status: completed on 2026-09-25 UTC; all 36 launches passed. See the
[results and hardware report](benchmark_results/a100_sxm_all_reduce/README.md).
This plan follows the requested `distributed_communication_single_node` problem.

## Objective and hardware

Measure how FP32 all-reduce latency changes with tensor size and process count.
Rent one physical host with **6 NVIDIA A100 SXM GPUs**, and confirm NVSwitch
connectivity with the provider. Keep the same allocation for every comparison,
using subsets of 2, 4, and 6 GPUs with one process per GPU. If only an eight-GPU
host is available, use six of its GPUs and account for the full rental cost.

Record `nvidia-smi` and `nvidia-smi topo -m` output, selected GPU IDs, GPU memory
capacity, driver, CUDA, PyTorch and NCCL versions, and the repository commit.
Verify the actual topology rather than assuming that SXM guarantees NVSwitch.

## Experiment matrix

Use contiguous `torch.float32` tensors. Sizes below are **per rank**, in decimal
units: 1 MB = 1,000,000 bytes and 1 GB = 1,000,000,000 bytes.

| Tensor size per rank | FP32 elements | Process/GPU counts |
| --- | ---: | --- |
| 1 MB | 250,000 | 2, 4, 6 |
| 10 MB | 2,500,000 | 2, 4, 6 |
| 100 MB | 25,000,000 | 2, 4, 6 |
| 1 GB | 250,000,000 | 2, 4, 6 |

This produces 12 configurations. Use NCCL on CUDA for the reported measurements;
use Gloo on CPU with small inputs for local debugging.

## Measurement protocol

1. Implement a configurable benchmark script with backend, tensor device, process
   count, tensor size, warmup count, measured iteration count, and output path.
   Initialize one process group per launch and bind each CUDA rank to its GPU
   before any NCCL collective, including object gathering.
2. Validate SUM all-reduce against a known expected result outside the timed
   region. Allocate tensors before timing; reset inputs outside timing to avoid
   repeatedly summing already-reduced values and overflowing.
3. Run at least 5 warmup all-reduces for each configuration. Start with 50 measured
   iterations per configuration and keep each launch under 5 minutes. Record
   actual iteration counts and any timeout or failure.
4. Before each measurement, prepare inputs, align ranks with a barrier, and call
   `torch.cuda.synchronize()` on the local device. Start a monotonic wall-clock
   timer, run `dist.all_reduce` with `async_op=False`, synchronize CUDA again,
   and stop the timer. The final synchronization is necessary because the
   collective returning does not imply GPU completion. Exclude the preparation
   and barrier from the timed region. Omit CUDA synchronization for CPU debugging.
5. Gather per-rank timing samples with `dist.all_gather_object` after timing is
   complete. Preserve raw samples. For each iteration, take the maximum latency
   across ranks; summarize those values with the median and interquartile range.
   Also report each rank's median to expose rank-to-rank variation. Label these
   as synchronized host-observed call latencies, not pure GPU kernel times.
6. Run the matrix three times on the same allocation, saving each repeat
   separately. Record GPU subsets and any competing workload. Destroy process
   groups cleanly after each launch.

## Deliverables

Save outputs under `benchmark_results/a100_sxm_all_reduce/` when the experiment
is run:

- Environment and topology metadata, launch settings, and logs.
- Raw per-rank timing samples and a summary CSV, with tensor bytes, world size,
  repeat, iteration counts, median latency, and interquartile range.
- A table comparing all 12 configurations and a latency-versus-tensor-size plot
  with separate curves for 2, 4, and 6 GPUs and clearly labeled units.
- Two or three sentences discussing the observed effect of tensor size and GPU
  count, referencing the measured topology and variation across repeats.

Small messages may be dominated by launch and synchronization overhead; larger
messages may expose communication bandwidth limits. More ranks can increase
coordination and communication costs, so lower latency is not guaranteed. These
are hypotheses to evaluate, not measured conclusions.

## Execution checklist

- [x] Implement the script and check correctness locally with Gloo/CPU.
- [ ] Confirm six A100 SXM GPUs share a host and verify NVSwitch connectivity.
- [x] Capture environment metadata and run a short NCCL smoke test.
- [x] Run all 12 configurations and repeats on the same allocation.
- [x] Export the table, plot, raw data, and results commentary.
- [ ] Download results and terminate the rental promptly.

## Prepared runner and environment

Run from the repository root. Install the exact `uv.lock` environment with:

```sh
bash scripts/setup_all_reduce.sh
```

The setup script uses copy mode and `/tmp/cs336-uv-cache` by default because the
standard cache failed with `Not supported (os error 95)` on this host. Override
`UV_CACHE_DIR` if needed. The benchmark uses the resulting `.venv` directly.

CPU correctness check (small tensors, not experiment measurements):

```sh
.venv/bin/python scripts/benchmark_all_reduce.py \
  --backend gloo --device cpu --world-sizes 2 --sizes 4096 40000 \
  --iterations 3 --repeats 1 \
  --output benchmark_results/a100_all_reduce_setup/cpu_smoke
```

Short GPU check on every planned process count:

```sh
.venv/bin/python scripts/benchmark_all_reduce.py \
  --gpu-ids 0,1,2,3,4,5 --world-sizes 2 4 6 --sizes 1000000 \
  --iterations 3 --repeats 1 \
  --output benchmark_results/a100_all_reduce_setup/nccl_smoke
```

Full experiment (36 launches: 12 configurations, three repeats):

```sh
.venv/bin/python scripts/benchmark_all_reduce.py \
  --gpu-ids 0,1,2,3,4,5 \
  --output benchmark_results/a100_sxm_all_reduce
```

Defaults are NCCL/CUDA, 5 warmups, 50 measured iterations, and a 290-second
wall-clock timeout per launch. Each configuration starts a fresh process group.
The runner fails immediately on an error, records launch status and logs, and
terminates timed-out process trees. Select a **new output directory** for each
invocation; existing directories are rejected to preserve results.

Each output directory contains environment/topology metadata, the exact runner
and lockfile, per-launch commands and GPU selections, logs, raw timing JSON,
and `summary.csv`. JSON includes every rank's samples and median, per-iteration
rank maxima, median and IQR in milliseconds. SUM correctness is checked before
warmup and after measurement. Inputs are reset before every collective.
Document any known competing workload separately; `nvidia-smi` captures a
snapshot at startup. The completed report, table, and plot are in `benchmark_results/a100_sxm_all_reduce/`.
