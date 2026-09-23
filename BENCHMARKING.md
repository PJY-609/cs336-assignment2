# H200 attention benchmark

The archived run below used three implementations. The current runners also
include `flash_pytorch_compiled`, compiling the same tiled PyTorch forward and
manual backward with TorchInductor. See
[the expanded experiment instructions](COMPILED_ATTENTION_EXPERIMENT.md) for the
320-case main sweep, 120-case batch sweep, 120 tile cases, fresh output paths,
compile-inclusive first-call reporting, and compatible-environment guidance.
Those new measurements have not been run as part of this change. The historical
commands and counts below describe the archived three-method experiment.

The main sweep runs 240 cases on GPU 0: batch size 1, causal masking, sequence
lengths 128 through 65536 in powers of two, dimensions 16/32/64/128, and BF16/FP32.
The additional tile-sensitivity experiment runs 72 cases, for 312 total.
Every case records latency and separate forward / forward+backward peak memory.

Implementations:

- `naive`: the existing `NaiveAttention`, with dense score matrices and compiled
  PyTorch forward/backward. It does not call SDPA or FlashAttention.
- `flash_pytorch`: the existing eager, tiled `FlashAttention`, with Python loops.
- `flash_triton`: the existing `FlashAttentionTriton`, including its PyTorch
  preprocessing in backward.

## Run and inspect

```sh
source benchmark.env
uv run --locked python scripts/benchmark_attention.py --output benchmark_results/h200_memory_tiles --tile-sensitivity
```

The background launch saves its PID to `benchmark_results/h200_memory_tiles/run.pid` and its
console output to `run.log`. Inspect without attaching to the process:

```sh
cat benchmark_results/h200_memory_tiles/progress.json
tail -n 10 benchmark_results/h200_memory_tiles/run.log
```

`results.csv` and `results.md` are replaced atomically after each completed case.
`tile_sensitivity.csv` and `tile_sensitivity.md` contain the fixed-tile experiment.
`cases/` contains per-stage checkpoints, environment information, correctness
checks, timing sample counts, and detailed errors. `logs/` contains worker logs.
`tuning/` contains candidate timings, validation outcomes and selected tiles.
`manifest.json` records the sweep configuration, PID, start time and GPU/driver
information. Copies of the runner, attention source, lockfile and environment
configuration are stored with the results.

Plots are refreshed every 24 completed cases and at completion, as PNG and PDF:

- `plots/memory_forward.*` and `plots/memory_forward_backward.*`: total peak
  allocated memory versus sequence length, with panels for each dimension and
  precision and a curve for each implementation. OOMs without a measurement are
  annotated at the top of the axes, never assigned a fabricated memory value.
- `plots/tile_sensitivity_forward.*`, `plots/tile_sensitivity_backward.*`, and
  `plots/tile_sensitivity_forward_backward.*`: fixed-tile latency heatmaps by
  sequence length and precision. Cells show latency, OOM, LIMIT (kernel resource
  limit), ERR (other error), or a dash for pending.
  These appear once the tile experiment begins producing results.

Regenerate plots from saved case files without using the GPU:

```sh
uv run --locked python scripts/plot_attention_benchmarks.py benchmark_results/h200_memory_tiles
```

The superseded latency-only run remains in `benchmark_results/h200`; it was
stopped before restarting the expanded experiment in the new directory.

After a stopped run, use the same command with `--resume`. Do not resume while
another process is still running in that output directory. Resume reuses finished
cases, including recorded failures, and requires the same sweep configuration.
Use a new output directory to retry all cases or change settings.

## Measurement details

All timings use `triton.testing.do_bench`, with a 25 ms warmup budget and a 100 ms
repetition budget, reporting the median CUDA-event latency in milliseconds.
These are time budgets, not repetition counts; slow cases may have only one
measured repetition. Sample counts are saved in each case JSON. For more samples,
raise `--rep-ms`. CUDA-event measurements of these Python-dispatched operations
also include GPU idle gaps caused by host dispatch; these are implementation
latencies, not sums of individual kernel durations.

Inputs and upstream gradients are generated on the GPU before timing, using the
same seed for corresponding cases. Compilation, tile tuning, correctness checks,
and explicit warmup calls are outside the measurements. FP32 uses highest
matmul precision with TF32 disabled; existing Triton dot products use IEEE FP32.
No GPU clock changes are made.

- Forward measures the training-mode autograd function, including saved tensors.
- Backward reuses an already-computed forward result with `retain_graph=True`.
- Forward+backward creates a fresh result and computes its gradients each time.

Backward uses `torch.autograd.grad`, avoiding leaf `.grad` accumulation. The
combined timing is measured separately, not calculated by adding other columns.
Normal allocations inside the implementation remain part of the measurement.

Each case uses a separate process, releasing its CUDA allocator on completion.
OOMs and other errors are recorded rather than reported as numeric latencies;
completed stages remain available if a later stage fails. There is no default
case timeout. An optional `--timeout-seconds` can limit unusually slow cases.

## Peak memory

Memory is measured separately from timing, after compilation and warmup. Before
each measured memory pass, previous outputs and gradients are released, garbage
is collected, CUDA is synchronized, and unused allocator cache is emptied. The
current allocated baseline is recorded and `torch.cuda.reset_peak_memory_stats()`
is called. After one forward (or one fresh forward+backward), CUDA is synchronized
and `torch.cuda.max_memory_allocated()` is recorded before releasing the result.

Both measurements start with Q, K, V and the upstream gradient already allocated.
CSV/Markdown columns `forward_peak_gb` and `forward_backward_peak_gb` report the
total allocated peaks in **decimal GB (1 GB = 1,000,000,000 bytes)**. CSV also
includes incremental peaks above each pre-call baseline. Case JSON files preserve
the baseline and peak byte counts. Forward memory includes tensors saved for
training; combined memory includes the returned Q/K/V gradients.

These are PyTorch allocator measurements, not reserved-memory or `nvidia-smi`
process-memory readings. They exclude CUDA context and allocations outside the
PyTorch allocator. Memory passes do not include `do_bench`'s cache-flushing buffer
or alter the reported timing measurements. Successfully measured forward memory
is retained if backward later fails with an OOM.

## Tile selection and validation

Both attention classes now accept optional tile arguments; their previous
16-token defaults and existing call signatures remain supported. Backward uses
the configuration saved by its corresponding forward call.

For each dtype, dimension and sequence length up to 2048, tile tuning compares
end-to-end latency, outside the final measurement:

- PyTorch: square tiles 256, 512 and 1024, capped at the sequence length.
- Triton: query/key tiles `(16,16)`, `(32,32)`, `(64,32)` and `(64,64)`.

Candidates are correctness-checked and timed with `do_bench` using 5 ms warmup
and 15 ms repetition budgets. Failed candidates are recorded and excluded.
Longer sequences reuse the corresponding 2048-token choice. This bounded search
is not exhaustive autotuning and may miss better long-sequence configurations.

## Fixed-tile sensitivity experiment

`--tile-sensitivity` adds an independent Triton experiment at dimension 64:

- Sequence lengths: 1024, 4096, 16384, 65536.
- Both BF16 and FP32.
- Every pair from query tile BM in {32, 64, 128} and key tile BN in {32, 64, 128}.

Each pair is measured at the **actual requested sequence length**, without
replacing it with an autotuned tile. It records forward, backward, combined
latency, and both peak-memory measurements. Backward uses the same BM/BN as
forward, and all other kernel launch settings remain unchanged. Numerical,
compilation/resource, and OOM failures are preserved in the table and worker log.
For example, the FP32 128x128 tile smoke test at d=64 exceeded the H200's
per-block shared-memory limit with the existing launch settings; such cases are
recorded as `resource_limit`, distinct from exhausting GPU device memory (`oom`).
This experiment runs after the main sweep. `--tile-seq-lengths` and `--tile-sizes`
can narrow it for smoke tests.

These comparisons establish tile-dependent performance; attributing differences
specifically to occupancy, register spills or memory traffic requires separate
profiler measurements rather than inferring those quantities from latency alone.

## Correctness checks

Validation compares outputs and all three input gradients against an eager,
dense FP32 reference on a smaller sequence (at least 128 tokens, and up to twice
the tile size, capped by the requested sequence length). Absolute and relative
tolerances are 0.001 for FP32 and 0.05 for BF16. This is a smoke correctness check,
not proof of accuracy across the entire sweep. Each full-size case also checks
that outputs and gradients are finite before reporting all three latencies.
