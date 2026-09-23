# Completing the attention comparison with compiled tiled PyTorch

## Status and comparison

The implementation, runners, tests, and plotting support are ready for an H200
run. No new GPU measurements are included by this change. The archived three-way
results remain historical measurements; do not infer results for the new method.

All new main and batch sweeps compare four implementations:

| Result identifier | Forward and backward execution |
|---|---|
| `naive` | Existing compiled dense PyTorch |
| `flash_pytorch` | Existing eager tiled PyTorch |
| `flash_pytorch_compiled` | The same tiled algorithm, compiled with TorchInductor |
| `flash_triton` | Existing explicit Triton kernels and backward preprocessing |

`CompiledFlashAttention` compiles the complete tiled forward and manually derived
tiled backward separately with `torch.compile(backend="inductor", fullgraph=True,
dynamic=False)`. The eager and compiled classes share their algorithm functions.
There is no SDPA substitution, per-tile-only compilation, or eager fallback on
graph breaks. Compiler errors are failed cases. Large statically unrolled graphs
can take substantial time or memory to compile; failure and timeout results are
part of the experiment, not evidence of successful steady-state performance.

## Coverage and tiles

All cases use causal masking. The main and tile sweeps use batch size 1.

| Sweep | Grid | Cases |
|---|---|---:|
| Main | N = 128 through 65,536 in powers of two; D = 16/32/64/128; BF16/FP32; four methods | 320 |
| Batch | B = 1/2/4/8/16; N = 128/2,048/16,384; D = 64; BF16/FP32; four methods | 120 |
| Triton tile sensitivity | N = 1,024/4,096/16,384/65,536; D = 64; BF16/FP32; BM and BN independently 32/64/128 | 72 |
| PyTorch tile sensitivity | Same N/D/precision as Triton; square tiles 256/512/1,024; eager and compiled | 48 |

The full experiment has **560 configurations**. Naive dense attention has no tile
parameter, so it participates in the main and batch sweeps only.

For the main sweep, compiled tiled PyTorch reuses the eager implementation's
selected tile for each shape. If its cache does not yet exist, the runner tunes
the **eager** implementation and saves that choice under the eager cache key.
Both methods use the original bounded tuning policy: tune through N=2,048 and
reuse that choice at longer lengths. This isolates compilation from tile choice.

For the batch sweep, both tiled PyTorch variants use the archived eager baseline
tiles, fixed across batch sizes. For tile sensitivity, both run the same explicit
square tiles. Triton's rectangular tile search is reported separately because
the current PyTorch algorithm accepts one square tile size.

## Run on the H200

Use a Python environment with PyTorch, Triton, NumPy, and Matplotlib that has
already passed CUDA preflight on the server. The archived batch sweep succeeded
with PyTorch 2.8.0+cu128 and Triton 3.4.0; the repository's copied lockfile targeted
a different stack that failed on that server. Do not assume `uv run --locked`
selects the successful environment. Actual package versions are recorded in each
case. Run all four methods in the same environment for the new comparison.

From the repository root, first inspect the grids without GPU work:

```sh
python scripts/benchmark_attention.py --tile-sensitivity --dry-run
python scripts/benchmark_batch_sizes.py --dry-run
```

The first command lists 440 main-plus-tile cases; the second lists 120 batch
cases. Then run a small, isolated GPU smoke sweep and the multi-batch validation:

```sh
python scripts/benchmark_attention.py --seq-lengths 128 --dims 64 --output benchmark_results/h200_compile_smoke
python scripts/benchmark_batch_sizes.py --smoke-only --output benchmark_results/h200_batch_compile_smoke
```

Inspect `progress.json`, `cases/`, and worker logs. The batch smoke command also
validates multiple-tile regimes and all Q/K/V gradients at batch size 2.
Do not proceed past a correctness failure.

Run the complete sweeps into fresh directories:

```sh
python scripts/benchmark_attention.py --tile-sensitivity --output benchmark_results/h200_compile_comparison
python scripts/benchmark_batch_sizes.py --output benchmark_results/h200_batch_compile_comparison
```

These runs rerun all four implementations, including batch-size-1 baselines.
Do not merge old timings from different software environments into the new
comparison. The runners expose GPU 0 to each isolated worker. Run sweeps serially
to avoid concurrent GPU contention.

The default per-case timeout is 600 seconds, including validation, compilation,
and measurements. On timeout, the worker process group is stopped so compiler
subprocesses are not left running. Main-sweep overrides use `--timeout-seconds`;
batch-sweep overrides belong in a copied JSON config passed with `--config`.
Use a new output directory when changing configuration or retrying failures.
`--resume` requires matching configuration and source hashes and preserves
completed successes and failures.

## Timing, compilation, memory, and validation

Forward, backward, and end-to-end latency are independent median CUDA-event
measurements. Compilation, validation, and first calls are outside those timing
regions. End-to-end throughput is `B * N * 1000 / forward_backward_ms`.
Forward and forward-plus-backward peak memory are separate untimed passes after
warmup. Memory is PyTorch-allocated GPU memory in decimal GB, including inputs;
compiler host memory is not included.

CSV and raw JSON retain these additional wall-clock measurements:

- `validation_seconds`: reference comparison and the implementation's first
  forward/backward on the validation shape, including compilation if needed.
- `forward_first_call_seconds`: first full-size forward, including compilation
  if that specialization was not already warmed by validation.
- `backward_first_call_seconds`: first full-size backward under the same rule.
- `forward_backward_first_call_seconds`: the independent end-to-end warmup call.

These are **compile-inclusive first-call costs, not pure compilation time**.
Compiler caches can be warm, and validation can compile the same shape as the
measurement. Keep those costs separate from steady-state latency in the report.
The raw compiled-case record includes backend, full-graph, and shape-specialization
settings. A checkpoint precedes each first call, preserving the stage if it times
out. Compiler errors include their exception message and worker log.

Existing checks compare outputs and Q/K/V gradients against dense FP32 on a
smaller validation shape, then check full-size outputs/gradients for finiteness.
They are smoke correctness checks, not exhaustive full-size numerical validation.
Inspect recorded sample counts: slow cases may have few repetitions under the
100 ms default budget. Increase `--rep-ms` (main) or `rep_ms` in a separate batch
config when needed, and document the changed budget. Batch `review.json` lists
failures, missing validations, and stages below `minimum_samples`.

## Complete the report after execution

The main runner writes the full results table, three latency figures, and two
memory figures with all four methods. Tile results include three Triton heatmaps
and three eager-versus-compiled PyTorch square-tile comparisons. The batch runner
writes three latency figures, throughput, and two memory figures. Both report
missing measurements as gaps and annotate OOM, resource limit, error, and timeout
outcomes rather than inventing values.

Regenerate figures from recorded cases without rerunning GPU work:

```sh
python scripts/plot_attention_benchmarks.py benchmark_results/h200_compile_comparison
python scripts/plot_batch_sizes.py benchmark_results/h200_batch_compile_comparison
```

Compare compiled against eager tiled PyTorch at the **same tiles**, then compare
it against naive and Triton. Discuss steady-state speed, first-call cost, peak
memory, and compilation feasibility separately. Retain the raw records, source
snapshots, environment details, tables, and PNG/PDF figures with the report.

Reference: [PyTorch torch.compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile).
