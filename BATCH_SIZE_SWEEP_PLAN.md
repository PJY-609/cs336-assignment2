# Representative batch-size sweep

## Purpose

Extend the completed H200 attention benchmark with a focused experiment on how
batch size affects latency, throughput, and peak GPU memory. Keep the assignment's
required batch-size-1 sweep as the main result; present this as an additional
experiment.

A full batch-size expansion of the existing sequence-length, dimension, and
precision grid would multiply runtime and make the results harder to interpret.
Start with representative short, intermediate, and long sequences instead.

## Initial experiment grid

| Parameter | Values |
|---|---|
| Batch size | 1, 2, 4, 8, 16 |
| Sequence length | 128, 2,048, 16,384 |
| Embedding dimension | 64 |
| Precision | `torch.bfloat16`, `torch.float32` |
| Implementation | Compiled dense PyTorch, eager tiled PyTorch, Triton FlashAttention |
| Masking | Causal for every case |
| Hardware | One NVIDIA H200, matching the original benchmark |

The Cartesian product contains **90 cases**: 5 batch sizes × 3 sequence lengths
× 1 dimension × 2 precisions × 3 implementations. Eighteen batch-size-1 cases
already exist in the original sweep, leaving 72 additional configurations.
Reuse those baseline measurements only if the hardware, code, and measurement
conditions match; otherwise rerun all 90 cases and retain the old results separately.

Add dimension 128 as a follow-up if the initial trends warrant it. Do not expand
to every dimension and sequence length by default.

## Tile policy

For the initial experiment, hold tiles fixed across batch sizes. Use each
implementation's existing batch-size-1 tile choice for the matching sequence
length, dimension, and precision. Preserve the original policy in which sequences
longer than 2,048 reuse the corresponding 2,048-token choice.

This isolates the effect of batch size under a fixed tile configuration. Record
query and key tile sizes for every case. If best achievable performance at each
batch size becomes the goal, conduct a separate experiment that retunes tiles
per batch size, and clearly distinguish it from this fixed-tile comparison.

## Measurements and validation

Follow [the existing benchmark methodology](BENCHMARKING.md), including precision
settings, untimed compilation and validation, separate case processes, and memory
accounting. Extend the runner to accept and record batch size before execution.
Smoke-test multi-batch outputs and all input gradients against a dense reference
before running the sweep; preserve the existing full-size finiteness checks.

Record the following for each case:

- Forward latency in milliseconds.
- Backward latency in milliseconds, using an already-computed forward result.
- End-to-end forward-plus-backward latency in milliseconds, measured independently
  rather than calculated by summing the other two measurements.
- End-to-end throughput in tokens per second:
  `batch_size * sequence_length * 1000 / forward_backward_ms`.
- Separate forward and forward-plus-backward peak allocated GPU memory in decimal
  GB, including resident inputs; retain baseline and incremental memory values.
- Tile configuration, timing sample counts, validation results, environment, and
  completion or failure status.

Keep input generation outside timing and use a consistent seed policy. Preserve
OOMs and kernel resource-limit failures as explicit outcomes, not numeric latency
or throughput values. Retain measurements from stages that completed before a
failure. Inspect sample counts for slow cases and increase the repetition budget
if needed, documenting any differences from the original run.

## Tables and visualization

Save a complete results table with batch size, sequence length, dimension,
precision, implementation, tiles, all three latencies, throughput, peak memory,
and status. Keep raw case records and the run configuration alongside the table
in a separate results directory, such as `benchmark_results/h200_batch_sweep/`.

For each figure, use batch size on the x-axis, sequence lengths as three columns,
and BF16/FP32 as two rows. Use consistent colors and marker shapes for the three
implementations. Show all five measured batch sizes explicitly.

Produce separate figures for:

1. Forward latency.
2. Backward latency.
3. End-to-end latency.
4. End-to-end throughput in tokens/s.
5. Forward peak allocated memory.
6. Forward-plus-backward peak allocated memory.

Use comparable axis scales within each figure and logarithmic latency axes when
needed to keep all implementations readable. Annotate failures without assigning
fabricated values or connecting lines across missing measurements. Export PNG
and PDF versions.

## Questions to answer

- Does increasing batch size improve throughput, and where does it level off?
- How do those trends differ between short and long sequences and between BF16
  and FP32?
- Does the relative ranking of the three implementations change with batch size?
- What latency and memory costs accompany any throughput gains?

Treat these as empirical questions. Do not attribute performance differences to
occupancy, register pressure, or memory traffic without separate profiler evidence.

## Completion criteria

The experiment is complete when all 90 configurations have either measurements
or explicit failure records, validation and timing sample counts have been
reviewed, and the results table and six figures are available. Document whether
batch-size-1 results were reused or rerun and identify any environment differences.
Decide on a dimension-128 extension or a separate retuned sweep only after reviewing
the initial results.

This document is an execution plan; it does not indicate that the batch-size
sweep has been run.
