# Completed H200 batch-size sweep

All 90 planned configurations completed successfully on September 23, 2026,
in approximately 11.9 minutes, including preflight validation and plotting.
The grid is batch sizes 1/2/4/8/16, sequence lengths 128/2048/16384, dimension 64,
BF16/FP32, and compiled dense PyTorch, eager tiled PyTorch, and Triton attention.
All cases use causal masking. All batch-size-1 measurements were rerun.
Tiles are fixed from the original `h200_memory_tiles` baseline, with sequences
longer than 2048 reusing its 2048-token choice; no batch-dependent retuning occurred.

## Read the results

- `results.csv` and `results.md`: latency, throughput, peak memory, tiles, and status.
- `cases/`: all 90 raw records, including sample counts, memory baselines,
  correctness checks, and per-case software/GPU environment.
- `plots/`: six figures, each in PNG and PDF.
- `smoke/cases/`: ten successful preflight checks at batch size 2, comparing
  outputs and Q/K/V gradients against the dense reference across tile regimes.
- `review.json`: 58 timing stages across 21 cases have fewer than five samples;
  some have only one. These remain valid recorded measurements but have limited
  sampling. No higher-repetition follow-up was run before archiving.
- `validation_report.json`: archival completeness and consistency checks.
- `manifest.json`, `baseline_manifest.json`, `config.json`, `tuning/`, and source
  snapshots: exact run configuration, provenance, and fixed tile selections.
- `environment_packages.json`: package versions visible to the actual run.

Throughput uses the independently measured end-to-end latency. Peak memory is
PyTorch-allocated memory in decimal GB, including resident inputs; it is not
reserved memory or total process memory. See the root `BENCHMARKING.md` for details.

## Actual execution environment

The successful run used Python 3.12.3, PyTorch **2.8.0+cu128**, Triton **3.4.0**,
and CUDA **12.8**, on one NVIDIA H200 with TF32 disabled. Plotting dependencies
were loaded from `/tmp/cs336-batch-plot-deps`; their exact versions are recorded
in `environment_packages.json`. That temporary directory is not needed to read
the archived results or regenerate figures after installing the plotting packages.

**The copied `uv.lock` was not the successful run's environment.** Its PyTorch
2.11/CUDA 13 build failed preflight because the available CUDA driver interface
was too old. The failed attempt is preserved alongside this directory and its logs
are included in the diagnostics archive. The runner was then relaunched with the
previously GPU-validated system Python environment. `launch_environment.json`
records that choice. Use the recorded successful package versions when reproducing
these measurements; `uv run --locked` selects the incompatible environment on this
server. Source hashes in `manifest.json` identify the exact benchmark implementation.

## Retained diagnostics and portability

`../archives/batch_sweep_diagnostics.tar.gz` contains the full run's ignored jobs
and logs, smoke logs, and the failed CUDA launch, including its logs.
`../archives/batch_sweep_validation_runs.tar.gz` preserves the preflight and
three-case end-to-end runs originally under `/tmp`, plus the dry-run job grid.
Synthetic plotting data are not experiment measurements and are not archived.

Absolute paths in job JSON and historical PIDs describe the original server.
They need not exist to inspect the results. To rerun, use the root
`scripts/benchmark_batch_sizes.py` with a compatible environment and a fresh
`--output` directory; it generates new paths and performs preflight validation.
To regenerate only the figures, install the recorded Matplotlib/NumPy versions
and run `python scripts/plot_batch_sizes.py benchmark_results/h200_batch_sweep`.

The repository also retains the earlier benchmark results and retirement archives.
`../artifact_inventory.json` provides SHA-256 hashes and sizes for preserved
artifacts. Runtime environments, downloaded wheels, compiler caches, PID/lock files,
and synthetic plot checks are disposable and are not required archival data.
