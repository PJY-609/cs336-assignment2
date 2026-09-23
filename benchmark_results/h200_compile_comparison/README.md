# Partial compiled attention comparison — stopped by request

The H200 run was stopped on September 23, 2026 after approximately 2 hours
27 minutes. This is a partial experiment, not completion of the 560-case plan.
Source revision: `19cd502c65158f73d4ca78254448e05659df74bc`.

## Stopping point

- 290 of 440 main-plus-tile cases reached terminal outcomes: 286 successful,
  4 timed out at the 600-second per-case limit. All completed cases are main cases.
- Interrupted: `flash_pytorch_compiled_n65536_d16_bfloat16`, during
  `forward_first_call`. Its raw nonterminal checkpoint remains in `cases/`.
  This user interruption is not a timeout or correctness failure.
- Main sweep: 30 configurations unfinished, including the interrupted case.
- Tile sensitivity: all 120 configurations unstarted.
- Batch comparison: all 120 configurations unstarted.
- Total remaining to fulfill the original plan: 270 configurations.
- Supervisor, runner, and active worker/compiler process groups were stopped;
  the GPU had no compute processes afterward. Recorded PIDs are historical.

`results.csv`, `results.md`, and five PNG/PDF figures were refreshed from the
290 terminal records. The interrupted checkpoint is excluded from these reports.
Empty tile tables indicate an unstarted sweep. Missing values are not zeros.
`validation_report.json` lists low-sample stages and archival checks.
`progress.json` and `../h200_compile_experiment_launch/status.json` record the stop.
Worker logs and jobs, including the interrupted case, are preserved in
`../archives/compiled_comparison_partial_diagnostics.tar.gz`.

## What these results establish

For N=128 through 16384, compiled tiled PyTorch beat matched-tile eager PyTorch
in all 64 completed configurations, with end-to-end speedups approximately
1.75–4.9x. At N=32768 four compiled configurations succeeded and four timed out.
Long-sequence timings can have only one or two samples; do not overstate their
precision. First-call costs include compilation and execution, and can reuse
compiler caches. They are not pure cold compilation times.
These results do not establish batch scaling or tile sensitivity for the compiled
implementation. Preserve every timeout when reporting compilation feasibility.

## Validation and environment

Before launch, all eight small GPU benchmark cases, 14 multi-batch output and
Q/K/V-gradient checks, and eight CPU orchestration/report tests passed. These
checks and their provenance remain in the sibling smoke directories.
The run used system Python 3.12.3, PyTorch 2.8.0+cu128, Triton 3.4.0 and CUDA 12.8.
`launch_environment.json` records exact plotting versions; the supervisor's
`environment.json` also records environment overrides and source revision.
The repository lockfile is a source snapshot, not the successful runtime.
Do not use `uv run --locked` to reproduce this server's environment.

## Resume on this server

Use the same hardware, software, source files, tiles, and timing settings:
25 ms warmup, 100 ms repetition budget, 600-second case timeout. The temporary
plotting dependencies currently live in `/tmp/cs336-compile-check-deps`.
From the repository root, resume both sweeps serially in the background:

```sh
export PYTHONPATH="/tmp/cs336-compile-check-deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
nohup bash benchmark_results/h200_compile_experiment_launch/resume.sh \
  > benchmark_results/h200_compile_experiment_launch/resume.log 2>&1 < /dev/null &
echo $!
```

The resume script checks for an existing main runner, then runs:

```sh
python scripts/benchmark_attention.py --tile-sensitivity --resume --output benchmark_results/h200_compile_comparison
python scripts/benchmark_batch_sizes.py --resume --output benchmark_results/h200_batch_compile_comparison
```

The main runner skips terminal outcomes, including the four timeouts, and retries
the interrupted case from scratch. It then completes the remaining main and tile
cases. The batch runner starts its 120-case run with fresh preflight validation;
its `--resume` also permits continuing a later interrupted batch run. If a runner
exits unsuccessfully, the script stops before starting the next sweep.
Use fresh output directories for retrying terminal failures or changing timing
budgets; resume rejects changed source hashes/configuration. Do not invoke the
original `run.py` to resume: it requires fresh output directories.

Monitor each output directory's `progress.json` and `resume.log`, plus the
supervisor `resume_status.txt`. Original `status.json` remains the historical
stop record. No resume was launched when this snapshot was committed.

## Resume after moving servers

First recreate a CUDA-compatible environment using the recorded versions and
validate it on an H200. Install the recorded plotting packages in a persistent
location and adjust `PYTHONPATH` accordingly. The temporary dependency directory,
compiler caches, historical absolute paths, and PID files are not portable.
The runners generate new job paths on resume; diagnostic archives need not be
extracted. Source fingerprints must match the manifest. A different environment
should use fresh result directories so the comparison does not mix environments.
