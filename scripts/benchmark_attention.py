"""Isolated, resumable H200 attention sweep; see BENCHMARKING.md."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = ('naive', 'flash_pytorch', 'flash_pytorch_compiled', 'flash_triton')
PYTORCH_TILED = ('flash_pytorch', 'flash_pytorch_compiled')
FIRST_CALL_FIELDS = ('validation_seconds', 'forward_first_call_seconds',
                     'backward_first_call_seconds', 'forward_backward_first_call_seconds')
FIELDS = ('implementation', 'sequence_length', 'embedding_dim', 'dtype', 'q_tile', 'k_tile',
          'forward_ms', 'backward_ms', 'forward_backward_ms',
          'forward_peak_gb', 'forward_backward_peak_gb',
          'forward_incremental_gb', 'forward_backward_incremental_gb',
          'experiment', 'status', 'wall_seconds', 'stage', 'error') + FIRST_CALL_FIELDS
TABLE_FIELDS = FIELDS[:11] + ('status',)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def write_table(directory, rows, name, title):
    temporary = directory / f'{name}.csv.tmp'
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / f'{name}.csv')
    lines = [f'# {title}', '',
             'Batch size 1; causal masking; median CUDA-event latency in milliseconds.',
             'Naive = existing compiled dense PyTorch; Flash PyTorch = eager tiled Python loops.',
             'flash_pytorch_compiled = the same tiled forward and backward with full-graph torch.compile.',
             'First-call wall times (including compilation when needed) are reported separately in CSV/JSON.',
             'Tile tuning, correctness checks, input generation and compilation are outside timing.',
             'Peak memory is total PyTorch-allocated GPU memory in decimal GB, including preallocated inputs.',
             'Forward and forward+backward memory are measured in separate untimed passes.',
             'Partial results are saved after each case. Blank cells are not measured values.', '',
             '| Implementation | N | D | Dtype | Q tile | K tile | Forward ms | Backward ms | Fwd+bwd ms | Forward peak GB | Fwd+bwd peak GB | Status |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        values = []
        for key in TABLE_FIELDS:
            value = row.get(key, '')
            values.append(f'{value:.4f}' if isinstance(value, float) else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    temporary = directory / f'{name}.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(directory / f'{name}.md')


def write_tables(directory, rows):
    write_table(directory, [row for row in rows if row['experiment'] == 'main'],
                'results', 'Attention benchmark on a single H200')
    write_table(directory, [row for row in rows if row['experiment'] == 'tile_sensitivity'],
                'tile_sensitivity', 'Tiled attention tile-size sensitivity on a single H200')


def tuning_implementation(implementation):
    """Share eager-selected tiles to isolate compilation from tile selection."""
    return 'flash_pytorch' if implementation == 'flash_pytorch_compiled' else implementation


def stop_worker(process):
    """Stop the worker and any compiler subprocesses it spawned."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def build_cases(args):
    cases = [(n, d, dtype, impl, None) for n, d, dtype, impl in
             itertools.product(args.seq_lengths, args.dims, args.dtypes, args.implementations)]
    if args.tile_sensitivity:
        for impl in args.tile_implementations:
            pairs = (itertools.product(args.tile_sizes, repeat=2) if impl == 'flash_triton'
                     else ((t, t) for t in args.tile_pytorch_sizes))
            cases.extend((n, 64, dtype, impl, tile) for n, dtype, tile in
                         itertools.product(args.tile_seq_lengths, args.dtypes, list(pairs)))
    return cases


def worker(job_path):
    # Heavy imports live in workers; each case gets its own CUDA context and allocator.
    import torch
    import triton
    from triton.runtime.errors import OutOfResources
    from triton.testing import do_bench
    from cs336_systems.flashattention import CompiledFlashAttention, FlashAttention, FlashAttentionTriton, NaiveAttention

    job = json.loads(Path(job_path).read_text())
    result_path = Path(job['result_path'])
    row = {key: job[key] for key in ('implementation', 'sequence_length', 'embedding_dim', 'dtype')}
    row['batch_size'] = job.get('batch_size', 1)
    row['experiment'] = job['experiment']
    started = time.monotonic()

    def checkpoint(status, **extra):
        row.update(status=status, wall_seconds=time.monotonic() - started, **extra)
        save_json(result_path, row)
        print(json.dumps(row), flush=True)

    try:
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(job['seed'])
        assert torch.cuda.device_count() == 1, 'Expose exactly one GPU with CUDA_VISIBLE_DEVICES.'
        assert 'H200' in torch.cuda.get_device_name(0), 'This run is configured for an H200.'
        row['environment'] = dict(python=sys.version, torch=torch.__version__, triton=triton.__version__,
                                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                  memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                                  tf32=False)
        dtype = getattr(torch, job['dtype'])
        n, d = job['sequence_length'], job['embedding_dim']
        impl = job['implementation']
        if impl not in IMPLEMENTATIONS:
            raise ValueError(f'Unknown implementation: {impl}')
        if impl == 'flash_pytorch_compiled':
            row['compile_config'] = dict(backend='inductor', fullgraph=True, dynamic=False,
                                         mode='default', forward=True, backward=True)
            row['tile_policy'] = 'fixed tile or eager PyTorch-selected tile; no compiled retuning'

        def inputs(length):
            # Deterministic, identical inputs for every implementation of a given shape.
            generator = torch.Generator(device='cuda').manual_seed(job['seed'])
            qkv = tuple(torch.randn(row['batch_size'], length, d, device='cuda', dtype=dtype, generator=generator,
                                    requires_grad=True) for _ in range(3))
            do = torch.randn(row['batch_size'], length, d, device='cuda', dtype=dtype, generator=generator)
            return qkv, do

        def apply(qkv, tile, implementation=None):
            selected = implementation or impl
            if selected == 'naive':
                return NaiveAttention.apply(*qkv, True)
            if selected == 'flash_pytorch':
                return FlashAttention.apply(*qkv, True, tile[0])
            if selected == 'flash_pytorch_compiled':
                return CompiledFlashAttention.apply(*qkv, True, tile[0])
            return FlashAttentionTriton.apply(*qkv, True, *tile)

        def end_to_end(qkv, do, tile, implementation=None):
            return torch.autograd.grad(apply(qkv, tile, implementation), qkv, do)

        def first_call(fn, label):
            # Wall time includes tracing/code generation, execution, and sync;
            # it is not a claim of pure compiler time or a cold cache.
            torch.cuda.synchronize()
            checkpoint('warming_up', stage=label)
            start = time.monotonic()
            result = fn()
            torch.cuda.synchronize()
            row[label + '_seconds'] = time.monotonic() - start
            return result

        def measure(fn, label):
            samples = do_bench(fn, warmup=job['warmup_ms'], rep=job['rep_ms'], return_mode='all')
            row[label + '_samples'] = len(samples)
            return statistics.median(samples)

        def measure_memory(fn, label):
            # Do not retain timing graphs, gradients, or do_bench cache buffers.
            # q/k/v and the upstream gradient remain resident for both measurements.
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = fn()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            row[label + '_baseline_bytes'] = baseline
            row[label + '_peak_bytes'] = peak
            row[label + '_peak_gb'] = peak / 1e9
            row[label + '_incremental_gb'] = (peak - baseline) / 1e9
            del result

        def validate(tile, implementation=None):
            # Exercise multiple tiles even when the tuned PyTorch tile is large.
            length = min(n, max(128, 2 * max(tile or (64,))))
            qkv, do = inputs(length)
            reference_inputs = tuple(x.detach().float().requires_grad_() for x in qkv)
            q, k, v = reference_inputs
            scores = (q @ k.transpose(-2, -1)) * d ** -0.5
            mask = torch.ones(length, length, device='cuda', dtype=torch.bool).triu(1)
            reference = scores.masked_fill(mask, float('-inf')).softmax(-1) @ v
            reference_grads = torch.autograd.grad(reference, reference_inputs, do.float())
            actual = apply(qkv, tile, implementation)
            actual_grads = torch.autograd.grad(actual, qkv, do)
            tolerance = 0.05 if dtype == torch.bfloat16 else 0.001
            for value, expected in zip((actual, *actual_grads), (reference, *reference_grads)):
                torch.testing.assert_close(value.float(), expected, atol=tolerance, rtol=tolerance)
            torch.cuda.synchronize()
            return dict(batch_size=row['batch_size'], sequence_length=length, atol=tolerance, rtol=tolerance,
                        max_abs_errors=[(x.float() - y).abs().max().item()
                                        for x, y in zip((actual, *actual_grads), (reference, *reference_grads))])

        tile = None
        if job.get('fixed_tile') is not None:
            tile = tuple(job['fixed_tile'])
            row.update(q_tile=tile[0], k_tile=tile[1])
        elif impl != 'naive':
            # Tune once per implementation, dtype, D and capped sequence length.
            # For longer inputs reuse the 2048-token choice; this is a bounded search,
            # not a claim of optimal tiles for every long sequence.
            tune_n = min(n, 2048)
            checkpoint('tuning', stage='tile_selection')
            cache_path = Path(job['tuning_path'])
            if cache_path.exists():
                tuning = json.loads(cache_path.read_text())
                tile = tuple(tuning['selected'])
            else:
                candidates = ([(min(tune_n, t),) * 2 for t in (256, 512, 1024)]
                              if impl in PYTORCH_TILED else [(16, 16), (32, 32), (64, 32), (64, 64)])
                candidates = list(dict.fromkeys(candidates))
                trials = []
                qkv_tune, do_tune = inputs(tune_n)
                for candidate in candidates:
                    trial = dict(tile=candidate)
                    try:
                        trial['validation'] = validate(candidate, tuning_implementation(impl))
                        def fn():
                            return end_to_end(qkv_tune, do_tune, candidate, tuning_implementation(impl))
                        fn()  # Compilation is excluded from timing.
                        torch.cuda.synchronize()
                        trial['ms'] = do_bench(fn, warmup=5, rep=15, return_mode='median')
                        trial['status'] = 'ok'
                    except Exception as exc:
                        trial.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                        torch.cuda.empty_cache()
                    trials.append(trial)
                    print('TILE ' + json.dumps(trial), flush=True)
                usable = [trial for trial in trials if trial['status'] == 'ok']
                if not usable:
                    raise RuntimeError('All tile candidates failed; see worker log.')
                tile = tuple(min(usable, key=lambda trial: trial['ms'])['tile'])
                tuning = dict(selected=tile, sequence_length=tune_n, trials=trials)
                save_json(cache_path, tuning)
                del qkv_tune, do_tune
            row['tuning'] = tuning
            row.update(q_tile=tile[0], k_tile=tile[1])
        row['validation'] = first_call(lambda: validate(tile), 'validation')
        if job.get('validation_only'):
            row['stage'] = 'complete'
            checkpoint('ok')
            return
        torch.cuda.empty_cache()
        qkv, do = inputs(n)
        checkpoint('warming_up')
        # Use autograd.grad so no leaf .grad accumulation or gradient clearing is timed.
        # Each timing stage warms its own compiled operations before do_bench.
        output = first_call(lambda: apply(qkv, tile), 'forward_first_call')
        assert torch.isfinite(output).all().item(), 'Nonfinite forward output'
        output = None
        row['stage'] = 'forward_memory'
        measure_memory(lambda: apply(qkv, tile), 'forward')
        checkpoint('forward_memory_done')
        row['stage'] = 'forward_timing'
        output = apply(qkv, tile)
        row['forward_ms'] = measure(lambda: apply(qkv, tile), 'forward')
        checkpoint('forward_done')
        gradients = first_call(lambda: torch.autograd.grad(output, qkv, do, retain_graph=True), 'backward_first_call')
        assert all(torch.isfinite(g).all().item() for g in gradients), 'Nonfinite gradients'
        del gradients
        row['stage'] = 'backward_timing'
        row['backward_ms'] = measure(lambda: torch.autograd.grad(output, qkv, do, retain_graph=True), 'backward')
        checkpoint('backward_done')
        output = None  # Avoid retaining an extra graph during end-to-end timing.
        def fn():
            return end_to_end(qkv, do, tile)
        first_call(fn, 'forward_backward_first_call')
        row['stage'] = 'forward_backward_memory'
        measure_memory(fn, 'forward_backward')
        checkpoint('forward_backward_memory_done')
        row['stage'] = 'forward_backward_timing'
        row['forward_backward_ms'] = measure(fn, 'forward_backward')
        row['tokens_per_second'] = row['batch_size'] * n * 1000 / row['forward_backward_ms']
        row['stage'] = 'complete'
        checkpoint('ok')
    except torch.cuda.OutOfMemoryError as exc:
        checkpoint('oom', error=str(exc))
    except OutOfResources as exc:
        checkpoint('resource_limit', error=f'{type(exc).__name__}: {exc}')
    except Exception as exc:
        traceback.print_exc()
        checkpoint('error', error=f'{type(exc).__name__}: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark_results' / 'h200_compile_comparison')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[2**i for i in range(7, 17)])
    parser.add_argument('--dims', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--dtypes', nargs='+', choices=['bfloat16', 'float32'], default=['bfloat16', 'float32'])
    parser.add_argument('--implementations', nargs='+', choices=IMPLEMENTATIONS, default=list(IMPLEMENTATIONS))
    parser.add_argument('--warmup-ms', type=float, default=25)
    parser.add_argument('--rep-ms', type=float, default=100)
    parser.add_argument('--timeout-seconds', type=float, default=600, help='Per-case limit including compilation; 0 means no limit.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tile-sensitivity', action='store_true', help='Also compare fixed tiles for eager/compiled PyTorch and Triton at D=64.')
    parser.add_argument('--tile-seq-lengths', type=int, nargs='+', default=[1024, 4096, 16384, 65536])
    parser.add_argument('--tile-sizes', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--tile-pytorch-sizes', type=int, nargs='+', default=[256, 512, 1024])
    parser.add_argument('--tile-implementations', nargs='+', choices=(*PYTORCH_TILED, 'flash_triton'),
                        default=[*PYTORCH_TILED, 'flash_triton'])
    parser.add_argument('--dry-run', action='store_true', help='Print the complete case grid without imports, GPU work, or writes.')
    parser.add_argument('--resume', action='store_true', help='Reuse completed cases and tile tuning in this output directory.')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if any(n < 16 or n & (n - 1) for n in args.seq_lengths + args.dims + args.tile_seq_lengths + args.tile_sizes + args.tile_pytorch_sizes):
        parser.error('Sequence lengths and embedding dimensions must be powers of two >= 16.')
    if args.rep_ms <= 0 or args.warmup_ms < 0 or args.timeout_seconds < 0:
        parser.error('Timing budgets must be positive (warmup and timeout may be zero).')
    directory = args.output.resolve()
    cases = build_cases(args)
    if args.dry_run:
        print(json.dumps(dict(output=str(directory), total_cases=len(cases), cases=cases), indent=2))
        return
    if (directory / 'manifest.json').exists() and not args.resume:
        parser.error('Output already contains a run; use --resume or choose a new --output.')
    for subdir in ('cases', 'logs', 'jobs', 'tuning'):
        (directory / subdir).mkdir(parents=True, exist_ok=True)
    config = dict(benchmark_version=3, sequence_lengths=args.seq_lengths, dimensions=args.dims, dtypes=args.dtypes,
                  implementations=args.implementations, warmup_ms=args.warmup_ms, rep_ms=args.rep_ms,
                  timeout_seconds=args.timeout_seconds, seed=args.seed, batch_size=1, causal=True,
                  tile_sensitivity=args.tile_sensitivity, tile_sequence_lengths=args.tile_seq_lengths,
                  tile_sizes=args.tile_sizes, tile_pytorch_sizes=args.tile_pytorch_sizes,
                  tile_implementations=args.tile_implementations,
                  memory_units='decimal GB, allocated (not reserved)')
    sources = (Path(__file__), ROOT / 'scripts' / 'plot_attention_benchmarks.py',
               ROOT / 'cs336_systems' / 'flashattention.py', ROOT / 'uv.lock', ROOT / 'benchmark.env')
    fingerprint = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if args.resume and (directory / 'manifest.json').exists():
        previous = json.loads((directory / 'manifest.json').read_text())
        if previous['config'] != config or previous.get('source_sha256') != fingerprint:
            parser.error('Resume requires identical configuration and source files.')
    started = time.time()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
    save_json(directory / 'manifest.json', dict(config=config, pid=os.getpid(), started_at=started,
                                               git_revision=revision, gpu_info=gpu,
                                               command=sys.argv, total_cases=len(cases), source_sha256=fingerprint))
    # Preserve the exact benchmark and implementation used for reproducibility.
    import shutil
    for source in sources:
        shutil.copy2(source, directory / source.name)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    from plot_attention_benchmarks import write_plots
    rows = []
    write_tables(directory, rows)
    for n, d, dtype, impl, fixed_tile in cases:
        key = f'{impl}_n{n}_d{d}_{dtype}'
        experiment = 'tile_sensitivity' if fixed_tile is not None else 'main'
        if fixed_tile is not None:
            key = f'tiles_{key}_bm{fixed_tile[0]}_bn{fixed_tile[1]}'
        result = directory / 'cases' / f'{key}.json'
        if args.resume and result.exists() and json.loads(result.read_text()).get('status') in ('ok', 'oom', 'error', 'timeout', 'resource_limit'):
            rows.append(json.loads(result.read_text()))
            write_tables(directory, rows)
            continue
        result.unlink(missing_ok=True)
        job = dict(implementation=impl, sequence_length=n, embedding_dim=d, dtype=dtype, seed=args.seed,
                   experiment=experiment, fixed_tile=fixed_tile,
                   warmup_ms=args.warmup_ms, rep_ms=args.rep_ms, result_path=str(result),
                   tuning_path=str(directory / 'tuning' / f'{tuning_implementation(impl)}_n{min(n, 2048)}_d{d}_{dtype}.json'))
        job_path = directory / 'jobs' / f'{key}.json'
        save_json(job_path, job)
        save_json(directory / 'progress.json', dict(status='running', pid=os.getpid(), current_case=key,
                                                   completed=len(rows), total=len(cases), elapsed_seconds=time.time()-started))
        print(f'[{len(rows)+1}/{len(cases)}] {key}', flush=True)
        case_start = time.monotonic()
        with (directory / 'logs' / f'{key}.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(job_path)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=args.timeout_seconds or None)
                timed_out = False
            except subprocess.TimeoutExpired:
                stop_worker(process)
                timed_out = True
            except BaseException:
                stop_worker(process)
                raise
        row = json.loads(result.read_text()) if result.exists() else {key: job[key] for key in FIELDS[:4]}
        row['experiment'] = experiment
        if fixed_tile is not None:
            row.update(q_tile=fixed_tile[0], k_tile=fixed_tile[1])
        if timed_out:
            row.update(status='timeout', error=f'Exceeded {args.timeout_seconds}s per-case limit')
        elif process.returncode != 0 or row.get('status') not in ('ok', 'oom', 'error', 'resource_limit'):
            row.update(status='error', error=f'Worker exited with code {process.returncode}; see log')
        row['wall_seconds'] = time.monotonic() - case_start
        save_json(result, row)
        rows.append(row)
        write_tables(directory, rows)
        if len(rows) % 24 == 0 and len(rows) < len(cases):
            write_plots(directory, rows)
        print(f"  {row['status']}; {row['wall_seconds']:.1f}s", flush=True)
    write_plots(directory, rows)
    save_json(directory / 'progress.json', dict(status='complete', pid=os.getpid(), completed=len(rows), total=len(cases),
                                               elapsed_seconds=time.time()-started,
                                               successful=sum(row['status']=='ok' for row in rows),
                                               failed=sum(row['status']!='ok' for row in rows)))
    print(f'Complete: {directory / "results.md"}', flush=True)


if __name__ == '__main__':
    main()
