"""Run the fixed-tile experiment in BATCH_SIZE_SWEEP_PLAN.md."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmark_attention import ROOT, IMPLEMENTATIONS, PYTORCH_TILED, FIRST_CALL_FIELDS, save_json, stop_worker, tuning_implementation

TERMINAL = {'ok', 'oom', 'resource_limit', 'error', 'timeout'}
FIELDS = ('batch_size', 'sequence_length', 'embedding_dim', 'dtype', 'implementation',
          'q_tile', 'k_tile', 'forward_ms', 'backward_ms', 'forward_backward_ms',
          'tokens_per_second', 'forward_peak_gb', 'forward_backward_peak_gb',
          'forward_baseline_bytes', 'forward_backward_baseline_bytes',
          'forward_incremental_gb', 'forward_backward_incremental_gb',
          'forward_samples', 'backward_samples', 'forward_backward_samples',
          'status', 'stage', 'wall_seconds', 'error') + FIRST_CALL_FIELDS


def prepare(config):
    for name in ('batch_sizes', 'sequence_lengths', 'dimensions'):
        values = config[name]
        if not values or any(type(v) is not int or v <= 0 for v in values) or len(set(values)) != len(values):
            raise ValueError(f'{name} must contain unique positive integers')
    for value in config['sequence_lengths'] + config['dimensions']:
        if value < 16 or value & (value - 1):
            raise ValueError('Sequence lengths and dimensions must be powers of two >= 16')
    for name, allowed in (('dtypes', {'bfloat16', 'float32'}), ('implementations', set(IMPLEMENTATIONS))):
        if not config[name] or not set(config[name]) <= allowed or len(set(config[name])) != len(config[name]):
            raise ValueError(f'Invalid {name}')
    if config['warmup_ms'] < 0 or config['rep_ms'] <= 0 or config['timeout_seconds'] < 0 or config['minimum_samples'] < 1:
        raise ValueError('Invalid timing or sample budget')
    if len(config['dimensions']) != 1:
        raise ValueError('Use a separate run per dimension to preserve the planned figure layout')
    baseline = ROOT / config['baseline_directory']
    tiles = {}
    jobs = []
    for b, n, d, dtype, impl in itertools.product(config['batch_sizes'], config['sequence_lengths'],
                                               config['dimensions'], config['dtypes'], config['implementations']):
        tile = None
        if impl != 'naive':
            name = f'{tuning_implementation(impl)}_n{min(n, 2048)}_d{d}_{dtype}.json'
            if name not in tiles:
                tiles[name] = json.loads((baseline / 'tuning' / name).read_text())
            tile = tiles[name]['selected']
            if len(tile) != 2 or any(type(t) is not int or t < 16 or t & (t - 1) for t in tile):
                raise ValueError(f'Invalid baseline tile in {name}')
            if impl in PYTORCH_TILED and tile[0] != tile[1]:
                raise ValueError(f'PyTorch requires square tiles: {name}')
        jobs.append(dict(batch_size=b, sequence_length=n, embedding_dim=d, dtype=dtype,
                         implementation=impl, fixed_tile=tile, experiment='batch_sweep',
                         seed=config['seed'], warmup_ms=config['warmup_ms'], rep_ms=config['rep_ms']))
    return jobs, tiles


def case_key(job):
    key = '{implementation}_b{batch_size}_n{sequence_length}_d{embedding_dim}_{dtype}'.format(**job)
    tile = job.get('fixed_tile')
    if tile is None and 'q_tile' in job:
        tile = (job['q_tile'], job['k_tile'])
    if tile is not None:
        key += f'_bm{tile[0]}_bn{tile[1]}'
    return key


def run_case(directory, job, timeout, resume):
    key = case_key(job)
    result = directory / 'cases' / f'{key}.json'
    if resume and result.exists():
        row = json.loads(result.read_text())
        if row.get('status') in TERMINAL:
            return row
    # Remove an interrupted checkpoint before a fresh attempt.
    result.unlink(missing_ok=True)
    job = dict(job, result_path=str(result))
    job_path = directory / 'jobs' / f'{key}.json'
    save_json(job_path, job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    started = time.monotonic()
    timed_out = False
    with (directory / 'logs' / f'{key}.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/benchmark_attention.py'), '--worker', str(job_path)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout or None)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_worker(process)
        except BaseException:
            stop_worker(process)
            raise
    row = {k: v for k, v in job.items() if k in FIELDS or k == 'experiment'}
    if job['fixed_tile'] is not None:
        row.update(q_tile=job['fixed_tile'][0], k_tile=job['fixed_tile'][1])
    if result.exists():
        row.update(json.loads(result.read_text()))
    if timed_out:
        row.update(status='timeout', error=f'Exceeded {timeout}s per-case limit')
    elif process.returncode or row.get('status') not in TERMINAL:
        row.update(status='error', error=f'Worker exited with code {process.returncode}; see log')
    row['wall_seconds'] = time.monotonic() - started
    save_json(result, row)
    return row


def write_results(directory, rows, minimum_samples):
    temporary = directory / 'results.csv.tmp'
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / 'results.csv')
    columns = FIELDS[:13] + ('status',)
    lines = ['# Batch-size sweep', '', 'Causal attention; fixed baseline tiles; all batch-size-1 timings rerun.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(f'{row[k]:.6g}' if isinstance(row.get(k), float) else str(row.get(k, '')) for k in columns) + ' |')
    temporary = directory / 'results.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(directory / 'results.md')
    save_json(directory / 'review.json', dict(
        failures=[dict(case=case_key(r), status=r['status'], stage=r.get('stage')) for r in rows if r['status'] != 'ok'],
        low_sample_counts=[dict(case=case_key(r), phase=p, samples=r[p + '_samples'])
                           for r in rows for p in ('forward', 'backward', 'forward_backward')
                           if p + '_samples' in r and r[p + '_samples'] < minimum_samples],
        missing_validation=[case_key(r) for r in rows if 'validation' not in r],
        minimum_samples=minimum_samples))


def execute(args, config, jobs, tiles, directory):
    sources = [ROOT / 'scripts' / name for name in
               ('benchmark_batch_sizes.py', 'benchmark_attention.py', 'plot_batch_sizes.py')]
    sources += [ROOT / 'cs336_systems/flashattention.py', ROOT / 'uv.lock', ROOT / 'benchmark.env']
    fingerprint = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    identity = dict(config=config, tiles=tiles, source_sha256=fingerprint, baseline_measurements='rerun')
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            raise ValueError('Output already contains a run; use --resume or a new --output')
        previous = json.loads(manifest_path.read_text())
        if previous['identity'] != identity:
            raise ValueError('Resume requires identical configuration, baseline tiles, and source files')
    for subdir in ('cases', 'jobs', 'logs', 'tuning', 'smoke/cases', 'smoke/jobs', 'smoke/logs'):
        (directory / subdir).mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    manifest = dict(identity=identity, gpu_info=gpu, git_revision=revision, total_cases=len(jobs),
                    pid=os.getpid(), started_at=time.time(), command=sys.argv)
    if manifest_path.exists():
        manifest['previous_started_at'] = previous['started_at']
    save_json(manifest_path, manifest)
    save_json(directory / 'config.json', config)
    for name, tuning in tiles.items():
        save_json(directory / 'tuning' / name, tuning)
    for source in sources:
        shutil.copy2(source, directory / source.name)
    shutil.copy2(ROOT / config['baseline_directory'] / 'manifest.json', directory / 'baseline_manifest.json')
    (directory / 'run.pid').write_text(str(os.getpid()) + '\n')
    # Validate every distinct tile regime at batch=2 before any full-size timing.
    smoke = {}
    for job in jobs:
        check = dict(job, batch_size=2, validation_only=True)
        check['sequence_length'] = min(job['sequence_length'], max(128, 2 * max(job['fixed_tile'] or [64])))
        smoke[(case_key(check), tuple(check['fixed_tile'] or []))] = check
    for i, job in enumerate(smoke.values(), 1):
        print(f'Smoke [{i}/{len(smoke)}] {case_key(job)}', flush=True)
        save_json(directory / 'progress.json', dict(status='validating', current_case=case_key(job)))
        row = run_case(directory / 'smoke', job, config['timeout_seconds'], False)
        if row['status'] != 'ok':
            save_json(directory / 'progress.json', dict(status='validation_failed', case=case_key(job), result=row))
            raise RuntimeError(f'Multi-batch validation failed: {case_key(job)}; see smoke/logs')
    if args.smoke_only:
        save_json(directory / 'progress.json', dict(status='smoke_complete', validated=len(smoke)))
        return
    from plot_batch_sizes import write_plots
    rows = []
    for i, job in enumerate(jobs, 1):
        print(f'[{i}/{len(jobs)}] {case_key(job)}', flush=True)
        save_json(directory / 'progress.json', dict(status='running', current_case=case_key(job), completed=len(rows), total=len(jobs)))
        rows.append(run_case(directory, job, config['timeout_seconds'], args.resume))
        write_results(directory, rows, config['minimum_samples'])
        print(f"  {rows[-1]['status']}", flush=True)
        if i % 24 == 0:
            write_plots(directory, rows, config)
    write_plots(directory, rows, config)
    save_json(directory / 'progress.json', dict(status='complete', completed=len(rows), total=len(jobs),
                                               successful=sum(r['status'] == 'ok' for r in rows)))
    print(f'Complete: {directory}. Inspect review.json for failures and low sample counts.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/batch_size_sweep.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate config and baseline tiles; print jobs without GPU work or writes.')
    parser.add_argument('--smoke-only', action='store_true', help='Only run multi-batch correctness validation.')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    jobs, tiles = prepare(config)
    directory = (args.output or ROOT / config['output_directory']).resolve()
    if args.dry_run:
        print(json.dumps(dict(output=str(directory), total_cases=len(jobs), jobs=jobs), indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another sweep is using this output directory')
        execute(args, config, jobs, tiles, directory)


if __name__ == '__main__':
    main()
