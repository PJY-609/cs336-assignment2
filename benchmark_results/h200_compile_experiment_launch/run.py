import json, os, subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
launch=Path(__file__).resolve().parent
stages=[('main_and_tiles', 'h200_compile_comparison', ['scripts/benchmark_attention.py', '--tile-sensitivity']), ('batch', 'h200_batch_compile_comparison', ['scripts/benchmark_batch_sizes.py'])]
state=dict(status='running', pid=os.getpid(), started_at=time.time(), completed_stages=[])
def save():
    temporary=launch/'status.json.tmp'
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    temporary.replace(launch/'status.json')
try:
    for stage, output, args in stages:
        directory=root/'benchmark_results'/output
        directory.mkdir(exist_ok=False)
        (directory/'launch_environment.json').write_text((launch/'environment.json').read_text())
        state.update(stage=stage, output=str(directory))
        save()
        print(f'Starting {stage}: {directory}', flush=True)
        with (directory/'run.log').open('w') as log:
            child=subprocess.Popen([sys.executable, *args, '--output', str(directory)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            state['runner_pid']=child.pid
            save()
            code=child.wait()
        if code:
            raise RuntimeError(f'{stage} runner exited with code {code}; inspect {directory}/run.log')
        state['completed_stages'].append(stage)
        print(f'Finished {stage}', flush=True)
    state.update(status='complete', finished_at=time.time())
    save()
except BaseException as error:
    state.update(status='failed', error=str(error), finished_at=time.time())
    save()
    raise
