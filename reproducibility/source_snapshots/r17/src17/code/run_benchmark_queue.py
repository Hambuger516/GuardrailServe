"""Run a frozen calibration/formal plan after dependencies and idle checks.

No retries or selection by outcomes. Any failed/contaminated run stops this
queue and leaves its artifacts for inspection. Locks only coordinate our jobs.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from runtime_smoke import sha
from gpu_run_observer import gpu_command


def main():
    parser = argparse.ArgumentParser()
    for name in ['python', 'model', 'pool', 'histogram', 'plan', 'out']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--require-success', action='append', default=[])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    original_plan = Path(args.plan).resolve()
    plan = json.loads(original_plan.read_text())
    if not plan['jobs'] or len({j['name'] for j in plan['jobs']}) != len(plan['jobs']):
        raise ValueError('empty plan or duplicate job names')
    code = out / 'code'
    for name in ['ablation_policy.py', 'runtime_ablation.py', 'ceiling_policy.py', 'runtime_ceiling.py', 'hysteresis_policy.py', 'runtime_hysteresis.py', 'estimator_policy.py', 'runtime_r15.py', 'r16_policy.py', 'runtime_r16.py', 'slack_policy.py', 'runtime_slack.py', 'revision_policy.py', 'runtime_revision.py', 'admission_reuse.py', 'runtime_scheduler_reuse.py', 'factorial_policy.py', 'runtime_factorial.py', 'benchmark_runtime.py', 'runtime_scheduler.py', 'runtime_pfs.py', 'pfs_reference.py',
                 'policy_core.py', 'gpu_run_observer.py', 'serving_metrics.py', 'runtime_smoke.py', 'artifact_files.py',
                 'sim/__init__.py', 'sim/corrected_engine.py', 'sim/prediction.py', 'run_benchmark_queue.py']:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent / name, target)
    inputs = out / 'inputs'
    inputs.mkdir()
    shutil.copy2(args.pool, inputs / 'pool.json')
    shutil.copy2(original_plan, inputs / 'plan.json')
    for job in plan['jobs']:
        if not job['name'].replace('_', '').isalnum():
            raise ValueError('invalid job name')
        for key in ['config', 'trace']:
            src = (original_plan.parent / job[key]).resolve()
            if not src.is_relative_to(original_plan.parent):
                raise ValueError('input outside plan directory')
            shutil.copy2(src, inputs / f'{job["name"]}_{key}.json')
    if plan.get('source_hashes'):
        for name, digest in plan['source_hashes'].items():
            if sha(code / name) != digest:
                raise ValueError(f'source differs from registered release: {name}')
    max_hours = plan.get('max_queue_hours', 12)
    if not isinstance(max_hours, (int, float)) or not 1 <= max_hours <= 72:
        raise ValueError('queue deadline must be within 1--72 hours')
    state = {'kind': plan['kind'], 'pid': os.getpid(), 'completed': [], 'status': 'waiting',
             'plan_sha256': sha(inputs / 'plan.json'), 'max_wait_hours': max_hours}
    def save(**fields):
        state.update(fields, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        tmp = out / 'state.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out / 'state.json')
        print(json.dumps(state), flush=True)
    deadline = time.monotonic() + max_hours * 3600
    lock = (out.parent / 'integration_queue.lock').open('a')
    try:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('queue readiness deadline exceeded')
            pending = [p for p in args.require_success if not Path(p).exists()]
            for p in args.require_success:
                if Path(p).exists() and Path(p).read_text().strip() != '0':
                    raise RuntimeError(f'prerequisite failed: {p}')
            if not pending:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pending = ['another project-owned queue holds lock']
            save(reason='waiting prerequisite or lock', pending=pending)
            time.sleep(30)
        shutil.copy2(args.histogram, inputs / 'histogram.json')
        save(reason='inputs frozen', pending=[], histogram_sha256=sha(inputs / 'histogram.json'))
        for job in plan['jobs']:
            idle_since = None
            while True:
                if shutil.disk_usage(out).free < plan.get('minimum_free_bytes', 1024 ** 3):
                    raise RuntimeError('less than registered free disk reserve; preserve artifacts and stop')
                if time.monotonic() > deadline:
                    raise TimeoutError('queue readiness deadline exceeded')
                processes = subprocess.run(gpu_command(['--query-compute-apps=pid', '--format=csv,noheader']),
                                           capture_output=True, text=True, timeout=15)
                gpu = subprocess.run(gpu_command(['--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits']), capture_output=True, text=True, timeout=15)
                rows = [line.split(',') for line in gpu.stdout.strip().splitlines()]
                ready = (not processes.returncode and not processes.stdout.strip() and not gpu.returncode
                         and bool(rows) and all(int(m) < 512 and int(u) < 5 for m, u in rows))
                if not ready:
                    idle_since = None
                elif idle_since is None:
                    idle_since = time.monotonic()
                elapsed = time.monotonic() - idle_since if idle_since is not None else 0
                save(status='waiting', next_job=job['name'], reason='idle_sample' if ready else 'GPU_busy_or_query_failed',
                     idle_seconds=elapsed)
                if elapsed >= (300 if not state['completed'] else 60):
                    break
                time.sleep(30)
            save(status='running', next_job=job['name'])
            command = ['timeout', '--signal=TERM', '--kill-after=30s', str(plan['run_timeout_s']) + 's',
                       args.python, '-u', str(code / 'benchmark_runtime.py'), '--model', args.model,
                       '--pool', str(inputs / 'pool.json'), '--histogram', str(inputs / 'histogram.json'),
                       '--trace', str(inputs / f'{job["name"]}_trace.json'),
                       '--config', str(inputs / f'{job["name"]}_config.json'), '--out', str(out / job['name'])]
            with (out / f'{job["name"]}.log').open('x') as stream:
                result = subprocess.run(command, cwd=code,
                                        env=dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_BATCH_INVARIANT='0',
                                                 OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                                 OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'),
                                        stdout=stream, stderr=subprocess.STDOUT)
            (out / f'{job["name"]}.exit').write_text(str(result.returncode) + '\n')
            if result.returncode:
                raise RuntimeError(f'{job["name"]} exited {result.returncode}; no automatic retry')
            state['completed'].append(job['name'])
        save(status='completed', next_job=None)
        (out / 'queue.exit').write_text('0\n')
    except Exception as exc:
        save(status='stopped', reason=str(exc))
        (out / 'queue.exit').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
