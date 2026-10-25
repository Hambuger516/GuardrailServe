"""step5: launch one frozen queue on one idle GPU UUID; no implicit retries.

The plan leaves gpu_uuid null; the operator supplies --gpu-uuid after checking
nvidia-smi. The chosen UUID, host and idle observation are recorded."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('plan','python','model'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--libs')
    p.add_argument('--gpu-uuid', required=True)
    args = p.parse_args()
    manifest = json.loads((HERE/'bundle_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(HERE/name)!=digest:
            raise ValueError('Frozen bundle changed: '+name)
    plan_path = (HERE/args.plan).resolve()
    if not plan_path.is_relative_to(HERE) or plan_path.name not in manifest['files']:
        raise ValueError('Unregistered plan')
    plan = json.loads(plan_path.read_text())
    if plan['phase'] != 'smoke':
        gate = json.loads((HERE/'SMOKE_AUDIT_PASS.json').read_text())
        if not gate['all_checks_pass'] or gate['bundle_sha256'] != sha(HERE/'bundle_manifest.json'):
            raise ValueError('A complete matching smoke audit is required')
    if plan.get('gpu_uuid') is not None:
        raise ValueError('step5 plans must not pre-bind a GPU')
    uuid = args.gpu_uuid
    if not uuid.startswith('GPU-'):
        raise ValueError('pass the full nvidia-smi UUID (GPU-...)')
    def smi(query):
        result = subprocess.run(['nvidia-smi','-i',uuid,query,'--format=csv,noheader,nounits'],
                                capture_output=True,text=True,timeout=15)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout.strip()
    state = smi('--query-gpu=uuid,name,memory.used,utilization.gpu')
    fields = [s.strip() for s in state.split(',')]
    if (len(fields)!=4 or fields[0]!=uuid or 'A800' not in fields[1] or int(fields[2])>=512 or int(fields[3])>=5
            or smi('--query-compute-apps=pid')):
        raise RuntimeError('Selected GPU is occupied or idle telemetry is inconsistent')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, TPDS_GPU_UUID=uuid,
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    env.pop('TRITON_PTXAS_PATH',None)
    if args.libs:
        env['LD_LIBRARY_PATH']=args.libs
    temporary = Path(tempfile.mkdtemp(prefix='tpds-s5-',dir='/tmp'))
    info = temporary.lstat()
    if (temporary.is_symlink() or info.st_uid!=os.geteuid() or stat.S_IMODE(info.st_mode)!=0o700
            or len(os.fsencode(str(temporary)))>40):
        raise RuntimeError('Private short IPC path failed checks')
    env['TMPDIR']=str(temporary)
    preflight = subprocess.run([args.python,'-c',
        "import json,torch,vllm,runtime_revision,runtime_slack,runtime_ablation,runtime_pfs,runtime_ceiling,runtime_hysteresis,runtime_r15,runtime_r16; print(json.dumps({'vllm':vllm.__version__,'torch':torch.__version__,'cuda':torch.version.cuda}))"],
        cwd=HERE/'code',env=env,capture_output=True,text=True,timeout=900)
    if preflight.returncode:
        raise RuntimeError('Runtime import failed: '+preflight.stderr[-3000:])
    versions = json.loads(preflight.stdout.strip().splitlines()[-1])
    if versions!={'vllm':'0.29.0','torch':'2.13.0+cu130','cuda':'13.0'}:
        raise ValueError('Version differs from registered stack: '+str(versions))
    slot = HERE/'execution'/plan['phase']/plan['slot']
    slot.mkdir(parents=True,exist_ok=False)
    env.update(TRITON_CACHE_DIR=str(slot/'triton_cache'),VLLM_CACHE_ROOT=str(slot/'vllm_cache'))
    command = [args.python,'-u',str(HERE/'code/run_benchmark_queue.py'),
               '--python',args.python,'--model',args.model,'--pool',str(HERE/plan['pool']),
               '--histogram',str(HERE/'inputs/histogram.json'),'--plan',str(plan_path),'--out',str(slot/'runs')]
    with (slot/'queue.log').open('x') as log:
        child = subprocess.Popen(command,cwd=HERE/'code',env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    import socket
    record = dict(host=socket.gethostname(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,
                  command=command,plan_sha256=sha(plan_path),bundle_sha256=sha(HERE/'bundle_manifest.json'),
                  gpu_uuid=uuid,idle_observation=state,versions=versions,jobs=len(plan['jobs']),
                  environment={k:env[k] for k in ('TMPDIR','TPDS_GPU_UUID','CUDA_VISIBLE_DEVICES',
                      'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    (slot/'launch.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))


if __name__=='__main__':
    main()
