"""Environment/inference gate only. Never use as a GuardrailServe effect result."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {"argv": args, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--inventory-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = {"kind": "environment_and_stock_vllm_smoke_NOT_POLICY_EVIDENCE",
              "started_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "script_sha256": sha(__file__),
              "model_path": args.model, "status": "started"}
    report['runtime_environment'] = {k: os.environ.get(k) for k in [
        'VLLM_USE_FLASHINFER_SAMPLER', 'CUDA_VISIBLE_DEVICES',
        'VLLM_WORKER_MULTIPROC_METHOD', 'CUDA_HOME']}
    for k, argv in {
        "gpu": ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap', '--format=csv,noheader'],
        "processes_before": ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        "disk": ['df', '-h', '/', '/root/autodl-tmp'],
    }.items():
        report[k] = command(argv)
    report['packages'] = {}
    for name in ['torch', 'vllm', 'transformers', 'numpy', 'scipy', 'triton']:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    if args.inventory_only:
        report['status'] = 'inventory_complete'
    elif report['processes_before']['exit_code'] or report['processes_before']['stdout'].strip():
        report['status'] = 'blocked_gpu_not_idle'
    else:
        try:
            model = Path(args.model)
            report['model_sha256'] = {str(p.relative_to(model)): sha(p)
                                       for p in sorted(model.rglob('*'))
                                       if p.is_file() and p.suffix in {'.json', '.safetensors'}}
            import torch
            from vllm import LLM, SamplingParams
            import vllm
            report['torch_cuda'] = torch.version.cuda
            report['vllm_path'] = vllm.__file__
            report['config'] = dict(model=args.model, dtype='bfloat16',
                                    tensor_parallel_size=1, enforce_eager=True,
                                    max_model_len=1024, max_num_seqs=8,
                                    gpu_memory_utilization=0.35, seed=4242,
                                    enable_prefix_caching=False)
            t = time.perf_counter()
            llm = LLM(**report['config'])
            report['load_seconds'] = time.perf_counter() - t
            prompts = ['Explain continuous batching briefly.', 'What is a key value cache?',
                       'Write a short greeting.', 'Count from one to ten.']
            outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32,
                                                           ignore_eos=True), use_tqdm=False)
            rows = [{'request_id': o.request_id, 'finished': o.finished,
                     'prompt_tokens': len(o.prompt_token_ids),
                     'output_tokens': len(o.outputs[0].token_ids),
                     'finish_reason': o.outputs[0].finish_reason} for o in outputs]
            report['requests'] = rows
            assert len(rows) == 4 and all(r['finished'] and r['output_tokens'] == 32 for r in rows)
            report['status'] = 'stock_inference_pass'
        except Exception:
            report['status'] = 'failed'
            report['traceback'] = traceback.format_exc()
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'manifest': str(out / 'manifest.json')}), flush=True)
    return 0 if report['status'] in {'inventory_complete', 'stock_inference_pass'} else 1


if __name__ == '__main__':
    sys.exit(main())
