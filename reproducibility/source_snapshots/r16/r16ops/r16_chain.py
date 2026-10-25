"""R16 operator (frozen 2026-10-16). Phase A: smoke (seven R16 arms) + Azure-2024-Friday calibration -> smoke gates ->
frozen budget decision -> build formal bundles. Phase B: formal queues (D, E, C2) on genuinely idle GPUs (two consecutive
idle observations; a GPU is reused once its queue exits). Unlike R15, a failed queue does not stop the launch of the
other, independent queues (lesson of AMENDMENT_R15_002); failed queues are never retried and their bundle is not
audited. Each bundle whose queues all exited 0 is audited and analysed."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r16_common import *
import r12_chain as L
L.LOG = OPS / 'r16_state.json'
L.OPS = OPS


def run_all(queues, phase, max_gpus=8):
    pending = list(queues); exits = {}
    while pending or L.state['running']:
        for key, v in list(L.state['running'].items()):
            rc = L.qexit(v['bundle'], phase, v['slot'])
            if rc is not None:
                exits[key] = rc; L.state['exits'][key] = rc; L.state['running'].pop(key); L.note(f'{key} exit {rc}')
        while pending and len(L.state['running']) < max_gpus:
            c = L.idle()
            if not c: break
            key, b, plan, slot, model = pending[0]
            try:
                p = subprocess.run([L.PY, 'launch_batch.py', '--plan', plan, '--gpu-uuid', c[0], '--python', L.PY,
                                    '--model', str(L.MODELS / model), '--libs', L.LIBS], cwd=b, capture_output=True, text=True, timeout=1500)
            except subprocess.TimeoutExpired as exc:
                p = subprocess.CompletedProcess(exc.cmd, 99, exc.stdout or '', 'launch_batch timeout')
            (OPS / f'launch_{key}.out').write_text(p.stdout + p.stderr)
            if p.returncode == 0:
                L.state['running'][key] = dict(bundle=str(b), slot=slot, gpu=c[0]); pending.pop(0); L.seen.pop(c[0], None)
                L.note(f'{key} launched on {c[0]}')
            else:
                L.note(f'{key} launch refused rc={p.returncode}'); time.sleep(30); break
        L.save(); time.sleep(60)
    return exits


def main():
    L.save(stage='phase_a')
    qa = [(f'{k}_{SHORT[m]}', X / f'r16_{k}_{SHORT[m]}', 'smoke_s1_plan.json', 's1', m) for k in ('smoke', 'realcal') for m in (QW, LL)]
    ex = run_all(qa, 'smoke')
    if any(v != '0' for v in ex.values()):
        return L.save(stage='stopped', final='phase A queue failure')
    if not all(L.step(['smoke_check12.py'], X / f'r16_{k}', f'smoke_check_{k}', 3600) for k, *_ in qa):
        return L.save(stage='stopped', final='smoke gate failed')
    if not L.step(['decide_r16.py'], OPS, 'decide_r16', 600):
        return L.save(stage='stopped', final='decision failed')
    if not L.step(['build_r16.py', 'formal'], OPS, 'build_r16_formal', 3600):
        return L.save(stage='stopped', final='formal build failed')
    import build_r16 as B
    qb = [(f'{e}_{SHORT[m]}_s{i + 1}', X / f'r16_{e}_{SHORT[m]}', f'formal_s{i + 1}_plan.json', f's{i + 1}', m)
          for e in B.ORDER for i in range(B.EXPS[e]['slots']) for m in (QW, LL)]
    L.save(stage='phase_b')
    ex = run_all(qb, 'formal')
    for e in B.ORDER:
        for m in (QW, LL):
            key = f'{e}_{SHORT[m]}'; bd = X / f'r16_{key}'
            if all(ex.get(f'{key}_s{i + 1}') == '0' for i in range(B.EXPS[e]['slots'])):
                if L.step(['audit_r12.py'], bd, f'audit_{key}', 14400):
                    L.step(['analyze_r12.py'], bd, f'analysis_{key}', 3600)
                L.state['done'].append(key)
            else:
                L.note(f'{key}: a queue failed; not audited')
    L.save(stage='done', final='done')


if __name__ == '__main__':
    main()
