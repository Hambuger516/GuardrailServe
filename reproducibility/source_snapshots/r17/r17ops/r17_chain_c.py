"""AMENDMENT_R17_002 resumption (2026-10-18 ~13:55 UTC). Six G queues (g_qw7 s2/s3/s4, g_ll8 s1/s2/s3) exited 1 at
12:53-12:54 UTC on their FIRST job: vLLM engine start-up refused because other processes held ~18 GB on the GPU
("Free memory 61.1/79.25 GiB < 0.8 utilisation"). No request was served and no job of these queues completed, so no
outcome exists. Outcome-blind resumption: each failed slot directory is kept as execution/formal/<slot>_void_a002 and the
SAME frozen plan (formal_<slot>_plan.json, unchanged bundle) is launched again on a genuinely idle GPU; no other change.
Runs alongside r17_chain_b.py without sharing its GPUs, then audits + analyses r17_g_qw7 and r17_g_ll8."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r17_common import *
import r12_chain as L
import r17_chain as R
L.LOG = OPS / 'r17_state_c.json'
L.OPS = OPS
FAILED = {'g_qw7': ['s2', 's3', 's4'], 'g_ll8': ['s1', 's2', 's3']}
_idle = L.idle


def idle():
    other = json.loads((OPS / 'r17_state.json').read_text())
    held = {v['gpu'] for v in other.get('running', {}).values()}
    return [u for u in _idle() if u not in held]


L.idle = idle


def main():
    L.save(stage='c_prepare', amendment='AMENDMENT_R17_002')
    for b, slots in FAILED.items():
        for s in slots:
            src = X / f'r17_{b}' / 'execution/formal' / s; dst = src.with_name(f'{s}_void_a002')
            if src.exists() and not dst.exists():
                st = json.loads((src / 'runs/state.json').read_text())
                assert st['status'] == 'stopped' and not st['completed'], (b, s, st['status'])
                src.rename(dst)
            write(X / f'r17_{b}' / f'AMENDMENT_R17_002_{s}.json', dict(void=str(dst), reason='vLLM start-up OOM from other '
                  'processes on the first job; no job completed; same plan relaunched'))
    # wait until r17_chain_b has launched all of its queues (so GPU choice never conflicts with its pending queues)
    while True:
        o = json.loads((OPS / 'r17_state.json').read_text())
        nformal = len([k for k in o['exits'] if k[0] in 'gh']) + len(o.get('running', {}))
        if nformal >= 16 or o.get('stage') in ('done', 'stopped'):
            break
        time.sleep(120)
    qc = [(f'{b}_{s}', X / f'r17_{b}', f'formal_{s}_plan.json', s, QW if b.endswith('qw7') else LL)
          for b, slots in FAILED.items() for s in slots]
    L.save(stage='c_run')
    ex = R.run_all(qc, 'formal')
    others = [f'{b}_{s}' for b, sl in FAILED.items() for s in ('s1', 's2', 's3', 's4') if s not in sl]
    while not all(k in json.loads((OPS / 'r17_state.json').read_text())['exits'] for k in others):
        time.sleep(120)
    o = json.loads((OPS / 'r17_state.json').read_text())['exits']
    for b in FAILED:
        allx = {s: ex.get(f'{b}_{s}', o.get(f'{b}_{s}')) for s in ('s1', 's2', 's3', 's4')}
        if all(v == '0' for v in allx.values()):
            if L.step(['audit_r12.py'], X / f'r17_{b}', f'audit_{b}', 14400):
                L.step(['analyze_r12.py'], X / f'r17_{b}', f'analysis_{b}', 3600)
            L.state['done'].append(b)
        else:
            L.note(f'{b}: queue exits {allx}; not audited')
    L.save(stage='done', final='done')


if __name__ == '__main__':
    main()
