"""R15 operator (frozen 2026-10-16). No retries; stops on any queue failure.
phase A: smoke (six A arms), C-geometry integrity run, Azure-2024-Thursday calibration -> smoke gates -> frozen budget
decision -> build formal bundles; phase B: formal queues (A, then B, then C) on genuinely idle GPUs (two consecutive idle
observations; a GPU is reused once its queue exits) -> per-bundle audit + registered analysis.
Usage: r15_chain.py [--phase-b-only]"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r15_common import *
import r12_chain as L
L.LOG = OPS / 'r15_state.json'
L.OPS = OPS


def phase_b():
    import build_r15 as B
    qb = [(f'{e}_{SHORT[m]}_s{i + 1}', X / f'r15_{e}_{SHORT[m]}', f'formal_s{i + 1}_plan.json', f's{i + 1}', m)
          for e in B.ORDER for i in range(B.EXPS[e]['slots']) for m in (QW, LL)]
    L.save(stage='phase_b')
    ex = L.run_queues(qb, 'formal', 8)
    for e in B.ORDER:
        for m in (QW, LL):
            key = f'{e}_{SHORT[m]}'; bd = X / f'r15_{key}'
            if all(ex.get(f'{key}_s{i + 1}') == '0' for i in range(B.EXPS[e]['slots'])):
                if L.step(['audit_r12.py'], bd, f'audit_{key}', 14400):
                    L.step(['analyze_r12.py'], bd, f'analysis_{key}', 3600)
                L.state['done'].append(key)
    L.save(stage='done', final='done')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--phase-b-only', action='store_true'); args = ap.parse_args()
    if args.phase_b_only:
        prev = json.loads(L.LOG.read_text()); assert prev.get('stage') == 'phase_a_done'
        L.state['events'] = prev['events']; L.state['exits'] = prev['exits']
        return phase_b()
    L.save(stage='phase_a')
    qa = [(f'{k}_{SHORT[m]}', X / f'r15_{k}_{SHORT[m]}', 'smoke_s1_plan.json', 's1', m)
          for k in ('csmoke', 'smoke', 'realcal') for m in (QW, LL)]
    ex = L.run_queues(qa, 'smoke', 8)
    if any(v != '0' for v in ex.values()):
        return L.save(stage='stopped', final='phase A queue failure')
    if not all(L.step(['smoke_check12.py'], X / f'r15_{k}', f'smoke_check_{k}', 3600) for k, *_ in qa):
        return L.save(stage='stopped', final='smoke gate failed')
    if not L.step(['decide_r15.py'], OPS, 'decide_r15', 600):
        return L.save(stage='stopped', final='decision failed')
    if not L.step(['build_r15.py', 'formal'], OPS, 'build_r15_formal', 3600):
        return L.save(stage='stopped', final='formal build failed')
    L.save(stage='phase_a_done')
    return phase_b()


if __name__ == '__main__':
    main()
