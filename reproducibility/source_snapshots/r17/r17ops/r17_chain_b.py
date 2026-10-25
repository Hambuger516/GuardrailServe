"""AMENDMENT_R17_001 resumption (2026-10-18 ~12:50 UTC; before any formal R17 run). Phase A completed and passed every
gate (four queues exit 0, four SMOKE_AUDIT_PASS, decide_r17 -> R17_G_DECISION.json), but build_r17.py formal read the
decision under its R16 name (R17_D_DECISION.json) and raised FileNotFoundError, so the chain stopped before building any
formal bundle. The only change is that file name in build_r17.py line 82; the frozen decision, arms, families, seeds,
traces and the phase-B procedure (r17_chain.run_all, audit, analysis) are unchanged. Phase A is not rerun."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r17_common import *
import r17_chain as R
L = R.L


def main():
    L.state.update(json.loads((OPS / 'r17_state.json').read_text()))
    L.save(stage='phase_b_build', amendment='AMENDMENT_R17_001')
    for k in ('smoke', 'realcal'):
        for m in (QW, LL):
            assert json.loads((X / f'r17_{k}_{SHORT[m]}' / 'SMOKE_AUDIT_PASS.json').read_text())['all_checks_pass']
    if not L.step(['build_r17.py', 'formal'], OPS, 'build_r17_formal_a001', 3600):
        return L.save(stage='stopped', final='formal build failed (A001)')
    import build_r17 as B
    qb = [(f'{e}_{SHORT[m]}_s{i + 1}', X / f'r17_{e}_{SHORT[m]}', f'formal_s{i + 1}_plan.json', f's{i + 1}', m)
          for e in B.ORDER for i in range(B.EXPS[e]['slots']) for m in (QW, LL)]
    L.save(stage='phase_b')
    ex = R.run_all(qb, 'formal')
    for e in B.ORDER:
        for m in (QW, LL):
            key = f'{e}_{SHORT[m]}'; bd = X / f'r17_{key}'
            if all(ex.get(f'{key}_s{i + 1}') == '0' for i in range(B.EXPS[e]['slots'])):
                if L.step(['audit_r12.py'], bd, f'audit_{key}', 14400):
                    L.step(['analyze_r12.py'], bd, f'analysis_{key}', 3600)
                L.state['done'].append(key)
            else:
                L.note(f'{key}: a queue failed; not audited')
    L.save(stage='done', final='done')


if __name__ == '__main__':
    main()
