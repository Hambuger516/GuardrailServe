"""R15 B budget rule for the unseen Azure-2024 Thursday workload: identical to the frozen R12/R14 REAL rule
(decide_real.py): D = time-weighted mean held KV blocks under unconstrained Priority (K = 12,000) on the two calibration
windows; K_j = round(r_j * D) with the frozen XM reference ratios (591, 800)/D_ref, floored at ceil(4096/16) + 2."""
import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, '/data/hanbo_futureserve/tpds/r12ops')
from r15_common import *
from decide_real import measure


def main():
    ref = json.loads((X / 'XM_DECISION.json').read_text())['reference']; ratios = ref['ratios']; floor = math.ceil(4096 / 16) + 2
    out = dict(rule=__doc__, reference=dict(model=ref['model'], D=ref['D'], K=ref['K'], ratios=ratios), models={})
    for model in (QW, LL):
        b = X / f'r15_realcal_{SHORT[model]}'
        assert json.loads((b / 'SMOKE_AUDIT_PASS.json').read_text())['all_checks_pass'], model
        plan = json.loads((b / 'smoke_s1_plan.json').read_text())
        per = [measure(b / 'execution/smoke/s1/runs' / j['name']) for j in plan['jobs']]
        D = sum(p['D'] for p in per) / len(per)
        out['models'][model] = dict(D=D, K=[max(floor, round(r * D)) for r in ratios], calibration=per)
    write(X / 'R15_B_DECISION.json', out)
    print(json.dumps({m: v['K'] for m, v in out['models'].items()}))


if __name__ == '__main__':
    main()
