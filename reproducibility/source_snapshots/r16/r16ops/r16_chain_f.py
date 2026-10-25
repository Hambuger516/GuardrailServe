"""AMENDMENT_R16_001 operator: waits until r16_chain.py has finished (no interleaving on shared GPUs), builds r16_f_ll8,
runs its four queues on idle GPUs (no retries), then audits and analyses it."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r16_common import *
import r12_chain as L
import r16_chain as R
L.LOG = OPS / 'r16_state_f.json'
L.OPS = OPS


def main():
    L.save(stage='waiting_for_r16_chain')
    while json.loads((OPS / 'r16_state.json').read_text()).get('stage') not in ('done', 'stopped'):
        time.sleep(120)
    if not L.step(['build_r16f.py'], OPS, 'build_r16f', 1800):
        return L.save(stage='stopped', final='build failed')
    q = [(f'f_ll8_s{i}', X / 'r16_f_ll8', f'formal_s{i}_plan.json', f's{i}', LL) for i in range(1, 5)]
    L.save(stage='running')
    ex = R.run_all(q, 'formal')
    if all(ex.get(k) == '0' for k, *_ in q):
        if L.step(['audit_r12.py'], X / 'r16_f_ll8', 'audit_f_ll8', 14400):
            L.step(['analyze_r12.py'], X / 'r16_f_ll8', 'analysis_f_ll8', 3600)
    L.save(stage='done', final='done')


if __name__ == '__main__':
    main()
