"""R15 resumption operator (AMENDMENT_R15_002, 2026-10-16). Waits until r15_chain.py has finished (stage done/stopped),
then: (1) in r15_b_ll8, moves the trace block hit by the external-process exclusion (K7724 t00, all four arms, so the
block stays on one GPU) and every later job of slot s2 to a resumption slot s2r (new plan file; superseded runs are kept
but never analysed), installs the amended audit, re-freezes the bundle manifest and the same-code smoke gate;
(2) runs r15_b_ll8/s2r and the eight C queues on genuinely idle GPUs (no retries); (3) audits + analyses r15_b_ll8,
r15_c_qw7, r15_c_ll8. No outcome is read before or during these steps."""
import json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r15_common import *
import r12_chain as L
L.LOG = OPS / 'r15_state2.json'
L.OPS = OPS
FIRST_MOVED = 'R15B_ll8_K7724_t00_priority'


def amend_b_ll8():
    b = X / 'r15_b_ll8'
    if (b / 'formal_s2r_plan.json').exists():
        return
    plan = json.loads((b / 'formal_s2_plan.json').read_text())
    names = [j['name'] for j in plan['jobs']]
    i = names.index(FIRST_MOVED)
    done = {n for n in names if (b / 'execution/formal/s2/runs' / f'{n}.exit').exists()}
    assert set(names[:i]) <= done, 'earlier blocks of s2 must have completed'
    new = dict(plan, slot='s2r', jobs=plan['jobs'][i:])
    write(b / 'formal_s2r_plan.json', new)
    write(b / 'AMENDMENT_R15_002.json', dict(
        reason='R15B_ll8_K7724_t00_protected on s2 was excluded by the GPU observer (another user started processes on '
               'the GPU during the run); the queue stopped by design. Outcome-blind resumption: the whole affected trace block '
               '(all four arms) and every later s2 job run in slot s2r on an idle GPU; the superseded s2 runs of that block '
               '(K7724_t00 priority completed, protected excluded) are kept and never analysed.',
        moved={j['name']: 's2r' for j in new['jobs']}, superseded_in_s2=sorted(n for n in names[i:] if n in done)))
    shutil.copy2(OPS / 'audit_r12.py', b / 'audit_r12.py')
    # Re-freeze: original file list (outputs under execution/ were never part of it) plus the amended/new files.
    files = json.loads((b / 'bundle_manifest.json').read_text())['files']
    assert not any(k.startswith('execution/') for k in files)
    for f in ('formal_s2r_plan.json', 'AMENDMENT_R15_002.json', 'audit_r12.py'):
        files[f] = sha(b / f)
    write(b / 'bundle_manifest.json', dict(files=dict(sorted(files.items()))))
    gate = json.loads((b / 'SMOKE_AUDIT_PASS.json').read_text())
    gate.update(bundle_sha256=sha(b / 'bundle_manifest.json'), amendment='AMENDMENT_R15_002 (plan s2r + audit; code unchanged)')
    (b / 'SMOKE_AUDIT_PASS.json').write_text(json.dumps(gate, indent=2))


def main():
    L.save(stage='waiting_for_chain1')
    while json.loads((OPS / 'r15_state.json').read_text()).get('stage') not in ('done', 'stopped'):
        time.sleep(120)
    amend_b_ll8(); L.note('amendment 002 applied to r15_b_ll8')
    import build_r15 as B
    q = [('b_ll8_s2r', X / 'r15_b_ll8', 'formal_s2r_plan.json', 's2r', LL)]
    q += [(f'c_{SHORT[m]}_s{i + 1}', X / f'r15_c_{SHORT[m]}', f'formal_s{i + 1}_plan.json', f's{i + 1}', m)
          for i in range(B.EXPS['c']['slots']) for m in (QW, LL)]
    L.save(stage='running')
    ex = L.run_queues(q, 'formal', 8)
    prev = json.loads((OPS / 'r15_state.json').read_text())['exits']
    todo = {'b_ll8': ['b_ll8_s1', 'b_ll8_s3', 'b_ll8_s4', 'b_ll8_s2r'],
            'c_qw7': [f'c_qw7_s{i}' for i in range(1, 5)], 'c_ll8': [f'c_ll8_s{i}' for i in range(1, 5)]}
    for key, qs in todo.items():
        if all({**prev, **ex}.get(k) == '0' for k in qs):
            if L.step(['audit_r12.py'], X / f'r15_{key}', f'audit_{key}', 14400):
                L.step(['analyze_r12.py'], X / f'r15_{key}', f'analysis_{key}', 3600)
            L.state['done'].append(key)
        else:
            L.note(f'{key}: not all queues exited 0; not audited')
    L.save(stage='done', final='done')


if __name__ == '__main__':
    main()
