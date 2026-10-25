"""AMENDMENT_R17_003 resumption (2026-10-18 ~18:55 UTC; outcome-blind). In r17_g_ll8 slot s3, job 12/18
R17G_ll8_K3406_t05_final failed at 18:35 UTC at vLLM start-up (another user's processes took the GPU between jobs:
"Free memory 45.4/79.3 GiB < 0.8"); the queue stopped by design. As in AMENDMENT_R15_002, the whole affected trace block
(K3406 t05, all three arms, so the block stays on one GPU) and every later s3 job run in a resumption slot s3r on a
genuinely idle GPU; the superseded s3 runs of that block (t05 protected, priority) are kept and never analysed. Plan
s3r = the frozen s3 plan from that block on; amended audit (reads the moved map); bundle manifest and same-code gate
re-frozen. Then audit + analysis of r17_g_ll8. No outcome is read before or during these steps."""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r17_common import *
import r12_chain as L
import r17_chain as R
L.LOG = OPS / 'r17_state_d.json'
L.OPS = OPS
FIRST = 'R17G_ll8_K3406_t05_protected'


def amend():
    b = X / 'r17_g_ll8'
    if (b / 'formal_s3r_plan.json').exists():
        return
    plan = json.loads((b / 'formal_s3_plan.json').read_text())
    names = [j['name'] for j in plan['jobs']]; i = names.index(FIRST)
    done = {n for n in names if (b / 'execution/formal/s3/runs' / f'{n}.exit').exists()}
    assert set(names[:i]) <= done
    write(b / 'formal_s3r_plan.json', dict(plan, slot='s3r', jobs=plan['jobs'][i:]))
    write(b / 'AMENDMENT_R17_003.json', dict(reason=__doc__, moved={j['name']: 's3r' for j in plan['jobs'][i:]},
          superseded_in_s3=sorted(n for n in names[i:] if n in done)))
    shutil.copy2(OPS / 'audit_r12.py', b / 'audit_r12.py')
    files = json.loads((b / 'bundle_manifest.json').read_text())['files']
    assert not any(k.startswith('execution/') for k in files)
    for f in ('formal_s3r_plan.json', 'AMENDMENT_R17_003.json', 'audit_r12.py'):
        files[f] = sha(b / f)
    write(b / 'bundle_manifest.json', dict(files=dict(sorted(files.items()))))
    gate = json.loads((b / 'SMOKE_AUDIT_PASS.json').read_text())
    gate.update(bundle_sha256=sha(b / 'bundle_manifest.json'), amendment='AMENDMENT_R17_003 (plan s3r + audit; code unchanged)')
    (b / 'SMOKE_AUDIT_PASS.json').write_text(json.dumps(gate, indent=2))


def main():
    L.save(stage='d_prepare', amendment='AMENDMENT_R17_003')
    amend()
    L.save(stage='d_run')
    ex = R.run_all([('g_ll8_s3r', X / 'r17_g_ll8', 'formal_s3r_plan.json', 's3r', LL)], 'formal', max_gpus=1)
    if ex.get('g_ll8_s3r') == '0':
        if L.step(['audit_r12.py'], X / 'r17_g_ll8', 'audit_g_ll8', 14400):
            L.step(['analyze_r12.py'], X / 'r17_g_ll8', 'analysis_g_ll8', 3600)
        L.save(stage='done', final='done')
    else:
        L.save(stage='stopped', final=f's3r exit {ex}')


if __name__ == '__main__':
    main()
