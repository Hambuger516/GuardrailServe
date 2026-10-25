"""R17 shared definitions (frozen 2026-10-17 before any R17 outcome). Reuses the R12 machinery with code src17."""
import json, sys
from pathlib import Path
sys.path.insert(0, '/data/hanbo_futureserve/tpds/r12ops')
import r12_common as C
C.SRC12 = C.X / 'src17'
C.OPS = C.X / 'r17ops'
C.QUEUE_FILES = C.QUEUE_FILES + ['ceiling_policy.py', 'runtime_ceiling.py', 'hysteresis_policy.py', 'runtime_hysteresis.py',
                                 'estimator_policy.py', 'runtime_r15.py', 'r16_policy.py', 'runtime_r16.py']
sys.path.insert(0, str(C.SRC12 / 'code')); sys.path.insert(0, str(C.SRC12))
X, QW, LL, SHORT, K_SYN, N_TRACES, OPS = C.X, C.QW, C.LL, C.SHORT, C.K_SYN, C.N_TRACES, C.OPS
AZS = X / 'r17_azure'
sha, write, manifest, slot_plans, base_bundle = C.sha, C.write, C.manifest, C.slot_plans, C.base_bundle

R16_ARMS = {'protected': ('protected', 'on'), 'protected_f2p': ('protected', 'pressure'),
            'hyst': ('hyst', 'on'), 'hyst_nof2': ('hyst', 'off'), 'hyst_f2p': ('hyst', 'pressure'),
            'adapt': ('adapt', 'on'), 'adapt_f2p': ('adapt', 'pressure'), 'final': ('adapt', 'regime')}
SCHED = {'priority': 'runtime_revision.RevisionScheduler', **{a: 'runtime_r16.R16Scheduler' for a in R16_ARMS}}


def config(arm):
    if arm == 'priority':
        return C.config('priority')
    c = C.config('protected')                         # C4 + warning 0.8K + wait 0.5xTTFT + F1 + F2=512
    mode, f2 = R16_ARMS[arm]
    c['r15'] = dict(control_estimator='chance', proactive_recovery_hook=True)
    c['r16'] = dict(mode=mode, f2=f2)
    if f2 == 'off':
        c.pop('high_decode_token_budget')
    return c
