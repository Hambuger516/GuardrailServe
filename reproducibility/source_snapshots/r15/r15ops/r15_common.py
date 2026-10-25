"""R15 shared definitions (frozen 2026-10-16 before any R15 outcome). Reuses the R12 machinery with code src15."""
import json, sys
from pathlib import Path
sys.path.insert(0, '/data/hanbo_futureserve/tpds/r12ops')
import r12_common as C
C.SRC12 = C.X / 'src15'
C.OPS = C.X / 'r15ops'
C.QUEUE_FILES = C.QUEUE_FILES + ['ceiling_policy.py', 'runtime_ceiling.py', 'hysteresis_policy.py', 'runtime_hysteresis.py',
                                 'estimator_policy.py', 'runtime_r15.py']
sys.path.insert(0, str(C.SRC12 / 'code')); sys.path.insert(0, str(C.SRC12))
X, QW, LL, SHORT, K_SYN, N_TRACES, OPS = C.X, C.QW, C.LL, C.SHORT, C.K_SYN, C.N_TRACES, C.OPS
AZT = X / 'r15_azure'
sha, write, manifest, slot_plans, base_bundle = C.sha, C.write, C.manifest, C.slot_plans, C.base_bundle

R15_ARMS = {'protected': ('chance', True), 'norec': ('chance', False), 'mean': ('point', True),
            'hard': ('hard', True), 'nohighx': ('chance', True), 'minru': ('min_chance_hard', True)}
SCHED = {'priority': 'runtime_revision.RevisionScheduler', 'hyst': 'runtime_hysteresis.HysteresisScheduler',
         **{a: 'runtime_r15.R15Scheduler' for a in R15_ARMS}}


def config(arm):
    if arm == 'priority':
        return C.config('priority')
    c = C.config('protected')                         # C4 + warning 0.8K + wait 0.5xTTFT + F1 + F2=512
    if arm == 'hyst':                                 # R14 hysteresis arm, unchanged
        c.update(admission_ceiling=0.85, recovery_target=0.90)
        return c
    est, hook = R15_ARMS[arm]
    c['r15'] = dict(control_estimator=est, proactive_recovery_hook=hook)
    if arm == 'nohighx':
        c['protect_high_victims'] = False
    return c
