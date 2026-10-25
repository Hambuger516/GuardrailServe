"""R16 code = R15 code (src15) + one default-off extension (frozen 2026-10-16). Usage: patch_code16.py SRC15 DST R16DIR
r16_policy.py / runtime_r16.py: recovery-depth regime (protected / hyst / adapt) and F2 mode (on / off / pressure) on the
R15 adapter; dispatched only when config['r16'] is set together with config['r15'] (protected/slack configuration)."""
import shutil, sys
from pathlib import Path
SRC, DST, R16 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for f in ['r16_policy.py', 'runtime_r16.py', 'test_r16.py']:
    shutil.copy2(R16 / 'code_add' / f, DST / 'code' / f)


def edit(path, pairs):
    p = DST / path; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    p.write_text(s)


OLD = "'estimator_policy.py', 'runtime_r15.py', "
NEW = "'estimator_policy.py', 'runtime_r15.py', 'r16_policy.py', 'runtime_r16.py', "
edit('code/benchmark_runtime.py', [
    ("""        report['r15'] = dict(cfg['r15'])
""", """        report['r15'] = dict(cfg['r15'])
    if cfg.get('r16') is not None:
        # R16: recovery-depth regime and F2 mode on the R15 adapter.
        if engine_cfg['scheduler_cls'] != 'runtime_r15.R15Scheduler':
            raise ValueError('r16 arms are registered only together with r15 on the protected configuration')
        policy_cfg['r16'] = dict(cfg['r16'])
        engine_cfg['scheduler_cls'] = 'runtime_r16.R16Scheduler'
        report['r16'] = dict(cfg['r16'])
"""),
    (OLD, NEW),
])
edit('code/run_benchmark_queue.py', [(OLD, NEW)])
edit('launch_batch.py', [("runtime_hysteresis,runtime_r15;", "runtime_hysteresis,runtime_r15,runtime_r16;")])
print('patched16', DST)
