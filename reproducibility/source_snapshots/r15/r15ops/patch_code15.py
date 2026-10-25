"""R15 code = R14 code (src14) + one default-off extension (frozen 2026-10-16). Usage: patch_code15.py SRC14 DST R15DIR
estimator_policy.py / runtime_r15.py: control-quantity choice and proactive-recovery hook switch on the protected
adapter; dispatched only when config['r15'] is set on the protected/slack configuration (no ablation/ceiling)."""
import shutil, sys
from pathlib import Path
SRC, DST, R15 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for f in ['estimator_policy.py', 'runtime_r15.py', 'test_r15.py']:
    shutil.copy2(R15 / 'code_add' / f, DST / 'code' / f)


def edit(path, pairs):
    p = DST / path; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    p.write_text(s)


OLD = "'hysteresis_policy.py', 'runtime_hysteresis.py', "
NEW = "'hysteresis_policy.py', 'runtime_hysteresis.py', 'estimator_policy.py', 'runtime_r15.py', "
edit('code/benchmark_runtime.py', [
    ("""    if cfg['policy'] in {'PFS-port-cold', 'PFS-port-warm'}:
        if cfg.get('ablation') is not None:""",
     """    if cfg.get('r15') is not None:
        # R15: control-quantity choice / proactive-recovery hook on the protected (slack) adapter.
        if engine_cfg['scheduler_cls'] != 'runtime_slack.SlackScheduler' or cfg.get('ablation') is not None:
            raise ValueError('r15 arms are registered only on the protected/slack configuration')
        policy_cfg['r15'] = dict(cfg['r15'])
        engine_cfg['scheduler_cls'] = 'runtime_r15.R15Scheduler'
        report['r15'] = dict(cfg['r15'])
    if cfg['policy'] in {'PFS-port-cold', 'PFS-port-warm'}:
        if cfg.get('ablation') is not None:"""),
    (OLD, NEW),
])
edit('code/run_benchmark_queue.py', [(OLD, NEW)])
edit('launch_batch.py', [("runtime_ceiling,runtime_hysteresis;", "runtime_ceiling,runtime_hysteresis,runtime_r15;")])
print('patched15', DST)
