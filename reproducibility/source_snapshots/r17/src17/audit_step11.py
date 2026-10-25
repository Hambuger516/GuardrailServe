"""step11 formal audit: raw reconstruction + physical accounting for every registered run.
Writes FORMAL_AUDIT_PASS.json and STEP11_COMPLETE_EVIDENCE.json; no effect estimates."""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'audit_tools'))
from audit_openloop import audit as audit_raw
from audit_serving_scheduler import audit as audit_physical
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SCHED = {"priority": "runtime_revision.RevisionScheduler", "slack_f12": "runtime_slack.SlackScheduler", "f12_f3a": "runtime_slack.SlackScheduler", "f12_f3ab": "runtime_slack.SlackScheduler"}


def main():
    reg = json.loads((HERE / 'registry.json').read_text())
    manifest = json.loads((HERE / 'bundle_manifest.json').read_text())['files']
    runs, problems, slot_gpu = [], [], {}
    for slot in reg['slots']:
        plan = json.loads((HERE / f'formal_{slot}_plan.json').read_text())
        sdir = HERE / 'execution/formal' / slot
        launch = json.loads((sdir / 'launch.json').read_text())
        slot_gpu[slot] = (launch['host'], launch['gpu_uuid'])
        for job in plan['jobs']:
            d = sdir / 'runs' / job['name']
            try:
                assert (sdir / 'runs' / (job['name'] + '.exit')).read_text().strip() == '0', 'nonzero exit'
                m = json.loads((d / 'manifest.json').read_text())
                assert m['status'] == 'run_complete', m['status']
                assert sha(d / 'input_config.json') == manifest[job['config']], 'config differs'
                assert sha(d / 'input_trace.json') == manifest[job['trace']], 'trace differs'
                assert m['engine_config']['scheduler_cls'] == SCHED[job['treatment']], 'scheduler differs'
                assert m['engine_config']['num_gpu_blocks_override'] == job['capacity'] + 1
                assert json.loads((d / 'input_trace.json').read_text())['target_rate_rps'] == job['rate'], 'rate differs'
                assert m['runtime_environment']['TPDS_GPU_UUID'] == launch['gpu_uuid']
                assert not m.get('exclude_from_effect_estimate'), 'excluded by GPU observer'
                raw, phys = audit_raw(d), audit_physical(d)
                assert raw['all_checks_pass'] and phys['all_checks_pass'], 'raw/physical audit failed'
                metrics = json.loads((d / 'metrics.json').read_text())
                runs.append(dict(job=job, slot=slot, host=launch['host'], gpu_uuid=launch['gpu_uuid'],
                                 trace_sha256=manifest[job['trace']], manifest_sha256=sha(d / 'manifest.json'),
                                 metrics=metrics, physical=phys))
            except Exception as exc:
                problems.append(dict(run=job['name'], slot=slot, error=repr(exc)[:300]))
    n = reg['n_formal']
    ok = not problems and len(runs) == n
    gate = dict(all_checks_pass=ok, n_runs=len(runs), expected=n, problems=problems, slots=slot_gpu,
                bundle_sha256=sha(HERE / 'bundle_manifest.json'))
    (HERE / ('FORMAL_AUDIT_PASS.json' if ok else 'FORMAL_AUDIT_FAIL.json')).write_text(json.dumps(gate, indent=2))
    if ok:
        (HERE / 'STEP11_COMPLETE_EVIDENCE.json').write_text(json.dumps(dict(group='STEP11', n_runs=n, runs=runs,
            registry_sha256=sha(HERE / 'registry.json')), indent=1))
    print(json.dumps(dict(all_checks_pass=ok, runs=len(runs), problems=problems[:10]), indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
