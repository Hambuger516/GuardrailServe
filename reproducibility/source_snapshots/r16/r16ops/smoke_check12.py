"""R16 smoke/calibration integrity gate (bundle-local; also used by the formal audit). Checks that every run used the
registered scheduler class, R16 switches (regime mode, F2 mode and cap), R15 base switches, execution mode and length
mode. Integrity only; outcomes are never used for effects."""
import gzip, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R16 = 'runtime_r16.R16Scheduler'
ARMS = {'protected': ('protected', 'on'), 'protected_f2p': ('protected', 'pressure'), 'hyst': ('hyst', 'on'),
        'hyst_nof2': ('hyst', 'off'), 'hyst_f2p': ('hyst', 'pressure'), 'adapt': ('adapt', 'on'), 'adapt_f2p': ('adapt', 'pressure')}
SCHED = {'priority': 'runtime_revision.RevisionScheduler', **{a: R16 for a in ARMS}}


def first_events(path, names, limit=400):
    found = {}
    with gzip.open(path, 'rt') as f:
        for i, line in enumerate(f):
            e = json.loads(line)
            if e['event'] in names and e['event'] not in found:
                found[e['event']] = e
            if i > limit and len(found) == len(names):
                break
    return found


def count_events(path, names):
    n = {k: 0 for k in names}
    with gzip.open(path, 'rt') as f:
        for line in f:
            for k in names:
                if f'"event": "{k}"' in line:
                    n[k] += 1
    return n


def check_run(d, job, cfg):
    m = json.loads((d / 'manifest.json').read_text())
    assert m.get('status') == 'run_complete', m.get('status')
    assert not m.get('exclude_from_effect_estimate'), 'GPU observer exclusion'
    cls = m['engine_config']['scheduler_cls']; arm = job['treatment']
    assert cls == SCHED[arm], cls
    assert m['engine_config']['enforce_eager'] is job['eager'] is cfg['enforce_eager'], 'execution mode differs'
    assert m['engine_config']['num_gpu_blocks_override'] == job['capacity'] + 1
    fixed = job['length_mode'] == 'fixed_from_trace'
    assert m.get('length_mode') == job['length_mode'] and m.get('natural_eos') is (not fixed), 'length mode differs'
    ev_path = d / 'scheduler_events.jsonl.gz'
    want = {'r16_init', 'r15_init', 'r15_step', 'slack_init'} if cls == R16 else {'revision_init'}
    ev = first_events(ev_path, want)
    assert set(ev) == want, f'missing init events {want - set(ev)}'
    counts = {}
    if cls == R16:
        a, b = ev['r16_init'], ev['r15_init']; mode, f2 = ARMS[arm]
        assert cfg['r16'] == dict(mode=mode, f2=f2), cfg['r16']
        assert a['mode'] == mode and a['f2'] == f2 and a['high_decode_token_budget'] == (None if f2 == 'off' else 512), a
        assert (a['hyst_ceiling'] == 0.85 and a['hyst_target'] == 0.90 and a['on_tokens'] == 1000 and a['off_tokens'] == 800
                and a['tau_s'] == 10 and a['start_long'] is (mode == 'hyst') and a['protect_high_victims'] is True
                and a['warning_wait_fraction'] == 0.5), a
        assert (b['control_estimator'] == 'chance' and b['proactive_recovery_hook'] is True and b['warning_threshold'] == 0.8
                and b['recovery_trigger'] == 0.95 and b['inflight_protection'] is False), b
        counts = count_events(ev_path, ['slack_victim', 'danger', 'preempt', 'r16_regime', 'r15_shadow_error'])
        assert counts['r15_shadow_error'] == 0
        assert mode == 'adapt' or counts['r16_regime'] == 0, 'fixed regime switched'
    else:
        assert ev['revision_init']['treatment'] == 'priority'
    if fixed:
        met = json.loads((d / 'metrics.json').read_text())
        bad = [r['request_id'] for r in met['requests'] if r['status'] == 'completed'
               and r['output_tokens'] != r['target_output_tokens']]
        assert not bad, f'{len(bad)} completed outputs differ from their recorded length'
    return dict(scheduler=cls, counts=counts,
                init={k: {kk: vv for kk, vv in v.items() if kk not in ('monotonic_s', 'config')} for k, v in ev.items()})


def main():
    plan = json.loads((HERE / 'smoke_s1_plan.json').read_text()); runs = HERE / 'execution/smoke/s1/runs'
    report, ok = [], (runs / 'queue.exit').exists() and (runs / 'queue.exit').read_text().strip() == '0'
    for job in plan['jobs']:
        d = runs / job['name']; cfg = json.loads((HERE / job['config']).read_text())
        try:
            assert (runs / (job['name'] + '.exit')).read_text().strip() == '0', 'nonzero exit'
            report.append(dict(run=job['name'], passed=True, **check_run(d, job, cfg)))
        except Exception as exc:
            ok = False; report.append(dict(run=job['name'], passed=False, error=repr(exc)[:400]))
    out = dict(all_checks_pass=ok, runs=report, bundle_sha256=sha(HERE / 'bundle_manifest.json'),
               note='Integrity gate only; smoke and calibration outcomes never used for effects.')
    (HERE / ('SMOKE_AUDIT_PASS.json' if ok else 'SMOKE_AUDIT_FAIL.json')).write_text(json.dumps(out, indent=2))
    print(json.dumps(dict(ok=ok, runs=[(r['run'], r['passed'], r.get('error'), r.get('counts')) for r in report]), indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
