"""R16 formal audit (= R15 audit + R16 regime/F2 counters): raw reconstruction + physical accounting + registered scheduler/switches/execution/length mode for
every registered run, plus the registered supporting measurements (computed from raw logs before any analysis):
steady-segment goodput and 60-s bins, max client-observed token gap, proactive vs native preemptions, recomputation
before/after the service deadline, time-weighted held KV, R15 shadow/danger counters, controller and schedule time.
Writes FORMAL_AUDIT_PASS.json and R12_COMPLETE_EVIDENCE.json; no effect estimates."""
import gzip, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'audit_tools')); sys.path.insert(0, str(HERE))
from audit_openloop import audit as audit_raw
from audit_serving_scheduler import audit as audit_physical
from smoke_check12 import check_run
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CLASSES = ['HIGH', 'DEFAULT', 'LOW']


def q(values, p):
    v = sorted(values)
    if not v: return None
    x = (len(v) - 1) * p; lo = int(x); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (x - lo)


def support(d, m, metrics, cfg):
    W = json.loads((d / 'input_trace.json').read_text())['arrival_window_s']; drain = cfg['drain_s']
    reqs = metrics['requests']; out = {}
    groups = {'overall': reqs, **{c: [r for r in reqs if r['priority'] == i] for i, c in enumerate(CLASSES)}}
    if W >= 600:                                            # registered steady segment [120, W)
        for g, rs in groups.items():
            out[f'steady_{g}_gp_rps'] = sum(r['slo_met'] and r['finish_s'] is not None and 120 <= r['finish_s'] < W
                                            for r in rs) / (W - 120)
        bins = int((W + drain) // 60)
        out['bins60_overall'] = [sum(r['slo_met'] and r['finish_s'] is not None and b * 60 <= r['finish_s'] < (b + 1) * 60
                                     for r in reqs) / 60 for b in range(bins)]
        out['bins60_HIGH'] = [sum(r['slo_met'] and r['finish_s'] is not None and b * 60 <= r['finish_s'] < (b + 1) * 60
                                  for r in groups['HIGH']) / 60 for b in range(bins)]
        out['backlog60'] = [sum(r['planned_arrival_s'] < (b + 1) * 60 and (r['finish_s'] is None or r['finish_s'] >= (b + 1) * 60)
                                for r in reqs) for b in range(bins)]
        half = (W - 120) / 2
        out['drift_overall_gp_rps'] = (sum(r['slo_met'] and r['finish_s'] is not None and 120 + half <= r['finish_s'] < W for r in reqs)
                                       - sum(r['slo_met'] and r['finish_s'] is not None and 120 <= r['finish_s'] < 120 + half for r in reqs)) / half
    # max client-observed inter-token gap per completed request
    last, gap = {}, {}
    with gzip.open(d / 'client_events.jsonl.gz', 'rt') as f:
        for line in f:
            if '"token_chunk"' not in line: continue
            e = json.loads(line); rid = e['request_id']; t = e['elapsed_s']
            if rid in last: gap[rid] = max(gap.get(rid, 0.0), t - last[rid])
            last[rid] = t
    for g, rs in groups.items():
        v = [1000 * gap.get(r['request_id'], 0.0) for r in rs if r['status'] == 'completed' and r['output_tokens'] > 1]
        out[f'maxgap_p50_{g}_ms'], out[f'maxgap_p95_{g}_ms'], out[f'maxgap_max_{g}_ms'] = q(v, .5), q(v, .95), (max(v) if v else None)
    # scheduler events
    start = m['measurement_start_monotonic_s']; deadline = start + W + drain; wend = start + W
    cap = None; prev = None; area = 0.0; span = 0.0; n = dict(victims=0, preempt=0, danger=0, danger_shadow=0, rec_pre=0, rec_post=0)
    sched_us, step = [], dict(n=0, r_minus_u=0.0, r_minus_m=0.0, r_gt_u=0); last_counters = None; last16 = None; n_sched = 0
    with gzip.open(d / 'scheduler_events.jsonl.gz', 'rt') as f:
        for line in f:
            e = json.loads(line); ev = e['event']; t = e['monotonic_s']
            if ev == 'controller_init': cap = e['capacity_blocks']
            if t < start: continue
            if ev == 'slack_victim': n['victims'] += 1
            elif ev == 'preempt': n['preempt'] += 1
            elif ev == 'danger': n['danger'] += 1
            elif ev == 'danger_shadow': n['danger_shadow'] += 1
            elif ev == 'recompute_scheduled': n['rec_pre' if t <= deadline else 'rec_post'] += e['tokens']
            elif ev == 'schedule':
                if t <= wend:
                    sched_us.append(e['scheduler_us'])
                    held = cap - e['free_blocks']
                    if prev is not None: area += (t - prev[0]) * prev[1]; span += t - prev[0]
                    prev = (t, held)
                if 'r15_counters' in e: last_counters = e['r15_counters']
                if 'r16_counters' in e: last16 = e['r16_counters']; n_sched += 1
            elif ev == 'r15_step' and t <= wend:
                step['n'] += 1; step['r_minus_u'] += e['chance'] - e['hard']; step['r_minus_m'] += e['chance'] - e['point']
                step['r_gt_u'] += e['chance'] > e['hard']
    out.update(proactive_victims=n['victims'], native_preemptions=n['preempt'] - n['victims'], recovery_episodes=n['danger'],
               danger_shadow=n['danger_shadow'], recompute_pre_deadline=n['rec_pre'], recompute_post_deadline=n['rec_post'],
               held_mean_blocks=area / span if span else None, held_mean_frac=(area / span / cap) if span and cap else None,
               sched_us_mean=sum(sched_us) / len(sched_us) if sched_us else None, sched_us_p99=q(sched_us, .99))
    if step['n']:
        out.update(r_minus_u_mean=step['r_minus_u'] / step['n'], r_minus_m_mean=step['r_minus_m'] / step['n'],
                   r_gt_u_frac=step['r_gt_u'] / step['n'])
    if last_counters:
        out.update(shadow_checks=last_counters['shadow_checks'], shadow_disagreements=last_counters['shadow_disagreements'],
                   inner_ms_total=1000 * (last_counters['inner_admission_s'] + last_counters['inner_recover_s']),
                   budget_cap_steps=last_counters['budget_cap_steps'], admission_reasons=last_counters['reasons'])
    if last16:
        out.update(regime_switches=last16['switches'], long_frac=last16['long_steps'] / max(n_sched, 1),
                   f2_caps_skipped=last16['cap_skipped_low_pressure'])
    return out


def main():
    reg = json.loads((HERE / 'registry.json').read_text())
    manifest = json.loads((HERE / 'bundle_manifest.json').read_text())['files']
    runs, problems, slot_gpu = [], [], {}
    # AMENDMENT_R15_002: jobs moved to a resumption slot after an external-process exclusion (whole trace blocks).
    amend = HERE / 'AMENDMENT_R15_002.json'
    moved = json.loads(amend.read_text())['moved'] if amend.exists() else {}
    for slot in reg['slots']:
        plan = json.loads((HERE / f'formal_{slot}_plan.json').read_text())
        for job in plan['jobs']:
            actual = moved.get(job['name'], slot)
            sdir = HERE / 'execution/formal' / actual
            launch = json.loads((sdir / 'launch.json').read_text())
            slot_gpu[actual] = (launch['host'], launch['gpu_uuid'])
            d = sdir / 'runs' / job['name']
            try:
                assert (sdir / 'runs' / (job['name'] + '.exit')).read_text().strip() == '0', 'nonzero exit'
                m = json.loads((d / 'manifest.json').read_text())
                assert sha(d / 'input_config.json') == manifest[job['config']], 'config differs'
                assert sha(d / 'input_trace.json') == manifest[job['trace']], 'trace differs'
                assert m['engine_config']['scheduler_cls'] == reg['sched'][job['treatment']], 'scheduler differs'
                assert json.loads((d / 'input_trace.json').read_text())['target_rate_rps'] == job['rate'], 'rate differs'
                assert m['runtime_environment']['TPDS_GPU_UUID'] == launch['gpu_uuid']
                cfg = json.loads((d / 'input_config.json').read_text())
                assert cfg['stage'] == 'formal'
                init = check_run(d, job, cfg)
                raw, phys = audit_raw(d), audit_physical(d)
                assert raw['all_checks_pass'] and phys['all_checks_pass'], 'raw/physical audit failed'
                metrics = json.loads((d / 'metrics.json').read_text())
                extra = support(d, m, metrics, cfg)
                metrics.pop('requests', None)
                runs.append(dict(job=job, slot=actual, host=launch['host'], gpu_uuid=launch['gpu_uuid'],
                                 trace_sha256=manifest[job['trace']], manifest_sha256=sha(d / 'manifest.json'),
                                 metrics=metrics, physical=phys, init=init['init'], counts=init['counts'], extra=extra))
            except Exception as exc:
                problems.append(dict(run=job['name'], slot=actual, error=repr(exc)[:300]))
    n = reg['n_formal']
    ok = not problems and len(runs) == n
    gate = dict(all_checks_pass=ok, n_runs=len(runs), expected=n, problems=problems, slots=slot_gpu,
                bundle_sha256=sha(HERE / 'bundle_manifest.json'))
    (HERE / ('FORMAL_AUDIT_PASS.json' if ok else 'FORMAL_AUDIT_FAIL.json')).write_text(json.dumps(gate, indent=2))
    # Evidence is written for every audited run; analysis refuses to run unless the gate passed.
    (HERE / 'R12_COMPLETE_EVIDENCE.json').write_text(json.dumps(dict(group=reg['group'], n_runs=len(runs), runs=runs,
        complete=ok, registry_sha256=sha(HERE / 'registry.json')), indent=1))
    print(json.dumps(dict(all_checks_pass=ok, runs=len(runs), problems=problems[:10]), indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
