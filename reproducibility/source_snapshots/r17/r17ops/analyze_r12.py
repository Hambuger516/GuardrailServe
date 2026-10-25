"""R16 registered analysis (frozen 2026-10-16 before any outcome; PROTOCOL_R16.md; = R15 analysis + family pairs always contrasted). Unit = trace (all arms of a trace on
one GPU); 12 traces per budget. Primary families from registry['families'] with their endpoints; Holm within each family
per model. Two-sided families: exact sign-flip enumeration over all 2^n assignments. Non-inferiority family ('ni'):
per-trace d_i = T_i - (1 - margin) B_i, one-sided exact sign-flip p for mean(d) > 0 and one-sided 95% bootstrap lower
bound. Effects: mean paired difference, 20,000-draw paired bootstrap percentile 95% CI (seed 20261016).
Supporting: every arm vs priority (if present) and vs protected, on all endpoints including the audit's extra measures."""
import hashlib, itertools, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
from paired_analysis import paired_summary, holm
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEED, DRAWS = 20261016, 20000


def exact_p(d, one_sided=False):
    d = np.asarray(d, float); n = len(d)
    signs = np.array(list(itertools.product([1, -1], repeat=n)), float)
    null = signs @ d / n; obs = d.mean()
    if one_sided:
        return float(np.mean(null >= obs - 1e-12))
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def endpoints(run):
    m = run['metrics']; out = {}
    for name, g in [('overall', m['overall']), *m['per_class'].items()]:
        n = g['submitted']
        out[name + '_gp_rps'] = g['window_slo_goodput_rps']; out[name + '_attainment'] = g['slo_met_fraction'] or 0.0
        out[name + '_noncompletion'] = 1 - g['states'].get('completed', 0) / n if n else 0.0
        for lat in ('ttft_ms_received', 'tpot_ms_completed_multi_token'):
            v = g[lat]['p99']
            if v is not None: out[f'{name}_{lat.split("_")[0]}_p99_ms'] = v
    out['preemptions'] = run['physical']['measured_preemptions']; out['recomputed_tokens'] = run['physical']['measured_recomputed_tokens']
    for k, v in run.get('extra', {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
            out[k] = v
    return out


def main():
    ev = json.loads((HERE / 'R12_COMPLETE_EVIDENCE.json').read_text()); gate = json.loads((HERE / 'FORMAL_AUDIT_PASS.json').read_text())
    assert gate['all_checks_pass'] and ev['n_runs'] == len(ev['runs']) and ev.get('complete', True)
    reg = json.loads((HERE / 'registry.json').read_text()); ARMS = reg['arms']
    fam_pairs = [tuple(pr) for f in reg['families'].values() if f['kind'] == 'two_sided' for pr in f['pairs']]
    contrasts = list(dict.fromkeys(fam_pairs + ([('priority', a) for a in ARMS if a != 'priority'] if 'priority' in ARMS else [])
                                   + ([('protected', a) for a in ARMS if a not in ('priority', 'protected')] if 'protected' in ARMS else [])
                                   + ([('hyst', 'norec')] if {'hyst', 'norec'} <= set(ARMS) else [])))
    cells = defaultdict(dict)
    for r in ev['runs']:
        j = r['job']; cells[(j['capacity'], j['replicate'])][j['treatment']] = r
    for c in cells.values():
        assert set(c) == set(ARMS) and len({x['gpu_uuid'] for x in c.values()}) == 1
    rows, fams, collapse, descriptive = [], defaultdict(list), {}, {}
    for K in sorted({k[0] for k in cells}):
        reps = sorted(k[1] for k in cells if k[0] == K)
        E = {i: {a: endpoints(cells[(K, i)][a]) for a in ARMS} for i in reps}
        ref = 'priority' if 'priority' in ARMS else 'protected'
        collapse[K] = {a: sum(E[i][a]['overall_noncompletion'] >= 0.05 or E[i][a]['HIGH_gp_rps'] <= 0.85 * E[i][ref]['HIGH_gp_rps']
                              for i in reps) for a in ARMS}
        keys = sorted(set.intersection(*(set(E[i][a]) for i in reps for a in ARMS)))
        descriptive[K] = {a: {e: float(np.mean([E[i][a][e] for i in reps])) for e in keys} for a in ARMS}
        for left, right in contrasts:
            for e in keys:
                x = [E[i][left][e] for i in reps]; y = [E[i][right][e] for i in reps]
                est = paired_summary(x, y, seed=SEED, draws=DRAWS); est['exact_p'] = exact_p(est['differences'])
                fam = None
                for fname, f in reg['families'].items():
                    if f['kind'] == 'two_sided' and [left, right] in f['pairs'] and e in f['endpoints']:
                        fam = fname
                row = dict(group=reg['group'], model=reg['model'], capacity=K, baseline=left, treatment=right, endpoint=e,
                           estimate=est, family=fam)
                rows.append(row)
                if fam: fams[fam].append(row)
        for fname, f in reg['families'].items():
            if f['kind'] != 'ni': continue
            for left, right in f['pairs']:
                for e in f['endpoints']:
                    x = np.array([E[i][left][e] for i in reps]); y = np.array([E[i][right][e] for i in reps])
                    d = y - (1 - f['margin']) * x
                    rng = np.random.default_rng(SEED)
                    lb = float(np.quantile(d[rng.integers(0, len(d), size=(DRAWS, len(d)))].mean(axis=1), 0.05))
                    row = dict(group=reg['group'], model=reg['model'], capacity=K, baseline=left, treatment=right, endpoint=e,
                               family=fname, margin=f['margin'],
                               estimate=dict(n_pairs=len(d), baseline_mean=float(x.mean()), treatment_mean=float(y.mean()),
                                             ni_statistic_mean=float(d.mean()), bootstrap_lower95_one_sided=lb,
                                             exact_p=exact_p(d, one_sided=True), differences=d.tolist()))
                    rows.append(row); fams[fname].append(row)
    for fam, members in fams.items():
        for row, p in zip(members, holm([r['estimate']['exact_p'] for r in members])): row['holm_p'] = p
    out = dict(kind='r16_registered_analysis', group=reg['group'], model=reg['model'], families={f: len(v) for f, v in fams.items()},
               contrasts=rows, collapse=collapse, descriptive_means=descriptive,
               evidence_sha256=sha(HERE / 'R12_COMPLETE_EVIDENCE.json'), analysis_sha256=sha(Path(__file__)))
    with (HERE / 'R12_ANALYSIS001.json').open('x') as f: json.dump(out, f, indent=1)
    show = ('overall_gp_rps', 'HIGH_gp_rps', 'LOW_gp_rps', 'steady_overall_gp_rps', 'steady_HIGH_gp_rps', 'recomputed_tokens',
            'preemptions', 'proactive_victims', 'native_preemptions', 'maxgap_p95_overall_ms', 'held_mean_frac')
    for r in rows:
        if r['family'] or r['endpoint'] in show:
            e = r['estimate']
            if r['family'] and reg['families'][r['family']]['kind'] == 'ni':
                print(f"{reg['group']} {reg['model'][:12]:12s} K={r['capacity']} {r['treatment']:9s}-{r['baseline']:9s} {r['endpoint']:22s} "
                      f"NI(m={r['margin']}) b={e['baseline_mean']:.3f} t={e['treatment_mean']:.3f} lb95={e['bootstrap_lower95_one_sided']:+.4f} "
                      f"p1={e['exact_p']:.4f} holm[{r['family']}]={r['holm_p']:.4f}")
                continue
            print(f"{reg['group']} {reg['model'][:12]:12s} K={r['capacity']} {r['treatment']:9s}-{r['baseline']:9s} {r['endpoint']:22s} "
                  f"b={e['baseline_mean']:.3f} t={e['treatment_mean']:.3f} d={e['mean_difference']:+.4f} "
                  f"ci=[{e['paired_percentile_ci95'][0]:+.3f},{e['paired_percentile_ci95'][1]:+.3f}] p={e['exact_p']:.4f}"
                  + (f" holm[{r['family']}]={r['holm_p']:.4f}" if r['family'] else ''))
    print('collapse', json.dumps(collapse))


if __name__ == '__main__':
    main()
