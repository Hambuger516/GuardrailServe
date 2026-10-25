"""step11 registered analysis (fixed before any formal outcome exists).
Unit: one trace (four arms on one GPU); 12 traces per cell: (6 rps, 591), (8 rps, 800), (8 rps, 1000), (8 rps, 591).
Primary family (Holm over 6): LOW window goodput, fix - slack_f12, fix in {f12_f3a, f12_f3ab}, in the three LOW-loss
cells (6/591, 8/800, 8/1000).
Safety (separate family, Holm over 16): fix - priority for HIGH and overall goodput in all four cells.
Supporting: all pairwise contrasts listed below; per-class goodput/attainment/noncompletion; preemptions; recompute;
collapse counts (overall noncompletion >= 0.05 or HIGH goodput <= 0.85 x same-trace priority)."""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'code'))
from paired_analysis import paired_summary, holm
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ARMS = ['priority', 'slack_f12', 'f12_f3a', 'f12_f3ab']; FIX = ARMS[2:]
CONTRASTS = [('priority', a) for a in ARMS[1:]] + [('slack_f12', a) for a in FIX]


def endpoints(run):
    m = run['metrics']; out = {}
    for name, g in [('overall', m['overall']), *m['per_class'].items()]:
        n = g['submitted']
        out[name + '_gp_rps'] = g['window_slo_goodput_rps']
        out[name + '_attainment'] = g['slo_met_fraction']
        out[name + '_noncompletion'] = 1 - g['states'].get('completed', 0) / n if n else 0.0
    out['preemptions'] = run['physical']['measured_preemptions']
    out['recomputed_tokens'] = run['physical']['measured_recomputed_tokens']
    return out


def main():
    ev = json.loads((HERE / 'STEP11_COMPLETE_EVIDENCE.json').read_text())
    gate = json.loads((HERE / 'FORMAL_AUDIT_PASS.json').read_text())
    assert gate['all_checks_pass'] and ev['n_runs'] == len(ev['runs'])
    cells = defaultdict(dict)
    for r in ev['runs']:
        j = r['job']; key = ((j['rate'], j['capacity']), j['replicate'])
        assert j['treatment'] not in cells[key]; cells[key][j['treatment']] = r
    for c in cells.values():
        assert set(c) == set(ARMS) and len({x['gpu_uuid'] for x in c.values()}) == 1
    rows, prim, safety, collapse = [], [], [], {}
    for K in sorted({k[0] for k in cells}):  # K = (rate, capacity)
        reps = sorted(k[1] for k in cells if k[0] == K)
        E = {i: {a: endpoints(cells[(K, i)][a]) for a in ARMS} for i in reps}
        collapse[f'{K[0]}rps_K{K[1]}'] = {a: sum(E[i][a]['overall_noncompletion'] >= 0.05 or E[i][a]['HIGH_gp_rps'] <= 0.85 * E[i]['priority']['HIGH_gp_rps']
                              for i in reps) for a in ARMS}
        for left, right in CONTRASTS:
            for e in E[reps[0]]['priority']:
                x = [E[i][left][e] for i in reps]; y = [E[i][right][e] for i in reps]
                row = dict(rate=K[0], capacity=K[1], baseline=left, treatment=right, endpoint=e, n_pairs=len(x),
                           estimate=paired_summary(x, y, seed=20261015, draws=20000))
                rows.append(row)
                if left == 'slack_f12' and e == 'LOW_gp_rps' and K != (8, 591): prim.append(row)
                if left == 'priority' and right in FIX and e in ('HIGH_gp_rps', 'overall_gp_rps'): safety.append(row)
    for fam, key in ((prim, 'holm_p_primary'), (safety, 'holm_p_safety')):
        for row, p in zip(fam, holm([r['estimate']['signflip_p'] for r in fam])): row[key] = p
    out = dict(kind='step11_low_registered_analysis', contrasts=rows, collapse=collapse,
               primary_family='LOW goodput fix - slack_f12 in 6/591, 8/800, 8/1000; Holm over 6',
               safety_family='fix - priority HIGH and overall in 4 cells; Holm over 16',
               evidence_sha256=sha(HERE / 'STEP11_COMPLETE_EVIDENCE.json'), analysis_sha256=sha(Path(__file__)))
    with (HERE / 'STEP11_ANALYSIS001.json').open('x') as f: json.dump(out, f, indent=1)
    for tag, fam, key in (('PRIMARY', prim, 'holm_p_primary'), ('SAFETY', safety, 'holm_p_safety')):
        for r in fam:
            e = r['estimate']
            print(f"{tag} {r['rate']}rps K={r['capacity']} {r['treatment']:8s}-{r['baseline']:9s} {r['endpoint']:14s} b={e['baseline_mean']:.3f} t={e['treatment_mean']:.3f} "
                  f"d={e['mean_difference']:+.4f} CI={[round(v, 4) for v in e['paired_percentile_ci95']]} holm={r[key]:.4f}")
    print('collapse', json.dumps(collapse))


if __name__ == '__main__':
    main()
