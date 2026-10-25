from pathlib import Path
from collections import defaultdict, Counter
import hashlib, json, math, subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE
BUNDLES = ['abl_qw7', 'abl_ll8', 'graph_qw7', 'graph_ll8', 'sens_qw7']
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def holm(values):
    adjusted = [None] * len(values)
    last = 0.0
    for rank, i in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        last = min(1.0, max(last, (len(values)-rank)*values[i]))
        adjusted[i] = last
    return adjusted

def endpoints(run):
    out = {}
    for name, g in [('overall', run['metrics']['overall']), *run['metrics']['per_class'].items()]:
        n = g['submitted']
        out[name+'_gp_rps'] = g['window_slo_completions']/run['metrics']['arrival_window_s']
        out[name+'_attainment'] = g['slo_met']/n if n else 0.0
        out[name+'_noncompletion'] = 1-g['states'].get('completed', 0)/n if n else 0.0
        for lat in ['ttft_ms_received', 'tpot_ms_completed_multi_token']:
            if g[lat]['p99'] is not None:
                out[name+'_'+lat.split('_')[0]+'_p99_ms'] = g[lat]['p99']
        assert abs(out[name+'_gp_rps']-g['window_slo_goodput_rps']) < 1e-10
        assert abs(out[name+'_attainment']-(g['slo_met_fraction'] or 0.0)) < 1e-10
        assert 0 <= g['window_slo_completions'] <= g['slo_met'] <= g['states'].get('completed', 0) <= n
        assert sum(g['states'].values()) == n
    out['preemptions'] = run['physical']['measured_preemptions']
    out['recomputed_tokens'] = run['physical']['measured_recomputed_tokens']
    assert run['physical']['all_checks_pass']
    m = run['metrics']
    for key in ['submitted','slo_met','window_slo_completions']:
        assert m['overall'][key] == sum(g[key] for g in m['per_class'].values())
    return out

def main():
    remote = json.loads((HERE/'remote_hashes.json').read_text())
    rng = np.random.default_rng(20261013)
    boot = rng.integers(0, 12, size=(20000,12))
    signs_mc = rng.choice([-1,1], size=(20000,12))
    signs_exact = 2*((np.arange(4096)[:,None] >> np.arange(12)) & 1)-1
    report, allrows, means, flags = {}, [], [], []
    total_requests = Counter()
    for b in BUNDLES:
        p = SOURCE/b
        for f, digest in remote['bundles'][b].items(): assert SHA(p/f) == digest, (b,f)
        reg, ev, analysis, gate = [json.loads((p/f).read_text()) for f in ['registry.json','R12_COMPLETE_EVIDENCE.json','R12_ANALYSIS001.json','FORMAL_AUDIT_PASS.json']]
        assert gate['all_checks_pass'] and not gate['problems']
        assert reg['n_formal'] == ev['n_runs'] == gate['n_runs'] == len(ev['runs'])
        assert ev['registry_sha256'] == SHA(p/'registry.json')
        assert analysis['evidence_sha256'] == SHA(p/'R12_COMPLETE_EVIDENCE.json')
        assert gate['bundle_sha256'] == SHA(p/'bundle_manifest.json')
        assert analysis['analysis_sha256'] == '74057a6a2b12b15ce9efb5f5e2a73f700ea8ad508f9c8775ea12f880fb597a0a'
        cells = defaultdict(dict)
        for run in ev['runs']:
            j = run['job']; k = (j['capacity'],j['replicate'])
            assert j['treatment'] not in cells[k]
            assert j['eager'] == reg['eager']
            assert reg['source']=='poisson' and j['length_mode']=='natural_eos'
            cells[k][j['treatment']] = run
            total_requests['submitted'] += run['metrics']['overall']['submitted']
            total_requests.update(run['metrics']['overall']['states'])
        E = {}
        for k, c in cells.items():
            assert set(c) == set(reg['arms'])
            assert len({x['trace_sha256'] for x in c.values()}) == 1
            assert len({x['gpu_uuid'] for x in c.values()}) == 1
            assert len({x['metrics']['overall']['submitted'] for x in c.values()}) == 1
            E[k] = {arm:endpoints(run) for arm,run in c.items()}
        for K in reg['capacities']:
            reps = sorted(k[1] for k in cells if k[0]==K)
            assert len(reps) == 12
            assert len({cells[K,i]['priority']['trace_sha256'] for i in reps}) == 12
            for arm in reg['arms']:
                cnt = 0
                for i in reps:
                    e = E[K,i][arm]
                    collapse = e['overall_noncompletion'] >= .05 or e['HIGH_gp_rps'] <= .85*E[K,i]['priority']['HIGH_gp_rps']
                    cnt += collapse
                    if collapse:
                        flags.append(dict(bundle=b,capacity=K,arm=arm,replicate=i,noncompletion=e['overall_noncompletion'],HIGH_ratio=e['HIGH_gp_rps']/E[K,i]['priority']['HIGH_gp_rps']))
                assert cnt == analysis['collapse'][str(K)][arm]
                means.append(dict(bundle=b,capacity=K,arm=arm,**{e:float(np.mean([E[K,i][arm][e] for i in reps])) for e in ['overall_gp_rps','HIGH_gp_rps','DEFAULT_gp_rps','LOW_gp_rps','overall_noncompletion','HIGH_noncompletion','LOW_noncompletion','preemptions','recomputed_tokens']},service_flags=cnt))
        fams = defaultdict(list)
        for row in analysis['contrasts']:
            K = row['capacity']; reps=sorted(k[1] for k in cells if k[0]==K)
            baseline=np.asarray([E[K,i][row['baseline']][row['endpoint']] for i in reps])
            treatment=np.asarray([E[K,i][row['treatment']][row['endpoint']] for i in reps])
            d = treatment-baseline; x=row['estimate']
            assert len(d) == x['n_pairs'] == 12
            for actual,expected in [(baseline.mean(),x['baseline_mean']),(treatment.mean(),x['treatment_mean']),(d.mean(),x['mean_difference']),(np.median(d),x['median_difference'])]:
                assert np.isclose(actual,expected,rtol=1e-10,atol=1e-8),(b,row['endpoint'],actual,expected)
            assert np.allclose(d,x['differences'],rtol=1e-10,atol=1e-8)
            ci=np.quantile(d[boot].mean(axis=1),[.025,.975])
            assert np.allclose(ci,x['paired_percentile_ci95'],rtol=1e-10,atol=1e-8),(b,row['endpoint'],ci,x['paired_percentile_ci95'])
            p_mc=(1+np.count_nonzero(np.abs((d*signs_mc).mean(axis=1))>=abs(d.mean())-1e-12))/20001
            assert p_mc == x['signflip_p'],(b,row['endpoint'],p_mc,x['signflip_p'])
            p_exact=float(np.mean(np.abs((d*signs_exact).mean(axis=1))>=abs(d.mean())-1e-12))
            out=dict(bundle=b,capacity=K,baseline=row['baseline'],treatment=row['treatment'],endpoint=row['endpoint'],family=row['family'],baseline_mean=float(baseline.mean()),treatment_mean=float(treatment.mean()),difference=float(d.mean()),relative_pct=float((d.mean()/baseline.mean())*100) if baseline.mean() else None,ci95=ci.tolist(),p_mc=p_mc,p_exact=p_exact,holm_p=row.get('holm_p'),positive_pairs=int(sum(d>0)),negative_pairs=int(sum(d<0)))
            allrows.append(out)
            if row['family']: fams[row['family']].append(out)
        exact_changes=[]
        for fam, rows in fams.items():
            for row, pmc, pex in zip(rows,holm([x['p_mc'] for x in rows]),holm([x['p_exact'] for x in rows])):
                assert pmc == row['holm_p']
                row['holm_exact']=pex
                if (pmc<.05)!=(pex<.05):exact_changes.append(row)
        report[b]=dict(runs=ev['n_runs'],paired_cells=len(cells),families={f:dict(tests=len(rs),MC_significant=sum(r['holm_p']<.05 for r in rs),exact_significant=sum(r['holm_exact']<.05 for r in rs)) for f,rs in fams.items()},all_contrasts_reproduced=len(analysis['contrasts']),exact_decision_changes=exact_changes,flags=analysis['collapse'])
        print(b,json.dumps(report[b]),flush=True)
    receipt=dict(source_root=str(SOURCE),remote_snapshot=remote['retrieved_utc'],status='PASS',verification='Run-level exported metrics, paired contrasts, 20,000-draw bootstrap and sign-flip, registered Holm families, exact 4,096-sign check, pair identity, source/registry/evidence hashes. Raw per-request and physical events audited by the server; not re-audited locally in this pass.',formal_runs=sum(r['runs'] for r in report.values()),paired_cells=sum(r['paired_cells'] for r in report.values()),contrasts=len(allrows),request_records=dict(total_requests),bundles=report)
    (HERE/'verification_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (HERE/'verified_contrasts.json').write_text(json.dumps(allrows,indent=2)+'\n')
    (HERE/'cell_arm_means.json').write_text(json.dumps(means,indent=2)+'\n')
    (HERE/'service_flags.json').write_text(json.dumps(flags,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='bundles'}),flush=True)

if __name__ == '__main__': main()
