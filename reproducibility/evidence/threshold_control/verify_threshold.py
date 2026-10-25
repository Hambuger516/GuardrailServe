from pathlib import Path
from collections import defaultdict
import hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parent
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def holm(ps):
    out = [None] * len(ps); last = 0.
    for rank, i in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        last = min(1., max(last, (len(ps)-rank)*ps[i])); out[i] = last
    return out

def endpoints(r):
    out = {}
    for name, g in [('overall',r['metrics']['overall']), *r['metrics']['per_class'].items()]:
        n = g['submitted']; completed = g['states'].get('completed',0)
        assert sum(g['states'].values()) == n
        assert 0 <= g['window_slo_completions'] <= g['slo_met'] <= completed <= n
        out[name+'_gp_rps'] = g['window_slo_completions']/r['metrics']['arrival_window_s']
        out[name+'_attainment'] = g['slo_met']/n if n else 0.
        out[name+'_noncompletion'] = 1-completed/n if n else 0.
        for lat in ['ttft_ms_received','tpot_ms_completed_multi_token']:
            if g[lat]['p99'] is not None:
                out[name+'_'+lat.split('_')[0]+'_p99_ms'] = g[lat]['p99']
        assert np.isclose(out[name+'_gp_rps'],g['window_slo_goodput_rps'])
    out['preemptions'] = r['physical']['measured_preemptions']
    out['recomputed_tokens'] = r['physical']['measured_recomputed_tokens']
    assert r['physical']['all_checks_pass']
    return out

def main():
    rng = np.random.default_rng(20261013)
    boot = rng.integers(0,12,size=(20000,12))
    signs_mc = rng.choice([-1,1],size=(20000,12))
    signs = 2*((np.arange(4096)[:,None]>>np.arange(12))&1)-1
    reports, contrasts, means = {}, [], []
    for bundle in ['syn_qw7','syn_ll8']:
        p=SOURCE/bundle
        reg,ev,ana,gate=[json.loads((p/f).read_text()) for f in ['registry.json','R12_COMPLETE_EVIDENCE.json','R12_ANALYSIS001.json','FORMAL_AUDIT_PASS.json']]
        assert gate['all_checks_pass'] and not gate['problems']
        assert gate['n_runs']==reg['n_formal']==ev['n_runs']==len(ev['runs'])
        assert ev['registry_sha256']==SHA(p/'registry.json')
        assert ana['evidence_sha256']==SHA(p/'R12_COMPLETE_EVIDENCE.json')
        assert gate['bundle_sha256']==SHA(p/'bundle_manifest.json')
        cells=defaultdict(dict)
        for r in ev['runs']:
            j=r['job']; key=(j['capacity'],j['replicate'])
            assert j['treatment'] not in cells[key]
            cells[key][j['treatment']]=r
        E={}
        for key,c in cells.items():
            assert set(c)==set(reg['arms'])
            assert len({r['trace_sha256'] for r in c.values()})==1
            assert len({r['gpu_uuid'] for r in c.values()})==1
            E[key]={a:endpoints(r) for a,r in c.items()}
        for K in reg['capacities']:
            reps=sorted(k[1] for k in cells if k[0]==K)
            assert len(reps)==12
            for a in reg['arms']:
                ms={ep:float(np.mean([E[K,i][a][ep] for i in reps])) for ep in ['overall_gp_rps','HIGH_gp_rps','LOW_gp_rps','overall_noncompletion','recomputed_tokens','preemptions']}
                means.append(dict(bundle=bundle,capacity=K,arm=a,**ms))
        families=defaultdict(list)
        for r in ana['contrasts']:
            K=r['capacity']; reps=sorted(k[1] for k in cells if k[0]==K)
            b=np.array([E[K,i][r['baseline']][r['endpoint']] for i in reps])
            t=np.array([E[K,i][r['treatment']][r['endpoint']] for i in reps]);d=t-b;x=r['estimate']
            assert np.allclose(d,x['differences'],atol=1e-8)
            assert np.isclose(b.mean(),x['baseline_mean']) and np.isclose(t.mean(),x['treatment_mean'])
            ci=np.quantile(d[boot].mean(axis=1),[.025,.975])
            assert np.allclose(ci,x['paired_percentile_ci95'],atol=1e-8),(bundle,r['endpoint'])
            mc=(1+np.count_nonzero(abs((d*signs_mc).mean(axis=1))>=abs(d.mean())-1e-12))/20001
            assert mc==x['signflip_p']
            exact=float(np.mean(abs((d*signs).mean(axis=1))>=abs(d.mean())-1e-12))
            row=dict(bundle=bundle,capacity=K,baseline=r['baseline'],treatment=r['treatment'],endpoint=r['endpoint'],family=r['family'],baseline_mean=float(b.mean()),treatment_mean=float(t.mean()),difference=float(d.mean()),relative_pct=float(100*d.mean()/b.mean()) if b.mean() else None,ci95=ci.tolist(),p_mc=mc,p_exact=exact,holm_p=r.get('holm_p'))
            contrasts.append(row)
            if row['family']: families[row['family']].append(row)
        changes=[]
        for f,rows in families.items():
            assert len(rows)==ana['families'][f]
            for r,pmc,pex in zip(rows,holm([r['p_mc'] for r in rows]),holm([r['p_exact'] for r in rows])):
                assert np.isclose(pmc,r['holm_p']);r['holm_exact']=pex
                if (pmc<.05)!=(pex<.05):changes.append({k:r[k] for k in ['capacity','baseline','treatment','endpoint','holm_p','holm_exact']})
        reports[bundle]={'runs':len(ev['runs']),'contrasts':len(ana['contrasts']),'families':ana['families'],'exact_decision_changes':changes,'source_sha256':{f:SHA(p/f) for f in ['registry.json','R12_COMPLETE_EVIDENCE.json','R12_ANALYSIS001.json','FORMAL_AUDIT_PASS.json','bundle_manifest.json']}}
    receipt={'status':'PASS','new_GPU_runs':0,'scope':'Independent reconstruction of existing R13 exported run counters, paired bootstrap/sign tests and registered families; no causal or non-inferiority claim inferred.','runs':sum(r['runs'] for r in reports.values()),'contrasts':len(contrasts),'bundles':reports}
    for name,data in [('r13_receipt.json',receipt),('r13_contrasts.json',contrasts),('r13_means.json',means)]:
        assert name != 'r13_receipt.json' or data['runs'] == 192
        if name == 'r13_contrasts.json':
            expected=json.loads((HERE/'contrasts.json').read_text())
            assert data == expected
        if name == 'r13_means.json':
            expected=json.loads((HERE/'means.json').read_text())
            assert data == expected
    print(json.dumps({k:v for k,v in receipt.items() if k!='bundles'}))
    print('Exact decision changes:',{n:r['exact_decision_changes'] for n,r in reports.items()})

if __name__=='__main__': main()
