"""Reconstruct all reported XM/BASE/BURST effects from compact run counters.
No GPU, network, original event logs, or local absolute paths are needed.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np

HERE=Path(__file__).resolve().parent
def holm(ps):
    out=np.empty(len(ps));last=0
    for k,i in enumerate(np.argsort(ps)):
        last=max(last,(len(ps)-k)*ps[i]);out[i]=min(1,last)
    return out

def ep(run):
    out={}
    for name,g in [('overall',run['overall']),*run['per_class'].items()]:
        n=g['submitted'];assert sum(g['states'].values())==n
        assert g['window_slo_completions']<=g['slo_met']<=g['states'].get('completed',0)<=n
        assert np.isclose(g['window_slo_goodput_rps'],g['window_slo_completions']/120,atol=1e-12)
        out[name+'_gp_rps']=g['window_slo_goodput_rps']
        out[name+'_attainment']=g['slo_met_fraction']
        out[name+'_noncompletion']=1-g['states'].get('completed',0)/n if n else 0
    out['preemptions']=run['physical']['measured_preemptions']
    out['recomputed_tokens']=run['physical']['measured_recomputed_tokens']
    assert run['physical']['all_checks_pass']
    return out

rng=np.random.default_rng(20261012)
boot=rng.integers(0,12,size=(20000,12))
signs=rng.choice([-1,1],size=(20000,12))
exact=2*((np.arange(4096)[:,None]>>np.arange(12))&1)-1
report=[]
for name in ['xm_compact.json','extra_compact.json']:
    source=json.loads((HERE/name).read_text())
    for bundle,study in source.items():
        assert study['gate']['all_checks_pass']
        cells=defaultdict(dict)
        for run in study['runs']:
            j=run['job'];k=(j['capacity'],j['replicate'])
            assert j['treatment'] not in cells[k]
            cells[k][j['treatment']]=run
        for k,rs in cells.items():
            assert set(rs)==set(study['registry']['arms'])
            assert len({r['trace_sha256'] for r in rs.values()})==1
            assert len({r['gpu_uuid'] for r in rs.values()})==1
        E={k:{a:ep(r) for a,r in rs.items()} for k,rs in cells.items()}
        primary=[]
        for row in study['analysis']['contrasts']:
            keys=sorted(k for k in E if k[0]==row['capacity']);assert len(keys)==12
            x=np.array([E[k][row['baseline']][row['endpoint']] for k in keys])
            y=np.array([E[k][row['treatment']][row['endpoint']] for k in keys])
            d=y-x;est=row['estimate']
            assert np.allclose(d,est['differences'],rtol=1e-11,atol=1e-9)
            assert np.allclose([x.mean(),y.mean(),d.mean()],
                               [est['baseline_mean'],est['treatment_mean'],est['mean_difference']],atol=1e-10)
            ci=np.quantile(d[boot].mean(1),[.025,.975])
            assert np.allclose(ci,est['paired_percentile_ci95'],rtol=1e-11,atol=1e-9)
            p=(1+np.count_nonzero(abs((d*signs).mean(1))>=abs(d.mean())-1e-12))/20001
            assert p==est['signflip_p']
            px=float(np.mean(abs((d*exact).mean(1))>=abs(d.mean())-1e-12))
            if row['primary']:primary.append((p,px,row['holm_p_primary']))
        mc=holm([r[0] for r in primary]);ex=holm([r[1] for r in primary])
        assert np.allclose(mc,[r[2] for r in primary],rtol=0,atol=1e-12)
        assert np.all((mc<.05)==(ex<.05))
        report.append(dict(bundle=bundle,runs=len(study['runs']),matched_sets=len(cells),
                           contrasts=len(study['analysis']['contrasts']),primary_tests=len(primary),
                           exact_decisions_unchanged=True))
receipt=dict(status='PASS',runs=sum(r['runs'] for r in report),contrasts=sum(r['contrasts'] for r in report),
             primary_tests=sum(r['primary_tests'] for r in report),studies=report,
             scope='Compact run counters and registered statistics; not a repeat of raw request/event audits.')
assert receipt['runs']==720 and receipt['contrasts']==1008 and receipt['primary_tests']==56
(HERE/'verification_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
