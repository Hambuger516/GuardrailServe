"""Verify packaged inputs and exact tests. No GPU, network or NumPy required."""
from pathlib import Path
from collections import defaultdict
from itertools import product
import hashlib, json, math

HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
rows=json.loads((HERE/'evidence/r15_r17/registered_contrasts.json').read_text())
signs=list(product((-1,1),repeat=12)); families=defaultdict(list)
for row in rows:
    est=row['estimate'];d=est['differences'];assert len(d)==12
    observed=sum(d)/12
    assert math.isclose(observed,est.get('mean_difference',est.get('ni_statistic_mean')),abs_tol=1e-10)
    if row['family']=='ni':
        p=sum(sum(v*s for v,s in zip(d,ss))/12>=observed-1e-12 for ss in signs)/4096
    else:
        p=sum(abs(sum(v*s for v,s in zip(d,ss))/12)>=abs(observed)-1e-12 for ss in signs)/4096
    assert p==est['exact_p'],(row['bundle'],row['endpoint'],p,est['exact_p'])
    families[row['bundle'],row['family']].append((p,row['holm_p']))
for key,values in families.items():
    last=0
    for rank,(p,expected) in enumerate(sorted(values)):
        last=min(1,max(last,(len(values)-rank)*p))
        assert math.isclose(last,expected,abs_tol=1e-12),(key,last,expected)
print(json.dumps({'status':'PASS','integrity_files':len(manifest['files']),'registered_exact_tests':len(rows),'Holm_families':len(families),'bootstrap_rerun_by_this_script':False},indent=2))
