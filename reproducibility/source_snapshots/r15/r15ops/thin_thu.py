"""R15 B subset (frozen rule, PROTOCOL_R15.md): block 2024-05-16 12:00-16:00 UTC of AzureLLMInferenceTrace_conv_1week.csv
(2024), Bernoulli-thinned in time order with p = 6*14400/rows_in_block, u ~ random.Random(int(sha256('20261016:r15:thin'))).
Usage: thin_thu.py SRC_CSV OUT_DIR"""
import datetime, hashlib, json, random, sys
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
SHA = 'a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941'
h = hashlib.sha256()
with src.open('rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''): h.update(chunk)
assert h.hexdigest() == SHA, 'source differs from the R14 file'
t0 = datetime.datetime(2024, 5, 16, 12, tzinfo=datetime.timezone.utc); t1 = t0 + datetime.timedelta(hours=4)
lo, hi = '2024-05-16 12:', '2024-05-16 16:'
rows = []
with src.open() as f:
    next(f)
    for line in f:
        if line < lo: continue
        if line >= hi: break
        ts, c, g = line.rstrip('\n').split(',')
        t = datetime.datetime.fromisoformat(ts)
        assert t0 <= t < t1
        rows.append(((t - t0).total_seconds(), int(c), int(g)))
assert all(b[0] >= a[0] for a, b in zip(rows, rows[1:]))
p = 6 * 14400 / len(rows)
rng = random.Random(int.from_bytes(hashlib.sha256(b'20261016:r15:thin').digest(), 'big'))
kept = [r for r in rows if rng.random() < p]
out.mkdir(parents=True, exist_ok=True)
csv = out / 'conv2024_thu12_16_thin6.csv'
csv.write_text('t_s,ContextTokens,GeneratedTokens\n' + ''.join(f'{t:.6f},{c},{g}\n' for t, c, g in kept))
meta = dict(source='AzureLLMInferenceTrace_conv_1week.csv (2024)', source_sha256=SHA, block='2024-05-16T12:00Z..16:00Z',
            rows_in_block=len(rows), block_rate_rps=len(rows) / 14400, thinning_p=p, seed='20261016:r15:thin',
            kept=len(kept), kept_rate_rps=len(kept) / 14400, rule=__doc__,
            subset_sha256=hashlib.sha256(csv.read_bytes()).hexdigest())
(out / 'conv2024_subset_meta.json').write_text(json.dumps(meta, indent=1))
print(json.dumps({k: v for k, v in meta.items() if k != 'rule'}, indent=1))
