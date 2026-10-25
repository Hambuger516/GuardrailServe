"""AMENDMENT_R16_001 (2026-10-16 22:45 UTC; before any formal R16 outcome). The frozen D budget rule gave Friday budgets
(Qwen 2392/3237, Llama 3272/4429) close to R14's KV-bound regime, so D cannot reproduce the low-KV-pressure condition in
which R15-B Llama lost to Priority (K 5706/7724, held KV 44-61%). Diagnostic experiment F re-runs exactly that condition:
Llama, the R15-B Thursday evaluation windows and budgets (development data for this question, not unseen validation),
arms priority, hyst, hyst_nof2, hyst_f2p (4 Williams sequences x 3), 2 x 12 x 4 = 96 runs.
Family "f2diag" (Holm): hyst_nof2 - hyst, hyst_f2p - hyst, hyst_nof2 - priority, hyst_f2p - priority; overall and HIGH;
both budgets (16 tests). Usage: build_r16f.py"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r16_common import *
import build_r16 as B

AZT = X / 'r15_azure'
ARMS = ['priority', 'hyst', 'hyst_nof2', 'hyst_f2p']
FAM = {'f2diag': dict(pairs=[['hyst', 'hyst_nof2'], ['hyst', 'hyst_f2p'], ['priority', 'hyst_nof2'], ['priority', 'hyst_f2p']],
                      endpoints=B.TWO, kind='two_sided')}


def main():
    model = LL; s = SHORT[model]; out = X / f'r16_f_{s}'; slots = ['s1', 's2', 's3', 's4']
    caps = json.loads((X / 'R15_B_DECISION.json').read_text())['models'][model]['K']
    az = AZT / s
    sources = base_bundle(out, model, histogram=az / 'histogram.json',
                          pools={'calibration_pool.json': az / 'calibration_pool.json', 'evaluation_pool.json': az / 'evaluation_pool.json'})
    orders = B.williams(len(ARMS)); blocks = []
    for ki, K in enumerate(caps):
        for i in range(N_TRACES):
            tr = f'traces/az_eva{i:02d}.json'; (out / 'traces').mkdir(exist_ok=True)
            (out / tr).write_bytes((az / 'traces' / f'az_eva{i:02d}.json').read_bytes())
            t = json.loads((out / tr).read_text())
            blocks.append(dict(rate=t['target_rate_rps'], capacity=K, level=['tighter', 'wider'][ki], replicate=i,
                               seed=t['seed'] * 100 + i, trace=tr, slot=slots[(i + ki) % 4],
                               order=[ARMS[x] for x in orders[i % len(orders)]]))
    random.Random(20261017 + 99).shuffle(blocks)
    queues = {sl: [] for sl in slots}
    for b in blocks:
        for arm in b['order']:
            name = f"R16F_{s}_K{b['capacity']}_t{b['replicate']:02d}_{arm}"; cfg = f'configs/{name}.json'
            c = config(arm); c.update(stage='formal', seed=b['seed'], num_gpu_blocks_override=b['capacity'] + 1, event_mode='compact')
            write(out / cfg, c)
            queues[b['slot']].append(dict(name=name, group='R16_f', model=model, rate=b['rate'], capacity=b['capacity'],
                                          level=b['level'], seed=b['seed'], replicate=b['replicate'], treatment=arm, eager=True,
                                          length_mode='fixed_from_trace', config=cfg, trace=b['trace']))
    slot_plans(out, 'R16_f_formal', 'formal', slots, queues, 'inputs/evaluation_pool.json', timeout=1500, sources=sources)
    n = sum(map(len, queues.values()))
    write(out / 'registry.json', dict(group='R16_f', experiment='f', model=model, capacities=caps, arms=ARMS, slots=slots,
          n_formal=n, eager=True, source='az15_thursday', families=FAM, sched={a: SCHED[a] for a in ARMS}, williams=orders,
          window_s=120, source_hashes=sources, amendment='AMENDMENT_R16_001',
          blocks=sorted(blocks, key=lambda b: (b['capacity'], b['replicate']))))
    manifest(out)
    ref = X / f'r16_smoke_{s}'
    g = json.loads((ref / 'SMOKE_AUDIT_PASS.json').read_text())
    assert g['all_checks_pass'] and g['bundle_sha256'] == sha(ref / 'bundle_manifest.json')
    write(out / 'SMOKE_AUDIT_PASS.json', dict(all_checks_pass=all(sha(out / 'code' / f) == sha(ref / 'code' / f) for f in C.QUEUE_FILES),
          kind='r16_smoke_same_code', bundle_sha256=sha(out / 'bundle_manifest.json'), smoke_bundles=[str(ref)]))
    print(out, n, {sl: len(q) for sl, q in queues.items()})


if __name__ == '__main__':
    main()
