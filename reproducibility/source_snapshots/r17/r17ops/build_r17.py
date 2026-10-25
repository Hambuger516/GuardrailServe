"""R17 bundles (frozen 2026-10-17 before any R17 outcome; PROTOCOL_R17.md). Usage: build_r17.py phase_a | formal
phase A: r17_smoke_<m> (protected, final on a 60-s Dolly trace) and r17_realcal_<m> (Azure-2024 Saturday calibration,
unconstrained Priority) for the frozen budget rule (decide_r17.py).
formal:  r17_g_<m>  Azure Saturday: priority, protected, final x 2 budgets x 12 unseen windows (Williams 6 x 2)
         r17_h_<m>  Dolly Poisson 8 req/s, fresh seeds 2,600,000+: priority, protected, final x 2 budgets x 12."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r17_common import *
from build_trace import make_trace

SMOKE_ARMS = ['protected', 'final']
TWO = ['overall_gp_rps', 'HIGH_gp_rps']
ARMS = ['priority', 'protected', 'final']
EXPS = {'g': dict(arms=ARMS, slots=4, source='az', families={
            'final': dict(pairs=[['priority', 'final']], endpoints=TWO, kind='two_sided'),
            'ni': dict(pairs=[['priority', 'final']], endpoints=['overall_gp_rps'], kind='ni', margin=0.10)}),
        'h': dict(arms=ARMS, slots=4, source='poisson', seed0=2600000, window=120, drain=120, budgets='both', families={
            'final': dict(pairs=[['protected', 'final']], endpoints=TWO, kind='two_sided'),
            'ni': dict(pairs=[['protected', 'final']], endpoints=['overall_gp_rps'], kind='ni', margin=0.05)})}
ORDER = ['g', 'h']
CAL_K = 12000
TIMEOUT = {'g': 1500, 'h': 1500}


def williams(n):
    first = [0]; lo, hi = 1, n - 1
    while len(first) < n:
        first.append(lo); lo += 1
        if len(first) < n:
            first.append(hi); hi -= 1
    rows = [[(x + r) % n for x in first] for r in range(n)]
    return rows if n % 2 == 0 else rows + [list(reversed(r)) for r in rows]     # odd n: 2n sequences


def poisson_trace(out, name, pool_file, seed, window, rate=8):
    pool = json.loads((out / pool_file).read_text())
    write(out / name, dict(schema='guardrail_openloop_v1', pool_sha256=sha(out / pool_file), pool_split=pool['split'], seed=seed,
          arrival_window_s=window, target_rate_rps=rate, arrival_process='poisson', class_probabilities=[.2, .7, .1],
          source_sha256=sha(OPS / 'build_r17.py'), requests=make_trace(pool['requests'], rate, window, seed)))


def smoke(model):
    s = SHORT[model]; out = X / f'r17_smoke_{s}'; sources = base_bundle(out, model)
    seed = 3999400 + (0 if model == QW else 1); K = K_SYN[model][0]
    poisson_trace(out, 'traces/smoke.json', 'inputs/calibration_pool.json', seed, 60)
    jobs = []
    for arm in SMOKE_ARMS:
        name = f'R17S_{s}_{arm}'; cfg = f'configs/{name}.json'
        c = config(arm); c.update(stage='pilot', seed=seed, num_gpu_blocks_override=K + 1, event_mode='full', drain_s=60)
        write(out / cfg, c)
        jobs.append(dict(name=name, group='R17_smoke', model=model, rate=8, capacity=K, seed=seed, replicate=0, treatment=arm,
                         eager=True, length_mode='natural_eos', config=cfg, trace='traces/smoke.json'))
    slot_plans(out, 'R17_smoke', 'smoke', ['s1'], {'s1': jobs}, 'inputs/calibration_pool.json', timeout=1200, hours=24, sources=sources)
    write(out / 'registry.json', dict(group='R17_smoke', model=model, n=len(jobs), sched=SCHED)); manifest(out); return out


def az_inputs(s):
    az = AZS / s
    return dict(histogram=az / 'histogram.json',
                pools={'calibration_pool.json': az / 'calibration_pool.json', 'evaluation_pool.json': az / 'evaluation_pool.json'})


def realcal(model):
    s = SHORT[model]; out = X / f'r17_realcal_{s}'; sources = base_bundle(out, model, **az_inputs(s))
    jobs = []
    for label in ('cal00', 'cal01'):
        tr = f'traces/az_{label}.json'; (out / 'traces').mkdir(exist_ok=True)
        (out / tr).write_bytes((AZS / s / 'traces' / f'az_{label}.json').read_bytes()); t = json.loads((out / tr).read_text())
        name = f'R17C_{s}_{label}_priority'; cfg = f'configs/{name}.json'
        c = config('priority'); c.update(stage='calibration', seed=t['seed'], num_gpu_blocks_override=CAL_K + 1, event_mode='compact')
        write(out / cfg, c)
        jobs.append(dict(name=name, group='R17_realcal', model=model, rate=t['target_rate_rps'], capacity=CAL_K, seed=t['seed'],
                         replicate=0, treatment='priority', eager=True, length_mode='fixed_from_trace', config=cfg, trace=tr))
    slot_plans(out, 'R17_realcal', 'smoke', ['s1'], {'s1': jobs}, 'inputs/calibration_pool.json', timeout=1500, hours=24, sources=sources)
    write(out / 'registry.json', dict(group='R17_realcal', model=model, n=len(jobs), cal_k=CAL_K, sched=SCHED)); manifest(out); return out


def formal(exp, model):
    E = EXPS[exp]; s = SHORT[model]; out = X / f'r17_{exp}_{s}'; arms = E['arms']; slots = [f's{i + 1}' for i in range(E['slots'])]
    if E['source'] == 'az':
        caps = json.loads((X / 'R17_G_DECISION.json').read_text())['models'][model]['K']
        sources = base_bundle(out, model, **az_inputs(s))
    else:
        caps = {'both': K_SYN[model], 'tight': K_SYN[model][:1], 'wide': K_SYN[model][1:]}[E['budgets']]
        sources = base_bundle(out, model)
    pool = json.loads((out / 'inputs/evaluation_pool.json').read_text()); assert pool['split'] == 'evaluation'
    orders = williams(len(arms)); assert N_TRACES % len(orders) == 0
    blocks = []
    for ki, K in enumerate(caps):
        for i in range(N_TRACES):
            if E['source'] == 'az':
                tr = f'traces/az_eva{i:02d}.json'; (out / 'traces').mkdir(exist_ok=True)
                (out / tr).write_bytes((AZS / s / 'traces' / f'az_eva{i:02d}.json').read_bytes())
                t = json.loads((out / tr).read_text()); seed, rate = t['seed'] * 100 + i, t['target_rate_rps']
            else:
                seed = E['seed0'] + (0 if model == QW else 1000) + 100 * ki + i; rate = 8
                tr = f'traces/{exp}_{s}_{seed}.json'
                poisson_trace(out, tr, 'inputs/evaluation_pool.json', seed, E['window'])
            blocks.append(dict(rate=rate, capacity=K, level=(['tighter', 'wider'][ki] if len(caps) == 2 else E['budgets']),
                               replicate=i, seed=seed, trace=tr, slot=slots[(i + ki) % len(slots)],
                               order=[arms[x] for x in orders[i % len(orders)]]))
    random.Random(20261018 + ORDER.index(exp) * 10 + (0 if model == QW else 1)).shuffle(blocks)
    queues = {sl: [] for sl in slots}
    for b in blocks:
        for arm in b['order']:
            name = f"R17{exp.upper()}_{s}_K{b['capacity']}_t{b['replicate']:02d}_{arm}"; cfg = f'configs/{name}.json'
            c = config(arm); c.update(stage='formal', seed=b['seed'], num_gpu_blocks_override=b['capacity'] + 1, event_mode='compact')
            if E['source'] != 'az':
                c['drain_s'] = E['drain']
            write(out / cfg, c)
            queues[b['slot']].append(dict(name=name, group=f'R17_{exp}', model=model, rate=b['rate'], capacity=b['capacity'],
                                          level=b['level'], seed=b['seed'], replicate=b['replicate'], treatment=arm, eager=True,
                                          length_mode='fixed_from_trace' if E['source'] == 'az' else 'natural_eos',
                                          config=cfg, trace=b['trace']))
    slot_plans(out, f'R17_{exp}_formal', 'formal', slots, queues, 'inputs/evaluation_pool.json', timeout=TIMEOUT[exp], sources=sources)
    n = sum(map(len, queues.values()))
    write(out / 'registry.json', dict(group=f'R17_{exp}', experiment=exp, model=model, capacities=caps, arms=arms, slots=slots,
          n_formal=n, eager=True, source=E['source'], families=E['families'], sched={a: SCHED[a] for a in arms},
          williams=orders, window_s=E.get('window', 120), source_hashes=sources,
          blocks=sorted(blocks, key=lambda b: (b['capacity'], b['replicate']))))
    manifest(out)
    refs = [X / f'r17_smoke_{s}'] + ([X / f'r17_realcal_{s}'] if E['source'] == 'az' else [])
    for ref in refs:
        g = json.loads((ref / 'SMOKE_AUDIT_PASS.json').read_text())
        assert g['all_checks_pass'] and g['bundle_sha256'] == sha(ref / 'bundle_manifest.json'), ref
    write(out / 'SMOKE_AUDIT_PASS.json', dict(all_checks_pass=all(sha(out / 'code' / f) == sha(r / 'code' / f) for r in refs for f in C.QUEUE_FILES),
          kind='r17_smoke_same_code', bundle_sha256=sha(out / 'bundle_manifest.json'), smoke_bundles=[str(r) for r in refs]))
    return out, n, {sl: len(q) for sl, q in queues.items()}


def main():
    if sys.argv[1] == 'phase_a':
        for m in (QW, LL): print(smoke(m), realcal(m))
    else:
        for exp in ORDER:
            for m in (QW, LL): print(exp, m, *formal(exp, m)[1:])


if __name__ == '__main__':
    main()
