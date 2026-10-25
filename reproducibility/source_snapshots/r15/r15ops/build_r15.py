"""R15 bundles (frozen 2026-10-16 before any R15 outcome; PROTOCOL_R15.md). Usage: build_r15.py phase_a | formal
phase A: r15_smoke_<m>   60-s Dolly trace at the tighter budget: all six A arms (integrity, recovery-hook evidence)
         r15_csmoke_<m>  one 600-s Dolly protected run with the C geometry (600-s window, 600-s drain; integrity)
         r15_realcal_<m> Azure-2024 Thursday calibration windows under unconstrained Priority (K = 12,000) for the frozen
                         budget rule (decide_r15.py)
formal:  r15_a_<m>  six arms x 2 budgets x 12 fresh Poisson traces (288 runs over both models)
         r15_b_<m>  priority, protected, norec, hyst x 2 budgets x 12 unseen Azure windows (192)
         r15_c_<m>  priority, protected, tight budget, 12 fresh 600-s traces (48)
All arms of one trace run consecutively on one GPU in Williams order."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r15_common import *
from build_trace import make_trace

A_ARMS = ['protected', 'norec', 'mean', 'hard', 'nohighx', 'minru']
B_ARMS = ['priority', 'protected', 'norec', 'hyst']
C_ARMS = ['priority', 'protected']
TWO = lambda e: ['overall_gp_rps', 'HIGH_gp_rps']
EXPS = {'a': dict(arms=A_ARMS, slots=4, source='poisson', seed0=2200000, window=120, drain=120, budgets='both',
                  families={'nec': dict(pairs=[['protected', a] for a in A_ARMS[1:]], endpoints=TWO(0), kind='two_sided')}),
        'b': dict(arms=B_ARMS, slots=4, source='az', families={
                  'prod': dict(pairs=[['priority', 'norec'], ['protected', 'norec'], ['priority', 'hyst']], endpoints=TWO(0),
                               kind='two_sided'),
                  'ni': dict(pairs=[['priority', 'norec'], ['priority', 'hyst']], endpoints=['overall_gp_rps'], kind='ni',
                             margin=0.10)}),
        'c': dict(arms=C_ARMS, slots=4, source='poisson', seed0=2300000, window=600, drain=600, budgets='tight',
                  families={'steady': dict(pairs=[['priority', 'protected']],
                                           endpoints=['steady_overall_gp_rps', 'steady_HIGH_gp_rps'], kind='two_sided')})}
ORDER = ['a', 'b', 'c']
CAL_K = 12000
TIMEOUT = {'a': 1500, 'b': 1500, 'c': 2700}


def williams(n):
    first = [0]; lo, hi = 1, n - 1
    while len(first) < n:
        first.append(lo); lo += 1
        if len(first) < n:
            first.append(hi); hi -= 1
    return [[(x + r) % n for x in first] for r in range(n)]


def poisson_trace(out, name, pool_file, seed, window, rate=8):
    write(out / name, dict(schema='guardrail_openloop_v1', pool_sha256=sha(out / pool_file),
          pool_split=json.loads((out / pool_file).read_text())['split'], seed=seed, arrival_window_s=window,
          target_rate_rps=rate, arrival_process='poisson', class_probabilities=[.2, .7, .1],
          source_sha256=sha(OPS / 'build_r15.py'),
          requests=make_trace(json.loads((out / pool_file).read_text())['requests'], rate, window, seed)))


def smoke(model):
    s = SHORT[model]; out = X / f'r15_smoke_{s}'; sources = base_bundle(out, model)
    seed = 3999100 + (0 if model == QW else 1); K = K_SYN[model][0]
    poisson_trace(out, 'traces/smoke.json', 'inputs/calibration_pool.json', seed, 60)
    jobs = []
    for arm in A_ARMS:
        name = f'R15S_{s}_{arm}'; cfg = f'configs/{name}.json'
        c = config(arm); c.update(stage='pilot', seed=seed, num_gpu_blocks_override=K + 1, event_mode='full', drain_s=60)
        write(out / cfg, c)
        jobs.append(dict(name=name, group='R15_smoke', model=model, rate=8, capacity=K, seed=seed, replicate=0, treatment=arm,
                         eager=True, length_mode='natural_eos', config=cfg, trace='traces/smoke.json'))
    slot_plans(out, 'R15_smoke', 'smoke', ['s1'], {'s1': jobs}, 'inputs/calibration_pool.json', timeout=1200, hours=24, sources=sources)
    write(out / 'registry.json', dict(group='R15_smoke', model=model, n=len(jobs), sched=SCHED)); manifest(out); return out


def csmoke(model):
    s = SHORT[model]; out = X / f'r15_csmoke_{s}'; sources = base_bundle(out, model)
    seed = 3999200 + (0 if model == QW else 1); K = K_SYN[model][0]
    poisson_trace(out, 'traces/csmoke.json', 'inputs/calibration_pool.json', seed, 600)
    name = f'R15CS_{s}_protected'; cfg = f'configs/{name}.json'
    c = config('protected'); c.update(stage='pilot', seed=seed, num_gpu_blocks_override=K + 1, event_mode='compact', drain_s=600)
    write(out / cfg, c)
    jobs = [dict(name=name, group='R15_csmoke', model=model, rate=8, capacity=K, seed=seed, replicate=0, treatment='protected',
                 eager=True, length_mode='natural_eos', config=cfg, trace='traces/csmoke.json')]
    slot_plans(out, 'R15_csmoke', 'smoke', ['s1'], {'s1': jobs}, 'inputs/calibration_pool.json', timeout=TIMEOUT['c'], hours=24,
               sources=sources)
    write(out / 'registry.json', dict(group='R15_csmoke', model=model, n=1, sched=SCHED)); manifest(out); return out


def az_inputs(s):
    az = AZT / s
    return dict(histogram=az / 'histogram.json',
                pools={'calibration_pool.json': az / 'calibration_pool.json', 'evaluation_pool.json': az / 'evaluation_pool.json'})


def realcal(model):
    s = SHORT[model]; out = X / f'r15_realcal_{s}'; sources = base_bundle(out, model, **az_inputs(s))
    jobs = []
    for label in ('cal00', 'cal01'):
        tr = f'traces/az_{label}.json'; (out / 'traces').mkdir(exist_ok=True)
        (out / tr).write_bytes((AZT / s / 'traces' / f'az_{label}.json').read_bytes()); t = json.loads((out / tr).read_text())
        name = f'R15C_{s}_{label}_priority'; cfg = f'configs/{name}.json'
        c = config('priority'); c.update(stage='calibration', seed=t['seed'], num_gpu_blocks_override=CAL_K + 1, event_mode='compact')
        write(out / cfg, c)
        jobs.append(dict(name=name, group='R15_realcal', model=model, rate=t['target_rate_rps'], capacity=CAL_K, seed=t['seed'],
                         replicate=0, treatment='priority', eager=True, length_mode='fixed_from_trace', config=cfg, trace=tr))
    slot_plans(out, 'R15_realcal', 'smoke', ['s1'], {'s1': jobs}, 'inputs/calibration_pool.json', timeout=1500, hours=24, sources=sources)
    write(out / 'registry.json', dict(group='R15_realcal', model=model, n=len(jobs), cal_k=CAL_K, sched=SCHED)); manifest(out); return out


def formal(exp, model):
    E = EXPS[exp]; s = SHORT[model]; out = X / f'r15_{exp}_{s}'; arms = E['arms']; slots = [f's{i + 1}' for i in range(E['slots'])]
    if E['source'] == 'az':
        caps = json.loads((X / 'R15_B_DECISION.json').read_text())['models'][model]['K']
        sources = base_bundle(out, model, **az_inputs(s))
    else:
        caps = K_SYN[model] if E['budgets'] == 'both' else K_SYN[model][:1]
        sources = base_bundle(out, model)
    pool = json.loads((out / 'inputs/evaluation_pool.json').read_text()); assert pool['split'] == 'evaluation'
    orders = williams(len(arms))
    blocks = []
    for ki, K in enumerate(caps):
        for i in range(N_TRACES):
            if E['source'] == 'az':
                tr = f'traces/az_eva{i:02d}.json'; (out / 'traces').mkdir(exist_ok=True)
                (out / tr).write_bytes((AZT / s / 'traces' / f'az_eva{i:02d}.json').read_bytes())
                t = json.loads((out / tr).read_text()); seed, rate = t['seed'] * 100 + i, t['target_rate_rps']
            else:
                seed = E['seed0'] + (0 if model == QW else 1000) + 100 * ki + i; rate = 8
                tr = f'traces/{exp}_{s}_{seed}.json'
                poisson_trace(out, tr, 'inputs/evaluation_pool.json', seed, E['window'])
            seq = orders[i % len(orders)]
            blocks.append(dict(rate=rate, capacity=K, level=['tighter', 'wider'][ki], replicate=i, seed=seed, trace=tr,
                               slot=slots[(i + ki) % len(slots)], order=[arms[x] for x in seq]))
    random.Random(20261016 + ORDER.index(exp) * 10 + (0 if model == QW else 1)).shuffle(blocks)
    queues = {sl: [] for sl in slots}
    for b in blocks:
        for arm in b['order']:
            name = f"R15{exp.upper()}_{s}_K{b['capacity']}_t{b['replicate']:02d}_{arm}"; cfg = f'configs/{name}.json'
            c = config(arm); c.update(stage='formal', seed=b['seed'], num_gpu_blocks_override=b['capacity'] + 1, event_mode='compact')
            if E['source'] != 'az':
                c['drain_s'] = E['drain']
            write(out / cfg, c)
            queues[b['slot']].append(dict(name=name, group=f'R15_{exp}', model=model, rate=b['rate'], capacity=b['capacity'],
                                          level=b['level'], seed=b['seed'], replicate=b['replicate'], treatment=arm, eager=True,
                                          length_mode='fixed_from_trace' if E['source'] == 'az' else 'natural_eos',
                                          config=cfg, trace=b['trace']))
    slot_plans(out, f'R15_{exp}_formal', 'formal', slots, queues, 'inputs/evaluation_pool.json', timeout=TIMEOUT[exp], sources=sources)
    n = sum(map(len, queues.values()))
    write(out / 'registry.json', dict(group=f'R15_{exp}', experiment=exp, model=model, capacities=caps, arms=arms, slots=slots,
          n_formal=n, eager=True, source=E['source'], families=E['families'], sched={a: SCHED[a] for a in arms},
          williams=orders, window_s=E.get('window', 120), source_hashes=sources,
          blocks=sorted(blocks, key=lambda b: (b['capacity'], b['replicate']))))
    manifest(out)
    refs = [X / f'r15_smoke_{s}'] + ([X / f'r15_realcal_{s}'] if E['source'] == 'az' else []) + ([X / f'r15_csmoke_{s}'] if exp == 'c' else [])
    for ref in refs:
        g = json.loads((ref / 'SMOKE_AUDIT_PASS.json').read_text())
        assert g['all_checks_pass'] and g['bundle_sha256'] == sha(ref / 'bundle_manifest.json'), ref
    write(out / 'SMOKE_AUDIT_PASS.json', dict(all_checks_pass=all(sha(out / 'code' / f) == sha(r / 'code' / f) for r in refs for f in C.QUEUE_FILES),
          kind='r15_smoke_same_code', bundle_sha256=sha(out / 'bundle_manifest.json'), smoke_bundles=[str(r) for r in refs]))
    return out, n, {sl: len(q) for sl, q in queues.items()}


def main():
    if sys.argv[1] == 'phase_a':
        for m in (QW, LL): print(smoke(m), csmoke(m), realcal(m))
    else:
        for exp in ORDER:
            for m in (QW, LL): print(exp, m, *formal(exp, m)[1:])


if __name__ == '__main__':
    main()
