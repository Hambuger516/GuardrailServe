"""R16 D workload: Azure LLM inference trace 2024 (conversation, 1 week), unseen day. Frozen 2026-10-16 before any R16
outcome (PROTOCOL_R16.md). Subset: block 2024-05-17 12:00-16:00 UTC, seeded Bernoulli thinning to 6 req/s
(thin_fri.py; metadata in conv2024_subset_meta.json). Then exactly the R14 rules (prep_azure24): time split history
[0, 20%) / calibration [20%, 30%) (first two 120-s windows) / evaluation [30%, 100%] (12 disjoint 120-s windows drawn with
seed 20261017); context clipped to [32, 3072], output to [1, 1000]; recorded output length enforced; prompt text from the
model's own pools; 20/70/10 class stream. Clipping fractions are reported per split (AZURE16_PREP.json)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, '/data/hanbo_futureserve/tpds/r12ops')
from r16_common import *
import prep_azure as P
P.SEED = 20261017


def clip_stats(rows):
    n = len(rows) or 1
    return dict(n=len(rows), ctx_below=sum(r['ctx'] < P.CTX[0] for r in rows) / n, ctx_above=sum(r['ctx'] > P.CTX[1] for r in rows) / n,
                gen_above=sum(r['gen'] > P.GEN[1] for r in rows) / n,
                ctx_mean=sum(r['ctx'] for r in rows) / n, gen_mean=sum(r['gen'] for r in rows) / n)


def main():
    src = AZF / 'conv2024_fri12_16_thin6.csv'; meta = json.loads((AZF / 'conv2024_subset_meta.json').read_text())
    assert sha(src) == meta['subset_sha256']
    rows = []
    for line in src.read_text().splitlines()[1:]:
        t, c, g = line.split(','); rows.append(dict(t=float(t), ctx=int(c), gen=int(g)))
    T = 4 * 3600; h_end, c_end = 0.2 * T, 0.3 * T
    gens = [min(max(r['gen'], P.GEN[0]), P.GEN[1]) for r in rows if r['t'] < h_end]
    counts = {}
    for g in gens: counts[g] = counts.get(g, 0) + 1
    histogram = dict(kind='trace_output_lengths', source=meta['source'], source_sha256=meta['source_sha256'],
                     subset_sha256=sha(src), segment=[0, h_end], dtype='float16',
                     sampling=dict(temperature=0, max_tokens=2048, ignore_eos=True), training_prompt_ids=[],
                     samples=len(gens), counts={str(k): v for k, v in sorted(counts.items())},
                     probabilities={str(k): v / len(gens) for k, v in sorted(counts.items())},
                     pfs_initial_history=gens[-40:],
                     pfs_initial_history_provenance=dict(order='last40_in_trace_time_order', segment='history',
                                                         historical_samples=len(gens), source_sha256=sha(src)))
    cal_w = P.windows(rows, h_end, c_end)[:2]
    ev_all = P.windows(rows, c_end, T)
    ev_w = sorted(P.rng('eval_windows').sample(ev_all, N_TRACES))
    sel = lambda ws: [r for r in rows if any(w <= r['t'] < w + P.WINDOW for w in ws)]
    clipping = dict(history=clip_stats([r for r in rows if r['t'] < h_end]), calibration=clip_stats(sel(cal_w)),
                    evaluation=clip_stats(sel(ev_w)), whole_block=clip_stats(rows))
    for model in (QW, LL):
        s = SHORT[model]; d = AZF / s
        cal_src = json.loads(C.model_inputs(model)[0].read_text()); ev_src = json.loads(C.model_inputs(model)[1].read_text())
        pools = {'calibration': [], 'evaluation': []}; traces = {}
        for split, ws, srcpool in (('calibration', cal_w, cal_src), ('evaluation', ev_w, ev_src)):
            for j, w0 in enumerate(ws):
                label = f'{split[:3]}{j:02d}'
                reqs = P.window_requests(rows, w0, label)
                pools[split] += P.build_prompts(reqs, srcpool, label)
                traces[label] = (split, w0, reqs)
        for split in pools:
            write(d / f'{split}_pool.json', dict(split=split, requests=pools[split], source='azure_conv2024_fri_length_replay'))
        write(d / 'histogram.json', histogram)
        for label, (split, w0, reqs) in traces.items():
            write(d / 'traces' / f'az_{label}.json', dict(
                schema='guardrail_openloop_v1', pool_sha256=sha(d / f'{split}_pool.json'), pool_split=split,
                seed=P.SEED, window_label=label, window_start_s=w0, arrival_window_s=P.WINDOW,
                target_rate_rps=round(len(reqs) / P.WINDOW, 4), length_mode='fixed_from_trace',
                class_probabilities=[.2, .7, .1], arrival_process='recorded_azure_conv_2024_fri_thinned',
                source_sha256=sha(src), requests=[{k: v for k, v in r.items() if k != 'context_tokens'} for r in reqs]))
        print(model, {k: len(v[2]) for k, v in traces.items()})
    write(AZF / 'AZURE16_PREP.json', dict(subset=meta, subset_sha256=sha(src), calibration_windows=cal_w,
          evaluation_windows_available=len(ev_all), evaluation_windows=ev_w, clipping=clipping, rule=__doc__))
    print(json.dumps(clipping, indent=1))


if __name__ == '__main__':
    main()
