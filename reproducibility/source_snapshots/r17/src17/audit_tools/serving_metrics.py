"""Client-observed cohort metrics with explicit unfinished-request denominators."""
from collections import Counter
import math


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    def at(p):
        if not values:
            return None
        x = (len(values) - 1) * p
        lo = int(x)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return {'n': len(values), 'p50': at(.5), 'p95': at(.95), 'p99': at(.99)}


def summarize(rows, arrival_window_s, ttft_ms=(500, 2000, 10000), tpot_ms=(50, 100, 200)):
    if arrival_window_s <= 0 or not math.isfinite(arrival_window_s):
        raise ValueError('invalid observation window')
    records = []
    for r in rows:
        first, last, end = r.get('first_token_s'), r.get('last_token_s'), r.get('finish_s')
        ttft = None if first is None else 1000 * (first - r['planned_arrival_s'])
        # Completion notification is separate from the final token timestamp.
        tpot = None if r['output_tokens'] <= 1 or r['status'] != 'completed' else 1000 * (last - first) / (r['output_tokens'] - 1)
        met = (r['status'] == 'completed' and ttft is not None and ttft <= ttft_ms[r['priority']]
               and (r['output_tokens'] == 1 or tpot <= tpot_ms[r['priority']]))
        records.append(dict(r, ttft_ms=ttft, tpot_ms=tpot, slo_met=met,
                            completed_in_window=r['status'] == 'completed' and end <= arrival_window_s))
    def group(selected):
        n = len(selected)
        met = sum(r['slo_met'] for r in selected)
        in_window = sum(r['slo_met'] and r['completed_in_window'] for r in selected)
        return {'submitted': n, 'states': dict(Counter(r['status'] for r in selected)),
                'slo_met': met, 'slo_met_fraction': met / n if n else None,
                'window_slo_completions': in_window, 'window_slo_goodput_rps': in_window / arrival_window_s,
                'cohort_slo_goodput_per_arrival_second': met / arrival_window_s,
                'window_completion_throughput_rps': sum(r['completed_in_window'] for r in selected) / arrival_window_s,
                'ttft_ms_received': quantiles([r['ttft_ms'] for r in selected]),
                'tpot_ms_completed_multi_token': quantiles([r['tpot_ms'] for r in selected]),
                'submission_lag_ms': quantiles([1000 * (r['actual_submit_s'] - r['planned_arrival_s'])
                                              for r in selected if r.get('actual_submit_s') is not None])}
    return {'primary_endpoint': 'window_slo_goodput_rps', 'arrival_window_s': arrival_window_s,
            'overall': group(records), 'per_class': {name: group([r for r in records if r['priority'] == i])
                                                    for i, name in enumerate(['HIGH', 'DEFAULT', 'LOW'])},
            'requests': records,
            'definitions': {'ttft': 'first client-observed token minus planned open-loop arrival',
                            'tpot': '(last observed token-first observed token)/(output tokens-1); completed multi-token only',
                            'latency_selection': 'TTFT includes incomplete requests with a first token; TPOT is conditional on completion',
                            'cohort': 'all trace requests remain in denominator; drain-window completions included only in cohort metrics'}}
