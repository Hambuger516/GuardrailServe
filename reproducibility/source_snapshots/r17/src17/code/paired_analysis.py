"""Analyze an explicit preregistered pair list; never discover/select successful runs."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from runtime_smoke import sha


def paired_summary(baseline, treatment, seed=20260921, draws=10000):
    a, b = np.asarray(baseline, dtype=float), np.asarray(treatment, dtype=float)
    if len(a) != len(b) or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('at least two complete finite pairs required')
    d = b - a
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    # Paired sign-flip randomization, Monte Carlo plus-one correction.
    null = (d * rng.choice([-1, 1], size=(draws, len(d)))).mean(axis=1)
    p = (1 + np.count_nonzero(np.abs(null) >= abs(d.mean()) - 1e-12)) / (draws + 1)
    return {'n_pairs': len(d), 'baseline_mean': float(a.mean()), 'treatment_mean': float(b.mean()),
            'mean_difference': float(d.mean()), 'paired_percentile_ci95': [float(low), float(high)],
            'median_difference': float(np.median(d)), 'difference_iqr': [float(x) for x in np.quantile(d, [.25, .75])],
            'relative_difference_of_means': float(d.mean() / a.mean()) if a.mean() else None,
            'signflip_p': float(p), 'ci_interpretation': 'inconclusive' if low <= 0 <= high else ('positive' if low > 0 else 'negative'),
            'differences': d.tolist()}


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, previous = [None] * len(p_values), 0
    for rank, i in enumerate(order):
        previous = min(1, max(previous, (len(order) - rank) * p_values[i]))
        adjusted[i] = previous
    return adjusted


def read_formal_run(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('stage') != 'formal' or manifest.get('status') not in {'run_complete', 'request_errors'} or manifest.get('exclude_from_effect_estimate'):
        raise ValueError(f'not a valid formal run: {directory}')
    if sha(directory / 'metrics.json') != manifest['output_hashes']['metrics.json']:
        raise ValueError('metrics changed after run')
    metrics = json.loads((directory / 'metrics.json').read_text())
    trace = json.loads((directory / 'input_trace.json').read_text())
    if sha(directory / 'input_trace.json') != manifest['input_hashes']['trace']:
        raise ValueError('input trace changed')
    if metrics['overall']['submitted'] != len(trace['requests']):
        raise ValueError('submitted denominator does not match full trace')
    for name, digest in manifest['source_hashes'].items():
        if sha(directory / 'source' / name) != digest:
            raise ValueError('run source changed after freezing')
    if manifest['policy'] == 'PFS-port-warm':
        histogram = json.loads((directory / 'input_histogram.json').read_text())
        if sha(directory / 'input_histogram.json') != manifest['input_hashes']['histogram']:
            raise ValueError('historical predictor changed')
        gr = manifest['engine_config']['additional_config']['guardrail']
        if (gr['pfs_initial_history'] != histogram['pfs_initial_history']
                or gr['pfs_initial_history_provenance'] != histogram['pfs_initial_history_provenance']):
            raise ValueError('PFS warm history differs from common independent history')
    return manifest, metrics, trace


def comparison_identity(manifest):
    config = copy.deepcopy(manifest['engine_config'])
    gr = config['additional_config']['guardrail']
    policy = gr.pop('policy')
    if policy != manifest['policy']:
        raise ValueError('engine and reported policy differ')
    gr.pop('events_dir')
    scheduler = config.pop('scheduler_cls')
    if policy in {'C0', 'C1', 'C2', 'C3', 'C4', 'C5'}:
        if scheduler != 'runtime_scheduler.GuardrailScheduler':
            raise ValueError('unexpected Guardrail engine adapter')
    elif policy == 'PFS-port-warm':
        if scheduler != 'runtime_pfs.PFSControlledScheduler' or gr['pfs_seed'] != config['seed']:
            raise ValueError('unexpected PFS adapter/random seed')
        for key in ['pfs_seed', 'pfs_initial_length', 'pfs_initial_history', 'pfs_initial_history_provenance']:
            gr.pop(key)
    else:
        raise ValueError('unsupported formal treatment identity')
    # Preserve shared engine geometry, software and environment switches. Only
    # the declared scheduler implementation and its verified PFS initialization
    # differ. Dynamic pre-run occupancy is not part of hardware identity.
    gpu = [x.strip() for x in manifest['gpu_before']['stdout'].strip().split(',')]
    if len(gpu) != 6:
        raise ValueError('requires exactly one recorded GPU')
    return {'config': config, 'model': manifest['model_sha256'], 'source': manifest['source_hashes'],
            'versions': manifest['versions'], 'pool': manifest['input_hashes']['pool'],
            'histogram': manifest['input_hashes']['histogram'],
            'runtime_environment': manifest['runtime_environment'],
            'gpu_identity': gpu[:4]}


def analyze_cell(cell, root, registered_sources=None):
    if len(cell['pairs']) != cell['expected_pairs']:
        raise ValueError('pair count differs from preregistered sample size')
    a, b, trace_ids = [], [], set()
    common_cell, request_errors = None, {'baseline': 0, 'treatment': 0}
    for pair in cell['pairs']:
        ma, xa, trace_a = read_formal_run(root / pair['baseline'])
        mb, xb, trace_b = read_formal_run(root / pair['treatment'])
        if registered_sources is not None and (ma['source_hashes'] != registered_sources
                                               or mb['source_hashes'] != registered_sources):
            raise ValueError('run source differs from preregistered release')
        if ma['policy'] != cell['baseline_policy'] or mb['policy'] != cell['treatment_policy']:
            raise ValueError('policy label mismatch')
        if comparison_identity(ma) != comparison_identity(mb):
            raise ValueError('paired runs differ in more than declared treatment')
        condition = comparison_identity(ma)
        condition['config'].pop('seed')
        condition['arrival_condition'] = {k: trace_a[k] for k in ['arrival_window_s', 'target_rate_rps', 'class_probabilities']}
        if common_cell is None:
            common_cell = condition
        elif condition != common_cell:
            raise ValueError('different hardware/workloads/configurations mixed inside a cell')
        identity = ma['input_hashes']['trace']
        if identity != pair.get('trace_sha256'):
            raise ValueError('trace does not match preregistered pair identity')
        if identity != mb['input_hashes']['trace'] or identity in trace_ids:
            raise ValueError('unmatched trace or technical repeat counted as independent pair')
        trace_ids.add(identity)
        endpoint = cell.get('endpoint', 'window_slo_goodput_rps')
        group = cell.get('group', 'overall')
        ra = xa['overall'] if group == 'overall' else xa['per_class'][group]
        rb = xb['overall'] if group == 'overall' else xb['per_class'][group]
        a.append(ra[endpoint])
        b.append(rb[endpoint])
        request_errors['baseline'] += xa['overall']['states'].get('request_error', 0)
        request_errors['treatment'] += xb['overall']['states'].get('request_error', 0)
    return dict(cell_id=cell['cell_id'], endpoint=endpoint, group=group,
                request_errors_included=request_errors,
                confirmatory=cell.get('confirmatory', False), **paired_summary(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    registry = json.loads(Path(args.registry).read_text())
    results = [analyze_cell(cell, Path(args.root), registry.get('source_hashes')) for cell in registry['cells']]
    confirmatory = [r for r in results if r['confirmatory']]
    for row, p in zip(confirmatory, holm([r['signflip_p'] for r in confirmatory])):
        row['holm_p_in_registered_family'] = p
    report = {'registry_sha256': sha(args.registry), 'source_sha256': sha(__file__), 'cells': results,
              'unit': 'independent trace; paired resampling; 10000 draws',
              'limitation': '95% intervals are per-cell, not simultaneous; inferential claims use the registered family'}
    with Path(args.out).open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
