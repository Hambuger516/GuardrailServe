"""Check physical event accounting for open-loop serving runs, including compact logs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from artifact_files import open_text
from runtime_smoke import sha


def audit(directory):
    directory = Path(directory)
    m = json.loads((directory / 'manifest.json').read_text())
    metrics = json.loads((directory / 'metrics.json').read_text())
    with open_text(directory, 'scheduler_events.jsonl') as stream:
        events = [json.loads(line) for line in stream]
    capacity = next(e['capacity_blocks'] for e in events if e['event'] == 'controller_init')
    assert all(a['monotonic_s'] <= b['monotonic_s'] for a, b in zip(events, events[1:]))
    assert all(0 <= e['free_blocks'] <= capacity for e in events if 'free_blocks' in e)
    preemptions = [e for e in events if e['event'] == 'preempt']
    assert all(e['blocks_released'] > 0 for e in preemptions)
    assert all(e.get('exact_output_prefix_preserved') is True for e in preemptions)
    invalidated = sum(e['invalidated_kv_tokens'] for e in preemptions)
    recomputed = sum(e['tokens'] for e in events if e['event'] == 'recompute_scheduled')
    assert 0 <= recomputed <= invalidated
    schedules = [e for e in events if e['event'] == 'schedule']
    counter = schedules[-1].get('cumulative_admission_counts')
    if counter:
        assert all(c['checked'] == c['allowed'] + c['deferred'] for c in counter.values())
        if m['engine_config']['additional_config']['guardrail'].get('event_mode', 'full') == 'full':
            for phase, c in counter.items():
                checks = [e for e in events if e['event'] == 'admission_check'
                          and ('warmup' if e['request_id'].startswith('warmup-') else 'measurement') == phase]
                assert c['checked'] == len(checks)
                assert c['allowed'] == sum(e['allow'] for e in checks)
    completed = metrics['overall']['states'].get('completed', 0)
    final = [e for e in events if 'free_blocks' in e][-1]
    release_verified = final['free_blocks'] == capacity and final.get('running') == 0
    if completed == metrics['overall']['submitted']:
        assert release_verified
    measured = [e for e in events if e['monotonic_s'] >= m['measurement_start_monotonic_s']]
    return {'kind': 'serving_scheduler_physical_accounting', 'all_checks_pass': True,
            'policy': m['policy'], 'capacity_blocks': capacity,
            'measured_events': dict(Counter(e['event'] for e in measured)),
            'measured_preemptions': sum(e['event'] == 'preempt' for e in measured),
            'measured_recomputed_tokens': sum(e['tokens'] for e in measured if e['event'] == 'recompute_scheduled'),
            'final_release_verified_in_trace': release_verified, 'cumulative_admission_counts': counter,
            'all_preemptions_exact_prefix_checked': len(preemptions),
            'manifest_sha256': sha(directory / 'manifest.json'), 'auditor_sha256': sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    result = audit(args.directory)
    (Path(args.directory) / 'scheduler_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
