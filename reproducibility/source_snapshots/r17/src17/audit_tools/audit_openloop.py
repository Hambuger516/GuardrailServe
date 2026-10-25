"""Reconstruct measurements from raw token chunks; no effect-size claims."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from runtime_smoke import sha
from serving_metrics import summarize


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    trace = json.loads((directory / 'input_trace.json').read_text())
    config = json.loads((directory / 'input_config.json').read_text())
    planned = {r['request_id']: r for r in trace['requests']}
    states = {rid: {'tokens': [], 'times': [], 'submitted': 0, 'terminal': None} for rid in planned}
    previous = -1
    end_count = 0
    with gzip.open(directory / 'client_events.jsonl.gz', 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            assert event['elapsed_s'] >= previous
            previous = event['elapsed_s']
            if event['event'] == 'measurement_end':
                end_count += 1
                continue
            rid = event['request_id']
            state = states[rid]
            assert state['terminal'] is None
            if event['event'] == 'submit':
                state['submitted'] += 1
                assert state['submitted'] == 1
                assert event['elapsed_s'] >= planned[rid]['arrival_s']
            elif event['event'] == 'token_chunk':
                assert state['submitted'] == 1 and event['token_ids']
                state['tokens'].extend(event['token_ids'])
                state['times'].append(event['elapsed_s'])
                assert len(state['tokens']) == event['cumulative_tokens']
            elif event['event'] == 'request_terminal':
                digest = hashlib.sha256()
                for token in state['tokens']:
                    digest.update(int(token).to_bytes(4, 'big'))
                assert digest.hexdigest() == event['output_sha256']
                assert len(state['tokens']) == event['output_tokens'] <= config['max_output_tokens']
                assert event['planned_arrival_s'] == planned[rid]['arrival_s']
                assert event['priority'] == planned[rid]['priority']
                if state['times']:
                    # emit() timestamps just after the timestamp stored in the record.
                    assert 0 <= state['times'][0] - event['first_token_s'] < 1
                    assert 0 <= state['times'][-1] - event['last_token_s'] < 1
                state['terminal'] = {k: v for k, v in event.items() if k not in {'event', 'elapsed_s'}}
            else:
                raise AssertionError('unknown event type')
    assert end_count == 1 and all(s['terminal'] for s in states.values())
    computed = summarize([s['terminal'] for s in states.values()], trace['arrival_window_s'])
    assert computed == json.loads((directory / 'metrics.json').read_text())
    for name, digest in manifest['output_hashes'].items():
        assert sha(directory / name) == digest, name
    for name, digest in manifest['source_hashes'].items():
        assert sha(directory / 'source' / name) == digest, name
    for record in manifest.get('compressed_outputs', {}).values():
        digest, size = hashlib.sha256(), 0
        with gzip.open(directory / record['file'], 'rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(chunk)
                size += len(chunk)
        assert digest.hexdigest() == record['raw_sha256'] and size == record['raw_bytes']
    observer = manifest['gpu_observer']
    return {'kind': 'raw_measurement_consistency_audit', 'all_checks_pass': True,
            'stage': manifest['stage'], 'status': manifest['status'], 'requests': len(planned),
            'completed': sum(s['terminal']['status'] == 'completed' for s in states.values()),
            'output_tokens': sum(len(s['tokens']) for s in states.values()),
            'length_cap_completions': sum(s['terminal'].get('finish_reason') == 'length' for s in states.values()),
            'occupancy_verified': not observer['foreign_observations'] and not observer['query_errors'],
            'source_manifest_sha256': sha(directory / 'manifest.json'),
            'auditor_sha256': sha(__file__),
            'metrics_implementation_sha256': sha(Path(__file__).parent / 'serving_metrics.py')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    result = audit(args.directory)
    (Path(args.directory) / 'measurement_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
