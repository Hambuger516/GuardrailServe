"""Open-loop Poisson arrivals with independent prompt/class random streams."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from runtime_smoke import sha


def make_trace(pool, rate, duration, seed):
    if not math.isfinite(rate) or not math.isfinite(duration) or min(rate, duration) <= 0:
        raise ValueError('positive finite rate and duration required')
    def rng(label):
        digest = hashlib.sha256(f'{seed}:{label}'.encode()).digest()
        return random.Random(int.from_bytes(digest, 'big'))
    arrival_rng, prompt_rng, class_rng = rng('arrival'), rng('prompt'), rng('class')
    result, at = [], 0.0
    while True:
        at += arrival_rng.expovariate(rate)
        if at >= duration:
            break
        row = prompt_rng.choice(pool)
        u = class_rng.random()
        result.append({'request_id': f'r{len(result):06d}', 'prompt_id': row['prompt_id'],
                       'arrival_s': at, 'priority': 0 if u < .2 else 1 if u < .9 else 2})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--duration', type=float, default=120)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    pool = json.loads(Path(args.pool).read_text())
    if pool['split'] not in {'calibration', 'evaluation'}:
        raise ValueError('effect traces must not use the historical-fit pool')
    trace = {'schema': 'guardrail_openloop_v1', 'pool_sha256': sha(args.pool),
             'pool_split': pool['split'], 'seed': args.seed, 'arrival_window_s': args.duration,
             'target_rate_rps': args.rate, 'class_probabilities': [.2, .7, .1],
             'source_sha256': sha(__file__),
             'requests': make_trace(pool['requests'], args.rate, args.duration, args.seed)}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(trace, stream, indent=2)
    print(json.dumps({'requests': len(trace['requests']), 'sha256': sha(target), 'out': str(target)}))


if __name__ == '__main__':
    main()
