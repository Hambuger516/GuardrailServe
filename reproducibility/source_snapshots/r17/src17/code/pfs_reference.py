"""Independent AE decision model: tokens, no GPU execution.

Preserves AE quirks deliberately: WINDOW=40, interpolation, max_output_len
endpoint, ignore_eos bypass, strict capacity inequality, one admit per call.
This is not native LightLLM throughput or a block-allocator runtime baseline.
"""
from collections import deque
from dataclasses import dataclass
import bisect
import random
import numpy as np


@dataclass
class Request:
    request_id: str
    prompt: int
    generated: int = 0
    max_output: int = 2048
    status: str = 'waiting'
    ignore_eos: bool = False

    def token_pair(self, sampled):
        length = self.max_output if self.ignore_eos else min(self.max_output, max(self.generated, sampled))
        if self.status in {'running', 'kvkeep'}:
            return self.prompt + self.generated, max(0, length - self.generated - 1)
        if self.status == 'waiting':
            return self.prompt + 1, max(0, length - 2)
        if self.status == 'offload':
            return self.prompt + self.generated + 1, max(0, length - self.generated - 2)
        raise ValueError(self.status)


def completion_peak(pairs, block_size=1):
    """Before-release peaks, including t=0; integer block extension is labelled.

    block_size=1 matches the AE's token formula. Larger values are an explicit
    paged-KV adaptation and must not be labelled byte-identical AE behavior.
    """
    if not pairs:
        return 0
    return max(sum((base + t + block_size - 1) // block_size
                   for base, left in pairs if left >= t)
               for t in {0, *(left for _, left in pairs)})


class PFSReference:
    def __init__(self, capacity, initial_length=6000, max_requests=128,
                 batch_max_tokens=18000, seed=0):
        self.capacity, self.max_requests = capacity, max_requests
        self.batch_max_tokens = batch_max_tokens
        self.history = deque([initial_length] * 20, maxlen=40)
        self.np_rng, self.py_rng = np.random.RandomState(seed), random.Random(seed)

    def update_history(self, lengths):
        self.history.extend(lengths)

    def active_samples(self, active):
        count = min(5, int(200 / len(active)) + 1) if active else 1
        samples = [[] for _ in range(count)]
        history = sorted(self.history)
        for r in active:
            surviving = history[bisect.bisect(history, r.generated):]
            endpoints = [r.generated, *surviving]
            if endpoints[-1] < r.max_output:
                endpoints.append(r.max_output)
            for sample in samples:
                u = self.np_rng.random_sample() * (len(endpoints) - 1)
                i = int(u)
                # AE raises for the degenerate completed-at-cap active request;
                # preserve that limitation rather than silently invent behavior.
                length = round(endpoints[i] + (endpoints[i + 1] - endpoints[i]) * (u - i))
                sample.append(r.token_pair(length))
        return samples

    def decide(self, active, waiting, block_size=1):
        if len(active) >= self.max_requests or not waiting:
            return [], {'reason': 'full_or_empty'}
        samples = self.active_samples(active)
        candidate = waiting[0]
        if candidate.prompt > self.batch_max_tokens:
            return [], {'reason': 'prefill_budget'}
        peaks = []
        for sample in samples:
            sample.append(candidate.token_pair(self.py_rng.choice(self.history)))
            peaks.append(completion_peak(sample, block_size))
        # capacity is in tokens for AE; in blocks for explicit block_size>1 port.
        allowed = max(peaks) < self.capacity * 0.95 and len(active) + 1 <= self.max_requests
        return ([candidate.request_id] if allowed else []), {'peaks': peaks, 'samples': samples}
