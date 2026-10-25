"""Decision-only comparisons against official unmodified AE class bodies.

AST extraction skips import-time GPU dependencies; runtime serving is not tested.
"""
import ast
import asyncio
import bisect
from collections import deque
import contextlib
import enum
import hashlib
import io
from pathlib import Path
import random
import time
from types import SimpleNamespace
import typing
import unittest
import uuid
import numpy as np
from pfs_reference import PFSReference, Request, completion_peak


ROOT = Path(__file__).parent / 'vendor_reference' / 'pfs_ae_d93ff69'


def official_classes():
    expected = {'io_struct.py': '5107b13ef94c861b86ff5e066c1c15749687f2a8d15816fc1d9d1e64979f2ccb',
                'req_queue.py': '79b3526ff983b759f41dd2fdc0a2eae1df8134486c11a4ee98c5e7b6d86bfbdc'}
    scope = dict(globals())
    scope.update(vars(typing))
    scope.update(SamplingParams=object, MultimodalParams=object)
    for name, selected in [('io_struct.py', {'ReqRunStatus', 'FinishStatus', 'Req', 'NormalReq', 'Batch'}),
                           ('req_queue.py', {'ReqQueue', 'FuturePastReqQueue'})]:
        data = (ROOT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected[name]
        tree = ast.parse(data.decode(), filename=name)
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in selected]
        assert len(tree.body) == len(selected)
        exec(compile(tree, str(ROOT / name), 'exec'), scope)
    return scope


OFFICIAL = official_classes()


def official_request(r):
    with contextlib.redirect_stdout(io.StringIO()):
        req = OFFICIAL['NormalReq'](r.request_id, [1] * r.prompt,
            SimpleNamespace(max_new_tokens=r.max_output, ignore_eos=r.ignore_eos), None, time.time())
    req.output_ids = [1] * r.generated
    req.req_status = getattr(OFFICIAL['ReqRunStatus'], {'running': 'RUNNING', 'waiting': 'WAIT_IN_QUEUE',
                            'offload': 'PAUSED_AND_OFFLOAD', 'kvkeep': 'PAUSED_AND_KVKEEP'}[r.status])
    req.cur_kv_len = r.prompt + max(0, r.generated - 1) if r.status == 'running' else 0
    return req


def queue(capacity, initial=16, max_requests=128, batch_budget=18000):
    args = SimpleNamespace(max_total_token_num=capacity, batch_max_tokens=batch_budget,
        running_max_req_size=max_requests, router_token_ratio=0.8, router_max_new_token_len=256,
        splitfuse_mode=False, splitfuse_block_size=256, max_req_total_len=initial + 100,
        max_req_input_len=100)
    return OFFICIAL['FuturePastReqQueue'](args, 0, 0)


class FidelityTests(unittest.TestCase):
    def test_completion_release_peak_by_hand(self):
        pairs = [(10, 1), (2, 8)]
        # t=0:12; t=1:11+3=14; t=8:10. Held-to-max would incorrectly be28.
        self.assertEqual(completion_peak(pairs), 14)
        self.assertEqual(queue(100)._calc_max_token_num_needed(list(pairs)), 14)
        self.assertEqual(completion_peak(pairs, block_size=4), 4)

    def test_strict_reserve_boundary(self):
        for capacity, expected in [(20, False), (21, True)]:
            # waiting prompt10, sampled length10 => pair(11,8), peak19.
            official = queue(capacity, initial=10)
            official.append(official_request(Request('r', 10, max_output=100)))
            self.assertEqual(official.generate_new_batch(None) is not None, expected)

    def test_one_admission_and_head_of_line(self):
        official = queue(1000)
        for i in range(3):
            official.append(official_request(Request(str(i), 4)))
        batch = official.generate_new_batch(None)
        self.assertEqual(len(batch.reqs), 1)
        self.assertEqual(len(official.waiting_req_list), 2)
        blocked = queue(20, initial=10)
        for r in [Request('large', 100), Request('small', 1)]:
            blocked.append(official_request(r))
        self.assertIsNone(blocked.generate_new_batch(None))

    def test_ignore_eos_bypasses_history(self):
        for ignore, expected in [(False, (11, 2)), (True, (11, 98))]:
            r = Request('r', 10, max_output=100, ignore_eos=ignore)
            self.assertEqual(official_request(r).get_tuple_tokens(False, 4, 1), expected)
            self.assertEqual(r.token_pair(4), expected)

    def test_cold_start_and_window(self):
        official = queue(1000, initial=16)
        reference = PFSReference(1000, initial_length=16)
        self.assertEqual(list(official.history_output_len), list(reference.history))
        official.record_output_lengths(list(range(1, 51)))
        reference.update_history(range(1, 51))
        self.assertEqual(list(official.history_output_len), list(reference.history))
        self.assertEqual(list(reference.history), list(range(11, 51)))

    def test_sampling_count_boundaries(self):
        for n, expected in [(1, 5), (50, 5), (51, 4), (67, 3), (101, 2), (201, 1)]:
            active = [Request(str(i), 8, generated=3, max_output=80, status='running')
                      for i in range(n)]
            official = queue(100000)
            official._init_cache_list(OFFICIAL['Batch']('b', [official_request(r) for r in active]), False)
            reference = PFSReference(100000, initial_length=16)
            self.assertEqual(len(official._cache_len_lists), expected)
            self.assertEqual(len(reference.active_samples(active)), expected)

    def test_seeded_sampling_and_admission_match_120_cases(self):
        for seed in range(120):
            active = [Request(str(i), 3 + i * 2, generated=(i * 7 + seed) % 40,
                              max_output=80, status='running', ignore_eos=seed % 3 == 0)
                      for i in range(1 + seed % 9)]
            waiting = [Request('new', 8, max_output=80, ignore_eos=seed % 3 == 0)]
            capacity = 50 + seed * 5
            reference = PFSReference(capacity, initial_length=16, seed=seed)
            official = queue(capacity)
            history = [4, 8, 16, 24, 32, 48]
            reference.update_history(history)
            official.record_output_lengths(history)
            real_active = [official_request(r) for r in active]
            real_waiting = official_request(waiting[0])
            batch = OFFICIAL['Batch']('b', real_active)
            np.random.seed(seed)
            random.seed(seed)
            official._init_cache_list(batch, False)
            self.assertEqual(official._cache_len_lists, reference.active_samples(active))
            # Reset both streams to compare the complete decision from entry.
            reference = PFSReference(capacity, initial_length=16, seed=seed)
            reference.update_history(history)
            np.random.seed(seed)
            random.seed(seed)
            official.append(real_waiting)
            real_result = official.generate_new_batch(batch)
            actual_ids = [] if real_result is None else [r.request_id for r in real_result.reqs]
            self.assertEqual(actual_ids, reference.decide(active, waiting)[0], f'seed={seed}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
