"""Offline tests for the R15 controller and recovery hook (no GPU). Registered gate (a)-(f) of PROTOCOL_R15.md."""
import importlib, random, sys, types, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_core import View
from slack_policy import SlackController
from estimator_policy import EstimatorController, ESTIMATORS

HIST = {str(n): 1 for n in range(1, 400)}
total = sum(HIST.values()); HIST = {k: v / total for k, v in HIST.items()}
BASE = dict(policy='C4', length_histogram=HIST, revision_arm='guardrail', horizon=10, epsilon=.05,
            theta_low=.8, theta_high=.95, weights=[3, 1, .3], tpot_ms=[50, 100, 200], gamma=.01, beta=.1,
            warning_wait_fraction=.5, protect_high_victims=True)


def r15(est='chance', hook=True, **kw):
    return dict(BASE, r15=dict(control_estimator=est, proactive_recovery_hook=hook), **kw)


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


def state(rng, clock, n=None):
    act = [View(f'a{i}', rng.randint(0, 300), rng.randint(20, 1500), rng.randint(0, 90), rng.randint(0, 2),
                clock.t - rng.random() * 8) for i in range(rng.randint(0, 22) if n is None else n)]
    cand = View('c', rng.choice([0, 0, 3]), rng.randint(20, 1500), rng.randint(0, 5), rng.randint(0, 2),
                clock.t - rng.random() * 8)
    return act, cand


class EstimatorTests(unittest.TestCase):
    def test_c_chance_identical_to_slack(self):
        rng = random.Random(15); clock = Clock()
        for protect in (True, False):
            a = SlackController(dict(BASE, protect_high_victims=protect), 16, 400, clock)
            for hook in (True, False):
                b = EstimatorController(r15('chance', hook, protect_high_victims=protect), 16, 400, clock)
                for _ in range(3000):
                    act, cand = state(rng, clock)
                    self.assertEqual(a.risk(act), b.risk(act))
                    self.assertEqual(a.admission_decision(act, cand), b.admission_decision(act, cand))
                    self.assertEqual(a.screen(act, cand, a.risk(act)), b.screen(act, cand, b.risk(act)))
                    self.assertEqual(a.choose_victim(act, 20.0), b.choose_victim(act, 20.0))

    def test_a_norec_controller_keeps_warning_semantics(self):
        clock = Clock(); c = EstimatorController(r15('chance', False), 16, 400, clock)
        self.assertTrue(c.recovery and c.hook is False)
        busy = [View(f'a{i}', 5, 300, 20, 1, clock.t) for i in range(17)]          # ~0.85K held
        new = View('n', 0, 100, 0, 1, clock.t)                                       # fresh non-HIGH, no wait yet
        self.assertGreater(c.risk(busy), c.low * c.capacity)
        self.assertEqual(c.admission_decision(busy, new)[1], 'warning_non_high')
        high = View('h', 0, 100, 0, 0, clock.t)
        self.assertNotEqual(c.admission_decision(busy, high)[1], 'warning_non_high')

    def test_d_hard_bound(self):
        rng = random.Random(16); clock = Clock()
        c = EstimatorController(r15('hard'), 16, 400, clock)
        for _ in range(1000):
            act, _ = state(rng, clock)
            b = c.bounds(act)
            hard = b['current'] + sum((r.prefix_tokens + 10 + 15) // 16 - (r.prefix_tokens + 15) // 16 for r in act)
            self.assertEqual(b['hard'], hard)
            self.assertGreaterEqual(b['hard'] + 1e-9, b['point'])
            self.assertEqual(b['min_chance_hard'], min(b['chance'], b['hard']))
            for r in act:                                                   # U covers the support maximum per request
                top = max(y for y, _ in c.histogram.truncated(r.generated, 10))
                self.assertLessEqual((r.prefix_tokens + top + 15) // 16 - (r.prefix_tokens + 15) // 16,
                                     (r.prefix_tokens + 10 + 15) // 16 - (r.prefix_tokens + 15) // 16)

    def test_estimator_is_the_compared_quantity(self):
        clock = Clock(); rng = random.Random(17)
        ctrl = {e: EstimatorController(r15(e), 16, 400, clock) for e in ESTIMATORS}
        seen = set()
        for _ in range(4000):
            act, cand = state(rng, clock)
            for e, c in ctrl.items():
                self.assertEqual(c.risk(act), c.bounds(act)[e])
                ok, reason, bound = c.admission_decision(act, cand)
                if reason.startswith('risk_bound'):
                    self.assertEqual(bound, c.bounds([*act, cand])[e])
            seen.add(tuple(ctrl[e].admission_decision(act, cand)[0] for e in ESTIMATORS))
        self.assertGreater(len(seen), 1)                                    # the quantities do change decisions

    def test_e_cache(self):
        clock = Clock(); c = EstimatorController(r15('hard'), 16, 400, clock)
        a = [View('x', 3, 100, 7, 1, clock.t), View('y', 0, 33, 3, 2, clock.t)]
        b1, b2 = c.bounds(a), c.bounds(list(a))
        self.assertEqual(b1, b2)
        b1['hard'] = -1
        self.assertNotEqual(c.bounds(a)['hard'], -1)                        # callers cannot mutate the cache
        moved = [View('x', 3, 160, 7, 1, clock.t), a[1]]
        self.assertNotEqual(c.bounds(moved)['hard'], c.bounds(a)['hard'])

    def test_shadow_joint_is_exact(self):
        rng = random.Random(19); clock = Clock(); c = EstimatorController(r15('chance'), 16, 400, clock)
        for _ in range(3000):
            act, cand = state(rng, clock)
            act = tuple(act)
            self.assertEqual(c._joint_bounds(c._active_bounds(act), cand),
                             {k: v for k, v in c.bounds([*act, cand]).items() if k in ESTIMATORS + ('current',)})

    def test_f_no_future_lengths(self):
        src = (Path(__file__).resolve().parent / 'estimator_policy.py').read_text()
        src += (Path(__file__).resolve().parent / 'runtime_r15.py').read_text()
        for word in ('target_output_tokens', "'recorded'", 'max_tokens', 'output_len', 'by_id'):
            self.assertNotIn(word, src)

    def test_validation(self):
        for bad in (dict(control_estimator='x', proactive_recovery_hook=True),
                    dict(control_estimator='point', proactive_recovery_hook=1),
                    dict(control_estimator='point')):
            with self.assertRaises(ValueError):
                EstimatorController(dict(BASE, r15=bad), 16, 400, Clock())
        with self.assertRaises(ValueError):
            EstimatorController(dict(r15(), warning_wait_fraction=None), 16, 400, Clock())

    def test_shadow_records_disagreement_without_side_effects(self):
        clock = Clock(); events = []
        c = EstimatorController(r15('chance'), 16, 400, clock); c.emit_evidence = lambda e, **f: events.append((e, f))
        c.held_fn = lambda: 123
        ref = EstimatorController(r15('chance'), 16, 400, clock)
        rng = random.Random(18)
        for _ in range(4000):                       # near-threshold states, where the quantities can disagree
            act = [View(f'a{i}', rng.randint(0, 380), rng.randint(200, 330), 0, rng.randint(0, 2),
                        clock.t - rng.random() * 8) for i in range(rng.randint(15, 22))]
            cand = View('c', 0, rng.randint(20, 200), 0, rng.randint(0, 2), clock.t - rng.random() * 8)
            self.assertEqual(c.screen(act, cand, c.risk(act)), ref.screen(act, cand, ref.risk(act)))
        names = {e for e, _ in events}
        self.assertIn('r15_shadow', names); self.assertNotIn('r15_shadow_error', names)
        for e, f in events:
            self.assertEqual(f['decisions']['chance'], f['actual'])
            self.assertEqual(f['held'], 123)


class HookTests(unittest.TestCase):
    """Scheduler hook with a stub parent (vLLM is not importable on CPU)."""

    def setUp(self):
        stub = types.ModuleType('runtime_slack')

        class SlackScheduler:
            def _gr_recover(self, timestamp):
                self.parent_called = True
                return ['victim']

            def _gr_emit(self, event, **fields):
                self.events.append((event, fields))
        stub.SlackScheduler = SlackScheduler
        self.saved = sys.modules.get('runtime_slack'); sys.modules['runtime_slack'] = stub
        sys.modules.pop('runtime_r15', None)
        self.mod = importlib.import_module('runtime_r15')

    def tearDown(self):
        sys.modules.pop('runtime_r15', None)
        if self.saved is not None: sys.modules['runtime_slack'] = self.saved
        else: sys.modules.pop('runtime_slack', None)

    def make(self, hook, n=20):
        clock = Clock(); s = self.mod.R15Scheduler.__new__(self.mod.R15Scheduler)
        s._gr = EstimatorController(r15('chance', hook), 16, 400, clock); s._gr.held_fn = lambda: 390
        s.events, s.parent_called, s._gr_decode_ms = [], False, 20.0
        s._r15_reasons, s._r15_inner_recover_s, s._r15_danger_shadow, s._gr_budget_steps = {}, 0.0, 0, 0
        s.running = [View(f'a{i}', 50, 330, 21, 1 + i % 2, clock.t) for i in range(n)]
        s._gr_view = lambda r: r
        s._preempt_request = lambda *a: self.fail('norec must never preempt')
        return s

    def test_a_hook_off_returns_no_victims_and_records_shadow(self):
        s = self.make(False)
        self.assertGreater(s._gr.risk(s.running), .95 * 400)
        self.assertEqual(s._gr_recover(0.0), [])
        self.assertFalse(s.parent_called)
        names = [e for e, _ in s.events]
        self.assertIn('r15_step', names); self.assertIn('danger_shadow', names)
        shadow = dict(s.events)['danger_shadow']
        self.assertTrue(shadow['would_evict'])
        self.assertTrue(all(v['priority'] > 0 for v in shadow['would_evict']))     # F1 replayed on views
        self.assertEqual(s._r15_danger_shadow, 1)

    def test_hook_on_delegates_to_audited_loop(self):
        s = self.make(True)
        self.assertEqual(s._gr_recover(0.0), ['victim'])
        self.assertTrue(s.parent_called)
        self.assertNotIn('danger_shadow', [e for e, _ in s.events])

    def test_schedule_event_carries_counters(self):
        s = self.make(False)
        s._gr_emit('admission_check', request_id='r1', allow=False, reason='warning_non_high')
        s._gr_emit('schedule', scheduled_tokens={})
        fields = dict(s.events)['schedule']
        self.assertEqual(fields['r15_counters']['reasons'], {'measurement:warning_non_high:defer': 1})


if __name__ == '__main__':
    unittest.main()
