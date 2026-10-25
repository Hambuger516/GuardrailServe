"""Offline tests for the R16 controller and adapter (no GPU)."""
import ast, importlib, random, sys, types, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy_core import View
from slack_policy import SlackController
from hysteresis_policy import HysteresisController
from r16_policy import R16Controller

HIST = {str(n): 1 for n in range(1, 400)}
total = sum(HIST.values()); HIST = {k: v / total for k, v in HIST.items()}
BASE = dict(policy='C4', length_histogram=HIST, revision_arm='guardrail', horizon=10, epsilon=.05,
            theta_low=.8, theta_high=.95, weights=[3, 1, .3], tpot_ms=[50, 100, 200], gamma=.01, beta=.1,
            warning_wait_fraction=.5, protect_high_victims=True)
R15 = dict(control_estimator='chance', proactive_recovery_hook=True)


def r16(mode, f2='on'):
    return dict(BASE, r15=R15, r16=dict(mode=mode, f2=f2))


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


def state(rng, clock):
    act = [View(f'a{i}', rng.randint(0, 380), rng.randint(150, 400), rng.randint(0, 25), rng.randint(0, 2),
                clock.t - rng.random() * 8) for i in range(rng.randint(10, 24))]
    cand = View('c', rng.choice([0, 0, 3]), rng.randint(20, 400), 0, rng.randint(0, 2), clock.t - rng.random() * 8)
    return act, cand


class ControllerTests(unittest.TestCase):
    def test_protected_equals_slack(self):
        rng = random.Random(16); clock = Clock()
        a = SlackController(BASE, 16, 400, clock); b = R16Controller(r16('protected'), 16, 400, clock)
        for _ in range(4000):
            act, cand = state(rng, clock)
            self.assertEqual(a.admission_decision(act, cand), b.admission_decision(act, cand))
            self.assertEqual(a.screen(act, cand, a.risk(act)), b.screen(act, cand, b.risk(act)))
            self.assertEqual(a.choose_victim(act, 20.0), b.choose_victim(act, 20.0))
        self.assertEqual(b.target(), .8)

    def test_hyst_equals_r14_hysteresis(self):
        rng = random.Random(17); clock = Clock(); seen = set()
        a = HysteresisController(dict(BASE, admission_ceiling=.85, recovery_target=.90), 16, 400, clock)
        b = R16Controller(r16('hyst'), 16, 400, clock)
        for _ in range(4000):
            act, cand = state(rng, clock)
            da, db = a.admission_decision(act, cand), b.admission_decision(act, cand)
            self.assertEqual(da, db); seen.add(da[1])
            self.assertEqual(a.choose_victim(act, 20.0), b.choose_victim(act, 20.0))
        self.assertIn('admission_ceiling', seen)
        self.assertEqual(b.target(), a.target)

    def test_adapt_switches_between_the_two(self):
        rng = random.Random(18); clock = Clock()
        p, h = R16Controller(r16('protected'), 16, 400, clock), R16Controller(r16('hyst'), 16, 400, clock)
        c = R16Controller(r16('adapt'), 16, 400, clock)
        self.assertFalse(c.long)
        for long in (False, True, False):
            c.long = long; ref = h if long else p
            for _ in range(1500):
                act, cand = state(rng, clock)
                self.assertEqual(c.admission_decision(act, cand), ref.admission_decision(act, cand))
            self.assertEqual(c.target(), ref.target())

    def test_ewma_regime_with_hysteresis(self):
        c = R16Controller(r16('adapt'), 16, 400, Clock())
        rows = lambda q: [View('x', 1, q, 1, 1, 0.0), View('y', 1, q, 1, 1, 0.0)]
        t = 0.0
        for _ in range(100): t += .1; c.observe(rows(500), t)
        self.assertFalse(c.long)
        for _ in range(600): t += .1; c.observe(rows(1500), t)          # 60 s -> EWMA ~1500
        self.assertTrue(c.long)
        for _ in range(600): t += .1; c.observe(rows(900), t)           # between 800 and 1000: stays long
        self.assertTrue(c.long)
        for _ in range(600): t += .1; c.observe(rows(600), t)
        self.assertFalse(c.long); self.assertEqual(c.switches, 2)
        c2 = R16Controller(r16('hyst'), 16, 400, Clock()); c2.observe(rows(10), 1.0)
        self.assertTrue(c2.long)                                        # fixed modes never switch

    def test_validation(self):
        for bad in (dict(mode='x', f2='on'), dict(mode='hyst', f2='x'), dict(mode='hyst')):
            with self.assertRaises(ValueError):
                R16Controller(dict(BASE, r15=R15, r16=bad), 16, 400, Clock())
        with self.assertRaises(ValueError):
            R16Controller(dict(BASE, r15=dict(R15, control_estimator='point'), r16=dict(mode='hyst', f2='on')), 16, 400, Clock())


def loop_lines(path, cls, fn, start='while len(rows) > 1'):
    tree = ast.parse(Path(path).read_text())
    node = next(f for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls for f in c.body
                if isinstance(f, ast.FunctionDef) and f.name == fn)
    src = ast.get_source_segment(Path(path).read_text(), node).splitlines()
    body = [l.split('#')[0].rstrip() for l in src if l.strip() and not l.strip().startswith('#')]
    return body[next(i for i, l in enumerate(body) if 'if not self._gr.recovery' in l):]


class AdapterTests(unittest.TestCase):
    def test_recovery_loop_verbatim_except_stop(self):
        a = loop_lines(HERE / 'runtime_scheduler.py', 'GuardrailScheduler', '_gr_recover')
        b = loop_lines(HERE / 'runtime_r16.py', 'R16Scheduler', '_r16_recover_loop')
        self.assertEqual(len(a), len(b))
        self.assertEqual([(x, y) for x, y in zip(a, b) if x != y],
                         [('        while len(rows) > 1 and bound > self._gr.low * self._gr.capacity:',
                           '        while len(rows) > 1 and bound > self._gr.target() * self._gr.capacity:')])

    def setUp(self):
        stub = types.ModuleType('runtime_r15')

        class R15Scheduler:
            def _gr_token_budget(self, budget): return 'parent'
            def _gr_emit(self, event, **f): self.events.append((event, f))
        stub.R15Scheduler = R15Scheduler
        self.saved = sys.modules.get('runtime_r15'); sys.modules['runtime_r15'] = stub
        sys.modules.pop('runtime_r16', None); self.mod = importlib.import_module('runtime_r16')

    def tearDown(self):
        sys.modules.pop('runtime_r16', None)
        if self.saved is not None: sys.modules['runtime_r15'] = self.saved
        else: sys.modules.pop('runtime_r15', None)

    def make(self, mode, f2, held, n=20):
        s = self.mod.R16Scheduler.__new__(self.mod.R16Scheduler)
        s._gr = R16Controller(r16(mode, f2), 16, 400, Clock()); s._gr.held_fn = lambda: held
        s.events, s._gr_decode_ms, s._gr_budget_cap = [], 20.0, 512
        s._r16_long_steps = s._r16_cap_skipped = 0; s._r15_inner_recover_s = 0.0
        s.running = [types.SimpleNamespace(request_id=f'a{i}', priority=i % 3, num_output_tokens=5) for i in range(n)]
        views = {r.request_id: View(r.request_id, 50, 330, 21, r.priority, 0.0) for r in s.running}
        s._gr_view = lambda r: views[r.request_id]
        s.preempted = []
        def pre(req, ts): s.preempted.append(req.request_id)
        s._preempt_request = pre
        return s

    def test_f2_pressure_gate(self):
        lo = self.make('hyst', 'pressure', held=100); hi = self.make('hyst', 'pressure', held=380)
        on = self.make('hyst', 'on', held=100)
        self.assertEqual(lo._gr_token_budget(4096), 4096); self.assertEqual(lo._r16_cap_skipped, 1)
        self.assertEqual(hi._gr_token_budget(4096), 'parent')
        self.assertEqual(on._gr_token_budget(4096), 'parent')

    def test_stop_levels(self):
        for mode, stop in (('protected', .80), ('hyst', .90)):
            s = self.make(mode, 'on', held=390)
            s._gr_recover(0.0)
            rows = [s._gr_view(r) for r in s.running]
            self.assertTrue(s.preempted)
            self.assertLessEqual(s._gr.risk(rows), stop * 400)
            # one fewer victim would have left the bound above the stop level
            self.assertTrue(all(v.split('a')[1] for v in s.preempted))
        deep, shallow = self.make('protected', 'on', 390), self.make('hyst', 'on', 390)
        deep._gr_recover(0.0); shallow._gr_recover(0.0)
        self.assertGreater(len(deep.preempted), len(shallow.preempted))


if __name__ == '__main__':
    unittest.main()
