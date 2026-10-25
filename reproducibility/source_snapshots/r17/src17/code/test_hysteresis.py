"""Offline tests for the R14 hysteresis controller and the recovery-loop copy (no GPU)."""
import inspect, random, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_core import View
from ceiling_policy import CeilingController
from hysteresis_policy import HysteresisController

HIST = {str(n): 1 for n in range(1, 400)}
total = sum(HIST.values()); HIST = {k: v / total for k, v in HIST.items()}
BASE = dict(policy='C4', length_histogram=HIST, revision_arm='guardrail', horizon=10, epsilon=.05,
            theta_low=.8, theta_high=.95, weights=[3, 1, .3], tpot_ms=[50, 100, 200], gamma=.01, beta=.1,
            warning_wait_fraction=.5, protect_high_victims=True)


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


class HysteresisTests(unittest.TestCase):
    def test_admission_identical_to_ceiling_controller(self):
        rng = random.Random(14); clock = Clock()
        a = CeilingController(dict(BASE, admission_ceiling=.85), 16, 400, clock)
        b = HysteresisController(dict(BASE, admission_ceiling=.85, recovery_target=.90), 16, 400, clock)
        for _ in range(4000):
            act = [View(f'a{i}', rng.randint(0, 300), rng.randint(20, 1500), rng.randint(0, 90), rng.randint(0, 2),
                        clock.t - rng.random() * 8) for i in range(rng.randint(0, 22))]
            cand = View('c', rng.choice([0, 0, 3]), rng.randint(20, 1500), rng.randint(0, 5), rng.randint(0, 2), clock.t - rng.random() * 8)
            self.assertEqual(a.admission_decision(act, cand), b.admission_decision(act, cand))
            self.assertEqual(a.choose_victim(act, 20.0), b.choose_victim(act, 20.0))
        self.assertEqual(b.target, .90)

    def test_validation(self):
        for t, c in ((.79, .85), (.95, .85), (.84, .85), (True, .85)):
            with self.assertRaises(ValueError):
                HysteresisController(dict(BASE, admission_ceiling=c, recovery_target=t), 16, 400, Clock())

    def test_recovery_loop_is_verbatim_except_stop_target(self):
        try:
            import runtime_scheduler, runtime_hysteresis
        except Exception as exc:                       # vLLM not importable in a plain CPU env
            self.skipTest(repr(exc)); return
        ref = inspect.getsource(runtime_scheduler.GuardrailScheduler._gr_recover)
        new = inspect.getsource(runtime_hysteresis.HysteresisScheduler._gr_recover)
        norm = lambda s: [l.split('#')[0].rstrip() for l in s.splitlines() if l.strip() and not l.strip().startswith('#')]
        a, b = norm(ref), norm(new)
        self.assertEqual(len(a), len(b))
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(diff, [('        while len(rows) > 1 and bound > self._gr.low * self._gr.capacity:',
                                 '        while len(rows) > 1 and bound > self._gr.target * self._gr.capacity:')])


if __name__ == '__main__':
    unittest.main()
