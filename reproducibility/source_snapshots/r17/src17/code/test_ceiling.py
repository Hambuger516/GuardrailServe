"""Offline tests for the R13 admission ceiling (no vLLM/GPU needed)."""
import random, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_core import View
from slack_policy import SlackController, SlackDeferralState
from ceiling_policy import CeilingController

HIST = {str(n): 1 for n in range(1, 400)}
total = sum(HIST.values()); HIST = {k: v / total for k, v in HIST.items()}
BASE = dict(policy='C4', length_histogram=HIST, revision_arm='guardrail', horizon=10, epsilon=.05,
            theta_low=.8, theta_high=.95, weights=[3, 1, .3], tpot_ms=[50, 100, 200], gamma=.01, beta=.1,
            warning_wait_fraction=.5, protect_high_victims=True)


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


def state(rng, now):
    act = [View(f'a{i}', rng.randint(0, 300), rng.randint(20, 1500), rng.randint(0, 90), rng.randint(0, 2),
                now - rng.random() * 8) for i in range(rng.randint(0, 22))]
    cand = View('c', rng.choice([0, 0, 0, 3]), rng.randint(20, 1500), rng.randint(0, 5), rng.randint(0, 2), now - rng.random() * 8)
    return act, cand


class CeilingTests(unittest.TestCase):
    def test_ceiling_one_is_identical_to_protected(self):
        rng = random.Random(13); clock = Clock()
        ref = SlackController(dict(BASE), 16, 400, clock); c = CeilingController(dict(BASE, admission_ceiling=1.0), 16, 400, clock)
        for _ in range(4000):
            act, cand = state(rng, clock.t)
            self.assertEqual(ref.admission_decision(act, cand), c.admission_decision(act, cand))
            self.assertEqual(SlackDeferralState(ref).decide(lambda: act, lambda: cand), SlackDeferralState(c).decide(lambda: act, lambda: cand))
            self.assertEqual(ref.choose_victim(act, 20.0), c.choose_victim(act, 20.0))

    def test_ceiling_only_restricts_non_high_allows(self):
        rng = random.Random(14); clock = Clock()
        ref = SlackController(dict(BASE), 16, 400, clock); c = CeilingController(dict(BASE, admission_ceiling=.95), 16, 400, clock)
        hits = 0
        for _ in range(6000):
            act, cand = state(rng, clock.t)
            r, d = ref.admission_decision(act, cand), c.admission_decision(act, cand)
            if not r[0] or cand.priority == 0:
                self.assertEqual(r, d)                       # never admits more; HIGH unchanged
            elif r[2] > .95 * 400:
                self.assertEqual(d, (False, 'admission_ceiling', r[2])); hits += 1
            else:
                self.assertEqual(r, d)
            self.assertEqual(ref.choose_victim(act, 20.0), c.choose_victim(act, 20.0))   # recovery untouched
        self.assertGreater(hits, 0)

    def test_inflight_inherited(self):
        c = CeilingController(dict(BASE, admission_ceiling=.95, inflight_protection=True), 16, 400, Clock())
        busy = View('l', 300, 400, 30, 2, 0.0); new = View('d', 0, 400, 25, 1, 1.0); high = View('h', 0, 100, 5, 0, 2.0)
        self.assertEqual(c.choose_victim([busy, new, high], 20.0).request_id, 'd')

    def test_validation(self):
        for bad in (0.8, 1.01, True, '0.95', None):
            with self.assertRaises(ValueError):
                CeilingController(dict(BASE, admission_ceiling=bad), 16, 400, Clock())
        with self.assertRaises(ValueError):
            CeilingController(dict(BASE, policy='C2', admission_ceiling=.95), 16, 400, Clock())


if __name__ == '__main__':
    unittest.main()
