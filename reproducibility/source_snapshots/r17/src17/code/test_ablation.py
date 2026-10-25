"""Offline tests for the R12 ablation controller (no vLLM/GPU needed)."""
import random, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_core import View
from slack_policy import SlackController, SlackDeferralState
from revision_policy import RevisionController, RevisionDeferralState
from ablation_policy import AblationController, validate_ablation

HIST = {str(n): 1 for n in range(1, 400)}
total = sum(HIST.values()); HIST = {k: v / total for k, v in HIST.items()}
BASE = dict(policy='C4', length_histogram=HIST, revision_arm='guardrail', horizon=10, epsilon=.05,
            theta_low=.8, theta_high=.95, weights=[3, 1, .3], tpot_ms=[50, 100, 200], gamma=.01, beta=.1)


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


def ab(mode, gate=False, wf=None, ph=False, cap=512):
    return dict(admission_mode=mode, warning_gate=gate, wait_fraction=wf, protect_high=ph, token_budget=cap)


def rand_state(rng, now):
    act = [View(f'a{i}', rng.randint(0, 300), rng.randint(20, 900), rng.randint(0, 60), rng.randint(0, 2),
                now - rng.random() * 8) for i in range(rng.randint(0, 22))]
    cand = View('c', rng.choice([0, 0, 0, 3]), rng.randint(20, 900), rng.randint(0, 5), rng.randint(0, 2),
                now - rng.random() * 8)
    return act, cand


class AblationTests(unittest.TestCase):
    def test_protected_configuration_identical_to_slack(self):
        rng = random.Random(12); clock = Clock()
        ref = SlackController(dict(BASE, warning_wait_fraction=.5, protect_high_victims=True), 16, 300, clock)
        abl = AblationController(dict(BASE, ablation=ab('C4', True, .5, True)), 16, 300, clock)
        for _ in range(4000):
            act, cand = rand_state(rng, clock.t)
            self.assertEqual(ref.admission_decision(act, cand), abl.admission_decision(act, cand))
            self.assertEqual(SlackDeferralState(ref).decide(lambda: act, lambda: cand),
                             SlackDeferralState(abl).decide(lambda: act, lambda: cand))
            self.assertEqual(ref.choose_victim(act, 20.0), abl.choose_victim(act, 20.0))
            self.assertEqual(ref.risk(act), abl.risk(act))
            self.assertEqual((ref.admission, ref.recovery), (abl.admission, abl.recovery))

    def test_c0_is_priority_admission_without_recovery(self):
        rng = random.Random(1); clock = Clock()
        pri = RevisionController(dict(BASE, policy='C0', revision_arm='priority'), 16, 300)
        abl = AblationController(dict(BASE, ablation=ab('C0')), 16, 300, clock)
        self.assertFalse(abl.recovery)
        for _ in range(2000):
            act, cand = rand_state(rng, clock.t)
            self.assertEqual(pri.admission_decision(act, cand), abl.admission_decision(act, cand))
            self.assertEqual(RevisionDeferralState(pri).decide(lambda: act, lambda: cand),
                             SlackDeferralState(abl).decide(lambda: act, lambda: cand))

    def test_c2_screen_without_warning_or_recovery(self):
        rng = random.Random(2); clock = Clock()
        abl = AblationController(dict(BASE, ablation=ab('C2')), 16, 300, clock)
        self.assertEqual((abl.admission, abl.recovery), ('chance', False))
        for _ in range(2000):
            act, cand = rand_state(rng, clock.t)
            allow, reason, bound = abl.admission_decision(act, cand)
            self.assertEqual(reason, 'risk_bound')
            self.assertEqual(allow, abl.bounds([*act, cand])['chance'] <= 300)

    def test_c5_current_fit_admission_with_gaussian_recovery_and_f1(self):
        rng = random.Random(3); clock = Clock()
        abl = AblationController(dict(BASE, ablation=ab('C5', ph=True)), 16, 300, clock)
        self.assertEqual((abl.admission, abl.recovery), ('current', True))
        for _ in range(2000):
            act, cand = rand_state(rng, clock.t)
            self.assertTrue(abl.admission_decision(act, cand)[0])
            self.assertEqual(abl.risk(act), abl.bounds(act)['chance'])
            v = abl.choose_victim(act, 20.0)
            if any(r.priority > 0 and r.allocated_blocks > 0 for r in act):
                self.assertTrue(v.priority > 0)

    def test_c4_without_warning_never_defers_by_warning(self):
        rng = random.Random(4); clock = Clock()
        prot = AblationController(dict(BASE, ablation=ab('C4', True, .5, True)), 16, 300, clock)
        nowarn = AblationController(dict(BASE, ablation=ab('C4', ph=True)), 16, 300, clock)
        seen = 0
        for _ in range(4000):
            act, cand = rand_state(rng, clock.t)
            d = nowarn.admission_decision(act, cand)
            self.assertNotEqual(d[1], 'warning_non_high')
            self.assertEqual(d[0], nowarn.bounds([*act, cand])['chance'] <= 300)
            seen += prot.admission_decision(act, cand)[1] == 'warning_non_high'
        self.assertGreater(seen, 0)   # the comparison state space does exercise the warning gate

    def test_validation(self):
        for bad in [ab('C1'), ab('C2', gate=True), ab('C0', ph=True), ab('C4', wf=.5), ab('C4', cap=32),
                    dict(ab('C4'), extra=1)]:
            with self.assertRaises(ValueError):
                validate_ablation(bad)
        self.assertEqual(validate_ablation(ab('C4', True, .5, True)), ('chance', True))


if __name__ == '__main__':
    unittest.main()
