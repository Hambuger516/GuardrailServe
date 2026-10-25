"""Offline tests for the slack-bounded warning gate (no vLLM/GPU needed)."""
import json, random, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_core import View
from revision_policy import RevisionController, RevisionDeferralState
from slack_policy import SlackController, SlackDeferralState

HIST = {str(n): 1 for n in range(1, 400)}
total = sum(HIST.values()); HIST = {k: v / total for k, v in HIST.items()}
BASE = dict(policy='C4', length_histogram=HIST, revision_arm='guardrail', horizon=10, epsilon=.05,
            theta_low=.8, theta_high=.95, weights=[3, 1, .3], tpot_ms=[50, 100, 200], gamma=.01, beta=.1)


class Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t


def rows(n, prefix=200, gen=20, blocks=13):
    return [View(f'a{i}', gen, prefix, blocks, 1, 0.0) for i in range(n)]


class SlackTests(unittest.TestCase):
    def make(self, fraction, capacity=200, clock=None):
        return SlackController(dict(BASE, warning_wait_fraction=fraction), 16, capacity, clock or Clock())

    def test_none_identical_to_parent_on_random_states(self):
        rng = random.Random(1)
        parent = RevisionController(dict(BASE), 16, 300)
        child = self.make(None, 300)
        for _ in range(3000):
            act = [View(f'a{i}', rng.randint(0, 300), rng.randint(20, 900), rng.randint(1, 60),
                        rng.randint(0, 2), rng.random()) for i in range(rng.randint(0, 20))]
            cand = View('c', rng.choice([0, 0, 3]), rng.randint(20, 900), rng.randint(0, 5), rng.randint(0, 2), rng.random())
            self.assertEqual(parent.admission_decision(act, cand), child.admission_decision(act, cand))
            ps, cs = RevisionDeferralState(parent), SlackDeferralState(child)
            self.assertEqual(ps.decide(lambda: act, lambda: cand), cs.decide(lambda: act, lambda: cand))

    def test_warning_defers_before_budget_then_risk_screen(self):
        clock = Clock(1000.0)
        c = self.make(0.5, capacity=200, clock=clock)
        act = rows(12)  # current risk > 0.8*200 assumed; verify
        cur = c.risk(act)
        self.assertGreater(cur, 0.8 * 200)
        cand = View('c', 0, 100, 0, 1, 1000.0 - 0.9)   # DEFAULT waited 0.9 s < 1.0 s
        self.assertEqual(c.admission_decision(act, cand)[1], 'warning_non_high')
        cand = View('c', 0, 100, 0, 1, 1000.0 - 1.0)   # waited exactly budget
        allow, reason, bound = c.admission_decision(act, cand)
        self.assertEqual(reason, 'risk_bound_after_wait_budget')
        self.assertEqual(allow, bound <= 200)
        self.assertEqual(bound, c.bounds([*act, cand])['chance'])

    def test_budget_scales_with_class_and_high_unaffected(self):
        clock = Clock(1000.0); c = self.make(0.5, 200, clock); act = rows(12)
        low = View('l', 0, 100, 0, 2, 1000.0 - 4.9)    # LOW budget 5 s
        self.assertEqual(c.admission_decision(act, low)[1], 'warning_non_high')
        low = View('l', 0, 100, 0, 2, 1000.0 - 5.0)
        self.assertEqual(c.admission_decision(act, low)[1], 'risk_bound_after_wait_budget')
        high = View('h', 0, 100, 0, 0, 1000.0)
        self.assertEqual(c.admission_decision(act, high)[1], 'risk_bound')

    def test_requests_with_tokens_keep_original_warning(self):
        c = self.make(0.5, 200, Clock(1000.0)); act = rows(12)
        resumed = View('r', 5, 300, 0, 1, 0.0)  # waited long, but already has output
        self.assertEqual(c.admission_decision(act, resumed)[1], 'warning_non_high')

    def test_below_warning_zone_unchanged(self):
        c = self.make(0.5, 10000, Clock(1000.0)); act = rows(3)
        cand = View('c', 0, 100, 0, 1, 0.0)
        self.assertEqual(c.admission_decision(act, cand)[1], 'risk_bound')

    def test_deferral_state_reuses_frozen_view(self):
        clock = Clock(1000.0); c = self.make(0.5, 200, clock); act = rows(12)
        st = SlackDeferralState(c); calls = []
        def af(): calls.append(1); return act
        d1, n1 = st.decide(af, lambda: View('x', 0, 100, 0, 1, 999.9))
        d2, n2 = st.decide(af, lambda: View('y', 0, 100, 0, 1, 999.95))
        self.assertEqual((d1[1], d2[1], len(calls), n1, n2), ('warning_non_high', 'warning_non_high', 1, 12, 12))
        d3, _ = st.decide(af, lambda: View('z', 0, 100, 0, 1, 990.0))
        self.assertEqual(d3[1], 'risk_bound_after_wait_budget')
        if d3[0]:
            self.assertIsNone(st.active)  # invalidated after an allow, like the parent

    def test_clock_mismatch_raises(self):
        c = self.make(0.5, 200, Clock(1000.0)); act = rows(12)
        with self.assertRaises(RuntimeError):
            c.admission_decision(act, View('m', 0, 100, 0, 1, 5_000_000.0))

    def test_cap_default_identical_to_slack(self):
        rng = random.Random(7)
        a = self.make(0.5, 300, Clock(1000.0))
        b = SlackController(dict(BASE, warning_wait_fraction=.5, bypass_bound_fraction=1.0), 16, 300, Clock(1000.0))
        for _ in range(2000):
            act = [View(f'a{i}', rng.randint(0, 300), rng.randint(20, 900), rng.randint(1, 60),
                        rng.randint(0, 2), 0.0) for i in range(rng.randint(0, 20))]
            cand = View('c', rng.choice([0, 0, 3]), rng.randint(20, 900), 0, rng.randint(0, 2), 1000.0 - rng.random() * 8)
            self.assertEqual(a.admission_decision(act, cand), b.admission_decision(act, cand))

    def test_cap_blocks_between_theta_high_and_capacity(self):
        clock = Clock(1000.0)
        cap = SlackController(dict(BASE, warning_wait_fraction=.5, bypass_bound_fraction=.95), 16, 200, clock)
        plain = self.make(0.5, 200, clock)
        found = 0
        for n in range(10, 16):
            for prefix in range(100, 400, 7):
                act = rows(n, prefix=prefix, blocks=(prefix + 15) // 16)
                cand = View('c', 0, 100, 0, 1, 1000.0 - 1.5)
                if not cap.warning_zone(cap.risk(act), cand):
                    continue
                bound = cap.bounds([*act, cand])['chance']
                d1, d2 = plain.admission_decision(act, cand), cap.admission_decision(act, cand)
                self.assertEqual(d2[1], 'risk_bound_after_wait_budget_capped')
                self.assertEqual(d2[0], bound <= 190)
                self.assertEqual(d1[0], bound <= 200)
                if 190 < bound <= 200:
                    found += 1; self.assertTrue(d1[0]); self.assertFalse(d2[0])
        self.assertGreater(found, 0)

    def test_cap_does_not_touch_other_paths(self):
        clock = Clock(1000.0); act = rows(12)
        cap = SlackController(dict(BASE, warning_wait_fraction=.5, bypass_bound_fraction=.95), 16, 200, clock)
        self.assertEqual(cap.admission_decision(act, View('c', 0, 100, 0, 1, 999.9))[1], 'warning_non_high')
        self.assertEqual(cap.admission_decision(act, View('h', 0, 100, 0, 0, 1000.0))[1], 'risk_bound')
        self.assertEqual(cap.admission_decision(act, View('r', 5, 300, 0, 1, 0.0))[1], 'warning_non_high')
        with self.assertRaises(ValueError):
            SlackController(dict(BASE, bypass_bound_fraction=.95), 16, 200)

    def test_protect_high_victims(self):
        base = SlackController(dict(BASE, warning_wait_fraction=.5), 16, 200, Clock())
        prot = SlackController(dict(BASE, warning_wait_fraction=.5, protect_high_victims=True), 16, 200, Clock())
        long_high = View('h', 300, 1200, 76, 0, 0.0)          # long HIGH, many blocks -> low density
        short_def = View('d', 20, 190, 12, 1, 1.0)
        rows = [long_high, short_def]
        self.assertEqual(base.choose_victim(rows, 14.0).request_id, 'h')   # reproduces the diagnosed pathology
        self.assertEqual(prot.choose_victim(rows, 14.0).request_id, 'd')
        only_high = [long_high, View('h2', 5, 100, 7, 0, 2.0)]
        self.assertIsNotNone(prot.choose_victim(only_high, 14.0))
        self.assertEqual(prot.choose_victim(only_high, 14.0), base.choose_victim(only_high, 14.0))

    def test_protect_default_identical(self):
        rng = random.Random(3)
        a = self.make(0.5, 300, Clock()); b = SlackController(dict(BASE, warning_wait_fraction=.5, protect_high_victims=False), 16, 300, Clock())
        for _ in range(1000):
            rows = [View(f'r{i}', rng.randint(0, 400), rng.randint(20, 1500), rng.randint(0, 90), rng.randint(0, 2), rng.random()) for i in range(rng.randint(1, 15))]
            self.assertEqual(a.choose_victim(rows, 14.0), b.choose_victim(rows, 14.0))

    def test_validation(self):
        for bad in (0, -0.1, 1.5, True):
            with self.assertRaises(ValueError): self.make(bad)
        with self.assertRaises(ValueError):
            SlackController(dict(BASE, revision_arm='rank_no_slack', warning_wait_fraction=.5), 16, 200)


class Step9Tests(unittest.TestCase):
    def ctl(self, clock, **kw):
        return SlackController(dict(BASE, warning_wait_fraction=.5, protect_high_victims=True, **kw), 16, 200, clock)

    def test_switches_off_identical_to_v3_on_random_states(self):
        rng = random.Random(7); clock = Clock(1000.0)
        a = self.ctl(clock); b = self.ctl(clock, requeue_wait_budget=False, adaptive_gate_window_s=None)
        for _ in range(3000):
            clock.t = 1000 + rng.random() * 50
            act = [View(f'a{i}', rng.randint(0, 300), rng.randint(20, 900), rng.randint(1, 60), rng.randint(0, 2), rng.random() * 1000)
                   for i in range(rng.randint(0, 20))]
            cand = View('c', rng.choice([0, 0, 3]), rng.randint(20, 900), rng.randint(0, 5), rng.randint(0, 2), 1000 - rng.random() * 8)
            self.assertEqual(a.admission_decision(act, cand), b.admission_decision(act, cand))
            if act:
                self.assertEqual(a.choose_victim(act, 20.0), b.choose_victim(act, 20.0))

    def test_requeue_budget(self):
        clock = Clock(1000.0); c = self.ctl(clock, requeue_wait_budget=True); act = rows(12)
        low = View('r', 5, 300, 0, 2, 0.0)
        self.assertEqual(c.admission_decision(act, low)[1], 'warning_non_high')   # timer starts
        clock.t = 1004.9
        self.assertEqual(c.admission_decision(act, low)[1], 'warning_non_high')
        clock.t = 1005.0
        self.assertEqual(c.admission_decision(act, low)[1], 'risk_bound_after_wait_budget')
        off = self.ctl(Clock(2000.0))
        self.assertEqual(off.admission_decision(act, low)[1], 'warning_non_high')

    def test_requeue_timer_resets_on_victim(self):
        clock = Clock(1000.0); c = self.ctl(clock, requeue_wait_budget=True); act = rows(12)
        r = View('a0', 5, 300, 13, 2, 0.0)
        c.admission_decision(act, r); clock.t = 1010.0
        c.choose_victim([r, *act[1:]], 20.0)     # a0 (LOW, lowest density) preempted again -> timer reset
        self.assertEqual(c.admission_decision(act, r)[1], 'warning_non_high')

    def test_adaptive_gate(self):
        clock = Clock(1000.0); c = self.ctl(clock, adaptive_gate_window_s=5.0); act = rows(12)
        cand = View('c', 0, 100, 0, 1, 1000.0)
        self.assertEqual(c.admission_decision(act, cand)[1], 'risk_bound')        # no recent recovery: no gate
        c.choose_victim(act, 20.0)
        self.assertEqual(c.admission_decision(act, cand)[1], 'warning_non_high')  # recovery just happened
        clock.t = 1005.0
        self.assertEqual(c.admission_decision(act, View('c', 0, 100, 0, 1, 1005.0))[1], 'warning_non_high')
        clock.t = 1005.1
        self.assertEqual(c.admission_decision(act, View('c', 0, 100, 0, 1, 1005.1))[1], 'risk_bound')
        high = View('h', 0, 100, 0, 0, 1000.0)
        self.assertEqual(c.admission_decision(act, high)[1], 'risk_bound')

    def test_deferral_state_uses_switches(self):
        clock = Clock(1000.0); c = self.ctl(clock, adaptive_gate_window_s=5.0); act = rows(12)
        st = SlackDeferralState(c)
        d, _ = st.decide(lambda: act, lambda: View('x', 0, 100, 0, 1, 1000.0))
        self.assertEqual(d[1], 'risk_bound')


class Step11Tests(unittest.TestCase):
    def ctl(self, clock, **kw):
        return SlackController(dict(BASE, warning_wait_fraction=.5, protect_high_victims=True, **kw), 16, 200, clock)

    def test_switches_off_identical_to_v4(self):
        rng = random.Random(11); clock = Clock(1000.0)
        a = self.ctl(clock); b = self.ctl(clock, inflight_protection=False, victim_cooldown_s=None)
        for _ in range(3000):
            act = [View(f'a{i}', rng.randint(0, 300), rng.randint(20, 900), rng.randint(0, 60), rng.randint(0, 2), rng.random())
                   for i in range(rng.randint(1, 20))]
            self.assertEqual(a.choose_victim(act, 20.0), b.choose_victim(act, 20.0))

    def test_inflight_prefers_unstarted_then_fewest_tokens(self):
        c = self.ctl(Clock(), inflight_protection=True)
        low_busy = View('l', 300, 400, 30, 2, 0.0)       # LOW, lowest density, in flight
        dflt_new = View('d', 0, 400, 25, 1, 1.0)          # DEFAULT, not started
        high = View('h', 0, 100, 5, 0, 2.0)
        self.assertEqual(c.choose_victim([low_busy, dflt_new, high], 20.0).request_id, 'd')
        d2 = View('d2', 5, 400, 25, 1, 1.0)
        self.assertEqual(c.choose_victim([low_busy, d2, high], 20.0).request_id, 'd2')
        self.assertEqual(c.choose_victim([high], 20.0).request_id, 'h')   # only HIGH holds blocks
        off = self.ctl(Clock())
        self.assertEqual(off.choose_victim([low_busy, dflt_new, high], 20.0).request_id, 'l')

    def test_cooldown(self):
        clock = Clock(1000.0)
        c = SlackController(dict(BASE, warning_wait_fraction=.5, protect_high_victims=True, requeue_wait_budget=True,
                                 victim_cooldown_s=10.0), 16, 10000, clock)
        allowed = c.admission_decision(rows(3), View('r', 5, 300, 0, 2, 0.0))   # resumed request admitted
        self.assertTrue(allowed[0])
        held = View('r', 5, 300, 20, 2, 0.0); other = View('o', 50, 300, 20, 1, 0.0)
        clock.t = 1009.9
        self.assertEqual(c.choose_victim([held, other], 20.0).request_id, 'o')   # r cooling down
        clock.t = 1010.0
        self.assertEqual(c.choose_victim([held, other], 20.0).request_id, 'r')
        clock.t = 1000.0; c.admission_decision(rows(3), View('r', 6, 300, 0, 2, 0.0))
        self.assertEqual(c.choose_victim([held], 20.0).request_id, 'r')          # only option

    def test_validation(self):
        with self.assertRaises(ValueError): self.ctl(Clock(), inflight_protection=1)
        with self.assertRaises(ValueError): self.ctl(Clock(), victim_cooldown_s=0)


if __name__ == '__main__':
    unittest.main(verbosity=1)
