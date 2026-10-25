"""Slack-bounded warning gate (step3, 2026-10-08).

Single change relative to the registered revision 'guardrail' arm (C4, reuse):
the HIGH-only warning gate may defer a non-HIGH candidate that has not yet
received its first token only while its engine-side wait is below
``warning_wait_fraction`` x its class TTFT target. After that wait budget the
candidate skips the warning gate but must still pass the unchanged Gaussian
risk-bound screen; native allocation remains the final physical check.
Requests that already emitted tokens (e.g. preempted and requeued) keep the
original warning behaviour. Recovery, ranking and all parameters are unchanged.
With ``warning_wait_fraction=None`` the controller is identical to the parent.
"""
import math, time
from revision_policy import RevisionController, RevisionDeferralState


class SlackController(RevisionController):
    def __init__(self, config, block_size, capacity, clock=time.time):
        super().__init__(config, block_size, capacity)
        fraction = config.get('warning_wait_fraction')
        if fraction is not None and (isinstance(fraction, bool) or not 0 < float(fraction) <= 1):
            raise ValueError('warning_wait_fraction must be in (0, 1] or None')
        if fraction is not None and self.arm != 'guardrail':
            raise ValueError('wait budget only applies to the registered guardrail arm')
        self.wait_fraction = None if fraction is None else float(fraction)
        cap = config.get('bypass_bound_fraction', 1.0)
        if isinstance(cap, bool) or not 0 < float(cap) <= 1:
            raise ValueError('bypass_bound_fraction must be in (0, 1]')
        if float(cap) != 1.0 and fraction is None:
            raise ValueError('bypass cap requires a wait budget')
        # step4: admissions that skip the warning gate must not themselves push the
        # Gaussian bound past this fraction of K (e.g. theta_high, the recovery trigger).
        self.bypass_cap = float(cap)
        self.ttft_ms = list(config.get('ttft_ms', [500, 2000, 10000]))
        # step5 F1: proactive recovery never selects HIGH victims unless only HIGH holds blocks
        # (mirrors vLLM's native lowest-priority-first preemption order).
        self.protect_high = config.get('protect_high_victims', False)
        if type(self.protect_high) is not bool:
            raise ValueError('protect_high_victims must be bool')
        if len(self.ttft_ms) != 3 or any(t <= 0 for t in self.ttft_ms):
            raise ValueError('three positive TTFT targets required')
        self.clock = clock
        # step9 V1: requeued requests (already emitted tokens) also get the class wait budget,
        # measured from when they are first seen waiting again.
        self.requeue_budget = config.get('requeue_wait_budget', False)
        if type(self.requeue_budget) is not bool:
            raise ValueError('requeue_wait_budget must be bool')
        if self.requeue_budget and fraction is None:
            raise ValueError('requeue budget requires a wait budget')
        self._requeue_since = {}
        # step9 V2: warning gate active only within W seconds after a proactive-recovery victim.
        window = config.get('adaptive_gate_window_s')
        if window is not None and (isinstance(window, bool) or not 0 < float(window) <= 3600):
            raise ValueError('adaptive_gate_window_s must be in (0, 3600] or None')
        self.gate_window = None if window is None else float(window)
        self._last_recovery = -math.inf
        # step11 F3a / F3b (victim selection)
        self.inflight = config.get('inflight_protection', False)
        if type(self.inflight) is not bool:
            raise ValueError('inflight_protection must be bool')
        cool = config.get('victim_cooldown_s')
        if cool is not None and (isinstance(cool, bool) or not 0 < float(cool) <= 600):
            raise ValueError('victim_cooldown_s must be in (0, 600] or None')
        self.cooldown = None if cool is None else float(cool)
        self._resumed_at = {}

    def warning_zone(self, current, candidate):
        """Original warning condition (unchanged arithmetic and comparison)."""
        zone = self.recovery and current > self.low * self.capacity and candidate.priority > 0
        if zone and self.gate_window is not None:
            return self.clock() - self._last_recovery <= self.gate_window
        return zone

    def wait_budget_spent(self, candidate):
        if self.wait_fraction is None:
            return False
        if candidate.generated > 0:
            if not self.requeue_budget:
                return False
            now = self.clock()
            waited = now - self._requeue_since.setdefault(candidate.request_id, now)
            return waited >= self.wait_fraction * self.ttft_ms[candidate.priority] / 1000
        waited = self.clock() - candidate.arrival_time
        if not -1.0 <= waited <= 3600.0:
            # Engine arrival_time must share the wall clock used here.
            raise RuntimeError(f'arrival clock mismatch: waited={waited!r}s')
        return waited >= self.wait_fraction * self.ttft_ms[candidate.priority] / 1000

    def screen(self, active, candidate, current):
        decision = self._screen(active, candidate, current)
        if decision[0] and self.requeue_budget:
            self._requeue_since.pop(candidate.request_id, None)
        if decision[0] and self.cooldown is not None and candidate.generated > 0:
            self._resumed_at[candidate.request_id] = self.clock()
        return decision

    def _screen(self, active, candidate, current):
        if self.warning_zone(current, candidate):
            if not self.wait_budget_spent(candidate):
                return False, 'warning_non_high', current
            bound = self.bounds([*active, candidate])[self.admission]
            if self.bypass_cap != 1.0:
                return bound <= self.bypass_cap * self.capacity, 'risk_bound_after_wait_budget_capped', bound
            return bound <= self.capacity, 'risk_bound_after_wait_budget', bound
        bound = self.bounds([*active, candidate])[self.admission]
        return bound <= self.capacity, 'risk_bound', bound

    def admission_decision(self, active, candidate):
        if self.arm != 'guardrail' or self.admission == 'current':
            return super().admission_decision(active, candidate)
        return self.screen(active, candidate, self.risk(active))


    def choose_victim(self, rows, decode_ms):
        chosen = self._choose_victim(rows, decode_ms)
        if chosen is not None:
            self._last_recovery = self.clock()
            self._requeue_since.pop(chosen.request_id, None)
        return chosen

    def _choose_victim(self, rows, decode_ms):
        pool = rows
        if self.protect_high:
            non_high = [r for r in rows if r.priority > 0 and r.allocated_blocks > 0]
            if non_high:
                pool = non_high
        if self.cooldown is not None:
            now = self.clock()
            warm = [r for r in pool if r.allocated_blocks > 0
                    and now - self._resumed_at.get(r.request_id, -math.inf) >= self.cooldown]
            if warm:
                pool = warm
        if self.inflight:
            eligible = [r for r in pool if r.allocated_blocks > 0]
            if eligible:
                fresh = [r for r in eligible if r.generated == 0]
                if fresh:
                    return super().choose_victim(fresh, decode_ms)
                fewest = min(r.generated for r in eligible)
                return super().choose_victim([r for r in eligible if r.generated == fewest], decode_ms)
        return super().choose_victim(pool, decode_ms)


class SlackDeferralState(RevisionDeferralState):
    """Reuse the frozen active view/risk exactly as the parent; only the
    warning shortcut consults the per-candidate wait budget."""

    def decide(self, active_factory, candidate_factory):
        c = self.controller
        if not isinstance(c, SlackController) or c.arm != 'guardrail' or c.admission == 'current':
            return super().decide(active_factory, candidate_factory)
        if self.active is None:
            self.active = tuple(active_factory())
            self.current_risk = c.risk(self.active)
        candidate = candidate_factory()
        count = len(self.active)
        decision = c.screen(self.active, candidate, self.current_risk)
        if decision[0]:
            self.invalidate()
        return decision, count
