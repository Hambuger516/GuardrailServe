"""R13 admission ceiling (2026-10-15). Default-off extension of the protected controller (SlackController).

Diagnosis (R12 REAL, Azure conversation trace, long prompts ~1.2k tokens): 92% of proactive-recovery episodes start in
the scheduling step right after an admission; each episode evicts 4-5 requests that are already decoding (99% have
emitted output), each invalidating ~1.3k KV tokens. This is the admission/recovery threshold overlap of Section 4.5.2:
admission accepts up to K while recovery starts above theta_high*K, so one long-prompt admission can place the active
set in the recovery zone and the recovery then discards several in-flight requests' KV caches.

Change: a non-HIGH candidate (new or readmitted) is admitted only if its joint Gaussian bound stays at or below
admission_ceiling * K (registered value: theta_high = 0.95). HIGH candidates keep the unchanged K limit, so HIGH
admissions may still trigger recovery, which (with F1) evicts non-HIGH requests by design. Everything else -- forecast,
warning gate, wait budget, recovery, F1/F2/F3a victim rules -- is inherited unchanged.
"""
import time
from slack_policy import SlackController


class CeilingController(SlackController):
    def __init__(self, config, block_size, capacity, clock=time.time):
        super().__init__(config, block_size, capacity, clock)
        c = config.get('admission_ceiling')
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not self.low < float(c) <= 1:
            raise ValueError('admission_ceiling must lie in (theta_low, 1]')
        if self.admission != 'chance' or not self.recovery:
            raise ValueError('the admission ceiling is registered only for Gaussian admission with recovery (C4)')
        self.ceiling = float(c)

    def _screen(self, active, candidate, current):
        allow, reason, bound = super()._screen(active, candidate, current)
        if allow and candidate.priority > 0 and bound > self.ceiling * self.capacity:
            return False, 'admission_ceiling', bound
        return allow, reason, bound
