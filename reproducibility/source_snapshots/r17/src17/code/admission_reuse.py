"""Reuse a frozen active view only within consecutive rejected candidates.

The caller owns the synchronous schedule lifetime and must invalidate before
any possible active-set or KV mutation. No incremental floating-point sums,
candidate-result cache, queue skipping, or controller parameter changes.
"""


class ConsecutiveDeferralState:
    def __init__(self, controller):
        self.controller = controller
        self.invalidate()

    def invalidate(self):
        self.active = None
        self.current_risk = None

    def decide(self, active_factory, candidate_factory):
        if self.active is None:
            self.active = tuple(active_factory())
            if self.controller.admission != 'current':
                self.current_risk = self.controller.risk(self.active)
        # Preserve the candidate read, including native request lookup behavior.
        candidate = candidate_factory()
        count = len(self.active)
        c = self.controller
        if (c.admission != 'current' and c.recovery
                and self.current_risk > c.low * c.capacity and candidate.priority > 0):
            decision = False, 'warning_non_high', self.current_risk
        else:
            # Delegate the original arithmetic and exact accumulation order.
            decision = c.admission_decision(self.active, candidate)
        if decision[0]:
            # Native allocation follows after return and may fail or succeed.
            # Invalidate for either outcome, before it can change any state.
            self.invalidate()
        return decision, count
