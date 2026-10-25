"""Orthogonal warning/recovery factors; Gaussian candidate admission is fixed."""
from admission_reuse import ConsecutiveDeferralState
from policy_core import Controller


class FactorialController(Controller):
    def __init__(self, config, block_size, capacity):
        if config.get("policy") != "C4":
            raise ValueError("Factorial arms use the same C4 Gaussian formulation")
        for key in ("warning_gate_enabled", "proactive_recovery_enabled"):
            if type(config.get(key)) is not bool:
                raise ValueError("Explicit Boolean factor required: " + key)
        super().__init__(config, block_size, capacity)
        self.warning_gate_enabled = config["warning_gate_enabled"]
        self.recovery = config["proactive_recovery_enabled"]

    def admission_decision(self, active, candidate):
        # Compute the identical risk even when the warning factor is off.
        # Only the class gate is toggled, not prediction or Gaussian admission.
        current = self.risk(active)
        if (self.warning_gate_enabled and current > self.low * self.capacity
                and candidate.priority > 0):
            return False, "warning_non_high", current
        bound = self.bounds([*active, candidate])["chance"]
        return bound <= self.capacity, "risk_bound", bound


class FactorialDeferralState(ConsecutiveDeferralState):
    def decide(self, active_factory, candidate_factory):
        if self.active is None:
            self.active = tuple(active_factory())
            self.current_risk = self.controller.risk(self.active)
        candidate = candidate_factory()
        count = len(self.active)
        controller = self.controller
        if (controller.warning_gate_enabled
                and self.current_risk > controller.low * controller.capacity
                and candidate.priority > 0):
            decision = False, "warning_non_high", self.current_risk
        else:
            decision = controller.admission_decision(self.active, candidate)
        if decision[0]:
            self.invalidate()
        return decision, count
