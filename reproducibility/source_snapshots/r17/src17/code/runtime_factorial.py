"""Four-factor-arm adapter on the validated synchronous state-reuse path.

All four arms use this class and identical instrumentation. The two explicit
Boolean factors are the only policy differences. Native allocation/repair,
victim density, output preservation, and reuse invalidation are inherited.
"""
import hashlib
from pathlib import Path
import factorial_policy
from factorial_policy import FactorialController, FactorialDeferralState
from runtime_scheduler import GuardrailScheduler
from runtime_scheduler_reuse import ReuseGuardrailScheduler


class FactorialScheduler(ReuseGuardrailScheduler):
    def _gr_controller_config(self, config):
        # The base constructor creates the original temporary C4 controller;
        # replacing it after initialization does not execute a schedule call.
        return config

    def __init__(self, *args, **kwargs):
        self._factor_reason_counts = {
            phase: {"allowed": 0, "warning_non_high": 0, "risk_bound": 0}
            for phase in ("warmup", "measurement")
        }
        super().__init__(*args, **kwargs)
        config = self.vllm_config.additional_config["guardrail"]
        self._gr = FactorialController(config, self.block_size, self._gr.capacity)
        self._gr_emit(
            "factorial_init", version="factorial20260927_v1",
            warning_gate_enabled=self._gr.warning_gate_enabled,
            proactive_recovery_enabled=self._gr.recovery,
            admission=self._gr.admission, horizon=self._gr.k,
            epsilon=self._gr.epsilon, low=self._gr.low, high=self._gr.high,
            policy_sha256=hashlib.sha256(
                Path(factorial_policy.__file__).read_bytes()).hexdigest(),
        )

    def schedule(self, *args, **kwargs):
        if self._reuse_window is not None:
            raise RuntimeError("Nested schedule invalidates active-view lifetime")
        self._reuse_window = FactorialDeferralState(self._gr)
        try:
            return GuardrailScheduler.schedule(self, *args, **kwargs)
        finally:
            self._reuse_window = None

    def _gr_emit(self, event, **fields):
        if event == "admission_check":
            phase = ("warmup" if fields["request_id"].startswith("warmup-")
                     else "measurement")
            reason = "allowed" if fields["allow"] else fields["reason"]
            if reason not in self._factor_reason_counts[phase]:
                raise ValueError("Unexpected factorial decision reason")
            self._factor_reason_counts[phase][reason] += 1
        elif event == "schedule":
            fields["cumulative_factorial_reason_counts"] = self._factor_reason_counts
        return super()._gr_emit(event, **fields)
