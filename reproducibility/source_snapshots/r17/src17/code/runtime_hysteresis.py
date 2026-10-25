"""R14 scheduler adapter: admission ceiling + shallow recovery target (2026-10-15).
Reuses the R13 CeilingScheduler (itself the audited protected adapter) and overrides only the proactive-recovery loop so
that eviction stops at recovery_target*K instead of theta_low*K. The loop is otherwise a verbatim copy of
GuardrailScheduler._gr_recover (same trigger, singleton exemption, victim choice, native preemption, events)."""
import hashlib
from pathlib import Path
import hysteresis_policy
from hysteresis_policy import HysteresisController
from runtime_ceiling import CeilingScheduler


class HysteresisScheduler(CeilingScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config.additional_config['guardrail']
        self._gr = HysteresisController(config, self.block_size, self._gr.capacity)
        self._gr.emit_evidence = self._gr_emit
        self._gr_emit('hysteresis_init', version='r14_hysteresis_v1', admission_ceiling=self._gr.ceiling,
                      recovery_target=self._gr.target, recovery_trigger=self._gr.high, warning_threshold=self._gr.low,
                      inflight_protection=self._gr.inflight, high_decode_token_budget=self._gr_budget_cap,
                      policy_sha256=hashlib.sha256(Path(hysteresis_policy.__file__).read_bytes()).hexdigest())

    def _gr_recover(self, timestamp):
        victims = []
        if not self._gr.recovery:
            return victims
        rows = [self._gr_view(r) for r in self.running]
        bound = self._gr.risk(rows)
        if bound <= self._gr.high * self._gr.capacity:
            return victims
        self._gr_emit('danger', bound=bound, active=len(rows))
        while len(rows) > 1 and bound > self._gr.target * self._gr.capacity:      # R14: shallow stop target
            chosen = self._gr.choose_victim(rows, self._gr_decode_ms)
            if chosen is None:
                break
            victim = next(r for r in self.running if r.request_id == chosen.request_id)
            self._gr_emit('slack_victim', request_id=victim.request_id,
                          density=self._gr.density(chosen, self._gr_decode_ms),
                          bound_before=bound, decode_ms=self._gr_decode_ms)
            self.running.remove(victim)
            self._preempt_request(victim, timestamp)
            victims.append(victim)
            rows = [self._gr_view(r) for r in self.running]
            bound = self._gr.risk(rows)
        self._gr_emit('recovery_end', bound=bound, victims=[r.request_id for r in victims],
                      singleton_exemption=len(rows) == 1)
        return victims
