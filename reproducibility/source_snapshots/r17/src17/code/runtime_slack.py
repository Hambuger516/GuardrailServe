"""Scheduler adapter for the slack-bounded warning gate (step3, 2026-10-08)."""
import hashlib
from pathlib import Path
import slack_policy
from slack_policy import SlackController, SlackDeferralState
import time
import vllm.v1.core.sched.scheduler as upstream
import runtime_scheduler
from runtime_revision import RevisionScheduler
from runtime_scheduler import GuardrailScheduler

# step5 F2: same audited patched schedule source plus ONE line that lets the adapter
# lower the per-step token budget. Used only when high_decode_token_budget is set.
_BUDGET_LINE = '    token_budget = self.max_num_scheduled_tokens\n'
if runtime_scheduler.PATCHED_SOURCE.count(_BUDGET_LINE) != 1:
    raise RuntimeError('vLLM schedule source does not match the audited token-budget line')
BUDGET_SOURCE = runtime_scheduler.PATCHED_SOURCE.replace(
    _BUDGET_LINE, _BUDGET_LINE + '    token_budget = self._gr_token_budget(token_budget)\n')
_scope = dict(vars(upstream))
exec(compile(BUDGET_SOURCE, '<guardrail-vllm-0.29-schedule-budget>', 'exec'), _scope)
_schedule_budget = _scope['schedule']


class SlackScheduler(RevisionScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config.additional_config['guardrail']
        if self._revision_variant != 'reuse' or config['revision_arm'] != 'guardrail':
            raise ValueError('slack gate is registered only for guardrail/reuse')
        self._gr = SlackController(config, self.block_size, self._gr.capacity)
        self._gr.emit_evidence = self._gr_emit
        cap = config.get('high_decode_token_budget')
        if cap is not None and (type(cap) is not int or not 64 <= cap <= self.max_num_scheduled_tokens):
            raise ValueError('high_decode_token_budget must be an int in [64, max_num_scheduled_tokens]')
        self._gr_budget_cap = cap
        self._gr_budget_steps = 0
        if cap is not None:
            (self._gr_dir / 'patched_schedule_budget.py').write_text(BUDGET_SOURCE)
        self._gr_emit('slack_init', version='step11_slack_warning_v5',
                      protect_high_victims=self._gr.protect_high, high_decode_token_budget=cap,
                      requeue_wait_budget=self._gr.requeue_budget, adaptive_gate_window_s=self._gr.gate_window,
                      inflight_protection=self._gr.inflight, victim_cooldown_s=self._gr.cooldown,
                      warning_wait_fraction=self._gr.wait_fraction, ttft_ms=self._gr.ttft_ms,
                      bypass_bound_fraction=self._gr.bypass_cap,
                      policy_sha256=hashlib.sha256(Path(slack_policy.__file__).read_bytes()).hexdigest())

    def _gr_token_budget(self, budget):
        """F2: while any HIGH request is decoding, cap the step's token budget."""
        cap = self._gr_budget_cap
        if cap is None or budget <= cap:
            return budget
        if any(r.priority == 0 and r.num_output_tokens > 0 for r in self.running):
            self._gr_budget_steps += 1
            return cap
        return budget

    def schedule(self, *args, **kwargs):
        if self._reuse_window is not None:
            raise RuntimeError('Nested schedule invalidates state lifetime')
        self._reuse_window = SlackDeferralState(self._gr)
        try:
            if self._gr_budget_cap is None:
                return GuardrailScheduler.schedule(self, *args, **kwargs)
            saved = runtime_scheduler._schedule
            runtime_scheduler._schedule = _schedule_budget   # one scheduler per engine process
            try:
                result = GuardrailScheduler.schedule(self, *args, **kwargs)
            finally:
                runtime_scheduler._schedule = saved
            if self.current_step % 500 == 0:
                self._gr_emit('budget_cap_steps', count=self._gr_budget_steps)
            return result
        finally:
            self._reuse_window = None
