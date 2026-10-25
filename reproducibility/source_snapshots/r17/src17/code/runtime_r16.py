"""R16 scheduler adapter (frozen 2026-10-16). Extends the audited R15 adapter (R15Scheduler -> SlackScheduler: same
patched schedule source, F2 hook, deferral lifetime, evidence) with:
  * the R16 controller (protected / hyst / adapt regime; see r16_policy.py),
  * a proactive-recovery loop whose stop level is the controller's current target (0.80K or 0.90K). The loop is a
    verbatim copy of GuardrailScheduler._gr_recover except for that stop line (CPU test), as in R14,
  * the F2 'pressure' rule: the HIGH-decode step budget applies only while held KV >= theta_low * K.
Evidence: per-step 'r15_step' (now with regime and EWMA), 'r16_regime' on every switch, cumulative R16 counters on
'schedule' events (regime switches, steps in the long regime, F2 caps skipped because pressure was low)."""
import hashlib, time
from pathlib import Path
import r16_policy
from r16_policy import R16Controller
from estimator_policy import ESTIMATORS
from runtime_r15 import R15Scheduler


class R16Scheduler(R15Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config.additional_config['guardrail']
        self._gr = R16Controller(config, self.block_size, self._gr.capacity)
        self._gr.emit_evidence = self._gr_emit
        self._gr.held_fn = lambda: self._gr.capacity - self.kv_cache_manager.block_pool.get_num_free_blocks()
        if (self._gr.f2 == 'off') != (self._gr_budget_cap is None):
            raise ValueError("f2='off' exactly when high_decode_token_budget is None")
        self._r16_long_steps = 0
        self._r16_cap_skipped = 0
        self._gr_emit('r16_init', version='r16_v1', mode=self._gr.mode, f2=self._gr.f2, high_decode_token_budget=self._gr_budget_cap,
                      hyst_ceiling=r16_policy.HYST_CEILING, hyst_target=r16_policy.HYST_TARGET, tau_s=r16_policy.TAU_S,
                      on_tokens=r16_policy.ON_TOKENS, off_tokens=r16_policy.OFF_TOKENS, start_long=self._gr.long,
                      protect_high_victims=self._gr.protect_high, warning_wait_fraction=self._gr.wait_fraction,
                      policy_sha256=hashlib.sha256(Path(r16_policy.__file__).read_bytes()).hexdigest())

    def _gr_emit(self, event, **fields):
        if event == 'schedule' and hasattr(self, '_r16_cap_skipped'):
            fields['r16_counters'] = dict(switches=self._gr.switches, long_steps=self._r16_long_steps,
                                          cap_skipped_low_pressure=self._r16_cap_skipped, long=self._gr.long)
        return super()._gr_emit(event, **fields)

    def _gr_token_budget(self, budget):
        if self._gr.f2 == 'regime' and self._gr.long:                 # R17: F2 follows the regime
            cap = self._gr_budget_cap
            if budget > cap and any(r.priority == 0 and r.num_output_tokens > 0 for r in self.running):
                self._r16_cap_skipped += 1
            return budget
        if self._gr.f2 == 'pressure' and self._gr.held_fn() < self._gr.low * self._gr.capacity:
            cap = self._gr_budget_cap
            if budget > cap and any(r.priority == 0 and r.num_output_tokens > 0 for r in self.running):
                self._r16_cap_skipped += 1
            return budget
        return super()._gr_token_budget(budget)

    def _gr_recover(self, timestamp):
        start = time.perf_counter()
        rows = [self._gr_view(r) for r in self.running]
        if self._gr.observe(rows, time.monotonic()):
            self._gr_emit('r16_regime', long=self._gr.long, ewma=self._gr.ewma, active=len(rows))
        self._r16_long_steps += self._gr.long
        b = self._gr.bounds(rows)
        self._gr_emit('r15_step', held=self._gr.held_fn(), active=len(rows), long=self._gr.long, ewma=self._gr.ewma,
                      **{e: b[e] for e in ESTIMATORS}, point_minus_base=b['point'] - b['current'])
        victims = self._r16_recover_loop(timestamp)
        self._r15_inner_recover_s += time.perf_counter() - start
        return victims

    def _r16_recover_loop(self, timestamp):
        victims = []
        if not self._gr.recovery:
            return victims
        rows = [self._gr_view(r) for r in self.running]
        bound = self._gr.risk(rows)
        if bound <= self._gr.high * self._gr.capacity:
            return victims
        self._gr_emit('danger', bound=bound, active=len(rows))
        # Hysteresis is refreshed after EACH actual release. A singleton is
        # allowed to progress past a watermark (never past physical capacity),
        # avoiding repeated recomputation of the only runnable request.
        while len(rows) > 1 and bound > self._gr.target() * self._gr.capacity:
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
