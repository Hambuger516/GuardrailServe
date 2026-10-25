"""R15 scheduler adapter (frozen 2026-10-16). Reuses the audited protected adapter (SlackScheduler: same patched
schedule source, F2 hook, deferral lifetime) and replaces only its controller with EstimatorController built from the
same configuration, plus:
  * proactive_recovery_hook=False: the proactive-recovery entry returns no victims. The would-be episode is evaluated
    on views only (no preemption, no controller bookkeeping) and recorded as 'danger_shadow'. Warning gate, wait
    budget, Gaussian/alternative screen, F2 and vLLM's native capacity preemption are untouched.
  * per-step 'r15_step' evidence (held blocks, M/R/U on the running set, active count) and cumulative counters
    (admission reasons, F2 cap steps, controller inner time, shadow checks/disagreements) on every 'schedule' event.
With control_estimator='chance' and proactive_recovery_hook=True the decisions equal SlackScheduler's."""
import hashlib, time
from pathlib import Path
import estimator_policy
from estimator_policy import EstimatorController, ESTIMATORS
from runtime_slack import SlackScheduler


class R15Scheduler(SlackScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config.additional_config['guardrail']
        self._gr = EstimatorController(config, self.block_size, self._gr.capacity)
        self._gr.emit_evidence = self._gr_emit
        self._gr.held_fn = lambda: self._gr.capacity - self.kv_cache_manager.block_pool.get_num_free_blocks()
        self._r15_reasons = {}
        self._r15_inner_recover_s = 0.0
        self._r15_danger_shadow = 0
        self._gr_emit('r15_init', version='r15_v1', control_estimator=self._gr.estimator,
                      proactive_recovery_hook=self._gr.hook, protect_high_victims=self._gr.protect_high,
                      warning_wait_fraction=self._gr.wait_fraction, warning_threshold=self._gr.low,
                      recovery_trigger=self._gr.high, high_decode_token_budget=self._gr_budget_cap,
                      inflight_protection=self._gr.inflight, capacity_blocks=self._gr.capacity,
                      policy_sha256=hashlib.sha256(Path(estimator_policy.__file__).read_bytes()).hexdigest())

    def _gr_emit(self, event, **fields):
        if event == 'admission_check':
            phase = 'warmup' if fields['request_id'].startswith('warmup-') else 'measurement'
            key = f"{phase}:{fields['reason']}:{'allow' if fields['allow'] else 'defer'}"
            self._r15_reasons[key] = self._r15_reasons.get(key, 0) + 1
        elif event == 'schedule' and hasattr(self, '_r15_reasons'):
            fields['r15_counters'] = dict(reasons=dict(self._r15_reasons), budget_cap_steps=self._gr_budget_steps,
                                          inner_admission_s=getattr(self._gr, 'inner_admission_s', 0.0),
                                          inner_recover_s=self._r15_inner_recover_s,
                                          shadow_checks=self._gr.shadow_checks,
                                          shadow_disagreements=self._gr.shadow_disagreements,
                                          danger_shadow=self._r15_danger_shadow)
        return super()._gr_emit(event, **fields)

    def _gr_recover(self, timestamp):
        start = time.perf_counter()
        rows = [self._gr_view(r) for r in self.running]
        b = self._gr.bounds(rows)
        self._gr_emit('r15_step', held=self._gr.held_fn(), active=len(rows),
                      **{e: b[e] for e in ESTIMATORS}, point_minus_base=b['point'] - b['current'])
        if self._gr.hook:
            victims = super()._gr_recover(timestamp)
        else:
            victims = []
            bound = b[self._gr.admission]
            if bound > self._gr.high * self._gr.capacity:
                # Pure replay of the stop loop on views: no preemption, no controller bookkeeping.
                sim, would = list(rows), []
                while len(sim) > 1 and bound > self._gr.low * self._gr.capacity:
                    chosen = self._gr._choose_victim(sim, self._gr_decode_ms)
                    if chosen is None:
                        break
                    would.append(dict(request_id=chosen.request_id, priority=chosen.priority,
                                      generated=chosen.generated, blocks=chosen.allocated_blocks))
                    sim = [r for r in sim if r.request_id != chosen.request_id]
                    bound = self._gr.risk(sim)
                self._r15_danger_shadow += 1
                self._gr_emit('danger_shadow', bound=b[self._gr.admission], active=len(rows), held=self._gr.held_fn(),
                              would_evict=would)
        self._r15_inner_recover_s += time.perf_counter() - start
        return victims
