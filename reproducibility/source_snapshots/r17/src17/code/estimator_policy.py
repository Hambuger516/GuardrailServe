"""R15 control-quantity and recovery-hook controller (frozen 2026-10-16). Default-off extension of the protected
controller (SlackController, C4 guardrail/reuse).

control_estimator selects the quantity the controller compares with its thresholds, everywhere it uses one (admission
screen, warning-zone test at theta_low*K, proactive-recovery trigger at theta_high*K and stop at theta_low*K):
  'chance'          R(A) = b + mean growth + z * sqrt(variance)   (unchanged paper bound; identical decisions)
  'point'           M(A) = b + mean growth
  'hard'            U(A) = b + sum_i (ceil((q_i + k)/B) - ceil(q_i/B))   (every request emits k tokens, none completes)
  'min_chance_hard' min(R(A), U(A))
with b = sum_i max(ceil(q_i/B), allocated_i). Only active-set geometry (generated, resident tokens, allocated blocks)
enters; no future output length is read.

The proactive-recovery hook switch lives in the scheduler adapter (runtime_r15.R15Scheduler); self.recovery is never
changed here, so the warning-zone semantics (which read self.recovery) are untouched.

Shadow evidence: for each admission screen the controller also evaluates the same screen under all four quantities
(no side effects) and emits 'r15_shadow' when they disagree, with the held KV blocks at that moment."""
from functools import lru_cache
import math, time
from slack_policy import SlackController

ESTIMATORS = ('chance', 'point', 'hard', 'min_chance_hard')


class EstimatorController(SlackController):
    def __init__(self, config, block_size, capacity, clock=time.time):
        super().__init__(config, block_size, capacity, clock)
        r15 = config.get('r15')
        if not isinstance(r15, dict) or set(r15) != {'control_estimator', 'proactive_recovery_hook'}:
            raise ValueError('r15 must define exactly control_estimator and proactive_recovery_hook')
        est, hook = r15['control_estimator'], r15['proactive_recovery_hook']
        if est not in ESTIMATORS or type(hook) is not bool:
            raise ValueError('invalid r15 configuration')
        if self.arm != 'guardrail' or self.admission != 'chance' or not self.recovery or self.wait_fraction is None:
            raise ValueError('R15 is registered only on the protected C4 guardrail/reuse configuration')
        if self.bypass_cap != 1.0 or self.gate_window is not None or self.requeue_budget or self.cooldown is not None:
            raise ValueError('R15 arms inherit the registered protected switches only')
        self.estimator, self.hook = est, hook
        self.admission = est          # risk(), screen bound key and warning 'current' all follow the chosen quantity
        self.held_fn = None           # set by the scheduler adapter: () -> held KV blocks
        self.shadow_checks = 0
        self.shadow_disagreements = 0

    # ---- control quantities ---------------------------------------------------------------------------------------
    def bounds(self, rows):
        geometry = tuple((r.generated, r.prefix_tokens, r.allocated_blocks) for r in rows)
        out = self._bounds_for_geometry(geometry).copy()
        hard = out['current'] + self._hard_growth(tuple(g[1] for g in geometry))
        out['hard'] = hard
        out['min_chance_hard'] = min(out['chance'], hard)
        return out

    @lru_cache(maxsize=2048)
    def _hard_growth(self, prefixes):
        B, k = self.block_size, self.k
        return sum((q + k + B - 1) // B - (q + B - 1) // B for q in prefixes)

    # ---- shadow screen (pure) ------------------------------------------------------------------------------------
    def _gated(self, est, b_active, candidate, spent):
        """True when the warning gate defers the candidate under quantity est (no joint bound needed)."""
        zone = self.recovery and b_active[est] > self.low * self.capacity and candidate.priority > 0
        return zone and not spent

    def screen(self, active, candidate, current):
        start = time.perf_counter()
        decision = super().screen(active, candidate, current)
        self.inner_admission_s = getattr(self, 'inner_admission_s', 0.0) + time.perf_counter() - start
        try:
            # AMENDMENT_R15_001: lazy shadow. The joint bound is evaluated only when at least one quantity's warning
            # gate would let the candidate through; otherwise all four shadow decisions are 'defer' (identical values).
            b_active = self._active_bounds(active)
            spent = self._spent_pure(candidate)
            gated = {e: self._gated(e, b_active, candidate, spent) for e in ESTIMATORS}
            if all(gated.values()):
                shadow, b_joint = {e: False for e in ESTIMATORS}, None
            else:
                b_joint = self._joint_bounds(b_active, candidate)
                shadow = {e: (not gated[e]) and b_joint[e] <= self.capacity for e in ESTIMATORS}
            self.shadow_checks += 1
            if len(set(shadow.values())) > 1 or shadow[self.estimator] != decision[0]:
                self.shadow_disagreements += 1
                if self.emit_evidence:
                    self.emit_evidence('r15_shadow', request_id=candidate.request_id, priority=candidate.priority,
                                       generated=candidate.generated, actual=decision[0], reason=decision[1],
                                       decisions=shadow, held=self.held_fn() if self.held_fn else None,
                                       active={e: b_active[e] for e in ESTIMATORS},
                                       joint=None if b_joint is None else {e: b_joint[e] for e in ESTIMATORS})
        except Exception as exc:          # evidence must never change a decision
            if self.emit_evidence:
                self.emit_evidence('r15_shadow_error', error=repr(exc)[:200])
        return decision

    def _active_bounds(self, active):
        # The deferral window passes the same frozen active tuple for consecutive candidates.
        if active is getattr(self, '_sa_obj', None):
            return self._sa_val
        val = self.bounds(active)
        self._sa_obj, self._sa_val = active, val
        return val

    def _joint_bounds(self, b_active, c):
        # Appending the candidate last reproduces the parent's sequential accumulation exactly.
        blocks = (c.prefix_tokens + self.block_size - 1) // self.block_size
        mu, var = self.histogram.block_moments(c.generated, c.prefix_tokens, self.block_size, self.k)
        base = b_active['current'] + max(blocks, c.allocated_blocks)
        mean, variance = b_active['mean_growth'] + mu, b_active['variance_growth'] + var
        chance = base + mean + self.z * math.sqrt(variance)
        hard = b_active['hard'] + max(blocks, c.allocated_blocks) + self._hard_growth((c.prefix_tokens,))
        return {'current': base, 'point': base + mean, 'chance': chance, 'hard': hard,
                'min_chance_hard': min(chance, hard)}

    def _spent_pure(self, candidate):
        """wait_budget_spent without mutating requeue bookkeeping (requeue budget is off for R15 arms)."""
        if candidate.generated > 0:
            return False
        return self.clock() - candidate.arrival_time >= self.wait_fraction * self.ttft_ms[candidate.priority] / 1000
