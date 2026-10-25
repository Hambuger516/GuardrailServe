"""Separately registered baseline and ranking treatments; original math unchanged."""
from dataclasses import asdict
import math
from admission_reuse import ConsecutiveDeferralState
from policy_core import Controller


ARMS = {'priority', 'guardrail', 'concurrency', 'reserve',
        'rank_no_slack', 'rank_no_prefix', 'rank_class_blocks'}


class RevisionController(Controller):
    def __init__(self, config, block_size, capacity):
        self.arm = config['revision_arm']
        if self.arm not in ARMS:
            raise ValueError('Unknown registered treatment')
        expected = 'C0' if self.arm in {'priority', 'concurrency', 'reserve'} else 'C4'
        if config['policy'] != expected:
            raise ValueError('Treatment and original policy disagree')
        super().__init__(config, block_size, capacity)
        self.concurrent_limit = config.get('concurrent_limit', 64)
        self.reserve_fraction = config.get('reserve_fraction', 0.0)
        if (type(self.concurrent_limit) is not int or not 1 <= self.concurrent_limit <= 64
                or not 0 <= self.reserve_fraction < 1):
            raise ValueError('Invalid fixed limit')
        self.audit_victims = config.get('audit_victims', False)
        if type(self.audit_victims) is not bool:
            raise ValueError('Explicit audit switch required')
        self.emit_evidence = None

    def admission_decision(self, active, candidate):
        if self.arm == 'concurrency':
            count = len(active) + 1
            return count <= self.concurrent_limit, 'fixed_concurrency', count
        if self.arm == 'reserve':
            # This fixed current logical-prefix budget has no output prediction.
            # Native allocation remains the final physical feasibility check.
            held = sum(max((r.prefix_tokens + self.block_size - 1) // self.block_size,
                           r.allocated_blocks) for r in (*active, candidate))
            limit = math.floor(self.capacity * (1 - self.reserve_fraction))
            return held <= limit, 'fixed_current_reserve', held
        return super().admission_decision(active, candidate)

    def components(self, row, decode_ms):
        slack = max(0, self.tpot_ms[row.priority] - decode_ms)
        weighted = self.weights[row.priority] * math.exp(-self.gamma * slack)
        prefix = self.beta * row.prefix_tokens / 1000
        blocks = max(row.allocated_blocks, 1)
        return dict(slack_ms=slack, priority_slack_cost=weighted,
                    prefix_cost=prefix, density=(weighted + prefix) / blocks,
                    no_slack=(self.weights[row.priority] + prefix) / blocks,
                    no_prefix=weighted / blocks)

    def density(self, row, decode_ms):
        if self.arm == 'rank_no_slack':
            return (self.weights[row.priority] + self.beta * row.prefix_tokens / 1000) / max(row.allocated_blocks, 1)
        if self.arm == 'rank_no_prefix':
            slack = max(0, self.tpot_ms[row.priority] - decode_ms)
            return self.weights[row.priority] * math.exp(-self.gamma * slack) / max(row.allocated_blocks, 1)
        # For class/blocks the inherited scalar is only descriptive, not its key.
        return super().density(row, decode_ms)

    @staticmethod
    def tie(row):
        return (-row.priority, -row.arrival_time, row.request_id)

    def choose_victim(self, rows, decode_ms):
        eligible = [r for r in rows if r.allocated_blocks > 0]
        if not eligible:
            return None
        if self.arm == 'rank_class_blocks':
            chosen = min(eligible, key=lambda r: (-r.priority, -r.allocated_blocks,
                                                -r.arrival_time, r.request_id))
        else:
            chosen = min(eligible, key=lambda r: (self.density(r, decode_ms), *self.tie(r)))
        if self.audit_victims and self.emit_evidence:
            self.emit_evidence('victim_candidates', treatment=self.arm,
                               decode_ms=decode_ms, chosen=chosen.request_id,
                               candidates=[dict(asdict(r), **self.components(r, decode_ms)) for r in eligible])
        return chosen


class RevisionDeferralState(ConsecutiveDeferralState):
    # Preserve the original active-risk cache as well as the active-view cache.
    # Fixed rules use admission='current', so the inherited warning shortcut is
    # naturally disabled for them. Their custom admission_decision still runs.
    pass
