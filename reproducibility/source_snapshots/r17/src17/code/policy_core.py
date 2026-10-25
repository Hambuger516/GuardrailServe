"""Shared pure policy mathematics. No access to future request output lengths."""
from dataclasses import dataclass
from functools import lru_cache
import math
from statistics import NormalDist
from sim.prediction import ResidualHistogram


@dataclass(frozen=True)
class View:
    request_id: str
    generated: int
    prefix_tokens: int
    allocated_blocks: int
    priority: int
    arrival_time: float


class Controller:
    MODES = {'C0': ('current', False), 'C1': ('point', False),
             'C2': ('chance', False), 'C3': ('point', True),
             'C4': ('chance', True), 'C5': ('current', True)}

    def __init__(self, config, block_size, capacity):
        self.admission, self.recovery = self.MODES[config['policy']]
        self.block_size, self.capacity = block_size, capacity
        self.histogram = ResidualHistogram({int(k): v for k, v in config['length_histogram'].items()})
        self.k = int(config.get('horizon', 10))
        self.epsilon = float(config.get('epsilon', 0.05))
        self.low = float(config.get('theta_low', 0.80))
        self.high = float(config.get('theta_high', 0.95))
        if not (0 < self.low < self.high < 1 and 0 < self.epsilon < 0.5
                and self.k > 0 and block_size > 0 and capacity > 0):
            raise ValueError('invalid controller geometry or risk configuration')
        self.z = NormalDist().inv_cdf(1 - self.epsilon)
        self.gamma, self.beta = float(config.get('gamma', 0.01)), float(config.get('beta', 0.1))
        self.weights = config.get('weights', [3, 1, 0.3])
        self.tpot_ms = config.get('tpot_ms', [50, 100, 200])

    def bounds(self, rows):
        # Repeated waiting-candidate checks often see identical active geometry.
        # Cache only pure arithmetic inputs, preserving row order and the exact
        # original accumulation order. Priority/arrival affect other decisions,
        # not this function. Return a copy so callers cannot mutate the cache.
        geometry = tuple((r.generated, r.prefix_tokens, r.allocated_blocks) for r in rows)
        return self._bounds_for_geometry(geometry).copy()

    @lru_cache(maxsize=2048)
    def _bounds_for_geometry(self, geometry):
        base = mean = variance = 0
        for generated, prefix_tokens, allocated_blocks in geometry:
            blocks = (prefix_tokens + self.block_size - 1) // self.block_size
            # Includes the sampled token awaiting its next forward pass. No
            # completion release is credited: this is a held-cache envelope.
            mu, var = self.histogram.block_moments(generated, prefix_tokens,
                                                  self.block_size, self.k)
            base += max(blocks, allocated_blocks)
            mean += mu
            variance += var
        return {'current': base, 'point': base + mean,
                'chance': base + mean + self.z * math.sqrt(variance),
                'mean_growth': mean, 'variance_growth': variance}

    def risk(self, rows):
        mode = self.admission if self.admission != 'current' else 'chance'
        return self.bounds(rows)[mode]

    def admission_decision(self, active, candidate):
        if self.admission == 'current':
            return True, 'native_current_fit', None
        current = self.risk(active)
        if self.recovery and current > self.low * self.capacity and candidate.priority > 0:
            return False, 'warning_non_high', current
        bound = self.bounds([*active, candidate])[self.admission]
        return bound <= self.capacity, 'risk_bound', bound

    def density(self, row, decode_ms):
        slack = max(0, self.tpot_ms[row.priority] - decode_ms)
        cost = (self.weights[row.priority] * math.exp(-self.gamma * slack)
                + self.beta * row.prefix_tokens / 1000)
        return cost / max(row.allocated_blocks, 1)

    def choose_victim(self, rows, decode_ms):
        reclaimable = [r for r in rows if r.allocated_blocks > 0]
        if not reclaimable:
            return None
        return min(reclaimable, key=lambda r: (self.density(r, decode_ms),
                                               -r.priority, -r.arrival_time, r.request_id))
