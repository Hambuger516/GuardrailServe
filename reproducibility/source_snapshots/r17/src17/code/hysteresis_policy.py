"""R14 admission/recovery hysteresis (2026-10-15). Default-off extension of the R13 admission ceiling.

R13 finding: a non-HIGH admission ceiling equal to the recovery trigger (0.95K) left zero headroom. Admission filled the
active set up to the trigger, after which ordinary decode growth (or a HIGH admission) re-triggered recovery; recovery
episodes not preceded by an admission rose from ~4 to ~31 per Qwen run, and each episode still evicted 4-5 decoding
requests down to 0.80K.

Change (three ordered levels, registered values): non-HIGH admission ceiling c = 0.85K < recovery stop target
t = 0.90K < recovery trigger theta_high = 0.95K. After a recovery the set sits at <= 0.90K, above the non-HIGH ceiling,
so non-HIGH admissions resume only after completions bring the bound below 0.85K; recovery evicts only down to 0.90K
(shallow) instead of 0.80K. The warning gate still uses theta_low = 0.80K, HIGH admissions keep the K limit, and the
forecast, wait budget, F1/F2/F3a victim rules are unchanged."""
import time
from ceiling_policy import CeilingController


class HysteresisController(CeilingController):
    def __init__(self, config, block_size, capacity, clock=time.time):
        super().__init__(config, block_size, capacity, clock)
        t = config.get('recovery_target')
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not self.low <= float(t) < self.high:
            raise ValueError('recovery_target must lie in [theta_low, theta_high)')
        if not self.ceiling <= float(t):
            raise ValueError('the non-HIGH admission ceiling must not exceed the recovery target')
        self.target = float(t)
