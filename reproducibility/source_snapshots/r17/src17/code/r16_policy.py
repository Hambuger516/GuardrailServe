"""R16 controller (frozen 2026-10-16 before any R16 outcome). Default-off extension of the R15 controller
(EstimatorController with control_estimator='chance', recovery hook on), configured by config['r16'] =
{'mode': 'protected' | 'hyst' | 'adapt', 'f2': 'on' | 'off' | 'pressure'}.

mode (recovery depth / non-HIGH admission ceiling; trigger stays at theta_high = 0.95K, warning at theta_low = 0.80K):
  protected  ceiling K, recovery stops at 0.80K                          (decisions identical to SlackController)
  hyst       non-HIGH ceiling 0.85K, recovery stops at 0.90K             (decisions identical to R14 HysteresisController)
  adapt      'long' regime -> hyst parameters, 'short' regime -> protected parameters. The regime follows a time-based
             EWMA (time constant 10 s) of the mean resident tokens per running request: enter 'long' above 1000 tokens,
             leave below 800 (hysteresis). Starts 'short'. Thresholds set from development data (R12-R15): mean
             invalidated tokens per proactive victim ~450-560 on Dolly vs ~2900-3000 on Azure conversation replay.
f2 (HIGH-decode step token budget, 512):
  on         unchanged F2 (cap whenever a HIGH request is decoding)
  off        no cap
  pressure   cap only while held KV >= theta_low * K (memory-pressure regime)
  regime     (R17) cap only in the short regime (protected parameters); no cap in the long regime
Only active-set geometry and the engine clock are read; no future output length."""
import math, time
from estimator_policy import EstimatorController
from slack_policy import SlackController

MODES, F2 = ('protected', 'hyst', 'adapt'), ('on', 'off', 'pressure', 'regime')
HYST_CEILING, HYST_TARGET = 0.85, 0.90
TAU_S, ON_TOKENS, OFF_TOKENS = 10.0, 1000.0, 800.0


class R16Controller(EstimatorController):
    def __init__(self, config, block_size, capacity, clock=time.time):
        super().__init__(config, block_size, capacity, clock)
        r16 = config.get('r16')
        if not isinstance(r16, dict) or set(r16) != {'mode', 'f2'} or r16['mode'] not in MODES or r16['f2'] not in F2:
            raise ValueError('r16 must define mode in %s and f2 in %s' % (MODES, F2))
        if self.estimator != 'chance' or not self.hook:
            raise ValueError('R16 arms use the Gaussian quantity with the recovery hook on')
        self.mode, self.f2 = r16['mode'], r16['f2']
        self.long = self.mode == 'hyst'
        self.ewma, self._ewma_t = None, None
        self.switches = 0

    # ---- regime ----------------------------------------------------------------------------------------------
    def observe(self, rows, now):
        """Update the resident-length EWMA from the running set (adapt only); returns True on a regime switch."""
        if self.mode != 'adapt' or not rows:
            return False
        x = sum(r.prefix_tokens for r in rows) / len(rows)
        if self.ewma is None:
            self.ewma = x
        else:
            a = 1 - math.exp(-max(0.0, now - self._ewma_t) / TAU_S)
            self.ewma += a * (x - self.ewma)
        self._ewma_t = now
        new = self.ewma > ON_TOKENS if not self.long else self.ewma >= OFF_TOKENS
        if new != self.long:
            self.long = new; self.switches += 1
            return True
        return False

    def ceiling(self):
        return HYST_CEILING if self.long else 1.0

    def target(self):
        return HYST_TARGET if self.long else self.low

    def screen(self, active, candidate, current):
        # R16 does not need the R15 four-quantity shadow (the ceiling would register as a disagreement on every check).
        start = time.perf_counter()
        decision = SlackController.screen(self, active, candidate, current)
        self.inner_admission_s = getattr(self, 'inner_admission_s', 0.0) + time.perf_counter() - start
        return decision

    # ---- admission (R13 ceiling rule, active only in the hyst regime) ---------------------------------------------
    def _screen(self, active, candidate, current):
        allow, reason, bound = super()._screen(active, candidate, current)
        if allow and self.long and candidate.priority > 0 and bound > HYST_CEILING * self.capacity:
            return False, 'admission_ceiling', bound
        return allow, reason, bound
