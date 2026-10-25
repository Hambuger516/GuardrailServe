"""R12 component-ablation controller (2026-10-13). Default-off; protected/slack paths are untouched.

Each ablation arm is the protected controller with exactly the listed components switched:
  admission_mode  C0 native current-fit admission, no proactive recovery
                  C2 Gaussian joint-demand screen, no proactive recovery (and therefore no warning gate)
                  C4 Gaussian screen + warning gate + proactive recovery (the protected base)
                  C5 native current-fit admission + Gaussian-triggered proactive recovery (no screen, no warning)
  warning_gate    HIGH-only warning gate above theta_low (C4 only)
  wait_fraction   new-request warning wait budget (fraction of class TTFT; requires the warning gate)
  protect_high    F1 HIGH-aware proactive victim eligibility (requires proactive recovery)
  token_budget    F2 conditional per-step token budget (applied by the scheduler adapter)
The forecast arithmetic, thresholds, victim density, tie rules and native capacity repair are inherited unchanged.
With admission_mode='C4', warning_gate=True, wait_fraction=0.5, protect_high=True the decisions are identical to
SlackController(protected); see test_ablation.py.
"""
import time
from policy_core import Controller
from slack_policy import SlackController

ABLATION_KEYS = {'admission_mode', 'warning_gate', 'wait_fraction', 'protect_high', 'token_budget'}


def validate_ablation(ab):
    if not isinstance(ab, dict) or set(ab) != ABLATION_KEYS:
        raise ValueError('ablation must define exactly ' + ', '.join(sorted(ABLATION_KEYS)))
    mode = ab['admission_mode']
    if mode not in {'C0', 'C2', 'C4', 'C5'}:
        raise ValueError('ablation admission_mode must be C0, C2, C4 or C5')
    for key in ('warning_gate', 'protect_high'):
        if type(ab[key]) is not bool:
            raise ValueError(key + ' must be bool')
    admission, recovery = Controller.MODES[mode]
    if ab['warning_gate'] and not (admission == 'chance' and recovery):
        raise ValueError('the warning gate exists only with Gaussian admission and proactive recovery (C4)')
    if ab['wait_fraction'] is not None and not ab['warning_gate']:
        raise ValueError('a wait budget requires the warning gate')
    if ab['protect_high'] and not recovery:
        raise ValueError('HIGH-aware victim eligibility requires proactive recovery')
    cap = ab['token_budget']
    if cap is not None and (type(cap) is not int or cap < 64):
        raise ValueError('token_budget must be None or an int >= 64')
    return admission, recovery


class AblationController(SlackController):
    def __init__(self, config, block_size, capacity, clock=time.time):
        ab = config['ablation']
        admission, recovery = validate_ablation(ab)
        # Construct through the registered protected parent (C4 guardrail/reuse) so that all inherited
        # validation, histogram handling and arithmetic are identical, then switch the declared components.
        base = dict(config, policy='C4', revision_arm='guardrail', warning_wait_fraction=ab['wait_fraction'],
                    protect_high_victims=ab['protect_high'], requeue_wait_budget=False,
                    adaptive_gate_window_s=None, inflight_protection=False, victim_cooldown_s=None)
        base.pop('ablation')
        super().__init__(base, block_size, capacity, clock)
        self.admission, self.recovery = admission, recovery
        self.warning_gate = ab['warning_gate']
        self.ablation = dict(ab)

    def warning_zone(self, current, candidate):
        if not self.warning_gate:
            return False
        return super().warning_zone(current, candidate)
