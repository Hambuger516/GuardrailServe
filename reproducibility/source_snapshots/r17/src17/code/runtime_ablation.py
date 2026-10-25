"""R12 scheduler adapter for component-ablation arms (2026-10-13).

Reuses the audited protected adapter (SlackScheduler: same patched schedule source, same one-line F2 token-budget
hook, same SlackDeferralState lifetime) and only replaces its controller with AblationController and its
token-budget cap with the arm's declared value. Registered only for config['ablation'] runs.
"""
import hashlib
from pathlib import Path
import ablation_policy
from ablation_policy import AblationController
from runtime_slack import SlackScheduler, BUDGET_SOURCE


class AblationScheduler(SlackScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config.additional_config['guardrail']
        if 'ablation' not in config or config.get('warning_wait_fraction') is not None \
                or config.get('high_decode_token_budget') is not None:
            raise ValueError('ablation arms declare their components only inside config["ablation"]')
        self._gr = AblationController(config, self.block_size, self._gr.capacity)
        self._gr.emit_evidence = self._gr_emit
        cap = config['ablation']['token_budget']
        if cap is not None and not 64 <= cap <= self.max_num_scheduled_tokens:
            raise ValueError('token_budget must lie in [64, max_num_scheduled_tokens]')
        self._gr_budget_cap = cap
        self._gr_budget_steps = 0
        if cap is not None:
            (self._gr_dir / 'patched_schedule_budget.py').write_text(BUDGET_SOURCE)
        self._gr_emit('ablation_init', version='r12_ablation_v1', ablation=self._gr.ablation,
                      admission=self._gr.admission, recovery=self._gr.recovery,
                      warning_gate=self._gr.warning_gate, wait_fraction=self._gr.wait_fraction,
                      protect_high=self._gr.protect_high, token_budget=cap,
                      policy_sha256=hashlib.sha256(Path(ablation_policy.__file__).read_bytes()).hexdigest())
