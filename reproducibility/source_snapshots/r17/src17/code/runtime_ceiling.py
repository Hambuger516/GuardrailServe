"""R13 scheduler adapter for the admission ceiling (2026-10-15). Reuses the audited protected adapter (SlackScheduler:
same patched schedule source, F2 hook and deferral lifetime) and only replaces its controller with CeilingController
built from the same configuration."""
import hashlib
from pathlib import Path
import ceiling_policy
from ceiling_policy import CeilingController
from runtime_slack import SlackScheduler


class CeilingScheduler(SlackScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config.additional_config['guardrail']
        if config.get('admission_ceiling') is None:
            raise ValueError('CeilingScheduler requires admission_ceiling')
        self._gr = CeilingController(config, self.block_size, self._gr.capacity)
        self._gr.emit_evidence = self._gr_emit
        self._gr_emit('ceiling_init', version='r13_ceiling_v1', admission_ceiling=self._gr.ceiling,
                      warning_wait_fraction=self._gr.wait_fraction, protect_high_victims=self._gr.protect_high,
                      inflight_protection=self._gr.inflight, high_decode_token_budget=self._gr_budget_cap,
                      policy_sha256=hashlib.sha256(Path(ceiling_policy.__file__).read_bytes()).hexdigest())
