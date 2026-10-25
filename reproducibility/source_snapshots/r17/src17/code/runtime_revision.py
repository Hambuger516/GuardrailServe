"""Shared version-pinned adapter for the September 29 reviewer experiments."""
import hashlib
from pathlib import Path
import revision_policy
from revision_policy import RevisionController, RevisionDeferralState
from runtime_scheduler import GuardrailScheduler
from runtime_scheduler_reuse import ReuseGuardrailScheduler


class RevisionScheduler(ReuseGuardrailScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config.additional_config['guardrail']
        self._revision_variant = config['adapter_variant']
        if self._revision_variant not in {'original', 'reuse'}:
            raise ValueError('Unknown state lifetime treatment')
        self._gr = RevisionController(config, self.block_size, self._gr.capacity)
        self._gr.emit_evidence = self._gr_emit
        self._gr_emit('revision_init', version='revision8_20260929_v1',
                      treatment=self._gr.arm, adapter_variant=self._revision_variant,
                      audit_victims=self._gr.audit_victims,
                      policy_sha256=hashlib.sha256(Path(revision_policy.__file__).read_bytes()).hexdigest())

    def schedule(self, *args, **kwargs):
        if self._reuse_window is not None:
            raise RuntimeError('Nested schedule invalidates state lifetime')
        if self._revision_variant == 'reuse':
            self._reuse_window = RevisionDeferralState(self._gr)
        try:
            return GuardrailScheduler.schedule(self, *args, **kwargs)
        finally:
            self._reuse_window = None
