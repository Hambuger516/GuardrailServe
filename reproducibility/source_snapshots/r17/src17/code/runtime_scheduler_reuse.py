"""Experimental, separately versioned adapter; release003 stays untouched.

Supports the audited synchronous ordinary text-decoding path only. This is not
performance evidence; real integration and a new registered comparison are
required before a manuscript benefit claim.
"""
import hashlib
import inspect
from pathlib import Path

import policy_core
import runtime_scheduler
import vllm.v1.core.kv_cache_manager as kv_manager
import vllm.v1.core.kv_cache_coordinator as kv_coordinator
from admission_reuse import ConsecutiveDeferralState
from runtime_scheduler import GuardrailScheduler


PINNED = {
    'runtime_scheduler': '8f46d9893082c7def8c17275c951e6810086d6213a9659042e3cf2843dc4d028',
    'policy_core': '2e38768f77055f6ce60316ca9e832ca33098d6689b3f0cdd392dddd1a17a11b6',
    'kv_manager': '2213a68ea1a2f05b1d99de25a3f1daf4040d8d7fdd0e271d80b7ba9e1977a08d',
    'kv_coordinator': '43883199a6230b4d9efd98b9702007862e2cfa5466a2dc0326bfd1e69b3e7617',
}


class ReuseGuardrailScheduler(GuardrailScheduler):
    def __init__(self, *args, **kwargs):
        for name, module in [('runtime_scheduler', runtime_scheduler),
                             ('policy_core', policy_core), ('kv_manager', kv_manager),
                             ('kv_coordinator', kv_coordinator)]:
            digest = hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
            if digest != PINNED[name]:
                raise ValueError('reuse adapter source differs from audited version: '+name)
        self._reuse_window = None
        super().__init__(*args, **kwargs)
        if (self.is_encoder_decoder or self.vllm_config.model_config.is_multimodal_model
                or self.ec_connector is not None or self.lora_config is not None
                or self.num_spec_tokens != 0 or self.num_prefill_lookahead != 0
                or self.requires_kv_delivery):
            raise ValueError('reuse adapter requires ordinary text decoding without auxiliary cache paths')
        self._gr_emit('admission_reuse_init', version='consecutive_deferral001', pinned=PINNED)

    def add_request(self, request):
        if (request.resumable or request.has_encoder_inputs or request.mm_features
                or request.lora_request is not None):
            raise ValueError('reuse adapter does not support streaming, multimodal, encoder or LoRA requests')
        return super().add_request(request)

    def schedule(self, *args, **kwargs):
        if self._reuse_window is not None:
            raise RuntimeError('nested schedule invalidates active-view lifetime')
        self._reuse_window = ConsecutiveDeferralState(self._gr)
        try:
            return super().schedule(*args, **kwargs)
        finally:
            # Includes exceptions, zero-token steps, preemption and pause paths.
            self._reuse_window = None

    def _preempt_request(self, *args, **kwargs):
        if self._reuse_window is not None:
            self._reuse_window.invalidate()
        return super()._preempt_request(*args, **kwargs)

    def _gr_admit(self, request):
        if self._reuse_window is None:
            return super()._gr_admit(request)
        (allow, reason, bound), count = self._reuse_window.decide(
            lambda: [self._gr_view(r) for r in self.running],
            lambda: self._gr_view(request))
        self._gr_emit('admission_check', request_id=request.request_id, allow=allow,
                      reason=reason, predicted_blocks=bound,
                      free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                      generated=request.num_output_tokens, active=count)
        return allow
