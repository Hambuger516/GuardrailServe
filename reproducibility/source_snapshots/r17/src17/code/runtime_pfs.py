"""Explicit PFS decision port to shared paged vLLM execution, NOT original AE.

Preserves conditional sampling, online 40-length window, strict 5% reserve,
head-candidate-only and at-most-one admit per scheduler step. Adaptations:
integer blocks, shared priority queue, chunked prefill/native recomputation,
one router invocation per vLLM step. Native recovery remains common C0 recovery.
"""
from vllm.v1.request import RequestStatus

from pfs_reference import PFSReference, Request
from runtime_scheduler import GuardrailScheduler


class PFSControlledScheduler(GuardrailScheduler):
    def _gr_controller_config(self, cfg):
        if cfg['policy'] not in {'PFS-port-cold', 'PFS-port-warm'}:
            raise ValueError('requires explicitly labelled controlled PFS port')
        return dict(cfg, policy='C0')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.vllm_config.additional_config['guardrail']
        self._pfs = PFSReference(self._gr.capacity,
                                 initial_length=cfg['pfs_initial_length'],
                                 max_requests=self.scheduler_config.max_num_seqs,
                                 batch_max_tokens=self.scheduler_config.max_num_batched_tokens,
                                 seed=cfg['pfs_seed'])
        if cfg['policy'] == 'PFS-port-warm':
            history = cfg.get('pfs_initial_history', [])
            if len(history) != 40 or any(type(x) is not int or x < 1 for x in history):
                raise ValueError('warm PFS port requires forty observed historical lengths')
            self._pfs.update_history(history)
        self._pfs_checked = False
        self._pfs_recorded = set()
        # R12: in length-faithful replay the port must not read per-request max_tokens/ignore_eos (an oracle of
        # the recorded output length); it then sees the global cap with natural-EOS semantics, as in natural runs.
        self._pfs_hide = bool(cfg.get('pfs_hide_length_oracle', False))
        self._pfs_max_output = int(cfg.get('pfs_max_output', 2048))
        self._gr_emit('pfs_port_init', history=list(self._pfs.history),
                      initialization=cfg['policy'],
                      label='controlled paged vLLM port; not native LightLLM AE',
                      capacity_blocks=self._pfs.capacity, block_size=self.block_size,
                      waiting_order='common vLLM priority', recovery='common native vLLM',
                      router_cadence='at most one admission per scheduler step',
                      hide_length_oracle=self._pfs_hide)

    def _pfs_request(self, r, active):
        hide = self._pfs_hide
        return Request(r.request_id, r.num_prompt_tokens, r.num_output_tokens,
                       self._pfs_max_output if hide else r.max_tokens,
                       'running' if active else ('offload' if r.num_output_tokens else 'waiting'),
                       False if hide else bool(r.sampling_params.ignore_eos))

    def _gr_admit(self, request):
        if self._pfs_checked:
            self._gr_emit('admission_check', request_id=request.request_id, allow=False,
                          reason='pfs_one_head_candidate_per_step', predicted_blocks=None,
                          free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                          generated=request.num_output_tokens, active=len(self.running))
            return False
        self._pfs_checked = True
        active = [self._pfs_request(r, True) for r in self.running]
        candidate = self._pfs_request(request, False)
        admitted, evidence = self._pfs.decide(active, [candidate], block_size=self.block_size)
        allow = bool(admitted)
        self._gr_emit('admission_check', request_id=request.request_id, allow=allow,
                      reason='pfs_sampled_peak', predicted_blocks=max(evidence.get('peaks', [0])),
                      free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                      generated=request.num_output_tokens, active=len(active),
                      pfs_active=[vars(r) for r in active], pfs_candidate=vars(candidate),
                      pfs_evidence=evidence, pfs_history=list(self._pfs.history))
        return allow

    def schedule(self, *args, **kwargs):
        self._pfs_checked = False
        return super().schedule(*args, **kwargs)

    def _free_request(self, request, delay_free_blocks=False):
        if (not request.request_id.startswith('warmup-')
                and request.request_id not in self._pfs_recorded and request.num_output_tokens > 0
                and request.status in {RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED}):
            self._pfs_recorded.add(request.request_id)
            self._pfs.update_history([request.num_output_tokens])
            self._gr_emit('pfs_history_update', request_id=request.request_id,
                          output_tokens=request.num_output_tokens, history=list(self._pfs.history))
        return super()._free_request(request, delay_free_blocks)
