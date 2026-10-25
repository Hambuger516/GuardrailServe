"""Version-bound vLLM 0.29 adapter; isolated from the installed package.

Two auditable insertion points in upstream schedule(): recovery at the empty
iteration boundary and guard immediately before waiting-request allocation.
All token execution, block allocation, native fallback, and recomputation use
upstream code. Unsupported execution modes fail closed at initialization.
"""
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time

import vllm.v1.core.sched.scheduler as upstream
from policy_core import Controller, View


def patched_schedule():
    source = textwrap.dedent(inspect.getsource(upstream.Scheduler.schedule))
    recovery = '    self.kv_cache_manager.new_step_starts()\n'
    allocation = ('            new_blocks = self.kv_cache_manager.allocate_slots(\n'
                  '                request,\n'
                  '                num_new_tokens,\n'
                  '                num_new_computed_tokens=num_new_local_computed_tokens,')
    if source.count(recovery) != 1 or source.count(allocation) != 1:
        raise RuntimeError('vLLM schedule source does not match audited insertion points')
    source = source.replace(recovery, recovery +
                            '    preempted_reqs.extend(self._gr_recover(scheduled_timestamp))\n')
    source = source.replace(allocation,
        '            if not self._gr_admit(request):\n'
        '                request_queue.pop_request()\n'
        '                step_skipped_waiting.prepend_request(request)\n'
        '                continue\n' + allocation)
    scope = dict(vars(upstream))
    exec(compile(source, '<guardrail-vllm-0.29-schedule>', 'exec'), scope)
    return scope['schedule'], source


_schedule, PATCHED_SOURCE = patched_schedule()


class GuardrailScheduler(upstream.Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.vllm_config.additional_config['guardrail']
        if (self.scheduler_config.async_scheduling or self.cache_config.enable_prefix_caching
                or self.vllm_config.speculative_config is not None or self.connector is not None
                or self.vllm_config.parallel_config.tensor_parallel_size != 1
                or len(self.kv_cache_config.kv_cache_groups) != 1):
            raise ValueError('adapter supports synchronous TP1 dense attention, no sharing/speculation/connectors')
        source_path = Path(inspect.getfile(upstream))
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_sha != cfg['upstream_sha256']:
            raise ValueError('installed scheduler changed after audit')
        self._gr_dir = Path(cfg['events_dir'])
        self._gr_dir.mkdir(parents=True, exist_ok=True)
        self._gr_events = (self._gr_dir / 'scheduler_events.jsonl').open('x', buffering=1)
        self._gr_event_mode = cfg.get('event_mode', 'full')
        if self._gr_event_mode not in {'full', 'compact'}:
            raise ValueError('unknown event mode')
        self._gr_admission_counts = {
            'warmup': {'checked': 0, 'allowed': 0, 'deferred': 0},
            'measurement': {'checked': 0, 'allowed': 0, 'deferred': 0}}
        (self._gr_dir / 'patched_schedule.py').write_text(PATCHED_SOURCE)
        capacity = self.kv_cache_manager.block_pool.get_num_free_blocks()
        self._gr = Controller(self._gr_controller_config(cfg), self.block_size, capacity)
        self._gr_decode_ms = float(cfg.get('initial_decode_ms', 10))
        self._gr_decode_only = False
        self._gr_compute_start = 0
        self._gr_recompute_pending = {}
        self._gr_emit('controller_init', config=cfg, capacity_blocks=capacity,
                      block_size=self.block_size, upstream_sha256=source_sha)

    def _gr_emit(self, event, **fields):
        if event == 'admission_check':
            phase = 'warmup' if fields['request_id'].startswith('warmup-') else 'measurement'
            count = self._gr_admission_counts[phase]
            count['checked'] += 1
            count['allowed' if fields['allow'] else 'deferred'] += 1
            if self._gr_event_mode == 'compact':
                return
        if event == 'schedule':
            fields['cumulative_admission_counts'] = self._gr_admission_counts
        self._gr_events.write(json.dumps({'event': event, 'monotonic_s': time.monotonic(),
                                         'step': self.current_step, **fields}) + '\n')

    def _gr_controller_config(self, cfg):
        return cfg

    def _gr_view(self, r):
        blocks = self.kv_cache_manager.get_blocks(r.request_id).blocks[0]
        return View(r.request_id, r.num_output_tokens, r.num_tokens, len(blocks),
                    r.priority, r.arrival_time)

    def _gr_recover(self, timestamp):
        victims = []
        if not self._gr.recovery:
            return victims
        rows = [self._gr_view(r) for r in self.running]
        bound = self._gr.risk(rows)
        if bound <= self._gr.high * self._gr.capacity:
            return victims
        self._gr_emit('danger', bound=bound, active=len(rows))
        # Hysteresis is refreshed after EACH actual release. A singleton is
        # allowed to progress past a watermark (never past physical capacity),
        # avoiding repeated recomputation of the only runnable request.
        while len(rows) > 1 and bound > self._gr.low * self._gr.capacity:
            chosen = self._gr.choose_victim(rows, self._gr_decode_ms)
            if chosen is None:
                break
            victim = next(r for r in self.running if r.request_id == chosen.request_id)
            self._gr_emit('slack_victim', request_id=victim.request_id,
                          density=self._gr.density(chosen, self._gr_decode_ms),
                          bound_before=bound, decode_ms=self._gr_decode_ms)
            self.running.remove(victim)
            self._preempt_request(victim, timestamp)
            victims.append(victim)
            rows = [self._gr_view(r) for r in self.running]
            bound = self._gr.risk(rows)
        self._gr_emit('recovery_end', bound=bound, victims=[r.request_id for r in victims],
                      singleton_exemption=len(rows) == 1)
        return victims

    def _gr_admit(self, request):
        rows = [self._gr_view(r) for r in self.running]
        candidate = self._gr_view(request)
        allow, reason, bound = self._gr.admission_decision(rows, candidate)
        self._gr_emit('admission_check', request_id=request.request_id, allow=allow,
                      reason=reason, predicted_blocks=bound,
                      free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                      generated=request.num_output_tokens, active=len(rows))
        return allow

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        generated, computed = request.num_output_tokens, request.num_computed_tokens
        retained_token_ids = tuple(request.output_token_ids)
        before = self.kv_cache_manager.block_pool.get_num_free_blocks()
        super()._preempt_request(request, timestamp, drop_stale_output)
        assert request.num_output_tokens == generated
        if tuple(request.output_token_ids) != retained_token_ids:
            raise RuntimeError('preemption changed an already emitted token prefix')
        self._gr_recompute_pending[request.request_id] = computed
        self._gr_emit('preempt', request_id=request.request_id, retained_outputs=generated,
                      exact_output_prefix_preserved=True,
                      invalidated_kv_tokens=computed,
                      blocks_released=self.kv_cache_manager.block_pool.get_num_free_blocks() - before)

    def schedule(self, *args, **kwargs):
        start = time.perf_counter()
        result = _schedule(self, *args, **kwargs)
        self._gr_decode_only = bool(result.num_scheduled_tokens) and all(
            n == 1 and self.requests[rid].num_output_tokens > 0
            for rid, n in result.num_scheduled_tokens.items())
        self._gr_last_batch = {'requests': len(result.num_scheduled_tokens),
                               'tokens': sum(result.num_scheduled_tokens.values()),
                               'decode_only': self._gr_decode_only}
        for rid, n in result.num_scheduled_tokens.items():
            pending = self._gr_recompute_pending.get(rid, 0)
            if pending:
                recomputed = min(n, pending)
                self._gr_recompute_pending[rid] -= recomputed
                self._gr_emit('recompute_scheduled', request_id=rid, tokens=recomputed)
        self._gr_emit('schedule', scheduled_tokens=result.num_scheduled_tokens,
                      free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                      running=len(self.running), scheduler_us=(time.perf_counter() - start) * 1e6)
        self._gr_compute_start = time.monotonic()
        return result

    def update_from_output(self, *args, **kwargs):
        elapsed_ms = (time.monotonic() - self._gr_compute_start) * 1000
        result = super().update_from_output(*args, **kwargs)
        self._gr_emit('step_complete', observed_execution_ms=elapsed_ms,
                      free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                      running=len(self.running), **self._gr_last_batch)
        if self._gr_decode_only:
            self._gr_decode_ms = 0.9 * self._gr_decode_ms + 0.1 * elapsed_ms
        return result
