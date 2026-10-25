"""Natural-EOS, open-loop AsyncLLM runner. Needs pilot validation before formal use.

Client concurrency is asynchronous; the engine scheduler remains synchronous,
matching the currently audited GuardrailScheduler adapter. No HTTP/network cost.
"""
import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from gpu_run_observer import Observer, gpu_command
from runtime_smoke import command, sha
from serving_metrics import summarize
from artifact_files import compress_scheduler_log


async def run(args, report, save):
    import torch
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    cfg = json.loads(Path(args.config).read_text())
    # The warm port passed the disjoint 7B pressure gate: 1,060 sampled
    # decisions replayed exactly and all 64 completion-history updates matched
    # client token counts. This validates this explicit port, not native AE.
    if cfg['policy'] == 'PFS-port-cold' and cfg['stage'] == 'formal':
        raise ValueError('cold-start PFS is a diagnostic, not the registered warm-history comparison')
    if (cfg['num_gpu_blocks_override'] - 1) * 16 < cfg['max_model_len']:
        raise ValueError('KV budget excludes one reserved null block; context cap cannot fit')
    pool = json.loads(Path(args.pool).read_text())
    trace = json.loads(Path(args.trace).read_text())
    histogram = json.loads(Path(args.histogram).read_text())
    length_mode = trace.get('length_mode', 'natural_eos')
    if length_mode not in {'natural_eos', 'fixed_from_trace'}:
        raise ValueError('unknown trace length mode')
    fixed = length_mode == 'fixed_from_trace'
    trace_hist = histogram.get('kind') == 'trace_output_lengths'
    if trace_hist and not fixed:
        raise ValueError('trace-length histograms are registered only for length-faithful replay')
    if sha(args.pool) != trace['pool_sha256']:
        raise ValueError('trace references a different prompt pool')
    if pool['split'] == 'history' or trace['pool_split'] != pool['split']:
        raise ValueError('historical/evaluation split mismatch')
    if cfg['stage'] == 'formal' and pool['split'] != 'evaluation':
        raise ValueError('formal effect measurements must use the evaluation pool')
    if cfg['stage'] == 'formal' and histogram.get('kind') != 'observed_training_outputs' and not trace_hist:
        raise ValueError('formal runs require an observed, independently fitted histogram')
    if histogram.get('training_prompt_ids'):
        if set(histogram['training_prompt_ids']) & {r['prompt_id'] for r in pool['requests']}:
            raise ValueError('historical predictor and target prompt pool overlap')
    if cfg['stage'] not in {'pilot', 'calibration', 'formal'}:
        raise ValueError('unknown run stage')
    requests = trace['requests']
    if not requests or len({r['request_id'] for r in requests}) != len(requests):
        raise ValueError('empty trace or duplicate IDs')
    by_id = {r['prompt_id']: r for r in pool['requests']}
    window = trace['arrival_window_s']
    for row in requests:
        if not (0 <= row['arrival_s'] < window and row['priority'] in {0, 1, 2}):
            raise ValueError('invalid arrival or priority')
        prompt = by_id[row['prompt_id']]
        limit = row.get('target_output_tokens') if fixed else cfg['max_output_tokens']
        if fixed and (type(limit) is not int or not 1 <= limit <= cfg['max_output_tokens']):
            raise ValueError('fixed-length replay requires an integer target within the output cap')
        if not fixed and 'target_output_tokens' in row:
            raise ValueError('target lengths are registered only for length-faithful replay')
        if len(prompt['prompt_token_ids']) + limit > cfg['max_model_len']:
            raise ValueError('request exceeds the declared context cap')
    out = Path(args.out)
    policy_cfg = dict(policy=cfg['policy'], events_dir=str(out), length_histogram=histogram['probabilities'],
                      upstream_sha256='abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983',
                      horizon=cfg.get('horizon', 10), epsilon=cfg.get('epsilon', .05),
                      theta_low=.80, theta_high=.95, weights=[3, 1, .3], tpot_ms=[50, 100, 200],
                      gamma=.01, beta=.1, event_mode=cfg.get('event_mode', 'full'))
    if cfg['policy'] in {'PFS-port-cold', 'PFS-port-warm'}:
        policy_cfg.update(pfs_initial_length=cfg['max_model_len'] - max(len(r['prompt_token_ids']) for r in pool['requests']),
                          pfs_seed=cfg['seed'], pfs_hide_length_oracle=fixed,
                          pfs_max_output=cfg['max_output_tokens'])
        if cfg['policy'] == 'PFS-port-warm':
            if histogram.get('kind') != 'observed_training_outputs' and not trace_hist:
                raise ValueError('PFS warm start requires disjoint observed training outputs')
            policy_cfg['pfs_initial_history'] = histogram['pfs_initial_history']
            policy_cfg['pfs_initial_history_provenance'] = histogram['pfs_initial_history_provenance']
    engine_cfg = {key: cfg[key] for key in ['dtype', 'max_model_len', 'max_num_seqs',
                   'max_num_batched_tokens', 'gpu_memory_utilization', 'num_gpu_blocks_override', 'enforce_eager']}
    engine_cfg.update(model=args.model, tensor_parallel_size=1, block_size=16,
                      enable_prefix_caching=False, enable_chunked_prefill=True, async_scheduling=False,
                      scheduling_policy='priority', scheduler_cls='runtime_scheduler.GuardrailScheduler',
                      additional_config={'guardrail': policy_cfg}, seed=cfg['seed'], disable_log_stats=True)
    if cfg.get('revision_arm') not in {'priority', 'guardrail', 'concurrency', 'reserve', 'rank_no_slack', 'rank_no_prefix', 'rank_class_blocks'}:
        raise ValueError('Explicit registered reviewer treatment required')
    policy_cfg.update({key: cfg[key] for key in ['revision_arm', 'adapter_variant', 'audit_victims']})
    for key in ['concurrent_limit', 'reserve_fraction', 'theta_low', 'theta_high']:
        if key in cfg:
            policy_cfg[key] = cfg[key]
    engine_cfg['scheduler_cls'] = 'runtime_revision.RevisionScheduler'
    report['revision_arm'] = cfg['revision_arm']
    if cfg.get('warning_wait_fraction') is not None:
        # step3: slack-bounded warning gate; only for guardrail/reuse.
        if cfg['revision_arm'] != 'guardrail' or cfg['adapter_variant'] != 'reuse' or cfg['policy'] != 'C4':
            raise ValueError('wait budget registered only for C4 guardrail/reuse')
        policy_cfg.update(warning_wait_fraction=cfg['warning_wait_fraction'], ttft_ms=[500, 2000, 10000],
                          bypass_bound_fraction=cfg.get('bypass_bound_fraction', 1.0),
                          protect_high_victims=cfg.get('protect_high_victims', False),
                          high_decode_token_budget=cfg.get('high_decode_token_budget'),
                          requeue_wait_budget=cfg.get('requeue_wait_budget', False),
                          adaptive_gate_window_s=cfg.get('adaptive_gate_window_s'),
                          inflight_protection=cfg.get('inflight_protection', False),
                          victim_cooldown_s=cfg.get('victim_cooldown_s'))
        engine_cfg['scheduler_cls'] = 'runtime_slack.SlackScheduler'
        report['warning_wait_fraction'] = cfg['warning_wait_fraction']
        report['bypass_bound_fraction'] = cfg.get('bypass_bound_fraction', 1.0)
        report['protect_high_victims'] = cfg.get('protect_high_victims', False)
        report['high_decode_token_budget'] = cfg.get('high_decode_token_budget')
        report['requeue_wait_budget'] = cfg.get('requeue_wait_budget', False)
        report['adaptive_gate_window_s'] = cfg.get('adaptive_gate_window_s')
        report['inflight_protection'] = cfg.get('inflight_protection', False)
        report['victim_cooldown_s'] = cfg.get('victim_cooldown_s')
    if cfg.get('ablation') is not None:
        # R12: component-ablation arm on the protected adapter; components only inside cfg['ablation'].
        from ablation_policy import validate_ablation
        if (cfg['revision_arm'] != 'guardrail' or cfg['adapter_variant'] != 'reuse' or cfg['policy'] != 'C4'
                or cfg.get('warning_wait_fraction') is not None or cfg.get('high_decode_token_budget') is not None):
            raise ValueError('ablation arms are registered only as C4 guardrail/reuse with components in cfg["ablation"]')
        validate_ablation(cfg['ablation'])
        policy_cfg.update(ablation=dict(cfg['ablation']), ttft_ms=[500, 2000, 10000])
        engine_cfg['scheduler_cls'] = 'runtime_ablation.AblationScheduler'
        report['ablation'] = cfg['ablation']
    if cfg.get('admission_ceiling') is not None:
        # R13: admission ceiling on the protected (slack) adapter; requires the slack configuration above.
        if engine_cfg['scheduler_cls'] != 'runtime_slack.SlackScheduler' or cfg.get('ablation') is not None:
            raise ValueError('admission ceiling is registered only on the protected/slack configuration')
        policy_cfg['admission_ceiling'] = cfg['admission_ceiling']
        engine_cfg['scheduler_cls'] = 'runtime_ceiling.CeilingScheduler'
        report['admission_ceiling'] = cfg['admission_ceiling']
    if cfg.get('recovery_target') is not None:
        # R14: shallow recovery stop target on top of the admission ceiling.
        if engine_cfg['scheduler_cls'] != 'runtime_ceiling.CeilingScheduler':
            raise ValueError('recovery_target is registered only together with admission_ceiling')
        policy_cfg['recovery_target'] = cfg['recovery_target']
        engine_cfg['scheduler_cls'] = 'runtime_hysteresis.HysteresisScheduler'
        report['recovery_target'] = cfg['recovery_target']
    if cfg.get('r15') is not None:
        # R15: control-quantity choice / proactive-recovery hook on the protected (slack) adapter.
        if engine_cfg['scheduler_cls'] != 'runtime_slack.SlackScheduler' or cfg.get('ablation') is not None:
            raise ValueError('r15 arms are registered only on the protected/slack configuration')
        policy_cfg['r15'] = dict(cfg['r15'])
        engine_cfg['scheduler_cls'] = 'runtime_r15.R15Scheduler'
        report['r15'] = dict(cfg['r15'])
    if cfg.get('r16') is not None:
        # R16: recovery-depth regime and F2 mode on the R15 adapter.
        if engine_cfg['scheduler_cls'] != 'runtime_r15.R15Scheduler':
            raise ValueError('r16 arms are registered only together with r15 on the protected configuration')
        policy_cfg['r16'] = dict(cfg['r16'])
        engine_cfg['scheduler_cls'] = 'runtime_r16.R16Scheduler'
        report['r16'] = dict(cfg['r16'])
    if cfg['policy'] in {'PFS-port-cold', 'PFS-port-warm'}:
        if cfg.get('ablation') is not None:
            raise ValueError('PFS and ablation arms are exclusive')
        engine_cfg['scheduler_cls'] = 'runtime_pfs.PFSControlledScheduler'
    report.update(stage=cfg['stage'], policy=cfg['policy'], engine_config=engine_cfg,
                  versions={'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda},
                  input_hashes={name: sha(getattr(args, name)) for name in ['pool', 'trace', 'histogram', 'config']},
                  model_sha256={str(p.relative_to(Path(args.model))): sha(p) for p in sorted(Path(args.model).glob('*'))
                                if p.is_file() and p.suffix in {'.json', '.safetensors'}},
                  total_trace_requests=len(requests), natural_eos=not fixed, length_mode=length_mode,
                  note='pretokenized engine-client measurements; excludes HTTP/network/tokenization')
    if cfg['stage'] == 'formal' or cfg['policy'] == 'PFS-port-warm':
        if histogram.get('dtype') != cfg['dtype'] or histogram.get('sampling', {}).get('max_tokens') != cfg['max_output_tokens']:
            raise ValueError('formal predictor precision/output cap differs from inference')
        if not trace_hist and (not histogram.get('model_identity') or any(report['model_sha256'].get(name) != value
                                                    for name, value in histogram['model_identity'].items())):
            raise ValueError('formal predictor was fitted using a different model')
    save()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_cfg))
    try:
        # Common fixed warm-up; excluded by request ID and measurement timestamps.
        async def warmup(i):
            prompt = [1229 + i % 4] * [64, 256, 512, 1024][i % 4]
            finished = False
            async for output in engine.generate({'prompt_token_ids': prompt},
                    SamplingParams(temperature=0, max_tokens=32, ignore_eos=True,
                                   output_kind=RequestOutputKind.DELTA, detokenize=False),
                    f'warmup-{i:03d}', priority=1):
                finished = output.finished
            return finished
        warm = await asyncio.wait_for(asyncio.gather(*(warmup(i) for i in range(16))), timeout=180)
        if not all(warm):
            raise RuntimeError('warmup did not complete')
        await asyncio.sleep(1)
        start = time.monotonic()
        report['measurement_start_monotonic_s'] = start
        report['status'] = 'measuring'
        save()
        records = [{'request_id': r['request_id'], 'prompt_id': r['prompt_id'], 'priority': r['priority'],
                    'planned_arrival_s': r['arrival_s'], 'actual_submit_s': None,
                    'first_token_s': None, 'last_token_s': None, 'finish_s': None,
                    'status': 'not_submitted', 'output_tokens': 0,
                    'target_output_tokens': r.get('target_output_tokens') if fixed else None} for r in requests]
        with gzip.open(out / 'client_events.jsonl.gz', 'wt', compresslevel=1) as events:
            def emit(event, **fields):
                events.write(json.dumps({'event': event, 'elapsed_s': time.monotonic() - start, **fields}) + '\n')
            async def serve(record):
                digest = hashlib.sha256()
                try:
                    await asyncio.sleep(max(0, record['planned_arrival_s'] - (time.monotonic() - start)))
                    record.update(status='running', actual_submit_s=time.monotonic() - start)
                    emit('submit', request_id=record['request_id'], planned_arrival_s=record['planned_arrival_s'])
                    prompt = by_id[record['prompt_id']]['prompt_token_ids']
                    params = SamplingParams(temperature=0, max_tokens=record['target_output_tokens'] if fixed
                                            else cfg['max_output_tokens'], ignore_eos=fixed,
                                            output_kind=RequestOutputKind.DELTA, detokenize=False)
                    async for output in engine.generate({'prompt_token_ids': prompt}, params,
                            record['request_id'], priority=record['priority']):
                        now = time.monotonic() - start
                        if len(output.outputs) != 1:
                            raise RuntimeError('expected one sequence per request')
                        part = output.outputs[0]
                        tokens = list(part.token_ids)
                        if tokens:
                            if record['first_token_s'] is None:
                                record['first_token_s'] = now
                            record['last_token_s'] = now
                            record['output_tokens'] += len(tokens)
                            for token in tokens:
                                digest.update(int(token).to_bytes(4, 'big'))
                            emit('token_chunk', request_id=record['request_id'], token_ids=tokens,
                                 cumulative_tokens=record['output_tokens'])
                        if output.finished:
                            record.update(finish_s=now, finish_reason=part.finish_reason,
                                          status='completed' if part.finish_reason in {'stop', 'length'} else 'aborted')
                    if record['status'] == 'running':
                        raise RuntimeError('stream ended without finish notification')
                except asyncio.CancelledError:
                    record['status'] = 'unfinished_at_deadline'
                    raise
                except Exception as exc:
                    record.update(status='request_error', error_type=type(exc).__name__, error=str(exc))
                finally:
                    record['output_sha256'] = digest.hexdigest()
                    emit('request_terminal', **record)
            tasks = [asyncio.create_task(serve(r)) for r in records]
            done, pending = await asyncio.wait(tasks, timeout=window + cfg['drain_s'])
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            emit('measurement_end', unfinished=len(pending))
        report['measurement_elapsed_s'] = time.monotonic() - start
        report['status'] = 'run_complete' if not any(r['status'] == 'request_error' for r in records) else 'request_errors'
        summary = summarize(records, window)
        (out / 'metrics.json').write_text(json.dumps(summary, indent=2))
        report['primary_endpoint'] = summary['overall']['window_slo_goodput_rps']
        report['cohort_slo_met_fraction'] = summary['overall']['slo_met_fraction']
        report['request_states'] = summary['overall']['states']
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    for name in ['model', 'pool', 'trace', 'histogram', 'config', 'out']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parent
    report = {'kind': 'openloop_runtime_measurement', 'status': 'starting',
              'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'runtime_environment': {k: os.environ.get(k) for k in [
                  'VLLM_USE_FLASHINFER_SAMPLER', 'VLLM_BATCH_INVARIANT', 'VLLM_WORKER_MULTIPROC_METHOD',
                  'CUDA_VISIBLE_DEVICES', 'TPDS_GPU_UUID', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}}
    def save():
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    report['source_hashes'] = {}
    for name in ['ablation_policy.py', 'runtime_ablation.py', 'ceiling_policy.py', 'runtime_ceiling.py', 'hysteresis_policy.py', 'runtime_hysteresis.py', 'estimator_policy.py', 'runtime_r15.py', 'r16_policy.py', 'runtime_r16.py', 'slack_policy.py', 'runtime_slack.py', 'revision_policy.py', 'runtime_revision.py', 'admission_reuse.py', 'runtime_scheduler_reuse.py', 'factorial_policy.py', 'runtime_factorial.py', 'run_benchmark_queue.py', 'benchmark_runtime.py', 'serving_metrics.py', 'runtime_scheduler.py', 'policy_core.py',
                 'gpu_run_observer.py', 'runtime_smoke.py', 'runtime_pfs.py', 'pfs_reference.py',
                 'artifact_files.py', 'sim/__init__.py', 'sim/corrected_engine.py', 'sim/prediction.py']:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code / name, target)
        report['source_hashes'][name] = sha(target)
    for name in ['trace', 'histogram', 'config']:
        shutil.copy2(getattr(args, name), out / f'input_{name}.json')
    observer = None
    save()
    try:
        report['gpu_before'] = command(gpu_command(['--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']))
        procs = command(gpu_command(['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']))
        report['processes_before'] = procs
        if procs['exit_code'] or procs['stdout'].strip():
            raise RuntimeError('GPU not idle or occupancy query failed')
        observer = Observer(out)
        observer.start()
        asyncio.run(run(args, report, save))
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc())
    finally:
        if observer:
            report['gpu_observer'] = observer.stop()
            if observer.foreign or observer.errors:
                report['exclude_from_effect_estimate'] = 'external GPU process or incomplete occupancy monitoring'
        report['utc_end'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        try:
            report['compressed_outputs'] = compress_scheduler_log(out)
        except Exception:
            report['archive_error'] = traceback.format_exc()
        report['output_hashes'] = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
        save()
    print(json.dumps({k: report.get(k) for k in ['status', 'primary_endpoint', 'cohort_slo_met_fraction', 'request_states', 'traceback']}))
    return 0 if report['status'] == 'run_complete' and 'exclude_from_effect_estimate' not in report else 1


if __name__ == '__main__':
    sys.exit(main())
