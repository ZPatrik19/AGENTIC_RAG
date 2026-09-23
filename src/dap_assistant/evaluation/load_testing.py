"""Load/performance evaluation, separated from functional scoring.

Offline metric helpers remain importable through evaluation.professional.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timezone
from hashlib import sha256
from random import Random
from statistics import mean
from time import perf_counter
import uuid

from ..documents.ingestion import load_chunks
from ..rag.dense_resources import acquire_dense, release_dense
from .preflight import acquire_verified_dense
from ..settings import Settings
from .dataset import active_dataset
from .metrics import context_window_utilization, generation_speed, latency_stats
from ..observability.telemetry import ResourceSampler, Telemetry


def _llm_perf(trace: dict, context_window: int, reserved_generation: int) -> dict:
    usages = trace.get('llm_usage', [])
    prompt_tokens = sum(item.get('prompt_eval_count') or 0 for item in usages)
    generated_tokens = sum(item.get('eval_count') or 0 for item in usages)
    eval_duration_s = sum((item.get('eval_duration') or 0) for item in usages) / 1e9
    ttfts = [item.get('ttft_s') for item in usages if item.get('ttft_s') is not None]
    # Different calls do not share one prompt window. Summing prompt tokens
    # across answer/tool calls falsely labels an agentic request as truncated.
    windows = [context_window_utilization(item.get('prompt_eval_count'),
        item.get('requested_num_ctx') or context_window,
        item.get('requested_num_predict') or reserved_generation) for item in usages]
    observed = [item for item in windows if item.get('ratio') is not None]
    peak_window = (max(observed, key=lambda item: item['ratio']) if observed
                   else context_window_utilization(None, context_window, reserved_generation))
    peak_window = {**peak_window, 'over_budget':
        any(item.get('over_budget') for item in observed) if observed else None,
        'measurement': 'maximum_per_call_requested_window_ratio_not_proof_of_truncation'}
    return {
        'prompt_tokens': prompt_tokens or None,
        'generated_tokens': generated_tokens or None,
        'ttft_s': mean(ttfts) if ttfts else None,
        'generation_tokens_per_second': generation_speed(generated_tokens or None, eval_duration_s or None),
        'context_window': context_window,
        'context_window_utilization': peak_window,
        'per_call_context_windows': windows,
    }


def summarize_load_llm(rows: list[dict]) -> dict:
    """Aggregate only LLM measurements actually returned by the provider."""
    from .metrics import percentile
    usages = [row['llm_performance'] for row in rows if row.get('llm_performance')]
    ttfts = [u['ttft_s'] for u in usages if u.get('ttft_s') is not None]
    speeds = [u['generation_tokens_per_second'] for u in usages
              if u.get('generation_tokens_per_second') is not None]
    windows = [u.get('context_window_utilization') or {} for u in usages]
    ratios = [w['ratio'] for w in windows if w.get('ratio') is not None]
    result = {}
    if ttfts:
        result.update(ttft_mean_s=mean(ttfts), ttft_p50_s=percentile(ttfts, .5),
                      ttft_p95_s=percentile(ttfts, .95))
    if speeds:
        result['generation_tokens_per_second_mean'] = mean(speeds)
    if ratios:
        result['context_window_utilization_mean'] = mean(ratios)
        result['context_truncation_count'] = sum(w.get('over_budget') is True for w in windows)
    return result


def _component_stats(rows: list[dict]) -> dict:
    values: dict[str, list[float]] = {}
    for row in rows:
        for span in row.get('node_execution_trace', {}).get('spans', []):
            values.setdefault(span['name'], []).append(span['duration_s'])
    result = {}
    from .metrics import percentile
    for name, durations in values.items():
        result[name] = {
            'invocations': len(durations), 'mean_s': mean(durations),
            'p50_s': percentile(durations, .5), 'p95_s': percentile(durations, .95),
            'max_s': max(durations), 'total_work_s': sum(durations),
        }
    return result


def run_load(settings: Settings, *, scope: str = 'full_workflow', target: str = 'agentic/full',
             topic: str = 'all', request_count: int = 50, concurrency: int = 1,
             timeout_s: float = 60.0, seed: int = 42, warmup: int = 1,
             progress=None) -> dict:
    if not 50 <= request_count <= 200:
        raise ValueError('request_count must be between 50 and 200')
    if concurrency not in (1, 2, 4):
        raise ValueError('concurrency must be 1, 2 or 4')
    from .professional import (
        _ollama_runtime_info, _retrieval_configuration, _prepare_rag_fixture,
        select_cases, available_targets, _environment,
    )
    cases = select_cases(topic=topic)
    runtime_info = (_ollama_runtime_info(settings) if scope == 'full_workflow' else {
        'available': False, 'configured_model': settings.ollama_model,
        'active_context_window': None, 'reason': 'retrieval_only_target',
    })
    active_context = runtime_info.get('active_context_window') or settings.ollama_num_ctx
    effective_settings = replace(
        settings,
        ollama_total_timeout_s=min(settings.ollama_total_timeout_s, timeout_s),
        ollama_read_timeout_s=min(settings.ollama_read_timeout_s, timeout_s),
    )
    chunks = load_chunks(settings.data_dir)
    if not chunks:
        raise RuntimeError('Processed official documents missing; download and index first')
    rng = Random(seed)
    sequence = [rng.choice(cases) for _ in range(request_count + warmup)]
    telemetry = Telemetry()
    dense = (acquire_verified_dense(
        effective_settings, require_inference=scope == 'full_workflow', real_only=True
    ) if settings.embedding_provider == 'sentence_transformers' else acquire_dense(settings))
    sampler = ResourceSampler(0.5)
    try:
        if scope == 'full_workflow':
            from ..workflow import build_workflow, initial_state
            app = build_workflow(effective_settings, dense=dense, telemetry=telemetry)
            def execute(case: dict, run_id: str) -> tuple[dict, str | None]:
                try:
                    state = initial_state(case['question'], reference_date=date.fromisoformat(case['reference_date']))
                    state['run_id'] = run_id
                    out = app.invoke(state, config={'configurable': {'thread_id': run_id}, 'recursion_limit': 30})
                    return out, None
                except Exception as exc:
                    return {}, f'{type(exc).__name__}: {exc}'
        else:
            mapping = available_targets().get(scope, {})
            if target not in mapping:
                raise ValueError('Invalid load-test target')
            nodes = mapping[target]
            from dap_assistant.rag.rag_graph import build_rag_graph
            app = build_rag_graph(effective_settings, dense=dense, telemetry=telemetry, selected_nodes=nodes)
            def execute(case: dict, run_id: str) -> tuple[dict, str | None]:
                try:
                    fixture = _prepare_rag_fixture(case, run_id, nodes, effective_settings, dense)
                    fixture['run_id'] = run_id
                    out = app.invoke(fixture, config={'recursion_limit': 16})
                    return out, None
                except Exception as exc:
                    return {}, f'{type(exc).__name__}: {exc}'

        # Warm-up is explicit and excluded from measured percentiles.
        warmup_rows = []
        for case in sequence[:warmup]:
            rid = str(uuid.uuid4())
            started = perf_counter()
            out, err = execute(case, rid)
            warmup_rows.append({'question_id': case['question_id'], 'latency_s': perf_counter() - started,
                                'success': err is None, 'error': err})

        rows: list[dict] = []
        sampler.start()
        wall_started = perf_counter()
        measured = sequence[warmup:]

        def one(index_case: tuple[int, dict]) -> dict:
            index, case = index_case
            rid = str(uuid.uuid4())
            began = perf_counter()
            out, err = execute(case, rid)
            latency = perf_counter() - began
            trace = telemetry.snapshot(rid)
            # Retrieval-only nodes emit no LLM usage; do not fabricate LLM metrics.
            llm_perf = (_llm_perf(trace, active_context, effective_settings.ollama_num_predict)
                        if trace.get('llm_usage') else None)
            generation_timeout = any(f.get('failure_kind') in ('http_timeout', 'total_timeout')
                                     for f in trace.get('llm_failures', []))
            evidence_count = len(out.get('evidence', [])) if out else 0
            return {
                'request_index': index, 'question_id': case['question_id'], 'topic': case['category'],
                'question': case['question'], 'latency_s': latency, 'success': err is None,
                'error': err, 'timeout': bool(generation_timeout or
                    (err and 'timeout' in err.casefold()) or latency > timeout_s),
                'generation_timeout': generation_timeout, 'sla_exceeded': latency > timeout_s,
                'answer_fallback': bool(out.get('answer_fallback')),
                'llm_failure_count': len(trace.get('llm_failures', [])),
                'response_status': out.get('response_status') if out else 'error',
                'evidence_count': evidence_count, 'node_execution_trace': trace,
                **({'llm_performance': llm_perf} if llm_perf is not None else {}),
            }

        if concurrency == 1:
            for index, case in enumerate(measured, 1):
                rows.append(one((index, case)))
                if progress:
                    progress(index, request_count)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(one, (index, case)): index
                           for index, case in enumerate(measured, 1)}
                done = 0
                for future in as_completed(futures):
                    rows.append(future.result())
                    done += 1
                    if progress:
                        progress(done, request_count)
            rows.sort(key=lambda row: row['request_index'])
        elapsed = perf_counter() - wall_started
        resource = sampler.stop()
    finally:
        if sampler._thread is not None and sampler._thread.is_alive():
            sampler.stop()
        release_dense(dense)

    latencies = [row['latency_s'] for row in rows]
    stats = latency_stats(latencies, elapsed)
    successful = sum(row['success'] for row in rows)
    component_stats = _component_stats(rows)
    return {
        'kind': 'load_v4', 'run_id': str(uuid.uuid4()),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': {
            'scope': scope, 'target': target, 'topic': topic,
            'request_count': request_count, 'concurrency': concurrency,
            'timeout_s': timeout_s, 'seed': seed, 'warmup': warmup,
            'timeout_semantics': 'latency_SLA_and_generation_timeouts_not_whole_workflow_cancellation',
            'model': settings.ollama_model, 'context_window': active_context,
            'configured_context_window': settings.ollama_num_ctx, 'model_runtime': runtime_info,
            'retrieval_configuration': _retrieval_configuration(settings),
            'dataset_version': '4.0', 'dataset_sha256': sha256(active_dataset().read_bytes()).hexdigest(),
            'question_distribution': {case['question_id']: sum(x['question_id'] == case['question_id'] for x in measured)
                                      for case in cases},
        },
        'execution_environment': _environment(settings, chunks),
        'rows': rows,
        'resources': resource,
        'summary': {
            **stats,
            'success_count': successful,
            'failure_count': request_count - successful,
            'error_rate': (request_count - successful) / request_count,
            'success_semantics': 'workflow_returned_without_exception_not_answer_correctness',
            'response_status_counts': {status: sum(r['response_status'] == status for r in rows)
                                       for status in sorted({str(r['response_status']) for r in rows})},
            'answer_fallback_count': sum(r['answer_fallback'] for r in rows),
            'generation_timeout_count': sum(r['generation_timeout'] for r in rows),
            'sla_exceeded_count': sum(r['sla_exceeded'] for r in rows),
            'llm_failure_count': sum(r['llm_failure_count'] for r in rows),
            'timeout_rate': sum(row['timeout'] for row in rows) / request_count,
            **summarize_load_llm(rows),
            'component_stats': component_stats,
            'warmup': warmup_rows,
            'resource_summary': resource['summary'],
        },
    }
