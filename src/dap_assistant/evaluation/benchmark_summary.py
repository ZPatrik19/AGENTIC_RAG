"""Deterministic load-benchmark aggregation; no model, database or graph imports.

Inclusive node timings are work, not additive wall time. The warm-up set is
reported separately and excluded from measured percentiles and throughput.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from .metrics import latency_stats, percentile


def summarize_benchmark_rows(
    rows: list[dict], *, mode: str, count: int, elapsed: float,
    concurrency: int, cpu_before: float, cpu_after: float,
    rss_before: int | None, rss_after: int | None,
    warmup: int, warmup_rows: list[dict],
) -> dict:
    """Summarize measured requests; adds unattributed_wall_s to each measured row."""
    successful = sum(r['success'] for r in rows)
    by_component: dict[str, list[float]] = defaultdict(list)
    ollama_usage = {'prompt_tokens': 0, 'generated_tokens': 0, 'eval_duration_ns': 0,
                    'prompt_eval_duration_ns': 0, 'total_duration_ns': 0,
                    'load_duration_ns': 0, 'llm_calls_with_usage': 0}
    for row in rows:
        for span in row['trace']['spans']:
            by_component[span['name']].append(span['duration_s'])
        for usage in row['trace']['llm_usage']:
            if usage.get('eval_count') is not None:
                ollama_usage['llm_calls_with_usage'] += 1
            for key, dest in [('prompt_eval_count', 'prompt_tokens'), ('eval_count', 'generated_tokens'),
                              ('eval_duration', 'eval_duration_ns'), ('prompt_eval_duration', 'prompt_eval_duration_ns'),
                              ('total_duration', 'total_duration_ns'), ('load_duration', 'load_duration_ns')]:
                ollama_usage[dest] += usage.get(key) or 0
    ollama_usage['generation_tokens_per_second'] = (
        ollama_usage['generated_tokens'] / (ollama_usage['eval_duration_ns'] / 1e9)
        if ollama_usage['eval_duration_ns'] else None)
    for row in rows:
        selection = ('main/' if mode == 'e2e' else 'rag/')
        intervals = [(span['start_s'], span['end_s']) for span in row['trace']['spans']
                     if span['name'].startswith(selection)]
        row['unattributed_wall_s'] = max(0.0, row['latency_s'] - _interval_union(intervals))
    component_stats = {name: {'total_work_s': sum(times), 'mean_invocation_s': mean(times),
                              'mean_work_per_request_s': sum(times) / count,
                              'p50_invocation_s': percentile(times, .5),
                              'p95_invocation_s': percentile(times, .95),
                              'max_invocation_s': max(times),
                              'invocations': len(times),
                              'work_over_wall_ratio': sum(times) / elapsed if elapsed else None}
                       for name, times in by_component.items()}
    atomic_names = {'llm_inference', 'embedding_query', 'dense_retrieval',
                    'bm25_retrieval', 'retrieval_fusion', 'fee_coverage_lookup',
                    'rag/process_query', 'rag/rerank_results', 'rag/evaluate_evidence',
                    'rag/prepare_context', 'main/classify_intent', 'main/plan_tasks',
                    'main/evidence_gate', 'main/execute_tools', 'main/answer_audit'}
    atomic = {k: v for k, v in component_stats.items() if k in atomic_names}
    # Aggregate inclusive spans in a separate list; overlapping nested/parallel spans are not additive.
    bottleneck = max(atomic or component_stats, key=lambda k: (atomic or component_stats)[k]['total_work_s'], default=None)
    observed_coverage = [
        coverage['coverage'] for row in rows
        if (coverage := row.get('request_coverage'))
        and coverage.get('coverage') is not None
    ]
    summary = {**latency_stats([r['latency_s'] for r in rows], elapsed),
               'success_count': successful, 'failure_count': count - successful,
               'error_rate': (count - successful) / count,
               'elapsed_wall_s': elapsed, 'concurrency': concurrency,
               'mean_unattributed_wall_s': mean(r['unattributed_wall_s'] for r in rows),
               'application_cpu_time_s': cpu_after - cpu_before,
               'application_cpu_time_note': 'Process CPU time, all Python threads; excludes separate Ollama process.',
               'application_rss_before_bytes': rss_before,
               'application_rss_after_bytes': rss_after,
               'application_rss_delta_bytes': (rss_after - rss_before if rss_before is not None
                                               and rss_after is not None else None),
               'memory_sampling_note': 'Application RSS sampled before and after measured run; '
                   'not peak memory or Ollama process/GPU memory.',
               'retrieval_empty_count': sum(not row['evidence_count'] for row in rows),
               'mean_evidence_count': mean(r['evidence_count'] for r in rows),
               'retrieval_queries_per_request': mean(sum(
                   span['name'] == 'dense_retrieval' for span in r['trace']['spans'])
                   for r in rows),
               'response_status_counts': dict(Counter(str(row['response_status']) for row in rows)),
               'error_type_counts': dict(Counter(
                   (row.get('error') or '').split(':', 1)[0] for row in rows if not row['success'])),
               'domain_breakdown': _domain_summary(rows),
               'request_facet_coverage_proxy': (mean(observed_coverage) if observed_coverage else None),
               'partial_answer_count': sum(r.get('response_status') == 'partial' for r in rows),
               'complete_answer_count': sum(r.get('response_status') == 'complete' for r in rows),
               'unattributed_wall_note': 'Approximate graph orchestration/setup/overhead outside timed node spans; not a pure scheduler measurement.',
               'warmup': {'count': warmup, 'rows': warmup_rows},
               'component_stats': component_stats, 'ollama_usage': ollama_usage,
               'bottleneck_by_total_work': bottleneck,
               'bottleneck_warning': 'Inclusive node and RAG spans overlap nested calls; work_over_wall_ratio is NOT additive and can exceed 100% under concurrency.',
               'optimization_candidates': _optimization_candidates(atomic)}
    return summary


def _domain_summary(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get('domain') or 'unknown'].append(row)
    return {
        domain: {
            'count': len(group),
            'p50_s': percentile([r['latency_s'] for r in group], .5),
            'p95_s': percentile([r['latency_s'] for r in group], .95),
            'errors': sum(not r['success'] for r in group),
        }
        for domain, group in sorted(groups.items())
    }


def _interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def _optimization_candidates(stats: dict) -> list[str]:
    if not stats:
        return ['No component timings were recorded; diagnose instrumentation before proposing optimization.']
    ranked = sorted(stats, key=lambda name: stats[name]['total_work_s'], reverse=True)
    suggestions = {
        'llm_inference': 'If LLM dominates, reduce duplicate classification/planning calls and shorten evidence context; re-measure on identical queries.',
        'embedding_query': 'If embedding is expensive, retain the embedding model in memory and consider query embedding cache.',
        'dense_retrieval': 'If dense search dominates, inspect payload filters and collection size.',
        'bm25_retrieval': 'If BM25 dominates, persist an inverted index rather than re-tokenizing all chunks per query.',
        'rag/rerank_results': 'If reranking dominates, reduce candidates while checking Recall@K on pinned labels.',
        'main/answer_audit': 'If auditing dominates, batch validation and avoid unnecessary duplicate model calls.',
    }
    result = []
    for key in ranked[:2]:
        observed = stats[key]
        detail = (f'{key}: {observed.get("mean_invocation_s", observed["total_work_s"]):.4f} s/hívás, '
                  f'{observed.get("invocations", 1)} hívás; összes mért munka '
                  f'{observed["total_work_s"]:.3f} s. ')
        result.append(detail + suggestions.get(key,
            'Vizsgáld meg az alkomponenseket külön; az inkluzív idők átfedhetnek.'))
    return result
