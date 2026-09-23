"""Offline scoring and telemetry aggregation for paired RAG comparisons.

This module does not initialize Qdrant, make Ollama calls, or execute graphs.
"""
from __future__ import annotations

from statistics import mean

from .dataset import reference_status, validate_relevant_chunks
from .metrics import (
    answer_completeness, average_available, cite_validity, citation_coverage,
    claim_support_breakdown, retrieval_metrics, source_precision, source_recall,
    workflow_success,
)
from .semantic_review import semantic_review
from ..response.quality import claim_provenance_report
from ..settings import Settings

_SIMPLE_METRICS = (
    'retrieval_recall_at_5', 'retrieval_precision_at_5', 'retrieval_mrr',
    'source_recall_at_5', 'source_precision_at_5',
    'answer_completeness', 'answer_text_integrity', 'answer_facet_coverage_proxy', 'faithfulness',
    'citation_integrity_proxy', 'semantic_review_coverage', 'citation_accuracy',
    'citation_coverage', 'unsupported_claim_rate', 'workflow_success_rate',
)


def _model_generated(trace: dict, settings: Settings) -> bool | None:
    """Valid Ollama answer JSON, independent of whether its claims are usable."""
    if settings.llm_provider != 'ollama' or settings.answer_mode == 'source':
        return None
    return any(item.get('phase') == 'answer' for item in trace.get('llm_successes', []))

def _simple_metrics(case: dict, output: dict, ranked_ids: list[str], versions: dict[str, str],
                    chunks: list[dict], runtime_error: str | None) -> tuple[dict, dict]:
    refs = reference_status(case, versions)
    if refs in ('pinned', 'automatic_proxy_pinned') and not validate_relevant_chunks(case, chunks):
        refs = 'invalid_chunk_reference'
    pinned = refs in ('pinned', 'automatic_proxy_pinned')
    retrieval = retrieval_metrics(ranked_ids, case.get('relevant_chunk_ids') if pinned else None, k=5)
    support = claim_support_breakdown(output)
    semantic = semantic_review(output, case)
    metrics = {
        'retrieval_recall_at_5': retrieval['recall_at_k'],
        'retrieval_precision_at_5': retrieval['precision_at_k'],
        'retrieval_mrr': retrieval['mrr'],
        'source_recall_at_5': (source_recall(output.get('evidence', [])[:5], case.get('expected_source_ids', []))
                               if pinned else None),
        'source_precision_at_5': (source_precision(output.get('evidence', [])[:5], case.get('expected_source_ids', []), 5)
                                  if pinned else None),
        'answer_text_integrity': (float(not any(output['answer_draft']['text_integrity'].get(k)
                                                 for k in ('incomplete_claims', 'model_warnings', 'duplicate_claims')))
                                  if output.get('answer_draft', {}).get('text_integrity')
                                  and output.get('answer_draft', {}).get('claims') else None),
        'answer_facet_coverage_proxy': output.get('answer_draft', {}).get('requested_facet_coverage', {}).get('coverage'),
        'answer_completeness': (answer_completeness(output.get('final_answer', ''), case.get('expected_facts', []))
                                if pinned and case.get('human_reviewed') else None),
        'faithfulness': semantic['faithfulness'],
        'citation_integrity_proxy': support['citation_integrity_proxy'],
        'semantic_review_coverage': semantic['review_coverage'],
        'citation_accuracy': cite_validity(output),
        'citation_coverage': citation_coverage(output),
        'unsupported_claim_rate': support['unsupported_claim_rate'],
        'workflow_success_rate': workflow_success(runtime_error, output.get('response_status')),
    }
    return metrics, {'reference_status': refs,
                     'reference_type': 'automatic_silver_proxy' if refs == 'automatic_proxy_pinned' else 'human_gold' if refs == 'pinned' else 'unavailable',
                     'retrieval': retrieval, 'claim_support': support,
                     'semantic_support': semantic,
                     'text_integrity': output.get('answer_draft', {}).get('text_integrity', {}),
                     'requested_facet_coverage': output.get('answer_draft', {}).get('requested_facet_coverage', {}),
                     'claim_provenance': claim_provenance_report(
                         output.get('answer_draft', {}).get('claims', []),
                         output.get('answer_draft', {}).get('model_context_evidence_ids', []))}

def _usage(trace: dict) -> dict:
    rows = trace.get('llm_usage', [])
    return {
        'llm_calls': len(trace.get('llm_successes', [])),  # successfully validated replies only
        'llm_responses_with_usage': len(rows),  # includes failed JSON/length responses
        'generated_tokens_observed': (sum(item['eval_count'] for item in rows
                                      if isinstance(item.get('eval_count'), int))
                                      if any(isinstance(item.get('eval_count'), int) for item in rows)
                                      else None),
        'token_usage_complete': (bool(rows) and all(isinstance(item.get('eval_count'), int)
                                                       for item in rows)),
        'llm_attempts': len(trace.get('llm_attempts', [])),
        'llm_failures': len(trace.get('llm_failures', [])),
        'answer_generation_completed': any(e.get('phase') == 'answer'
                                          for e in trace.get('llm_successes', [])),
        'prompt_tokens': sum(item.get('prompt_eval_count') or 0 for item in rows),
        'generated_tokens': sum(item.get('eval_count') or 0 for item in rows),
    }

def _mode_summary(rows: list[dict], metric_names: tuple[str, ...]) -> dict:
    return {
        'cases': len(rows),
        'successful_runs': sum(row['success'] for row in rows),
        'failed_runs': sum(not row['success'] for row in rows),
        'complete_answers': sum(row.get('response_status') == 'complete' for row in rows),
        'partial_answers': sum(row.get('response_status') == 'partial' for row in rows),
        'fallback_answers': sum(bool(row.get('answer_fallback')) for row in rows),
        'generation_successful': sum(row.get('generation_success') is True for row in rows),
        'usable_model_answers': sum(row.get('model_answer_usable') is True for row in rows),
        'llm_attempts_mean': mean(row['llm_usage'].get('llm_attempts', 0) for row in rows) if rows else None,
        'llm_failures_mean': mean(row['llm_usage'].get('llm_failures', 0) for row in rows) if rows else None,
        'latency_mean_s': mean(row['latency_s'] for row in rows) if rows else None,
        'llm_calls_mean': mean(row['llm_usage']['llm_calls'] for row in rows) if rows else None,
        'prompt_tokens_mean': mean(row['llm_usage']['prompt_tokens'] for row in rows) if rows else None,
        'generated_tokens_mean': mean(row['llm_usage']['generated_tokens'] for row in rows) if rows else None,
        'metrics': {name: average_available([row['metrics'] for row in rows], name)
                    for name in metric_names},
    }

def _paired_deltas(summary: dict) -> dict:
    baseline = summary.get('baseline_dense', {})
    out: dict[str, dict] = {}
    for mode in ('hybrid_rrf', 'agentic'):
        current = summary.get(mode, {})
        deltas = {
            'latency_mean_s': ((current.get('latency_mean_s') - baseline.get('latency_mean_s'))
                               if current.get('latency_mean_s') is not None
                               and baseline.get('latency_mean_s') is not None else None),
            'llm_calls_mean': ((current.get('llm_calls_mean') - baseline.get('llm_calls_mean'))
                               if current.get('llm_calls_mean') is not None
                               and baseline.get('llm_calls_mean') is not None else None),
        }
        metric_deltas = {}
        for name, item in current.get('metrics', {}).items():
            cur = item.get('value')
            base = baseline.get('metrics', {}).get(name, {}).get('value')
            metric_deltas[name] = cur - base if cur is not None and base is not None else None
        deltas['metrics'] = metric_deltas
        out[f'{mode}_minus_baseline_dense'] = deltas
    return out
