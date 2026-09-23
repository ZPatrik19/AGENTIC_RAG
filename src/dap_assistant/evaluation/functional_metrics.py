"""Calculate functional metrics from observed artifacts and reference labels, without model calls."""
from __future__ import annotations

import json

from .dataset import reference_status, validate_relevant_chunks
from .layers import per_task_retrieval
from .semantic_review import semantic_review
from .metrics import (
    abstention_accuracy, answer_completeness, cite_validity, citation_coverage,
    claim_support_breakdown, context_metrics, recovery_success, retrieval_metrics,
    source_precision, source_recall, task_completion_score, task_coverage,
    tool_accuracy, tool_call_efficiency, workflow_success,
)
from ..observability.telemetry import Telemetry
from ..context_engineering.information_needs import answer_facets


def _prompt_context_ids(telemetry: Telemetry, run_id: str) -> list[str]:
    """Evidence in the LAST answer prompt (including a successful recovery)."""
    ids: list[str] = []
    for request in telemetry.prompt_preview(run_id):
        if request.get('phase') != 'answer':
            continue
        messages = request.get('messages') or []
        if not messages:
            continue
        try:
            payload = json.loads(messages[-1].get('content', '{}'))
        except (json.JSONDecodeError, TypeError):
            continue
        current_ids: list[str] = []
        for item in payload.get('evidence', []):
            evidence_id = item.get('evidence_id')
            if evidence_id:
                current_ids.append(evidence_id)
        ids = current_ids
    return ids


def _context_chunk_ids(output: dict, telemetry: Telemetry, run_id: str) -> list[str]:
    evidence_by_id = {item.get('evidence_id'): item for item in output.get('evidence', [])}
    prompt_ids = _prompt_context_ids(telemetry, run_id)
    if prompt_ids:
        return list(dict.fromkeys(evidence_by_id[eid]['chunk_id'] for eid in prompt_ids
                                  if eid in evidence_by_id))
    # Source-only mode performs no LLM prompt; the prepared evidence is the effective context.
    return list(dict.fromkeys(item.get('chunk_id') for item in output.get('evidence', [])
                              if item.get('chunk_id')))




def request_coverage(question: str, output: dict) -> dict:
    """Measure requested facets from audited claims only.

    A failed answer audit contributes no accepted claims. Unsupported claims are
    excluded when an older workflow payload does not carry precomputed coverage.
    """
    validation = output.get('answer_validation') or {}
    recorded = validation.get('requested_facet_coverage') or validation.get('request_coverage')
    if isinstance(recorded, dict):
        return recorded
    if validation.get('status') == 'failed':
        return answer_facets(question, [])
    rejected = validation.get('unsupported_claims') or []
    claims = [
        claim for claim in output.get('answer_draft', {}).get('claims', [])
        if not any(str(claim.get('text', '')).startswith(str(text)) for text in rejected)
    ]
    return answer_facets(question, claims)


def _ranked_source_evidence(output: dict, chunks: list[dict]) -> list[dict]:
    """Resolve each branch's real pre-audit top-5 ranking to corpus chunks."""
    indexed = {chunk.get('chunk_id'): chunk for chunk in chunks}
    ranked: list[dict] = []
    for task in output.get('branch_results', {}).values():
        for chunk_id in task.get('ranked_chunk_ids', [])[:5]:
            chunk = indexed.get(chunk_id)
            if chunk is not None:
                ranked.append(chunk)
    return ranked

def _full_metrics(case: dict, output: dict, trace: dict, versions: dict[str, str], chunks: list[dict],
                  telemetry: Telemetry, run_id: str, error: str | None = None) -> tuple[dict, dict]:
    refs = reference_status(case, versions)
    if refs in ('pinned', 'automatic_proxy_pinned') and not validate_relevant_chunks(case, chunks):
        refs = 'invalid_chunk_reference'
    pinned = refs in ('pinned', 'automatic_proxy_pinned')
    evidence = output.get('evidence', [])
    ranked_source_evidence = _ranked_source_evidence(output, chunks)
    context_ids = _context_chunk_ids(output, telemetry, run_id)
    retrieval = per_task_retrieval(output, case, pinned=pinned)
    context = context_metrics(
        context_ids,
        case.get('relevant_chunk_ids') if pinned else None,
        case.get('expected_subtasks'),
        case.get('relevant_chunk_ids_by_task') if pinned else None,
    )
    subtask = task_coverage(output.get('subtasks', {}), case.get('expected_subtasks'))
    completeness = (answer_completeness(output.get('final_answer', ''), case.get('expected_facts', []))
                    if pinned and case.get('human_reviewed') else None)
    support = claim_support_breakdown(output)
    semantic = semantic_review(output, case)
    wf_success = workflow_success(error, output.get('response_status'))
    task_complete = task_completion_score(
        subtask_coverage=subtask,
        answer_completeness_score=completeness,
        workflow_success=wf_success,
        answerability=case.get('answerability'),
    )
    # A source-only fallback is useful, but it did not complete an answerable
    # user's task. Never equate successful retrieval with answer completion.
    if output.get('answer_fallback') or (
        case.get('answerability') == 'answerable'
        and output.get('response_status') not in ('complete', None)
    ):
        task_complete = 0.0
    efficiency = tool_call_efficiency(output.get('tool_results', {}), case.get('expected_tool_calls'))
    metrics = {
        'domain_accuracy': float(set(output.get('domains', [])) == set(case.get('expected_domains', []))),
        'intent_accuracy': (float(set(output.get('intents', [])) == set(case.get('expected_intents', [])))
                            if output.get('intents') is not None else None),
        'subtask_coverage': subtask,
        'task_decomposition_accuracy': subtask,
        'retrieval_recall_at_5': retrieval['recall_at_5'],
        'retrieval_precision_at_5': retrieval['precision_at_5'],
        'retrieval_mrr': retrieval['mrr'],
        'source_recall_at_5': (source_recall(ranked_source_evidence, case.get('expected_source_ids', []))
                               if pinned and ranked_source_evidence else None),
        'source_precision_at_5': (source_precision(ranked_source_evidence, case.get('expected_source_ids', []), 5)
                                  if pinned and ranked_source_evidence else None),
        'context_recall': context['context_recall'],
        'context_precision': context['context_precision'],
        'context_coverage': context['context_coverage'],
        'answer_completeness': completeness,
        'answer_facet_coverage_proxy': request_coverage(case.get('question', ''), output).get('coverage'),
        'answer_text_integrity': (float(not any(output['answer_validation']['text_integrity'].get(k)
                                                 for k in ('incomplete_claims', 'model_warnings', 'duplicate_claims')))
                                  if output.get('answer_validation', {}).get('text_integrity')
                                  and output.get('answer_draft', {}).get('claims') else None),
        'answer_correctness': None,
        'answer_relevancy': None,
        'faithfulness': semantic['faithfulness'],
        'citation_integrity_proxy': support['citation_integrity_proxy'],
        'semantic_review_coverage': semantic['review_coverage'],
        'unsupported_claim_rate': semantic['unsupported_claim_rate'],
        'contradicted_claim_rate': semantic['contradicted_claim_rate'],
        'citation_accuracy': cite_validity(output),
        'citation_coverage': citation_coverage(output),
        'abstention_accuracy': abstention_accuracy(output.get('response_status'),
                                                   case.get('expected_behavior'),
                                                   case.get('answerability')),
        'tool_selection_accuracy': tool_accuracy(output.get('tool_results', {}),
                                                 case.get('expected_tool_calls')),
        'tool_call_efficiency': efficiency['score'],
        'task_completion_rate': task_complete,
        'recovery_success_rate': recovery_success(trace, output.get('response_status')),
        'workflow_success_rate': wf_success,
    }
    details = {
        'reference_status': refs,
        'reference_type': 'automatic_silver_proxy' if refs == 'automatic_proxy_pinned' else 'human_gold' if refs == 'pinned' else 'unavailable',
        'retrieval': retrieval,
        'context': context,
        'claim_support': support,
        'semantic_support': semantic,
        'tool_utilization': output.get('answer_validation', {}).get('tool_utilization', {}),
        'text_integrity': output.get('answer_validation', {}).get('text_integrity', {}),
        'requested_facet_coverage': output.get('answer_validation', {}).get('requested_facet_coverage', {}),
        'tool_efficiency': efficiency,
        'final_context_chunk_ids': context_ids,
        'claim_provenance': output.get('answer_validation', {}).get('claim_provenance', {}),
        'native_tool_trace': output.get('native_tool_trace', {}),
    }
    return metrics, details


def _rag_metrics(case: dict, output: dict, versions: dict[str, str], chunks: list[dict],
                 *, nodes: tuple[str, ...] | None = None) -> tuple[dict, dict]:
    refs = reference_status(case, versions)
    if refs in ('pinned', 'automatic_proxy_pinned') and not validate_relevant_chunks(case, chunks):
        refs = 'invalid_chunk_reference'
    pinned = refs in ('pinned', 'automatic_proxy_pinned')
    # A standalone node is scored only on the artifacts that it actually
    # produces. A reranker/evidence gate receives upstream ranking as an input
    # fixture; that input is NOT a retrieval score for the downstream node.
    # None retains backwards compatibility for legacy direct metric callers.
    selected = set(nodes) if nodes is not None else {
        'hybrid_retrieval', 'rerank_results', 'evaluate_evidence', 'prepare_context',
    }
    if 'rerank_results' in selected:
        ranked = output.get('ranked_chunk_ids') or [
            item.get('chunk_id') for item in output.get('ranked', [])
        ]
    elif 'hybrid_retrieval' in selected:
        ranked = [item.get('chunk_id') for item in output.get('candidates', [])]
    else:
        ranked = []
    ranked = [item for item in ranked if item]
    retrieval = retrieval_metrics(
        ranked, case.get('relevant_chunk_ids') if pinned else None, 5
    ) if selected & {'hybrid_retrieval', 'rerank_results'} else {
        'recall_at_k': None, 'precision_at_k': None, 'mrr': None,
        'reason': 'node_did_not_run_retrieval_or_rerank',
    }
    context_ids = [item.get('chunk_id') for item in output.get('evidence', [])
                   if item.get('chunk_id')] if selected & {'evaluate_evidence', 'prepare_context'} else []
    context = context_metrics(
        context_ids,
        case.get('relevant_chunk_ids') if pinned else None,
        case.get('expected_subtasks'),
        case.get('relevant_chunk_ids_by_task') if pinned else None,
    ) if selected & {'evaluate_evidence', 'prepare_context'} else {
        'context_recall': None, 'context_precision': None, 'context_coverage': None,
        'reason': 'node_did_not_prepare_evidence',
    }
    metrics = {
        'retrieval_recall_at_5': retrieval['recall_at_k'],
        'retrieval_precision_at_5': retrieval['precision_at_k'],
        'retrieval_mrr': retrieval['mrr'],
        'context_recall': context['context_recall'],
        'context_precision': context['context_precision'],
        'context_coverage': context['context_coverage'],
        # Explicit N/A for metrics that cannot exist at RAG-node level.
        'answer_completeness': None,
        'faithfulness': None,
        'citation_accuracy': None,
        'answer_correctness': None,
        'answer_relevancy': None,
        'unsupported_claim_rate': None,
        'citation_coverage': None,
        'abstention_accuracy': None,
        'tool_selection_accuracy': None,
        'task_completion_rate': None,
        'subtask_coverage': None,
        'tool_call_efficiency': None,
        'recovery_success_rate': None,
        'workflow_success_rate': 1.0,
    }
    return metrics, {'reference_status': refs,
                     'reference_type': 'automatic_silver_proxy' if refs == 'automatic_proxy_pinned' else 'human_gold' if refs == 'pinned' else 'unavailable',
                     'retrieval': retrieval, 'context': context,
                     'final_context_chunk_ids': context_ids}


_FUNCTIONAL_METRIC_NAMES = (
    'domain_accuracy', 'intent_accuracy', 'task_decomposition_accuracy',
    'retrieval_recall_at_5', 'retrieval_precision_at_5', 'retrieval_mrr',
    'source_recall_at_5', 'source_precision_at_5',
    'context_recall', 'context_precision', 'context_coverage',
    'answer_completeness', 'faithfulness', 'citation_integrity_proxy',
    'semantic_review_coverage', 'citation_accuracy',
    'answer_correctness', 'answer_relevancy', 'unsupported_claim_rate',
    'contradicted_claim_rate', 'citation_coverage', 'abstention_accuracy',
    'tool_selection_accuracy', 'task_completion_rate', 'subtask_coverage',
    'tool_call_efficiency', 'recovery_success_rate', 'workflow_success_rate',
)
