"""Three independent evaluation layers; never promote proxies into gold quality."""
from __future__ import annotations

from statistics import mean

from .metrics import retrieval_metrics


def per_task_retrieval(output: dict, case: dict, *, pinned: bool) -> dict:
    """Use each worker's actual reranked retrieval order, not final evidence order.

    Multi-task cases require explicit mapping of source-backed reference chunks to task IDs.
    A global set of gold chunks does not identify which branch should find each.
    """
    branches = output.get('branch_results') or {}
    task_gold = case.get('relevant_chunk_ids_by_task') or {}
    if not pinned:
        return {'recall_at_5': None, 'precision_at_5': None, 'mrr': None, 'per_task': {},
                'reason': 'unreviewed_or_outdated_gold'}
    if not task_gold and len(branches) == 1 and case.get('relevant_chunk_ids'):
        task_gold = {next(iter(branches)): case['relevant_chunk_ids']}
    if not task_gold or set(task_gold) != set(branches):
        return {'recall_at_5': None, 'precision_at_5': None, 'mrr': None, 'per_task': {},
                'reason': 'gold_task_mapping_missing'}
    results = {}
    for task_id, branch in branches.items():
        ranked = branch.get('ranked_chunk_ids')
        if ranked is None:
            return {'recall_at_5': None, 'precision_at_5': None, 'mrr': None, 'per_task': results,
                    'reason': 'raw_retrieval_ranking_missing'}
        gold = task_gold[task_id]
        if not gold:
            return {'recall_at_5': None, 'precision_at_5': None, 'mrr': None, 'per_task': results,
                    'reason': 'empty_gold_for_task'}
        results[task_id] = retrieval_metrics(ranked, gold, k=5)
    return {'recall_at_5': mean(result['recall_at_k'] for result in results.values()),
            'precision_at_5': mean(result['precision_at_k'] for result in results.values()),
            'mrr': mean(result['mrr'] for result in results.values()),
            'per_task': results, 'reason': None}


def verified_task_execution(output: dict) -> float | None:
    tasks = output.get('subtasks') or {}
    if not tasks:
        return None
    branches = output.get('branch_results') or {}
    return sum(
        t.get('status') == 'complete'
        and branches.get(tid, {}).get('retrieval_status') == 'complete'
        and bool(branches.get(tid, {}).get('evidence'))
        for tid, t in tasks.items()
    ) / len(tasks)


def grouped_layers(metrics: dict) -> dict:
    return {
        'routing_planning': {k: metrics.get(k) for k in (
            'domain_accuracy', 'intent_accuracy', 'task_decomposition_accuracy',
            'tool_calling_accuracy', 'task_completion_rate')},
        'retrieval': {k: metrics.get(k) for k in (
            'source_recall_at_5', 'retrieval_recall_at_5', 'retrieval_mrr')},
        'answer_quality': {k: metrics.get(k) for k in (
            'answer_correctness', 'faithfulness', 'reference_fact_surface_coverage',
            'citation_validity', 'faithfulness_quote_proxy', 'request_facet_coverage_proxy')},
    }
