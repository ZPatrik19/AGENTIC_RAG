"""Deterministic metrics; return N/A when required references are unavailable."""
from __future__ import annotations

from math import log2
from hashlib import sha256
from ..response.quality import claim_identifier
import re
from statistics import mean


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    numbers = sorted(values)
    position = (len(numbers) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(numbers) - 1)
    return numbers[lo] + (numbers[hi] - numbers[lo]) * (position - lo)


def latency_stats(values: list[float], elapsed_s: float) -> dict:
    return {
        'count': len(values),
        'mean_s': mean(values) if values else None,
        'min_s': min(values) if values else None,
        'max_s': max(values) if values else None,
        'p50_s': percentile(values, .50),
        'p95_s': percentile(values, .95),
        'p99_s': percentile(values, .99),
        'throughput_qps': ratio(len(values), elapsed_s),
    }


def retrieval_metrics(ranked_chunk_ids: list[str], relevant_chunk_ids: list[str] | None,
                      k: int = 5) -> dict:
    """Chunk-level Recall@K, Precision@K and reciprocal rank.

    Precision uses the number of actually returned results (up to K) as denominator.
    If no result is returned while relevant chunks are expected, precision is 0.
    """
    if not relevant_chunk_ids:
        return {
            'recall_at_k': None,
            'precision_at_k': None,
            'mrr': None,
            'reason': 'no_pinned_relevant_chunk_ids',
        }
    relevant = set(relevant_chunk_ids)
    # Multiple retrieval branches can return the same chunk; never count a
    # duplicate as another result or penalize the next unique ranked hit.
    ranked_chunk_ids = list(dict.fromkeys(ranked_chunk_ids))
    top = ranked_chunk_ids[:k]
    hits = len(set(top) & relevant)
    first = next((rank for rank, chunk in enumerate(ranked_chunk_ids, 1)
                  if chunk in relevant), None)
    precision = hits / len(top) if top else 0.0
    return {
        'recall_at_k': hits / len(relevant),
        'precision_at_k': precision,
        'mrr': 1 / first if first is not None else 0.0,
        'reason': None,
    }


def source_recall(evidence: list[dict], expected_ids: list[str]) -> float | None:
    if not expected_ids:
        return None
    return len({e.get('document_id') for e in evidence} & set(expected_ids)) / len(set(expected_ids))


def source_precision(evidence: list[dict], expected_ids: list[str], k: int = 5) -> float | None:
    if not expected_ids:
        return None
    top = evidence[:k]
    if not top:
        return 0.0
    relevant = set(expected_ids)
    return sum(e.get('document_id') in relevant for e in top) / len(top)


def _binary_ndcg(ranked_ids: list[str], relevant_ids: set[str]) -> float | None:
    """Rank-sensitive binary nDCG used as Context Precision.

    This is intentionally separate from set precision: relevant chunks appearing
    earlier receive more credit. The score is normalized by the ideal ranking.
    """
    if not relevant_ids:
        return None
    if not ranked_ids:
        return 0.0
    gains = [1.0 if item in relevant_ids else 0.0 for item in ranked_ids]
    dcg = sum(gain / log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(len(relevant_ids), len(ranked_ids))
    ideal = sum(1.0 / log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal if ideal else 0.0


def context_metrics(context_chunk_ids: list[str], relevant_chunk_ids: list[str] | None,
                    expected_subtasks: list[dict] | None = None,
                    relevant_chunk_ids_by_task: dict[str, list[str]] | None = None) -> dict:
    """Context Recall, rank-sensitive Context Precision and task Context Coverage."""
    if not relevant_chunk_ids:
        return {
            'context_recall': None,
            'context_precision': None,
            'context_coverage': None,
            'reason': 'no_pinned_context_reference',
        }
    # Several answer attempts can reference one chunk. nDCG must count each
    # distinct evidence unit once, or repeated gold may exceed a score of 1.
    context_chunk_ids = list(dict.fromkeys(context_chunk_ids))
    relevant = set(relevant_chunk_ids)
    present = set(context_chunk_ids)
    recall = len(present & relevant) / len(relevant)
    precision = _binary_ndcg(context_chunk_ids, relevant)

    coverage = None
    if expected_subtasks and relevant_chunk_ids_by_task:
        covered = 0
        evaluable = 0
        for index, _task in enumerate(expected_subtasks, 1):
            gold = set(relevant_chunk_ids_by_task.get(f't{index}', []))
            if not gold:
                continue
            evaluable += 1
            covered += bool(present & gold)
        coverage = covered / evaluable if evaluable else None

    return {
        'context_recall': recall,
        'context_precision': precision,
        'context_coverage': coverage,
        'reason': None,
    }


def task_coverage(actual_tasks: dict, expected_subtasks: list[dict] | None) -> float | None:
    """One-to-one expected-to-actual mapping; dependent tasks must be truly dependent."""
    if not expected_subtasks:
        return None
    actual = list(actual_tasks.values())
    # For multi-branch benchmark cases, matching only the domain lets six
    # identical retrieval queries appear to cover six different information
    # needs. Use the planner's explicit, user-derived facet for the supported
    # single-label categories; keep legacy single-task matching unchanged.
    facet_by_label = {
        'steps': 'steps', 'registration': 'steps', 'documents': 'documents',
        'deadline': 'deadline', 'insurance': 'insurance', 'fees': 'costs',
        'benefits': 'supports', 'health': 'healthcare', 'financial': 'costs',
    }
    matched_actual: set[int] = set()
    matched_expected: set[int] = set()
    ordering = sorted(
        range(len(expected_subtasks)),
        key=lambda index: (
            bool(expected_subtasks[index].get('requires_dependencies')),
            len(expected_subtasks[index].get('keywords', [])),
        ),
        reverse=True,
    )
    for expected_index in ordering:
        expected = expected_subtasks[expected_index]
        for actual_index, task in enumerate(actual):
            if actual_index in matched_actual or task.get('domain') != expected['domain']:
                continue
            expected_facet = (facet_by_label.get(expected.get('label', ''))
                              if len(expected_subtasks) > 1 else None)
            if expected_facet and task.get('facet') != expected_facet:
                continue
            if expected.get('requires_dependencies') and not task.get('depends_on'):
                continue
            question = task.get('question', '').casefold()
            if not all(term.casefold() in question for term in expected.get('keywords', [])):
                continue
            matched_actual.add(actual_index)
            matched_expected.add(expected_index)
            break
    return len(matched_expected) / len(expected_subtasks)


def cite_validity(final: dict) -> float | None:
    claims = final.get('answer_draft', {}).get('claims', [])
    evidence = {e['evidence_id']: e for e in final.get('evidence', [])}
    if not claims:
        return None
    valid = 0
    for claim in claims:
        ids = claim.get('evidence_ids', [])
        quote = ' '.join(claim.get('supporting_quote', '').split()).casefold()
        if ids and quote and all(i in evidence for i in ids) and any(
            quote in ' '.join(evidence[i]['text'].split()).casefold() for i in ids
        ):
            valid += 1
    return valid / len(claims)


def citation_coverage(final: dict) -> float | None:
    """Share of factual claims that contain at least one evidence reference."""
    claims = final.get('answer_draft', {}).get('claims', [])
    if not claims:
        return None
    return sum(bool(claim.get('evidence_ids')) for claim in claims) / len(claims)


def claim_support_breakdown(final: dict) -> dict:
    """Classify citation integrity, not semantic entailment or independently reviewed faithfulness."""
    claims = final.get('answer_draft', {}).get('claims', [])
    evidence = {e['evidence_id']: e for e in final.get('evidence', [])}
    unsupported_from_audit = set(final.get('answer_validation', {}).get('unsupported_claims', []))
    rows = []
    for claim in claims:
        text = claim.get('text', '')
        ids = claim.get('evidence_ids', [])
        quote = ' '.join(claim.get('supporting_quote', '').split()).casefold()
        status = 'not_evaluable'
        if text in unsupported_from_audit:
            status = 'unsupported'
        elif ids and quote and all(i in evidence for i in ids) and any(
            quote in ' '.join(evidence[i].get('text', '').split()).casefold() for i in ids
        ):
            status = 'supported'
        elif ids or quote:
            status = 'unsupported'
        rows.append({'claim': text, 'claim_id': claim_identifier(claim),
                     'supporting_quote': claim.get('supporting_quote', ''),
                     'source_sha256': {eid: sha256(evidence[eid]['text'].encode('utf-8')).hexdigest()
                                       for eid in ids if eid in evidence},
                     'status': status,
                     'structural_status': {'supported': 'citation_valid',
                                           'unsupported': 'citation_invalid'}.get(status, 'not_evaluable'),
                     'semantic_status': 'not_evaluated', 'evidence_ids': ids})
    counts = {key: sum(row['status'] == key for row in rows)
              for key in ('supported', 'unsupported', 'contradicted', 'not_evaluable')}
    evaluable = counts['supported'] + counts['unsupported'] + counts['contradicted']
    return {
        'claims': rows,
        'counts': counts,
        'citation_integrity_proxy': counts['supported'] / evaluable if evaluable else None,
        'citation_invalid_rate': counts['unsupported'] / evaluable if evaluable else None,
        'faithfulness': None,
        'unsupported_claim_rate': None,
        'contradicted_claim_rate': None,
        'measurement': 'technical_citation_integrity_not_semantic_entailment',
    }


def faithfulness_proxy(final: dict) -> float | None:
    return claim_support_breakdown(final)['citation_integrity_proxy']


def fact_coverage(answer: str, expected_facts: list[dict]) -> float | None:
    """Deterministic pattern coverage, not semantic correctness.

    Semantic paraphrases can additionally be evaluated by the optional local judge.
    """
    if not expected_facts or not all(f.get('answer_pattern') for f in expected_facts):
        return None
    return sum(bool(re.search(f['answer_pattern'], answer, re.IGNORECASE))
               for f in expected_facts) / len(expected_facts)


def answer_completeness(answer: str, expected_facts: list[dict]) -> float | None:
    return fact_coverage(answer, expected_facts)


def tool_accuracy(actual: dict, expected: list[str] | None) -> float | None:
    if expected is None:
        return None
    names = [result.get('tool') for result in actual.values()]
    left, right = set(names), set(expected)
    if not left and not right:
        return 1.0
    return ratio(2 * len(left & right), len(left) + len(right))


def tool_call_efficiency(actual: dict, expected: list[str] | None) -> dict:
    """Efficiency without rewarding incomplete answers for simply using fewer tools."""
    if expected is None:
        return {'score': None, 'total': len(actual), 'successful': None,
                'failed': None, 'repeated': None, 'redundant': None}
    names = [item.get('tool') for item in actual.values()]
    successful = sum(item.get('status') == 'success' for item in actual.values())
    failed = sum(item.get('status') == 'error' for item in actual.values())
    repeated = len(names) - len(set(names))
    acceptable = set(expected)
    redundant = sum(name not in acceptable for name in names)
    # 1.0 only when all expected tools are present and no failed/redundant/repeated calls.
    coverage = len(set(names) & acceptable) / len(acceptable) if acceptable else (1.0 if not names else 0.0)
    penalty = failed + redundant + repeated
    score = coverage / (1 + penalty)
    return {'score': score, 'total': len(names), 'successful': successful,
            'failed': failed, 'repeated': repeated, 'redundant': redundant}


def abstention_accuracy(response_status: str | None, expected_behavior: str | None,
                        answerability: str | None) -> float | None:
    """Correct answer-vs-abstain/clarify decision for naturally limited cases."""
    if answerability not in ('answerable', 'partial', 'insufficient'):
        return None
    status = response_status or ''
    if answerability == 'answerable':
        return 1.0 if status in ('complete', 'partial') else 0.0
    # Partial/insufficient cases should not be confidently marked complete.
    return 1.0 if status in ('partial', 'unsupported') else 0.0


def task_completion_score(*, subtask_coverage: float | None,
                          answer_completeness_score: float | None,
                          workflow_success: float | None,
                          answerability: str | None) -> float | None:
    """Acceptance-criteria task completion, not merely 'a response was generated'."""
    if workflow_success == 0:
        return 0.0
    if answerability in ('partial', 'insufficient'):
        # Correctly partial/abstained execution can complete the *evaluation task*.
        return workflow_success
    components = [x for x in (subtask_coverage, answer_completeness_score) if x is not None]
    if not components:
        return None
    return min(components)


def recovery_success(trace: dict, response_status: str | None) -> float | None:
    """Recovery success only when a retry/recovery event actually occurred."""
    spans = trace.get('spans', []) if trace else []
    retrieval_calls = sum(span.get('name') == 'rag/hybrid_retrieval' for span in spans)
    if retrieval_calls <= 1:
        return None
    return 1.0 if response_status in ('complete', 'partial') else 0.0


def workflow_success(error: str | None, response_status: str | None) -> float:
    return 0.0 if error or response_status in ('error', None) else 1.0


def context_window_utilization(prompt_tokens: int | None, context_window: int | None,
                               reserved_generation_tokens: int | None = None) -> dict:
    if not prompt_tokens or not context_window:
        return {'ratio': None, 'prompt_tokens': prompt_tokens, 'context_window': context_window,
                'reserved_generation_tokens': reserved_generation_tokens, 'over_budget': None}
    reserved = reserved_generation_tokens or 0
    return {
        'ratio': prompt_tokens / context_window,
        'prompt_tokens': prompt_tokens,
        'context_window': context_window,
        'reserved_generation_tokens': reserved,
        'over_budget': prompt_tokens + reserved > context_window,
    }


def generation_speed(generated_tokens: int | None, generation_duration_s: float | None) -> float | None:
    if generated_tokens is None or not generation_duration_s:
        return None
    return generated_tokens / generation_duration_s


def ttft(first_content_s: float | None) -> float | None:
    return first_content_s if first_content_s is None or first_content_s >= 0 else None


def average_available(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows if row.get(key) is not None]
    return {'value': mean(values) if values else None, 'evaluated': len(values), 'total': len(rows)}


def diagnose(case: dict, output: dict | None, error: str | None) -> list[str]:
    if error:
        if 'vector' in error.casefold() or 'qdrant' in error.casefold():
            return ['retrieval_runtime_error']
        if 'timeout' in error.casefold():
            return ['timeout']
        return ['workflow_runtime_error']
    assert output is not None
    issues = []
    if set(output.get('domains', [])) != set(case.get('expected_domains', [])):
        issues.append('intent_classification_or_domain_routing')
    if task_coverage(output.get('subtasks', {}), case.get('expected_subtasks')) not in (None, 1.0):
        issues.append('task_planning')
    if output.get('validation', {}).get('status') == 'failed':
        issues.append('retrieval_or_insufficient_evidence')
    if output.get('answer_validation', {}).get('status') == 'failed':
        issues.append('answer_generation_or_citation')
    return issues
