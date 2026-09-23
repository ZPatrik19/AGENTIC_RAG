"""Bounded, traceable LangGraph context planning and generation diagnostics.

No gold labels, semantic entailment or effective Ollama window are inferred here.
"""
from __future__ import annotations

from .evidence_selection import (evidence_matches_facet, is_relevant_evidence,
                                 requested_facets)
from ..response.filters import relevant_unit


def context_budget(settings, question: str, domain: str, role: str = '') -> dict:
    """Estimate a per-question excerpt budget inside the *configured* window.

    The UTF-8 estimate is not a tokenizer; leave ample room for JSON schema,
    instructions, model output, and conversation metadata. Never raise num_ctx.
    """
    fast = settings.answer_mode == 'quick' and settings.quick_single_pass
    window = min(settings.ollama_num_ctx, settings.ollama_quick_num_ctx) if fast else settings.ollama_num_ctx
    facets = requested_facets(question, domain, role=role)
    ceiling = (min(700, max(320, (window - 900) // 2)) if fast
               else min(2200, max(384, (window - 900) // 3)))
    target = (320 + 80 * max(0, len(facets) - 1) if fast
              else 384 + 260 * max(0, len(facets) - 1))
    return {'requested_num_ctx': window, 'excerpt_budget_estimate': min(ceiling, target),
            'max_excerpt_budget_estimate': ceiling, 'requested_facets': list(facets),
            'budget_kind': 'conservative_utf8_estimate_not_qwen_tokens',
            'num_ctx_modified': False}


def _section_key(item: dict) -> tuple[str, str, str] | None:
    """Existing indexed metadata identifies one exact section at one doc version."""
    values = (item.get('document_id'), item.get('document_version'),
              item.get('section_text_sha256'))
    return tuple(values) if all(isinstance(v, str) and v for v in values) else None


def prepare_graph_context(evidence: list[dict], question: str, domain: str,
                          role: str, settings, *, section_chunks: list[dict] | None = None) -> tuple[list[dict], dict]:
    """De-duplicate a worker union; expand verified same-section siblings when needed.

    The original retrieved evidence remains the source of truth for audit and
    retrieval ranks. Expansion never assigns a search rank or invents a source.
    """
    budget = context_budget(settings, question, domain, role)
    selected: list[dict] = []
    ids: set[str] = set()
    repeated = 0
    for item in evidence:
        cid = item.get('chunk_id') or item.get('evidence_id')
        if not cid or cid in ids:
            repeated += 1
            continue
        if not is_relevant_evidence(item, question, domain, role):
            continue
        ids.add(cid)
        selected.append(item)

    def matched_facets(items: list[dict]) -> list[str]:
        return [facet for facet in budget['requested_facets']
                if any(evidence_matches_facet(item, facet) for item in items)]

    before = matched_facets(selected)
    expanded: list[dict] = []
    if section_chunks and len(selected) < 22:
        anchors = {_section_key(e): e for e in selected[:8] if _section_key(e)}
        for item in section_chunks:
            if len(expanded) >= 2 or len(selected) >= 22:
                break
            key = _section_key(item)
            if not key or key not in anchors or item.get('chunk_id') in ids:
                continue
            if not is_relevant_evidence(item, question, domain, role):
                continue
            missing = set(budget['requested_facets']) - set(matched_facets(selected))
            # A sibling is useful only when it supplies a missing information need.
            if not missing or not any(evidence_matches_facet(item, facet) for facet in missing):
                continue
            if not any(relevant_unit(unit.strip(), question, domain, role)
                       for unit in str(item.get('text', '')).splitlines() if unit.strip()):
                continue
            original = anchors[key]
            ids.add(item['chunk_id'])
            source = {**item, 'evidence_id': 'E_' + item['chunk_id'][:16],
                      'context_origin': 'same_version_parent_section_sibling'}
            selected.append(source)
            expanded.append({'chunk_id': item['chunk_id'], 'parent_section_sha256': key[2],
                             'anchor_chunk_id': original.get('chunk_id'),
                             'document_id': key[0], 'document_version': key[1]})

    after = matched_facets(selected)
    report = {**budget, 'raw_retrieved_chunks': len(evidence),
              'unique_retrieved_chunks': len(selected) - len(expanded),
              'duplicate_chunk_ids_removed': repeated,
              'parent_section_siblings_added': expanded,
              'context_candidate_chunks': len(selected),
              'lexically_covered_facets_before': before,
              'lexically_covered_facets_after': after,
              'missing_retrieved_facets': [f for f in budget['requested_facets'] if f not in after],
              'coverage_measurement': 'lexical_candidates_not_semantic_support_or_gold'}
    return selected, report


def generation_diagnostics(trace: dict, *, fallback: bool, reason: str,
                           model_claim_count: int, requested_facets: list[str],
                           draft_facets: list[str]) -> dict:
    """Separate a completed-but-incomplete answer from output-limit/timeout.

    Unknown usage stays null; received bytes are never mistaken for tokens.
    """
    responses = [r for r in trace.get('llm_usage', []) if r.get('phase') == 'answer']
    failures = [f for f in trace.get('llm_failures', []) if f.get('phase') == 'answer']
    last = responses[-1] if responses else {}
    failure_kinds = [f.get('failure_kind') for f in failures]
    missing = [facet for facet in requested_facets if facet not in draft_facets]
    if 'input_context_budget' in failure_kinds:
        status = 'input_context_budget_exceeded'
    elif 'output_token_limit' in failure_kinds or last.get('done_reason') == 'length':
        status = ('recovered_after_output_limit' if not fallback and model_claim_count else
                  'output_token_limit')
    elif 'total_timeout' in failure_kinds or 'http_timeout' in failure_kinds:
        status = 'timeout'
    elif failures and not model_claim_count:
        status = 'generation_failed_or_interrupted'
    elif not responses and not failures:
        status = 'not_instrumented_or_deterministic_response'
    elif fallback or not model_claim_count:
        status = 'completed_but_unusable' if last.get('done_reason') == 'stop' else 'model_unavailable_or_unobserved'
    elif missing and last.get('done_reason') == 'stop':
        status = 'completed_with_uncovered_facets'
    elif last.get('done_reason') == 'stop':
        status = 'completed_with_structural_coverage'
    else:
        status = 'generation_not_instrumented'
    return {'status': status, 'requested_num_ctx': last.get('requested_num_ctx'),
            'requested_num_predict': last.get('requested_num_predict'),
            'prompt_eval_count': last.get('prompt_eval_count'),
            'eval_count': last.get('eval_count'), 'done_reason': last.get('done_reason'),
            'prompt_eval_duration': last.get('prompt_eval_duration'),
            'eval_duration': last.get('eval_duration'),
            'total_duration': last.get('total_duration'),
            'attempts': len(responses) or len(failures), 'failure_kinds': failure_kinds,
            'model_claim_count': model_claim_count, 'uncovered_facets_proxy': missing,
            'prompt_near_requested_window': (last['prompt_eval_count'] >= 0.9 * last['requested_num_ctx']
                if isinstance(last.get('prompt_eval_count'), int) and
                   isinstance(last.get('requested_num_ctx'), int) and last['requested_num_ctx'] > 0 else None),
            'window_measurement': 'requested_config_and_ollama_usage_not_effective_context_proof',
            'coverage_measurement': 'lexical_facets_not_semantic_or_legal_completeness',
            'token_limit_vs_content_gap': 'input_budget' if 'input_context_budget' in failure_kinds
                else 'output_limit' if 'output_token_limit' in failure_kinds or last.get('done_reason') == 'length'
                else 'content_gap' if missing and last.get('done_reason') == 'stop' else 'undetermined'}
