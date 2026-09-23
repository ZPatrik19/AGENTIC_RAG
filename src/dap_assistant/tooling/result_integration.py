"""Grounded checklist utilization: tool output alone is never an answer claim.

Only exact, independent, question-relevant source units may be rendered.  Every
item remains traceable even when it is rejected. This is NOT semantic entailment.
"""
from __future__ import annotations

from ..response.filters import low_value_procedural_unit
from ..response.filters import relevant_unit
from ..response.quality import (claim_facets, claim_identifier, is_complete_statement,
                             non_independent_source_fragment, same_fact)


def _normal(text: str) -> str:
    return ' '.join(str(text).split()).casefold()


def integrate_checklist(*, question: str, domain: str, role: str,
                        evidence: list[dict], existing_claims: list[dict],
                        tool_results: dict, native_tool_results: dict | None = None,
                        native_tool_trace: dict | None = None,
                        max_added: int = 3, stage: str = '') -> tuple[list[dict], dict]:
    """Integrate literal claims only from executed, delivered calls and independently linked source evidence."""
    by_id = {e['evidence_id']: e for e in evidence if e.get('evidence_id')}
    trace = native_tool_trace or {}
    delivered = {tuple(call.get('result_evidence_ids') or []): call
                 for call in trace.get('calls', [])
                 if call.get('tool_name') == 'get_document_checklist'
                 and call.get('status') == 'executed'
                 and call.get('returned_to_model') is True}
    results: list[tuple[str, dict]] = []
    for key, result in (native_tool_results or {}).items():
        if (key.startswith('native_get_document_checklist_') and isinstance(result, dict)
                and result.get('tool') == 'get_document_checklist'):
            results.append(('native_tool', result))
    if isinstance(tool_results.get('checklist'), dict):
        results.append(('deterministic_checklist', tool_results['checklist']))
    added: list[dict] = []
    rows: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for origin, result in results:
        if result.get('status') != 'success':
            rows.append({'origin': origin, 'status': 'tool_not_successful',
                         'claim_id': None, 'evidence_ids': []})
            continue
        source_ids = tuple(result.get('source_evidence_ids') or [])
        if origin == 'native_tool' and source_ids not in delivered:
            rows.append({'origin': origin, 'status': 'not_delivered_to_model',
                         'claim_id': None, 'evidence_ids': list(source_ids)})
            continue
        for item in result.get('items', [])[:25]:
            if not isinstance(item, dict):
                continue
            text = ' '.join(str(item.get('text', '')).split())
            ids = tuple(item.get('evidence_ids') or
                        ([item['evidence_id']] if item.get('evidence_id') else source_ids))
            signature = (_normal(text), tuple(sorted(set(ids))))
            if signature in seen:
                rows.append({'origin': origin, 'status': 'duplicate_tool_item',
                             'claim_id': claim_identifier({'text': text,
                                 'supporting_quote': text, 'evidence_ids': list(ids)}),
                             'evidence_ids': list(ids)})
                continue
            seen.add(signature)
            candidate = {'text': text, 'supporting_quote': text,
                         'evidence_ids': list(ids), 'category': 'documents',
                         'origin': 'tool_extract', 'in_model_prompt': None}
            claim_id = claim_identifier(candidate) if text else None
            row = {'origin': origin, 'status': 'not_a_document_requirement',
                   'claim_id': claim_id, 'evidence_ids': list(ids)}
            if (not ids or any(eid not in by_id for eid in ids)
                    or not any(_normal(text) in _normal(by_id[eid].get('text', ''))
                               for eid in ids if eid in by_id)):
                row['status'] = 'source_mismatch'
            elif (not is_complete_statement(text, source_unit=True)
                  or non_independent_source_fragment(text)):
                row['status'] = 'incomplete_source_unit'
            elif not relevant_unit(text, question, domain, role, stage):
                row['status'] = 'irrelevant_to_life_event'
            elif (low_value_procedural_unit(text, question, domain, role)
                  or 'documents' not in claim_facets(candidate, domain)):
                row['status'] = 'not_a_document_requirement'
            elif any(same_fact(text, c.get('text', '')) or
                     (_normal(text) in _normal(c.get('supporting_quote', ''))
                      and 'documents' in claim_facets(c, domain))
                     for c in existing_claims + added):
                row['status'] = 'covered_by_answer'
            elif len(added) >= max_added:
                row['status'] = 'addition_limit'
            else:
                candidate['claim_id'] = claim_id
                added.append(candidate)
                row['status'] = 'added_verbatim'
            rows.append(row)
    return added, {
        'measurement': 'exact_source_unit_and_document_action_proxy_not_semantic_validation',
        'items': rows,
        'counts': {status: sum(item['status'] == status for item in rows)
                   for status in sorted({item['status'] for item in rows})},
        'visible_items': sum(item['status'] in ('covered_by_answer', 'added_verbatim')
                             for item in rows),
        'source_of_truth': 'final_validated_claims_and_verified_tool_extracts',
    }
