"""Prompt inputs and trusted, source-exact tool hints.

No model calls or retrieval side effects are performed here.
"""
from __future__ import annotations

from ..response.quality import is_complete_statement, non_independent_source_fragment
from ..response.filters import relevant_unit


def _verified_tool_hints(tools: list[dict], selected_evidence: list[dict],
                         question: str, role: str, stage: str) -> list[dict]:
    """Add source-exact, bounded tool snippets only for evidence IDs in the current prompt."""
    by_id = {e['evidence_id']: e for e in selected_evidence if e.get('evidence_id')}
    hints: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for result in tools:
        if (not isinstance(result, dict) or result.get('status') != 'success'
                or result.get('tool') not in ('get_document_checklist', 'build_document_checklist')):
            continue
        for item in result.get('items', [])[:12]:
            if not isinstance(item, dict):
                continue
            text = ' '.join(str(item.get('text', '')).split())
            ids = item.get('evidence_ids') or ([item['evidence_id']]
                   if item.get('evidence_id') else result.get('source_evidence_ids') or [])
            if (not text or len(text) > 220 or not is_complete_statement(text, source_unit=True)
                    or non_independent_source_fragment(text)):
                continue
            for eid in ids:
                source = by_id.get(eid)
                if (source is None or (eid, text) in seen
                        or text.casefold() not in ' '.join(source.get('text', '').split()).casefold()
                        or not relevant_unit(text, question, source.get('domain', ''), role, stage)):
                    continue
                hints.append({'evidence_id': eid, 'source_excerpt': text})
                seen.add((eid, text))
                break
            if len(hints) >= 2:
                return hints
    return hints


def answer_context(*, domains: list[str], role: str, stage: str,
                   strategy: str, reference_date: str, event_date: str | None,
                   subtasks: dict[str, dict], event_date_confirmed: bool = False,
                   event_date_source: str = 'default', focus: str = '') -> dict:
    """Explicit, inspectable task context shared by Qwen and the UI."""
    return {
        'life_events': domains,
        'life_event_labels': [{
            'vehicle': 'Autóvásárlás vagy -eladás',
            'employment': 'Munkahely elvesztése',
        }.get(domain, domain) for domain in domains],
        'role': role or 'not_specified',
        'stage': stage or 'unknown',
        'answer_strategy': strategy,
        'administrative_subtopic': focus or 'not_specified',
        'reference_date': reference_date,
        'event_date': event_date,
        'event_date_confirmed': event_date_confirmed,
        'event_date_source': event_date_source,
        'date_warning': ('The displayed event date is not verified as the legal start date. '
                         'Do not calculate an exact deadline from it.'
                         if not event_date_confirmed else
                         'User-selected date; the legal start event and calculation rules still require verification.'),
        'tasks': [{'id': key, 'domain': item.get('domain'), 'goal': item.get('question')}
                  for key, item in subtasks.items()],
    }
