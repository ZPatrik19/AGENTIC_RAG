"""Side-effect-free presentation data for a calm, evidence-first Streamlit chat.

No extra LLM calls, hidden source rewriting, or simulated workflow events.
"""
from __future__ import annotations

from collections import OrderedDict
from math import isfinite
from typing import Any


def official_source_overview(evidence: list[dict]) -> list[dict]:
    """Group retrieved chunks by *actual* document identity; keep their original URLs.

    A document with an unusable URL remains visible, but cannot create a link.
    Do not conflate distinct document IDs merely because they share a hostname.
    """
    docs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        raw_url = item.get('source_url')
        url = raw_url if isinstance(raw_url, str) and raw_url.startswith('https://') else ''
        doc_id = str(item.get('document_id') or url or item.get('title') or 'Névtelen forrás')
        if doc_id not in docs:
            docs[doc_id] = {
                'document_id': doc_id,
                'title': str(item.get('title') or 'Hivatalos forrás'),
                'source_url': url,
                'chunk_count': 0,
            }
        docs[doc_id]['chunk_count'] += 1
        # A source URL may be populated on another chunk of the same document.
        if not docs[doc_id]['source_url'] and url:
            docs[doc_id]['source_url'] = url
    return list(docs.values())


def insight_counts(final: dict, elapsed_s: float | None) -> dict:
    """Expose only measured counts, without interpreting model/tool success as answer quality."""
    evidence = final.get('evidence') or []
    native = final.get('native_tool_trace') or {}
    native_calls = native.get('calls') or []
    successful_native = sum(
        isinstance(call, dict) and call.get('status') == 'executed'
        and call.get('returned_to_model') is True
        for call in native_calls
    )
    # The local deterministic tool results may be a dict or empty, never inferred.
    tools = final.get('tool_results') or {}
    duration = (float(elapsed_s) if isinstance(elapsed_s, (int, float))
                and not isinstance(elapsed_s, bool) and isfinite(elapsed_s) and elapsed_s >= 0
                else None)
    return {
        'documents': len(official_source_overview(evidence)),
        'chunks': len(evidence),
        'tool_results': (len(tools) if isinstance(tools, (dict, list)) else 0) + successful_native,
        'elapsed_s': duration,
        'partial': bool(final.get('answer_fallback') or final.get('response_status') == 'partial'),
    }


# A sidebar uses examples only from the two supported life events. These are
# illustrative questions, not verified legal answers or a separate LLM prompt.
_EXAMPLE_QUESTIONS: dict[str, tuple[str, ...]] = {
    'vehicle': (
        'Eladtam az autómat. Mit kell bejelentenem, hol és meddig?',
        'Használt autót vettem. Milyen teendőim és határidőim vannak?',
        'Milyen költségekkel jár egy használt autó átírása?',
    ),
    'employment': (
        'Elvesztettem a munkámat. Mi legyen az első lépésem?',
        'Milyen iratok szükségesek az álláskeresési járadék igényléséhez?',
        'Hol regisztrálhatok álláskeresőként, és milyen határidőkre figyeljek?',
    ),
}


def example_questions_for_domain(domain_hint: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return relevant sidebar suggestions without calling the workflow."""
    categories = (
        ('vehicle', 'Autóvásárlás és -eladás'),
        ('employment', 'Munkahely elvesztése'),
    )
    return tuple(
        (label, _EXAMPLE_QUESTIONS[key])
        for key, label in categories
        if not domain_hint or domain_hint == key
    )


def submitted_question(typed_question: str | None, suggested_question: str | None) -> str | None:
    """Prefer explicit chat input; a clicked suggestion is consumed once by the UI."""
    return typed_question or suggested_question
