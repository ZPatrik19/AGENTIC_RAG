"""Explain benchmark N/A without treating retrieval suggestions as human gold.

The current corpus can be assessed offline. A high metric is NEVER synthesized
from expected URLs, model responses, or automatically proposed chunk labels.
"""
from __future__ import annotations

from collections import Counter

from .dataset import index_versions, reference_status, validate_relevant_chunks


REFERENCE_REASONS = {
    'unversioned_reference': 'Nincs jóváhagyott, dokumentumverzióhoz kötött referencia.',
    'incomplete_source_versions': 'A jóváhagyott források verziólistája hiányos.',
    'version_mismatch': 'A dokumentum megváltozott a jóváhagyás óta; újra kell ellenőrizni.',
    'unreviewed_reference': 'A forrásverzió megvan, de hiányzik az emberi jóváhagyás.',
    'pinned_source_only': 'Forrásverzió van, de ellenőrzött releváns chunklista nincs.',
    'task_mapping_missing': 'Hiányzik a releváns chunkok részfeladatonkénti hozzárendelése.',
    'invalid_chunk_reference': 'A jóváhagyott chunk nem illeszkedik az aktuális indexhez.',
    'pinned': 'Emberileg ellenőrzött, verzióhoz kötött referencia rendelkezésre áll.',
    'automatic_proxy_pinned': 'Automatikus, forrásverzióhoz kötött SILVER/proxy referencia – nem emberileg ellenőrzött gold.',
    'automatic_proxy_incomplete': 'Nem sikerült minden referenciaállításhoz és részfeladathoz önálló forrásbizonyítékot találni.',
    'invalid_reference_type': 'Ellentmondó automatikus és emberi jóváhagyási jelölés.',
    'synthetic': 'Szintetikus tesztadat; nem valódi dokumentumértékelés.',
}


def reference_readiness(cases: list[dict], chunks: list[dict]) -> dict:
    """Show actionable per-case reference status; do not mutate either input."""
    versions = index_versions(chunks)
    rows = []
    for case in cases:
        status = reference_status(case, versions)
        if status in ('pinned', 'automatic_proxy_pinned') and not validate_relevant_chunks(case, chunks):
            status = 'invalid_chunk_reference'
        missing = sorted(set(case.get('expected_source_ids', [])) - versions.keys())
        rows.append({
            'question_id': case['question_id'],
            'status': status,
            'reason': REFERENCE_REASONS.get(status, 'Ismeretlen referenciaállapot.'),
            'missing_source_ids': missing,
            'ready': status in ('pinned', 'automatic_proxy_pinned'),
            'reference_type': 'automatic_silver_proxy' if status == 'automatic_proxy_pinned' else 'human_gold' if status == 'pinned' else 'unavailable',
        })
    return {
        'total': len(rows),
        'ready': sum(row['ready'] for row in rows),
        'automatic_proxy_ready': sum(row['status'] == 'automatic_proxy_pinned' for row in rows),
        'human_gold_ready': sum(row['status'] == 'pinned' for row in rows),
        'status_counts': dict(Counter(row['status'] for row in rows)),
        'rows': rows,
    }


def run_reference_diagnostics(result: dict) -> dict:
    """Summarize the SAVED run, not today's possibly changed local index."""
    rows = result.get('rows') or []
    counts = Counter((row.get('details') or {}).get('reference_status') or 'execution_error'
                     for row in rows)
    failures = [{'question_id': row.get('question_id'), 'error': row.get('error')}
                for row in rows if row.get('error')]
    return {
        'total': len(rows),
        'ready': counts['pinned'] + counts['automatic_proxy_pinned'],
        'automatic_proxy_ready': counts['automatic_proxy_pinned'],
        'human_gold_ready': counts['pinned'],
        'status_counts': dict(counts),
        'failures': failures,
    }
