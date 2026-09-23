"""Optional evidence-pinned independent human semantic adjudication.

A correct quotation does NOT imply an entailed claim. Never infer reviews from
claim categories, citations, model outputs, or machine-generated fixtures.
"""
from __future__ import annotations

from hashlib import sha256
from .metrics import ratio
from ..response.quality import claim_identifier

_LABELS = frozenset({'entailed', 'contradicted', 'not_supported'})


def source_fingerprint(text: str) -> str:
    return sha256(text.encode('utf-8')).hexdigest()


def semantic_review(output: dict, case: dict | None = None) -> dict:
    """Score only explicitly approved, source-version-pinned human annotations; partial review stays item-level."""
    claims = output.get('answer_draft', {}).get('claims', [])
    by_id = {item.get('evidence_id'): item for item in output.get('evidence', [])}
    case = case or {}
    reviews = case.get('approved_semantic_reviews') or [] if case.get('human_reviewed') is True else []
    indexed = {r.get('claim_id'): r for r in reviews if isinstance(r, dict)}
    rows = []
    for claim in claims:
        cid = claim_identifier(claim)
        ids = claim.get('evidence_ids', [])
        review = indexed.get(cid, {})
        valid = (review.get('reviewer') and review.get('reviewed_at')
                 and review.get('approved') is True
                 and review.get('label') in _LABELS
                 and review.get('claim_text') == claim.get('text')
                 and review.get('supporting_quote') == claim.get('supporting_quote')
                 and review.get('evidence_ids') == ids
                 and isinstance(review.get('source_sha256'), dict)
                 and ids and all(eid in by_id and
                     review['source_sha256'].get(eid) == source_fingerprint(by_id[eid]['text'])
                     for eid in ids))
        rows.append({'claim_id': cid, 'label': review['label'] if valid else 'unreviewed',
                     'reviewed': bool(valid)})
    complete = bool(rows) and all(row['reviewed'] for row in rows)
    entailed = sum(row['label'] == 'entailed' for row in rows)
    return {'claims': rows, 'reviewed_claims': sum(row['reviewed'] for row in rows),
            'total_claims': len(rows), 'review_coverage': ratio(sum(row['reviewed'] for row in rows), len(rows)),
            'faithfulness': (entailed / len(rows)) if complete else None,
            'unsupported_claim_rate': (sum(row['label'] == 'not_supported' for row in rows) / len(rows)) if complete else None,
            'contradicted_claim_rate': (sum(row['label'] == 'contradicted' for row in rows) / len(rows)) if complete else None,
            'measurement': 'approved_human_entailment_source_sha256_pinned' if complete
                            else 'not_evaluated_or_partially_reviewed',
            'limitations': 'Reviewer judgments are external and cannot be validated automatically.'}
