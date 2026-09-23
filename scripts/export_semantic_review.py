"""Export an UNAPPROVED semantic-review worksheet from a comparison result.json.

The script never writes judgments or changes golden_v4.json. A reviewer must
inspect the full original source, then explicitly approve each claim/quote.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def review_template(result: dict) -> dict:
    rows = []
    for row in result.get('rows', []):
        claims = row.get('details', {}).get('claim_support', {}).get('claims', [])
        cases = []
        for claim in claims:
            if (not claim.get('claim_id') or not claim.get('supporting_quote')
                    or not claim.get('source_sha256')):
                raise ValueError('Benchmark lacks source-fingerprint review fields; rerun the comparison with P6.7.')
            cases.append({
                'claim_id': claim.get('claim_id'), 'claim_text': claim.get('claim', ''),
                'supporting_quote': claim.get('supporting_quote', ''),
                'evidence_ids': claim.get('evidence_ids', []),
                'source_sha256': claim.get('source_sha256', {}),
                'label': None, 'reviewer': '', 'reviewed_at': '', 'approved': False,
                'review_notes': '',
            })
        rows.append({'question_id': row.get('question_id'), 'mode': row.get('mode'),
                     'question': row.get('question', ''), 'claims': cases})
    return {'source_run_id': result.get('run_id'), 'status': 'unreviewed_template',
            'instructions': ('Inspect every exact source quote against the original, '
                             'then explicitly label entailed, contradicted or not_supported. '
                             'A correct citation is not evidence of semantic entailment. '
                             'Copy approved claims to approved_semantic_reviews in golden_v4.json '
                             'only after human review; do not set human_reviewed for unreviewed cases.'),
            'cases': rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    if data.get('kind') != 'rag_profile_comparison_v1':
        parser.error('Expected rag_profile_comparison_v1 JSON')
    if args.output.resolve() == args.input.resolve():
        parser.error('Refusing to overwrite the benchmark source')
    try:
        worksheet = review_template(data)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(worksheet, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    print(f'[REVIEW] Unapproved worksheet saved: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
