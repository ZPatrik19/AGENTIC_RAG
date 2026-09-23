"""Summarize saved real runs without rerunning inference or approving gold."""
from __future__ import annotations

import json
import re
from collections import Counter
from hashlib import sha256

from dap_assistant.evaluation.automatic_reference import corpus_fingerprint
from dap_assistant.evaluation.professional import _llm_perf
from dap_assistant.settings import Settings


def source_checks(spec: dict, rows: list[dict], chunks: list[dict]) -> dict:
    """Lexical indicators only: matching words cannot prove entailment."""
    indexed = {c['chunk_id']: c for c in chunks}
    answers = {r['question_id']: r for r in rows}
    results = []
    for qid, case in spec['cases'].items():
        row = answers.get(qid)
        checks = []
        for check in case['checks']:
            missing_sources = [cid for cid in check['source_chunk_ids'] if cid not in indexed]
            evaluable = row is not None and not missing_sources
            checks.append({
                'id': check['id'], 'source_chunk_ids': check['source_chunk_ids'],
                'missing_source_ids': missing_sources,
                'lexical_match': all(re.search(p, row['generated_answer'], re.IGNORECASE | re.DOTALL)
                                     is not None for p in check['all_patterns']) if evaluable else None,
                'source_text_sha256': {cid: sha256(indexed[cid]['text'].encode('utf-8')).hexdigest()
                                       for cid in check['source_chunk_ids'] if cid in indexed},
            })
        results.append({'question_id': qid, 'scope': case['scope'], 'checks': checks})
    return {'human_reviewed': False, 'method': spec['method'],
            'corpus_fingerprint': corpus_fingerprint(chunks), 'cases': results}


def functional_summary(run: dict, settings: Settings | None = None) -> dict:
    rows = run['rows']
    active = settings or Settings()
    perf = [_llm_perf(r['node_execution_trace'], active.ollama_num_ctx,
                      active.ollama_answer_num_predict) for r in rows]
    return {
        **run['summary'],
        'success_semantics': 'No workflow exception; not semantic correctness',
        'response_status_counts': dict(Counter(r['response_status'] for r in rows)),
        'llm_failure_kinds': dict(Counter(
            f.get('failure_kind', f.get('type', 'unknown'))
            for r in rows for f in r['node_execution_trace'].get('llm_failures', []))),
        'done_reasons': dict(Counter(u.get('done_reason') or 'unavailable'
                                    for r in rows for u in r['node_execution_trace'].get('llm_usage', []))),
        'observed_prompt_tokens': sum(p['prompt_tokens'] or 0 for p in perf),
        'observed_completion_tokens': sum(p['generated_tokens'] or 0 for p in perf),
        'token_note': 'Observed API usage only; missing or failed calls are not zero-token calls',
    }


def main() -> None:
    root = Settings().root
    output = root / 'reports' / 'final_validation'
    manifest = json.loads((output / 'completed_runs.json').read_text(encoding='utf-8'))
    runs = {stage: json.loads((root / manifest[stage] / 'result.json').read_text(encoding='utf-8'))
            for stage in ('functional', 'load') if stage in manifest}
    summary = {'corpus_fingerprint': manifest['corpus_fingerprint'], 'reports': manifest,
               'human_reviewed': False}
    if 'functional' in runs:
        summary['functional'] = functional_summary(runs['functional'], Settings())
        # Legacy human-review notes and old fixed-run check specs are intentionally
        # absent from the streamlined evaluation dataset. Never substitute made-up
        # qualitative scores for them.
        summary['qualitative_review_status'] = 'not_available_in_current_dataset'
    if 'load' in runs:
        summary['load'] = runs['load']['summary']
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
