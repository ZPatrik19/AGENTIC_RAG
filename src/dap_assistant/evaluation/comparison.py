"""Compare baseline, hybrid and Agentic RAG on paired golden cases using production components."""
from __future__ import annotations

from datetime import date, datetime, timezone
from hashlib import sha256

from time import perf_counter

import re
import uuid

from .dataset import active_dataset, index_versions
from .professional import _environment, _full_metrics, select_cases
from ..observability.telemetry import Telemetry
from .comparison_metrics import (
    _SIMPLE_METRICS, _model_generated, _simple_metrics, _usage,
    _mode_summary, _paired_deltas,
)
from dap_assistant.rag.dense_resources import acquire_dense, release_dense
from dap_assistant.documents.ingestion import load_chunks
from ..llm import LLMError, answer_context, get_llm, source_answer
from ..response.quality import (clean_claims, structural_facet_coverage,
                              missing_facets_hungarian, source_fallback_notice,
                              claim_identifier)
from ..context_engineering.evidence_selection import requested_facets
from dap_assistant.rag.retrieval import hybrid_search
from ..settings import Settings


COMPARISON_PROFILES: dict[str, dict[str, object]] = {
    'baseline_dense': {
        'label': 'Baseline RAG',
        'retrieval': 'dense_only_top5',
        'query_rewrite': False,
        'bm25': False,
        'rrf': False,
        'rerank': False,
        'retry': False,
        'agentic_routing': False,
        'tools': False,
    },
    'hybrid_rrf': {
        'label': 'Hybrid RAG',
        'retrieval': 'bm25_plus_dense_rrf_k60_top5',
        'query_rewrite': False,
        'bm25': True,
        'rrf': True,
        'rerank': False,
        'retry': False,
        'agentic_routing': False,
        'tools': False,
    },
    'agentic': {
        'label': 'Agentic RAG',
        'retrieval': 'production_langgraph_rag_subgraph',
        'query_rewrite': True,
        'bm25': True,
        'rrf': True,
        'rerank': True,
        'retry': True,
        'agentic_routing': True,
        'tools': True,
    },
}

def _role(case: dict) -> str:
    if case['category'] != 'vehicle':
        return ''
    lowered = case['question'].casefold()
    return 'seller' if any(term in lowered for term in ('eladtam', 'eladó', 'eladás')) else 'buyer'


def _as_evidence(items: list[dict], limit: int = 5) -> list[dict]:
    """Attach the same stable evidence IDs used by the production RAG subgraph."""
    result = []
    for item in items[:limit]:
        if not item.get('chunk_id'):
            continue
        result.append({**item, 'evidence_id': 'E_' + item['chunk_id'][:16]})
    return result


def _dense_retrieve(question: str, case: dict, chunks: list[dict], dense, *, telemetry: Telemetry,
                    run_id: str, limit: int = 5) -> tuple[list[dict], list[str]]:
    domain, role = case['category'], _role(case)
    by_id = {item['chunk_id']: item for item in chunks
             if item.get('domain') == domain
             and (not role or item.get('role', 'general') in ('general', role))}
    ranking = dense.search(question, domain, limit=limit, role=role,
                           telemetry=telemetry, run_id=run_id)
    candidates = []
    ranked_ids = []
    for rank, (chunk_id, raw_score) in enumerate(ranking, 1):
        item = by_id.get(chunk_id)
        if item is None:
            continue
        ranked_ids.append(chunk_id)
        candidates.append({
            **item,
            'dense_rank': rank,
            'dense_score': float(raw_score),
            'bm25_rank': None,
            'bm25_score': None,
            'rrf_components': {},
            'score': float(raw_score),
            'fusion_method': 'dense_cosine_similarity_not_probability',
        })
    return _as_evidence(candidates, limit), ranked_ids


def _hybrid_retrieve(question: str, case: dict, settings: Settings, dense, *, telemetry: Telemetry,
                     run_id: str, limit: int = 5) -> tuple[list[dict], list[str]]:
    candidates = hybrid_search(
        question, case['category'], settings, limit=limit, dense=dense,
        role=_role(case), telemetry=telemetry, run_id=run_id,
    )
    return _as_evidence(candidates, limit), [item['chunk_id'] for item in candidates]


def _audit_simple_draft(draft: dict, evidence: list[dict], *, question: str = '',
                        domain: str = '', role: str = '') -> tuple[dict, list[str]]:
    """Apply the production answer audit's structural provenance checks.

    This intentionally does not claim semantic/legal entailment.  The optional local
    judge or human-reviewed reference remains responsible for semantic correctness.
    """
    by_id = {item['evidence_id']: item for item in evidence}
    valid, rejected = [], []
    cleaned, integrity = clean_claims(draft.get('claims', []), question=question,
                                      domain=domain, role=role)
    integrity['model_warnings'] = (draft.get('quality_warnings', [])
                                   + (['healthcare_condition_unverified'] if integrity['healthcare_condition_issues'] else []))
    for claim in cleaned:
        ids = claim.get('evidence_ids', [])
        quote = ' '.join(claim.get('supporting_quote', '').split()).casefold()
        linked = ' '.join(' '.join(by_id[eid].get('text', '').split()).casefold()
                          for eid in ids if eid in by_id)
        claim_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', claim.get('text', ''))
        quote_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', quote)
        supported = (
            bool(ids) and all(eid in by_id for eid in ids) and bool(quote)
            and quote in linked and all(number in quote_numbers for number in claim_numbers)
            and bool(claim.get('text', '').strip())
        )
        if supported:
            valid.append(claim)
        else:
            rejected.append(claim.get('text', '')[:120])
    audited = {**draft, 'claims': valid, 'text_integrity': integrity}
    return audited, rejected


def _render_simple_answer(draft: dict, evidence: list[dict]) -> str:
    by_id = {item['evidence_id']: item for item in evidence}
    lines = []
    for claim in draft.get('claims', []):
        refs = ' '.join(
            f"[Forrás](<{by_id[eid].get('source_url', '')}>)"
            for eid in claim.get('evidence_ids', []) if eid in by_id
        )
        lines.append(f"- [{claim_identifier(claim)}] {claim.get('text', '').strip()} {refs}".rstrip())
    integrity = draft.get('text_integrity', {})
    if integrity.get('incomplete_claims') or 'model_incomplete_statement' in integrity.get('model_warnings', []):
        lines.extend(['', 'Részleges tájékoztatás: félbeszakadt állításokat kihagytam; a hiányzó részleteket az eredeti forrásban ellenőrizd.'])
    if integrity.get('low_value_claims') or 'model_low_value_statement' in integrity.get('model_warnings', []):
        lines.extend(['', 'A kérdéshez nem közvetlenül kapcsolódó háttérmondatokat kihagytam.'])
    if 'healthcare_condition_unverified' in integrity.get('model_warnings', []):
        lines.extend(['', 'Az egészségügyi jogosultság feltételei nem igazoltak az adott forrásrészletben; a feltétel nélküli állítást kihagytam.'])
    coverage = draft.get('requested_facet_coverage', {})
    if coverage.get('missing'):
        lines.extend(['', 'Nem igazolt témakörök: ' + missing_facets_hungarian(coverage['missing'])
                      + '. (Állítás–forrásidézet alapú szöveges becslés, nem teljes körű ellenőrzés.)'])
    if draft.get('disclaimer'):
        lines.extend(['', draft['disclaimer']])
    return '\n'.join(lines).strip()


def _simple_generate(settings: Settings, case: dict, evidence: list[dict], telemetry: Telemetry,
                     run_id: str, llm) -> dict:
    context = answer_context(
        domains=[case['category']], role=_role(case), stage='evaluation',
        strategy=settings.answer_mode, reference_date=case['reference_date'],
        event_date=None, event_date_confirmed=False, event_date_source='evaluation',
        subtasks={}, focus='',
    )
    try:
        draft_model = llm.answer(case['question'], evidence, [], run_id=run_id, context=context)
        fallback = 'model_no_grounded_complete_claims' in draft_model.quality_warnings
        fallback_reason = 'model_output_unusable' if fallback else ''
        error = None
    except LLMError as exc:
        # Same safe source-only fallback principle as the production workflow.
        draft_model = source_answer(case['question'], evidence)
        fallback = True
        fallback_reason = 'model_request_failed'
        error = f'{type(exc).__name__}: {exc}'
    audited, rejected = _audit_simple_draft(draft_model.model_dump(), evidence,
                                           question=case['question'], domain=case['category'],
                                           role=_role(case))
    audited['requested_facet_coverage'] = structural_facet_coverage(
        requested_facets(case['question'], case['category'], role=_role(case)), audited['claims'],
        domain=case['category'])
    answer = _render_simple_answer(audited, evidence)
    if fallback:
        answer = source_fallback_notice(fallback_reason) + '\n\n' + answer
    integrity = audited['text_integrity']
    has_fragments = bool(integrity['incomplete_claims'] or integrity['model_warnings'])
    status = ('complete' if audited.get('claims') and not rejected and not fallback
              and not has_fragments and not audited['requested_facet_coverage']['missing'] else 'partial')
    if not audited.get('claims'):
        status = 'unsupported'
    return {
        'evidence': evidence,
        'answer_draft': audited,
        'answer_validation': {'status': 'passed' if status == 'complete' else 'partial',
                              'completion_basis': 'structural_proxy_not_human_verified',
                              'unsupported_claims': rejected},
        'final_answer': answer,
        'response_status': status,
        'answer_fallback': fallback,
        'answer_fallback_reason': fallback_reason,
        'generation_error': error,
        'tool_results': {},
    }












def run_rag_comparison(settings: Settings, *, topic: str = 'all', question_ids: list[str] | None = None,
                       local_judge: bool = False, progress=None) -> dict:
    """Run all three profiles against exactly the same selected golden cases."""
    chunks = load_chunks(settings.data_dir)
    if not chunks:
        raise RuntimeError('Processed official documents missing; download and index first')
    from .automatic_reference import ensure_auto_reference
    ensure_auto_reference(settings.data_dir, chunks=chunks)
    cases = select_cases(topic=topic, question_ids=question_ids)
    versions = index_versions(chunks)
    telemetry = Telemetry()
    dense = acquire_dense(settings)
    simple_llm = None
    rows: list[dict] = []
    agentic_app = None
    try:
        # Ensure that even an LLM initialization failure releases the exclusive
        # embedded-Qdrant lease before propagating the error to the CLI.
        simple_llm = get_llm(settings, telemetry=telemetry)
        for mode in ('baseline_dense', 'hybrid_rrf', 'agentic'):
            if mode == 'agentic':
                from ..workflow import build_workflow, initial_state
                agentic_app = build_workflow(settings, dense=dense, telemetry=telemetry)
            for case_index, case in enumerate(cases, 1):
                run_id = str(uuid.uuid4())
                started = perf_counter()
                output: dict = {}
                ranked_ids: list[str] = []
                error = None
                try:
                    if mode == 'baseline_dense':
                        evidence, ranked_ids = _dense_retrieve(
                            case['question'], case, chunks, dense, telemetry=telemetry, run_id=run_id)
                        output = _simple_generate(settings, case, evidence, telemetry, run_id, simple_llm)
                        metrics, details = _simple_metrics(case, output, ranked_ids, versions, chunks, None)
                    elif mode == 'hybrid_rrf':
                        evidence, ranked_ids = _hybrid_retrieve(
                            case['question'], case, settings, dense, telemetry=telemetry, run_id=run_id)
                        output = _simple_generate(settings, case, evidence, telemetry, run_id, simple_llm)
                        metrics, details = _simple_metrics(case, output, ranked_ids, versions, chunks, None)
                    else:
                        state = initial_state(case['question'], reference_date=date.fromisoformat(case['reference_date']))
                        state['run_id'] = run_id
                        output = agentic_app.invoke(
                            state, config={'configurable': {'thread_id': run_id}, 'recursion_limit': 30})
                        trace = telemetry.snapshot(run_id)
                        metrics, details = _full_metrics(
                            case, output, trace, versions, chunks, telemetry, run_id, None)
                        ranked_ids = [cid for branch in output.get('branch_results', {}).values()
                                      for cid in branch.get('ranked_chunk_ids', [])]
                    if local_judge and output:
                        from .local_judge import judge_answer
                        judgment = judge_answer(
                            settings, question=case['question'], answer=output.get('final_answer', ''),
                            reference_facts=case.get('expected_facts', []), evidence=output.get('evidence', []),
                            reference_status=details.get('reference_status'),
                            human_reviewed=case.get('human_reviewed', False),
                        )
                        if judgment:
                            metrics['answer_correctness'] = judgment['correctness']
                            metrics['answer_completeness'] = judgment['completeness']
                            metrics['answer_relevancy'] = judgment['relevancy']
                            metrics['faithfulness'] = judgment['faithfulness']
                            details['local_judge'] = judgment
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    metrics = {name: None for name in _SIMPLE_METRICS}
                    metrics['workflow_success_rate'] = 0.0
                    details = {}
                trace = telemetry.snapshot(run_id)
                rows.append({
                    'mode': mode,
                    'mode_label': COMPARISON_PROFILES[mode]['label'],
                    'question_id': case['question_id'],
                    'topic': case['category'],
                    'question': case['question'],
                    'success': error is None,
                    'error': error,
                    'response_status': output.get('response_status') if output else 'error',
                    'latency_s': perf_counter() - started,
                    'ranked_chunk_ids': ranked_ids,
                    'retrieved_sources': sorted({item.get('document_id') for item in output.get('evidence', [])
                                                 if item.get('document_id')}) if output else [],
                    'evidence_count': len(output.get('evidence', [])) if output else 0,
                    'generated_answer': output.get('final_answer', '') if output else '',
                    'answer_fallback': bool(output.get('answer_fallback')) if output else False,
                    # LLM success and final answer usability are distinct events:
                    # valid JSON can still contain no usable grounded claims.
                    'generation_success': _model_generated(trace, settings),
                    'model_answer_usable': (any(c.get('origin') == 'model_generated'
                                                 for c in output.get('answer_draft', {}).get('claims', []))
                                            and not output.get('answer_fallback') if output else False),
                    'answer_fallback_reason': output.get('answer_fallback_reason', '') if output else '',
                    'llm_usage': _usage(trace),
                    'llm_failures': trace.get('llm_failures', []),
                    'answer_context_selection': trace.get('selection', {}),
                    'claim_provenance': details.get('claim_provenance', {}) if output else {},
                    'native_tool_trace': trace.get('native_tool_trace', {}) if output else {},
                    'ollama_diagnostics': {
                        'responses': trace.get('llm_usage', []),
                        'failures': trace.get('llm_failures', []),
                        'note': ('eval_count is authoritative only when supplied by Ollama; '
                                 'observed_content_chars/bytes are NOT token counts. '
                                 'Never treat missing usage as zero generated tokens.'),
                    },
                    'component_durations_s': {name: sum(
                        span['duration_s'] for span in trace.get('spans', [])
                        if span['name'] == name)
                        for name in ('llm_inference', 'evidence_selection_llm',
                                     'answer_generation_llm', 'main/generate_answer',
                                     'main/answer_audit', 'rag_subgraph',
                                     'rag/hybrid_retrieval', 'rag/rerank_results',
                                     'rag/evaluate_evidence', 'rag/prepare_context',
                                     'bm25_retrieval', 'dense_retrieval',
                                     'embedding_query', 'retrieval_fusion')},
                    'metrics': metrics,
                    'details': details,
                })
                if progress:
                    progress((('baseline_dense', 'hybrid_rrf', 'agentic').index(mode) * len(cases)) + case_index,
                             len(cases) * 3)
    finally:
        close = getattr(simple_llm, 'close', None)
        if callable(close):
            close()
        release_dense(dense)

    metric_names = tuple(sorted({name for row in rows for name in row['metrics']}))
    by_mode = {mode: _mode_summary([row for row in rows if row['mode'] == mode], metric_names)
               for mode in COMPARISON_PROFILES}
    return {
        'kind': 'rag_profile_comparison_v1',
        'run_id': str(uuid.uuid4()),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': {
            'topic': topic,
            'question_ids': question_ids or [case['question_id'] for case in cases],
            'profiles': COMPARISON_PROFILES,
            'same_generator_for_baseline_and_hybrid': True,
            'model': settings.ollama_model,
            'answer_mode': settings.answer_mode,
            'embedding_model': settings.embedding_model,
            'dataset_sha256': sha256(active_dataset().read_bytes()).hexdigest(),
            'local_judge': local_judge,
            'methodology_note': (
                'Baseline and hybrid use the same production Qwen answer adapter. Agentic uses the existing '
                'production LangGraph workflow, so its delta includes routing/planning/rewrite/retry/tool overhead. '
                'Valid Qwen JSON can still yield an unusable, source-only fallback. Compare generation_successful, '
                'usable_model_answers and complete_answers separately; claim-and-quote validity is reported as citation_integrity_proxy, not semantic faithfulness; human review is N/A until source-fingerprint-pinned approvals exist. '
                'Native tool calling is a separate, bounded Ollama tool_calls/role=tool round-trip in the agentic profile only; '
                'its attempts and token usage are included in the agentic latency/cost, not in baseline/hybrid. '
                'Retrieval metrics using pinned SILVER references are machine-estimated; incomplete or stale references remain N/A.'
            ),
        },
        'execution_environment': _environment(settings, chunks),
        'rows': rows,
        'summary': {
            'by_mode': by_mode,
            'paired_deltas': _paired_deltas(by_mode),
            'evaluated_cases_per_mode': len(cases),
            'total_executions': len(rows),
        },
    }
