"""Self-contained offline checks: no customer corpus, Ollama or network needed."""
from __future__ import annotations

from copy import deepcopy
import json

from dap_assistant.evaluation.automatic_reference import ensure_auto_reference
from dap_assistant.evaluation.dataset import (load_dataset, reference_status,
    index_versions, validate_relevant_chunks)
from dap_assistant.evaluation.layers import per_task_retrieval
from dap_assistant.evaluation.readiness import reference_readiness


def fixtures():
    cases, corpus = [], []
    for category, prefix in (('vehicle', 'AUTO'), ('employment', 'WORK')):
        for index in range(1, 11):
            qid = f'{prefix}_{index:03d}'
            doc = f'official-{qid}'
            cid = f'chunk-{qid}'
            cases.append({
                'question_id': qid, 'question': f'Mi a {index} határidő?',
                'expected_domains': [category], 'expected_intents': [],
                'expected_subtasks': [{'domain': category, 'keywords': ['határidő']}],
                'expected_source_ids': [doc],
                'expected_facts': [{'fact_id': f'F_{index}',
                                   'statement': f'Az ügyintézés határideje {index} nap.',
                                   'answer_pattern': str(index), 'source_ids': [doc]}],
                'expected_tool_calls': [], 'expected_behavior': 'answer',
                'reference_answer': f'{index} nap', 'reference_date': '2026-09-22',
                'answerability': 'answerable', 'difficulty': 'simple', 'category': category,
            })
            corpus.append({'chunk_id': cid, 'document_id': doc, 'document_version': f'version-{index}',
                           'domain': category, 'text': f'Az ügyintézés határideje {index} nap.',
                           'source_url': 'https://dap.gov.hu/', 'title': qid})
    return cases, corpus


def make_dataset(tmp_path):
    cases, corpus = fixtures()
    path = tmp_path / 'golden_v4.json'
    path.write_text(json.dumps({'version': '4.0', 'cases': cases}, ensure_ascii=False), encoding='utf-8')
    return path, corpus


def test_automatic_twenty_is_silver_not_falsely_human_reviewed(tmp_path):
    source, corpus = make_dataset(tmp_path)
    payload = ensure_auto_reference(tmp_path, chunks=corpus, baseline_path=source)
    assert payload['reference_ready_cases'] == 20
    assert len(payload['cases']) == 20
    assert payload['reference_type'] == 'automatic_silver_proxy'
    assert not payload['human_reviewed']
    assert source.read_bytes()  # baseline unchanged
    cases = load_dataset(tmp_path / 'golden_auto_v4.json')
    assert all(not c['human_reviewed'] for c in cases)
    assert all(reference_status(c, index_versions(corpus)) == 'automatic_proxy_pinned' for c in cases)
    assert all(validate_relevant_chunks(c, corpus) for c in cases)
    assert reference_readiness(cases, corpus)['automatic_proxy_ready'] == 20
    assert per_task_retrieval({'branch_results': {'t1': {'ranked_chunk_ids': ['chunk-AUTO_001']}}},
                              cases[0], pinned=True)['recall_at_5'] == 1.0


def test_missing_fact_is_not_fabricated_and_stays_na(tmp_path):
    source, corpus = make_dataset(tmp_path)
    corpus[0]['text'] = 'A tájékoztató kizárólag a rendszámot részletezi.'
    payload = ensure_auto_reference(tmp_path, chunks=corpus, baseline_path=source)
    case = payload['cases'][0]
    assert payload['reference_ready_cases'] == 19
    assert case['reference_status'] == 'automatic_proxy_incomplete'
    assert case['reference_audit']['missing_fact_ids'] == ['F_1']
    assert reference_status(case, index_versions(corpus)) == 'automatic_proxy_incomplete'
    assert per_task_retrieval({'branch_results': {'t1': {'ranked_chunk_ids': []}}},
                              case, pinned=False)['recall_at_5'] is None


def test_source_text_change_invalidates_saved_proxy_and_rebuilds(tmp_path):
    source, corpus = make_dataset(tmp_path)
    path = tmp_path / 'golden_auto_v4.json'
    original = ensure_auto_reference(tmp_path, chunks=corpus, baseline_path=source)
    assert ensure_auto_reference(tmp_path, chunks=corpus, baseline_path=source)['generated_at_utc'] == original['generated_at_utc']
    changed = deepcopy(corpus)
    changed[0]['text'] += ' Második mondat.'
    assert not validate_relevant_chunks(original['cases'][0], changed)
    updated = ensure_auto_reference(tmp_path, chunks=changed, baseline_path=source)
    assert updated['corpus_fingerprint'] != original['corpus_fingerprint']
    assert validate_relevant_chunks(updated['cases'][0], changed)
    assert path.exists()


def test_two_task_branches_are_evaluated_independently():
    output = {'branch_results': {
        't1': {'ranked_chunk_ids': ['noise', 'a']},
        't2': {'ranked_chunk_ids': ['b']},
    }}
    result = per_task_retrieval(output, {'relevant_chunk_ids_by_task': {'t1': ['a'], 't2': ['b']}}, pinned=True)
    assert result['recall_at_5'] == 1
    assert result['precision_at_5'] == 0.75
    assert result['mrr'] == 0.75
    assert per_task_retrieval({'branch_results': {'other': {'ranked_chunk_ids': ['a']}}},
                              {'relevant_chunk_ids_by_task': {'t1': ['a']}}, pinned=True)['precision_at_5'] is None


def test_previous_real_human_gold_is_preserved_when_current(tmp_path):
    source, corpus = make_dataset(tmp_path)
    human_case = deepcopy(json.loads(source.read_text(encoding='utf-8'))['cases'][0])
    human_case.update({
        'human_reviewed': True,
        'reference_status': 'human_reviewed_pinned',
        'expected_source_versions': {'official-AUTO_001': 'version-1'},
        'relevant_chunk_ids': ['chunk-AUTO_001'],
        'relevant_chunk_ids_by_task': {'t1': ['chunk-AUTO_001']},
    })
    payload = json.loads(source.read_text(encoding='utf-8'))
    payload['cases'][0] = human_case
    (tmp_path / 'golden_reviewed_v4.json').write_text(
        json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    result = ensure_auto_reference(tmp_path, chunks=corpus, baseline_path=source)
    assert result['reference_ready_cases'] == 20
    assert result['human_gold_ready_cases'] == 1
    assert result['cases'][0]['human_reviewed'] is True
    assert result['cases'][1]['human_reviewed'] is False
    assert reference_status(result['cases'][0], index_versions(corpus)) == 'pinned'
    # On a changed source version, old human labels must not override new proxy.
    changed = deepcopy(corpus)
    changed[0]['document_version'] = 'updated'
    updated = ensure_auto_reference(tmp_path, chunks=changed, baseline_path=source)
    assert updated['human_gold_ready_cases'] == 0
    assert updated['cases'][0]['human_reviewed'] is False


def test_professional_rag_metrics_use_proxy_with_explicit_status(tmp_path):
    from dap_assistant.evaluation.professional import _rag_metrics
    source, corpus = make_dataset(tmp_path)
    packet = ensure_auto_reference(tmp_path, chunks=corpus, baseline_path=source)
    case = packet['cases'][0]
    metrics, details = _rag_metrics(case, {'ranked_chunk_ids': ['noise', 'chunk-AUTO_001'],
                                           'evidence': [{'chunk_id': 'chunk-AUTO_001'}]},
                                   index_versions(corpus), corpus)
    assert metrics['retrieval_recall_at_5'] == 1.0
    assert metrics['retrieval_precision_at_5'] == 0.5
    assert metrics['retrieval_mrr'] == 0.5
    assert metrics['context_recall'] == 1.0
    assert details['reference_type'] == 'automatic_silver_proxy'
    assert metrics['answer_completeness'] is None


def test_professional_full_metrics_use_independent_branch_rankings(tmp_path):
    from dap_assistant.evaluation.professional import _full_metrics
    from dap_assistant.observability.telemetry import Telemetry
    source, corpus = make_dataset(tmp_path)
    case = ensure_auto_reference(tmp_path, chunks=corpus, baseline_path=source)['cases'][0]
    output = {
        'domains': ['vehicle'], 'intents': [],
        'branch_results': {'t1': {'ranked_chunk_ids': ['noise', 'chunk-AUTO_001']}},
        'evidence': [{'chunk_id': 'chunk-AUTO_001', 'evidence_id': 'E1',
                      'document_id': 'official-AUTO_001',
                      'text': corpus[0]['text']}],
        'subtasks': {'t1': {'domain': 'vehicle', 'question': 'Határidő?',
                            'status': 'complete', 'depends_on': []}},
        'answer_draft': {'claims': []}, 'answer_validation': {},
        'tool_results': {}, 'final_answer': 'Az ügyintézés határideje 1 nap.',
        'response_status': 'complete',
    }
    metrics, details = _full_metrics(case, output, {}, index_versions(corpus), corpus,
                                     Telemetry(), 'synthetic-case')
    assert metrics['retrieval_recall_at_5'] == 1.0
    assert metrics['retrieval_precision_at_5'] == 0.5
    assert metrics['retrieval_mrr'] == 0.5
    assert metrics['context_recall'] == 1.0
    assert details['reference_status'] == 'automatic_proxy_pinned'
    assert metrics['answer_completeness'] is None
