"""UI progress regression tests, intentionally offline and without model calls."""
from __future__ import annotations

from dap_assistant.presentation.runtime_view import event_from_update, grouped_progress_milestones


def _view(events: list[dict], spans: list[dict] | None = None) -> str:
    return '\n'.join(grouped_progress_milestones(events, {'spans': spans or []}))


def test_six_sections_show_only_real_completed_nodes():
    events = [
        event_from_update((), 'classify_intent', {'domains': ['vehicle'], 'role': 'seller',
                                                 'question_analysis': {'goal': 'deadline', 'stage': 'after_event'}}),
        event_from_update((), 'plan_tasks', {'subtasks': {
            't1': {'domain': 'vehicle', 'question': 'secret', 'depends_on': [], 'status': 'running'},
            't2': {'domain': 'vehicle', 'question': 'other secret', 'depends_on': [], 'status': 'running'},
            't3': {'domain': 'vehicle', 'question': 'other secret', 'depends_on': ['t2'], 'status': 'pending'},
        }}),
        event_from_update((), 'evidence_gate', {'evidence': [
            {'document_id': 'a', 'source_url': 'https://example.org'},
            {'document_id': 'a', 'source_url': 'https://example.org'},
        ]}),
        event_from_update((), 'engineer_context', {'context_evidence': [{}, {}],
            'context_engineering': {'context_candidate_chunks': 3, 'missing_retrieved_facets': ['documents']}}),
        event_from_update((), 'execute_tools', {'tool_results': {
            'checklist': {'status': 'success', 'items': []},
            'deadline_1': {'status': 'incomplete', 'reason': 'no date'},
        }}),
        event_from_update((), 'generate_answer', {'answer_strategy': 'source'}),
        event_from_update((), 'answer_audit', {'answer_validation': {'status': 'partial'}}),
    ]
    text = _view(events)
    assert len(grouped_progress_milestones(events)) == 6
    assert 'Autóvásárlás vagy -eladás' in text and 'Eladó' in text
    assert '2 függőség nélküli ág' in text and '3 tervezett feladatág' in text
    assert '2 dokumentum' not in text and '1 dokumentum · 2 összegyűjtött részlet' in text
    assert '2 kontextushoz kiválasztott részlet' in text
    assert '1 témához nincs lexikai forrásegyezés' in text
    assert 'Iratlista, Határidő-ellenőrzés' in text
    assert '1 sikeres helyi eszközeredmény' in text
    assert '1 további adatot igénylő / nem számítható' in text
    assert 'helyi, modell nélküli válasz' in text and 'részleges eredmény' in text
    assert 'secret' not in text and 'no date' not in text
    assert 'forrás ellenőrzött' not in text.casefold()


def test_internal_rag_measured_nodes_show_after_completed_worker_only():
    spans = [
        {'name': 'rag/process_query', 'duration_s': .002},
        {'name': 'rag/hybrid_retrieval', 'duration_s': .6},
        {'name': 'rag/rerank_results', 'duration_s': .02},
        {'name': 'rag/evaluate_evidence', 'duration_s': .003},
        {'name': 'rag/prepare_context', 'duration_s': .002},
        {'name': 'rag_subgraph', 'duration_s': 10},
    ]
    assert _view([], spans) == ''
    text = _view([{'node': 'rag_worker'}], spans)
    for name in ('Keresőkifejezések előkészítése', 'Dokumentumtalálatok',
                 'Találatok újrarangsorolása', 'Forráslefedettség', 'RAG-találatok'):
        assert name in text
    assert '10,0 s' not in text  # Parent span is not an additive child step.
    assert len(grouped_progress_milestones([{'node': 'rag_worker'}], {'spans': spans})) == 1


def test_only_real_node_updates_or_worker_spans_visible():
    text = _view([{'node': 'classify_intent'}], [
        {'name': 'rag/hybrid_retrieval', 'duration_s': 1},
        {'name': 'main/plan_tasks', 'duration_s': 12},
    ])
    assert 'Részfeladatok megtervezése' not in text
    assert 'Dokumentumtalálatok' not in text
    assert '12,0 s' not in text and '1,0 s' not in text


def test_repeated_search_is_sum_of_node_work_not_wall_time():
    text = _view([{'node': 'rag_worker'}], [
        {'name': 'rag/hybrid_retrieval', 'duration_s': .2},
        {'name': 'rag/hybrid_retrieval', 'duration_s': .3},
        {'name': 'rag_subgraph', 'duration_s': 8},
    ])
    assert '0,5 s' in text and '2 végrehajtás (összesített node-idő)' in text
    assert '8,0 s' not in text


def test_native_tool_failures_not_presented_as_executed():
    event = event_from_update((), 'execute_tools', {'tool_results': {}, 'native_tool_trace': {
        'calls': [{'status': 'failed', 'returned_to_model': False}]
    }})
    text = _view([event])
    assert '1 nem sikeres / nem átadott natív hívás' in text
    assert 'sikeres natív eszközhívás' not in text
    assert 'nem volt szükség' not in text


def test_no_tool_call_message_for_real_empty_tool_node():
    text = _view([event_from_update((), 'execute_tools', {
        'tool_results': {}, 'native_tool_trace': {'calls': []}
    })])
    assert 'nem volt szükség külön eszközre' in text


def test_invalid_rag_timings_are_hidden():
    text = _view([{'node': 'rag_worker'}], [
        {'name': 'rag/hybrid_retrieval', 'duration_s': float('nan')},
        {'name': 'rag/rerank_results', 'duration_s': -1},
    ])
    assert text == ''


def test_retrieval_breakdown_is_measured_and_does_not_claim_missing_dense_branch():
    events = [{'node': 'hybrid_retrieval'}]
    spans = [
        {'name': 'rag/hybrid_retrieval', 'duration_s': .5},
        {'name': 'bm25_retrieval', 'duration_s': .2},
        {'name': 'retrieval_fusion', 'duration_s': .01},
    ]
    text = _view(events, spans)
    assert 'BM25: 0,2 s' in text
    assert 'RRF összevezetés: 0,01 s' in text
    assert 'Vektoros keresés:' not in text
    assert 'Embedding:' not in text
    assert text.count('Dokumentumtalálatok visszakeresése') == 1
