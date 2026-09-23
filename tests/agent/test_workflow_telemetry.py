"""Measured workflow presentation with synthetic events/monotonic spans, no Ollama."""
from __future__ import annotations

from dataclasses import replace

from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.presentation.runtime_view import (event_from_update, execution_steps, progress_milestones,
                                   duration_label, run_export)
from dap_assistant.settings import Settings


def _events():
    return [
        event_from_update((), 'classify_intent', {'question_analysis': {
            'goal': 'cost', 'stage': 'after_event', 'raw_question': 'PRIVATE PROMPT'}}),
        event_from_update((), 'plan_tasks', {'subtasks': {
            't1': {'domain': 'vehicle', 'question': 'Do not expose this', 'status': 'running'}}}),
        event_from_update(('rag_worker:UUID',), 'hybrid_retrieval', {}),
        event_from_update((), 'evidence_gate', {'evidence': [
            {'document_id': 'test-doc', 'title': 'Test', 'source_url': 'https://example.org',
             'evidence_id': 'E_aaaaaaaaaaaaaaaa'}]}),
        event_from_update((), 'execute_tools', {'tool_results': {'t': {'status': 'success'}}}),
        event_from_update((), 'generate_answer', {}),
        event_from_update((), 'answer_audit', {'answer_validation': {'status': 'partial'}}),
    ]


def test_real_monotonic_spans_populate_correct_steps_without_double_counting():
    events = _events()
    trace = {'spans': [
        {'name': 'main/classify_intent', 'duration_s': 0.008},
        {'name': 'main/plan_tasks', 'duration_s': 0.25},
        {'name': 'main/rag_worker', 'duration_s': 3.4},  # parent; never count as search
        {'name': 'rag_subgraph', 'duration_s': 3.4},  # parent; never count as search
        {'name': 'rag/hybrid_retrieval', 'duration_s': 0.63},
        {'name': 'main/evidence_gate', 'duration_s': 0.04},
        {'name': 'main/execute_tools', 'duration_s': 1.12},
        {'name': 'native_tool/get_document_checklist', 'duration_s': 0.7},  # nested
        {'name': 'main/generate_answer', 'duration_s': 12.45},
        {'name': 'main/answer_audit', 'duration_s': 0.091},
    ]}
    rows = execution_steps(events, trace)
    assert len(rows) == 7
    assert [row['seconds'] for row in rows] == [0.008, 0.25, 0.63, 0.04, 1.12, 12.45, 0.091]
    lines = progress_milestones(events, trace)
    assert '0,008 s' in lines[0] and 'Cél: költség' in lines[0]
    assert '0,6 s' in lines[2] and 'rag_worker:' not in '\n'.join(lines)
    assert '12,4 s' in lines[5]
    assert 'részleges eredmény' in lines[-1] and '0,09 s' in lines[-1]
    assert 'PRIVATE PROMPT' not in repr(rows) and 'Do not expose this' not in '\n'.join(lines)


def test_missing_or_corrupt_timing_is_not_invented_and_missing_node_stays_hidden():
    events = _events()[:4] + _events()[5:]
    rows = execution_steps(events, {'spans': [
        {'name': 'rag/hybrid_retrieval', 'duration_s': -1},
        {'name': 'main/generate_answer', 'duration_s': float('nan')},
        {'name': 'main/classify_intent', 'duration_s': 1.4},
        {'name': 'main/execute_tools', 'duration_s': 3.0},  # no update, no display
    ]})
    assert len(rows) == 6
    assert rows[0]['seconds'] == 1.4
    assert rows[2]['seconds'] is None and rows[4]['seconds'] is None
    assert all(row['node'] != 'execute_tools' for row in rows)
    summary = '\n'.join(progress_milestones(events, {'spans': []}))
    assert ' s' not in summary and 'eszköz' not in summary
    assert duration_label(0.0032) == '0,003 s'


def test_multiple_search_spans_are_explicit_node_work_not_full_elapsed():
    events = _events()
    events.insert(3, event_from_update(('rag_worker:UUID2',), 'hybrid_retrieval', {}))
    trace = {'spans': [
        {'name': 'rag/hybrid_retrieval', 'duration_s': 0.12},
        {'name': 'rag/hybrid_retrieval', 'duration_s': 0.23},
    ]}
    row = execution_steps(events, trace)[2]
    assert round(row['seconds'], 4) == 0.35 and row['event_runs'] == 2
    assert '2 lezárt futás' in progress_milestones(events, trace)[2]
    assert row['measured_runs'] == 2


def test_telemetry_measure_and_export_retain_exact_monotonic_duration():
    events = _events()
    telemetry = Telemetry()
    with telemetry.measure('run-1', 'main/answer_audit'):
        sum(range(5))
    trace = telemetry.snapshot('run-1')
    rows = execution_steps(events, trace)
    assert rows[-1]['seconds'] is not None and rows[-1]['seconds'] >= 0
    export = run_export(events=events, trace=trace, final={'response_status': 'partial'}, elapsed_s=2)
    assert export['telemetry']['spans'] == trace['spans']


def test_production_rag_worker_reports_own_measured_task_duration(monkeypatch, tmp_path):
    from test_workflow_answer_nodes import _graph

    with monkeypatch.context() as patch:
        module, _ = _graph(patch, object(), tmp_path)

        class FakeRag:
            def invoke(self, payload, config):
                assert payload['task_id'] == 'synthetic_task'
                return {'retrieval_status': 'complete', 'search_attempt': 1,
                        'evidence': [{'evidence_id': 'E_aaaaaaaaaaaaaaaa'}]}

        patch.setattr(module, 'build_rag_graph', lambda *_args, **_kwargs: FakeRag())
        settings = replace(Settings(), embedding_provider='dummy', data_dir=tmp_path)
        graph = module.build_workflow(settings, telemetry=Telemetry())
        result = graph.nodes['rag_worker']({
            'task': {'task_id': 'synthetic_task', 'domain': 'employment', 'question': 'Synthetic'},
            'run_id': 'run-1'})
        branch = result['branch_results']['synthetic_task']
        assert branch['elapsed_s'] >= 0
        assert branch['search_attempt'] == 1
        assert len(branch['evidence']) == 1
