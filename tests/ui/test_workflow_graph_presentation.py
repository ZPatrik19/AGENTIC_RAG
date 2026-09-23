"""The workflow dashboard shows planned tasks without inventing RAG executions."""
from pathlib import Path

import pytest

from dap_assistant.presentation.workflow_visualization import execution_dot, rag_subgraph_dot


def _events(*nodes):
    return [{'node': name} for name in nodes]


def test_comprehensive_plan_displays_independent_workers_and_real_results():
    tasks = {f't{i}': {
        'domain': 'employment', 'facet': facet, 'status': 'complete', 'depends_on': []
    } for i, facet in enumerate(('steps', 'documents', 'supports', 'deadline',
                                 'healthcare', 'costs'), 1)}
    branches = {key: {'search_attempt': 1, 'retrieval_status': 'complete',
                      'evidence': [{'chunk_id': key}]} for key in tasks}
    dot = execution_dot({'subtasks': tasks, 'branch_results': branches},
                        _events('classify_intent', 'plan_tasks', 'rag_worker', 'evidence_gate'))
    assert dot.count('-> "evidence_gate";') == 6
    assert '"rag_worker" [label=' not in dot
    for key in tasks:
        assert f'"task_{key}" -> "rag_{key}";' in dot
        assert f'"rag_{key}" -> "evidence_gate";' in dot
    assert 'healthcare' in dot and 'supports' in dot


def test_pending_task_does_not_get_fabricated_worker_and_dependencies_are_visible():
    dot = execution_dot({
        'subtasks': {
            't1': {'domain': 'vehicle', 'facet': 'steps', 'status': 'complete', 'depends_on': []},
            't2': {'domain': 'vehicle', 'facet': 'costs', 'status': 'pending',
                   'depends_on': ['t1']},
        },
        'branch_results': {'t1': {'search_attempt': 1, 'retrieval_status': 'complete'}},
    }, _events('classify_intent', 'plan_tasks', 'rag_worker', 'evidence_gate'))
    assert '"task_t1" -> "task_t2" [style=dashed,label="függőség"]' in dot
    assert '"task_t2" -> "rag_t2"' not in dot
    assert '"task_t1" -> "rag_t1"' in dot


def test_no_graph_events_does_not_guess_executed_nodes():
    assert 'Még nincs végrehajtási esemény' in execution_dot(
        {'subtasks': {'t1': {'status': 'complete'}}}, [])


def test_rag_subgraph_is_canonical_and_conditional_edge_is_schematic():
    diagram = rag_subgraph_dot('t1', {'search_attempt': 2, 'retrieval_status': 'complete'})
    for node in ('process_query', 'hybrid_retrieval', 'rerank_results',
                 'evaluate_evidence', 'prepare_context'):
        assert f'"{node}" [label=' in diagram
    assert '"evaluate_evidence" -> "process_query" [style=dashed' in diagram
    assert '2 keresési kör' in diagram
    with pytest.raises(ValueError):
        rag_subgraph_dot('t1', {})


def test_render_moves_workflow_diagram_to_flow_tab_without_duplicate_dev_chart():
    ui = (Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant' / 'ui.py').read_text(
        encoding='utf-8')
    flow_tab = (Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant' / 'presentation' / 'ui_flow.py').read_text(encoding='utf-8')
    report_source = (Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant'
                     / 'presentation' / 'insight_report.py').read_text(encoding='utf-8')
    dev_report = report_source.split('def render_report(', 1)[1]
    assert 'from dap_assistant.presentation.insight_report import' in ui
    assert 'Technikai LangGraph-gráf (node-ok, függőségek és eszközök)' in flow_tab
    assert 'rag_subgraph_dot(selected, branch)' in flow_tab
    assert 'Lépések és idők' in flow_tab and 'LangGraph-gráf' in flow_tab
    assert 'st.graphviz_chart' not in dev_report
