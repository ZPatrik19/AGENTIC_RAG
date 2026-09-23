"""Offline tests for scope-aware RAG metrics and responsive dashboard grouping."""
from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

import pytest

from dap_assistant.evaluation.presentation.ui_metric_layout import (
    AGENTIC,
    CONTEXT,
    RETRIEVAL,
    execution_success,
    metric_sections,
    node_latency_rows,
    selected_rag_nodes,
)
from dap_assistant.evaluation.presentation.ui_metric_visibility import visible_functional_result, visible_functional_csv


DASHBOARD = (Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant'
             / 'pages' / '1_Értékelés_és_teljesítmény.py')


@pytest.mark.parametrize(('target', 'expected'), [
    ('rag/process_query', ()),
    ('rag/hybrid_retrieval', (RETRIEVAL,)),
    ('rag/rerank_results', (RETRIEVAL,)),
    ('rag/evaluate_evidence', (CONTEXT,)),
    ('rag/prepare_context', (CONTEXT,)),
])
def test_isolated_nodes_expose_only_attributable_metrics(target, expected):
    assert tuple(section.keys for section in metric_sections('single_node', target)) == expected


def test_scope_and_subflow_do_not_display_upstream_or_full_workflow_metrics():
    assert selected_rag_nodes('single_node', 'rag/hybrid_retrieval') == ('hybrid_retrieval',)
    assert selected_rag_nodes('single_node', 'rag/unknown') == ()
    assert selected_rag_nodes('subflow', 'rag/full_subgraph') == (
        'process_query', 'hybrid_retrieval', 'rerank_results',
        'evaluate_evidence', 'prepare_context',
    )
    assert tuple(section.keys for section in metric_sections('subflow', 'rag/full_subgraph')) == (
        RETRIEVAL, CONTEXT,
    )
    assert tuple(section.keys for section in metric_sections('subflow', 'rag/query_to_rerank')) == (
        RETRIEVAL,
    )
    assert tuple(section.keys for section in metric_sections('full_workflow', 'agentic/full')) == (
        RETRIEVAL, CONTEXT, ('citation_accuracy', 'citation_coverage', 'abstention_accuracy'), AGENTIC,
    )
    forbidden = {'task_completion_rate', 'recovery_success_rate', 'answer_completeness', 'faithfulness'}
    assert not any(forbidden.intersection(section.keys)
                   for section in metric_sections('full_workflow', 'agentic/full'))


def test_measured_node_latency_excludes_fixture_and_missing_spans():
    rows = [
        {'success': True, 'latency_s': 5.0, 'node_execution_trace': {'spans': [
            {'name': 'rag/process_query', 'duration_s': 3.0},
            {'name': 'rag/rerank_results', 'duration_s': 0.2},
        ]}},
        {'success': False, 'latency_s': 9.0, 'node_execution_trace': {'spans': [
            {'name': 'rag/rerank_results', 'duration_s': 0.4},
            {'name': 'rag/rerank_results', 'duration_s': -1},
        ]}},
    ]
    result = node_latency_rows(rows, ('rerank_results', 'prepare_context'))
    assert result[0]['calls'] == 2
    assert result[0]['mean_s'] == pytest.approx(0.3)
    assert result[1]['calls'] == 0
    assert result[1]['mean_s'] is None
    assert execution_success(rows) == 0.5
    assert execution_success([]) is None


def test_dashboard_renders_scoped_sections_and_compact_even_rows():
    source = DASHBOARD.read_text(encoding='utf-8')
    ast.parse(source)
    assert 'metric_sections(scope, target)' in source
    assert 'st.columns(len(section.keys), gap=' in source
    assert 'max-width: 1360px' in source
    assert "with container.container(border=True)" in source
    assert 'node_latency_rows' in source
    assert 'local_judge=False' in source
    assert 'task_completion_rate' not in source
    assert 'recovery_success_rate' not in source


def test_retrieval_and_context_from_actual_single_node_outputs():
    # No model calls. The upstream artifacts are deliberately injected into
    # downstream fixtures to prove they cannot be wrongly scored there.
    from dap_assistant.evaluation.professional import _rag_metrics

    chunk = {'chunk_id': 'gold', 'document_id': 'official', 'document_version': 'v1',
             'domain': 'vehicle', 'text': 'Hivatalos információ'}
    case = {
        'automatic_reference': True, 'human_reviewed': False,
        'reference_status': 'automatic_proxy_pinned',
        'expected_source_ids': ['official'], 'expected_source_versions': {'official': 'v1'},
        'relevant_chunk_ids': ['gold'], 'reference_chunk_sha256': {
            'gold': sha256(chunk['text'].encode('utf-8')).hexdigest()},
        'expected_subtasks': [{'domain': 'vehicle'}],
        'relevant_chunk_ids_by_task': {'t1': ['gold']},
    }
    versions = {'official': 'v1'}
    candidates = [{'chunk_id': 'noise'}, {'chunk_id': 'gold'}]

    query, _ = _rag_metrics(case, {'ranked_chunk_ids': ['gold'], 'evidence': [chunk]},
                            versions, [chunk], nodes=('process_query',))
    assert query['retrieval_recall_at_5'] is None
    assert query['context_recall'] is None

    hybrid, _ = _rag_metrics(case, {'candidates': candidates},
                             versions, [chunk], nodes=('hybrid_retrieval',))
    assert hybrid['retrieval_recall_at_5'] == 1.0
    assert hybrid['retrieval_mrr'] == 0.5
    assert hybrid['context_recall'] is None

    rerank, _ = _rag_metrics(case, {'ranked_chunk_ids': ['gold', 'noise'], 'evidence': [chunk]},
                             versions, [chunk], nodes=('rerank_results',))
    assert rerank['retrieval_mrr'] == 1.0
    assert rerank['context_recall'] is None

    assessor, _ = _rag_metrics(case, {'ranked_chunk_ids': ['gold'], 'evidence': [chunk]},
                               versions, [chunk], nodes=('evaluate_evidence',))
    assert assessor['retrieval_recall_at_5'] is None
    assert assessor['context_recall'] == 1.0

    preparer, _ = _rag_metrics(case, {'ranked_chunk_ids': ['gold'], 'evidence': [chunk]},
                               versions, [chunk], nodes=('prepare_context',))
    assert preparer['retrieval_mrr'] is None
    assert preparer['context_precision'] == 1.0


def test_old_saved_report_unchanged_while_two_new_metrics_filtered():
    original = {
        'kind': 'functional_v4', 'summary': {'metrics': {
            'task_completion_rate': {'value': 0.0}, 'recovery_success_rate': {'value': None},
            'retrieval_mrr': {'value': 0.75},
        }},
        'rows': [{'metrics': {'task_completion_rate': 0.0, 'recovery_success_rate': None,
                             'retrieval_mrr': 0.75}}],
    }
    filtered = visible_functional_result(original)
    assert set(filtered['summary']['metrics']) == {'retrieval_mrr'}
    assert set(filtered['rows'][0]['metrics']) == {'retrieval_mrr'}
    assert 'task_completion_rate' in original['summary']['metrics']
    csv = b'question_id,metric_task_completion_rate,metric_recovery_success_rate,metric_retrieval_mrr\r\nA,0,,0.75\r\n'
    exported = visible_functional_csv(csv).decode('utf-8-sig')
    assert 'metric_task_completion_rate' not in exported
    assert 'metric_recovery_success_rate' not in exported
    assert 'metric_retrieval_mrr' in exported


def test_legacy_node_runs_explicitly_require_new_evaluation():
    from dap_assistant.evaluation import professional

    assert 'node_metric_schema_version' in Path(professional.__file__).read_text(encoding='utf-8')
    assert "cfg.get('node_metric_schema_version') != 2" in DASHBOARD.read_text(encoding='utf-8')
