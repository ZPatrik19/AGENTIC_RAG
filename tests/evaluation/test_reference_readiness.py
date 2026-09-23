"""P6.13 offline regressions: meaningful N/A, unique context and severance RAG."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from dap_assistant.evaluation.dataset import DATASET, load_dataset
from dap_assistant.evaluation.metrics import context_metrics, retrieval_metrics
from dap_assistant.evaluation.professional import _context_chunk_ids
from dap_assistant.evaluation.readiness import reference_readiness, run_reference_diagnostics
from dap_assistant.settings import Settings


def _chunk():
    return {'chunk_id': 'gold-1', 'document_id': 'dap-vehicle-buyer',
            'document_version': 'sha-123', 'domain': 'vehicle',
            'text': 'Az átírásra 15 nap áll rendelkezésre.'}


def test_missing_review_remains_na_despite_official_source():
    case = deepcopy(load_dataset(DATASET)[1])
    diagnostics = reference_readiness([case], [_chunk()])
    assert diagnostics['total'] == 1 and diagnostics['ready'] == 0
    assert diagnostics['rows'][0]['status'] == 'unversioned_reference'
    assert retrieval_metrics(['gold-1'], case.get('relevant_chunk_ids'))['recall_at_k'] is None
    assert context_metrics(['gold-1'], case.get('relevant_chunk_ids'))['context_precision'] is None
    assert 'dap-vehicle-buyer' not in diagnostics['rows'][0]['missing_source_ids']


def test_reviewed_version_pinned_case_unlocks_retrieval_and_context():
    case = deepcopy(load_dataset(DATASET)[1])
    # Test fixture ONLY. Real labels must come from explicit reviewer approval.
    case.update(expected_source_ids=['dap-vehicle-buyer'],
                expected_source_versions={'dap-vehicle-buyer': 'sha-123'},
                relevant_chunk_ids=['gold-1'], human_reviewed=True)
    ready = reference_readiness([case], [_chunk()])
    assert ready['ready'] == 1 and ready['rows'][0]['status'] == 'pinned'
    scores = retrieval_metrics(['other', 'gold-1'], case['relevant_chunk_ids'])
    assert scores['recall_at_k'] == 1 and scores['precision_at_k'] == .5
    assert scores['mrr'] == .5
    assert context_metrics(['gold-1'], case['relevant_chunk_ids'])['context_recall'] == 1
    version_changed = reference_readiness([case], [{**_chunk(), 'document_version': 'sha-456'}])
    assert version_changed['rows'][0]['status'] == 'version_mismatch'
    assert version_changed['ready'] == 0
    orphan = reference_readiness([case], [{**_chunk(), 'chunk_id': 'different'}])
    assert orphan['rows'][0]['status'] == 'invalid_chunk_reference'
    assert orphan['ready'] == 0


def test_saved_report_does_not_relabel_unversioned_or_errors_as_zero_scores():
    report = {'rows': [
        {'question_id': 'AUTO_001', 'details': {'reference_status': 'unversioned_reference'},
         'error': None},
        {'question_id': 'WORK_006', 'details': {}, 'error': "KeyError: 'employment_severance'"},
    ]}
    result = run_reference_diagnostics(report)
    assert result['ready'] == 0 and result['total'] == 2
    assert result['status_counts'] == {'unversioned_reference': 1, 'execution_error': 1}
    assert result['failures'] == [{'question_id': 'WORK_006',
                                   'error': "KeyError: 'employment_severance'"}]


def test_duplicated_branches_do_not_count_extra_retrieval_or_context_hits():
    recall = retrieval_metrics(['noise', 'noise', 'gold', 'gold'], ['gold'], k=2)
    assert recall['recall_at_k'] == 1.0
    assert recall['precision_at_k'] == .5 and recall['mrr'] == .5
    context = context_metrics(['gold', 'gold'], ['gold'])
    assert context['context_recall'] == 1.0
    assert context['context_precision'] == pytest.approx(1.0)
    assert context['context_coverage'] is None  # No task-level labels supplied.


def test_context_uses_only_last_answer_prompt_and_deduplicates():
    class Trace:
        def prompt_preview(self, run_id):
            assert run_id == 'run-1'
            return [
                {'phase': 'answer', 'messages': [{'content': json.dumps({
                    'evidence': [{'evidence_id': 'E_old'}]})}]},
                {'phase': 'native_tool_selection', 'messages': [{'content': '{}'}]},
                {'phase': 'answer', 'messages': [{'content': json.dumps({
                    'evidence': [{'evidence_id': 'E_new'}, {'evidence_id': 'E_new'}]})}]},
            ]

    output = {'evidence': [
        {'evidence_id': 'E_old', 'chunk_id': 'old-chunk'},
        {'evidence_id': 'E_new', 'chunk_id': 'new-chunk'},
    ]}
    assert _context_chunk_ids(output, Trace(), 'run-1') == ['new-chunk']


def test_employment_severance_cost_question_does_not_use_vehicle_price_terms(monkeypatch):
    """Call the real RAG closures with a stubbed LangGraph, not a mocked reranker."""
    class Graph:
        def __init__(self, *_args):
            self.nodes = {}
        def add_node(self, name, fn):
            self.nodes[name] = fn
        def add_edge(self, *_args):
            pass
        def add_conditional_edges(self, *_args):
            pass
        def compile(self):
            return self

    graph_module = ModuleType('langgraph.graph')
    graph_module.StateGraph, graph_module.START, graph_module.END = Graph, 'START', 'END'
    pkg = ModuleType('langgraph')
    pkg.graph = graph_module
    monkeypatch.setitem(sys.modules, 'langgraph', pkg)
    monkeypatch.setitem(sys.modules, 'langgraph.graph', graph_module)
    import dap_assistant
    path = Path(dap_assistant.__file__).parent / 'rag' / 'rag_graph.py'
    spec = importlib.util.spec_from_file_location('dap_assistant._rag_graph_p613_test', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    graph = module.build_rag_graph(replace(Settings(), max_rag_attempts=1))
    question = 'Mikor járhat végkielégítés, és mitől függ az összege?'
    state = {'domain': 'employment', 'query': question, 'original_query': question,
             'role': '', 'search_attempt': 0}
    state.update(graph.nodes['process_query'](state))
    assert 'összeg mértéke számítása' in state['query']
    assert 'eredetiségvizsgálat' not in state['query']
    state['candidates'] = [{
        'chunk_id': 'severance-1', 'document_id': 'njt-labour-code-termination',
        'document_version': 'sha-law', 'domain': 'employment', 'role': '',
        'source_url': 'https://njt.jog.gov.hu/jogszabaly/2012-1-00-00',
        'text': 'A végkielégítés összege a munkaviszony időtartamától függ.',
        'score': 1 / 61, 'topics': ['severance'], 'section_path': ['Végkielégítés'],
    }]
    result = graph.nodes['rerank_results'](state)
    assert result['ranked_chunk_ids'] == ['severance-1']
