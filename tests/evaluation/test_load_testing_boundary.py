"""Existing evaluation API must forward to one canonical load implementation."""
import pytest

from dap_assistant.evaluation import professional
from dap_assistant.evaluation import load_testing


def test_load_api_is_reexported_not_duplicated():
    for name in ('run_load', '_llm_perf', 'summarize_load_llm', '_component_stats'):
        assert getattr(professional, name) is getattr(load_testing, name)


def test_invalid_load_size_fails_before_loading_models_or_documents():
    from dap_assistant.settings import Settings
    with pytest.raises(ValueError, match='request_count'):
        load_testing.run_load(Settings(), request_count=49)


def test_load_summary_does_not_invent_llm_telemetry():
    assert load_testing.summarize_load_llm([{'latency_s': 1.0}]) == {}


def test_single_rag_node_load_run_without_llm_or_vectorstore(monkeypatch, tmp_path):
    """Execute all 50 requests through the relocated load runner with a fake graph."""
    import sys
    from types import ModuleType
    from dataclasses import replace
    from dap_assistant.settings import Settings

    class FakeGraph:
        def invoke(self, state, config=None):
            assert state['task_id'] in {'AUTO_001', 'WORK_001'}
            return {'evidence': [], 'response_status': 'n/a'}

    graph_module = ModuleType('dap_assistant.rag.rag_graph')
    graph_module.build_rag_graph = lambda *_args, **_kwargs: FakeGraph()
    monkeypatch.setitem(sys.modules, 'dap_assistant.rag.rag_graph', graph_module)
    monkeypatch.setattr(load_testing, 'acquire_dense', lambda _settings: None)
    monkeypatch.setattr(load_testing, 'release_dense', lambda _dense: None)
    monkeypatch.setattr(load_testing, 'load_chunks', lambda _path: [{'chunk_id': 'offline'}])
    monkeypatch.setattr(professional, 'select_cases', lambda topic: [
        {'question_id': name, 'category': category, 'question': 'Tesztkérdés'}
        for name, category in [('AUTO_001', 'vehicle'), ('WORK_001', 'employment')]
    ])
    monkeypatch.setattr(professional, '_environment', lambda _settings, _chunks: {})
    settings = replace(Settings(), data_dir=tmp_path, llm_provider='dummy',
                       embedding_provider='dummy')
    report = load_testing.run_load(settings, scope='single_node', target='rag/process_query',
                                   request_count=50, concurrency=2, warmup=1, seed=3)
    assert report['kind'] == 'load_v4'
    assert report['summary']['success_count'] == 50
    assert len(report['rows']) == 50
    assert len(report['summary']['warmup']) == 1
    assert 'ttft_mean_s' not in report['summary']
    assert all('llm_performance' not in row for row in report['rows'])
