"""If LangGraph is installed, keep the existing Send worker and execute_tools node."""
from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip('langgraph')

from dap_assistant import workflow
from dap_assistant.llm import DummyAdapter
from dap_assistant.settings import Settings


def test_existing_graph_uses_native_tool_result_and_retains_rag_worker(monkeypatch, tmp_path):
    evidence = {'evidence_id': 'E_aaaaaaaaaaaaaaaa', 'chunk_id': 'aaaaaaaaaaaaaaaa',
                'document_id': 'buyer', 'domain': 'vehicle', 'role': 'buyer',
                'text': 'Az adásvételi szerződés kötelező; 15 napon belül át kell íratni.',
                'source_url': 'https://dap.gov.hu/vehicle', 'title': 'Forrás',
                'document_version': 'test', 'retrieved_at': '2026-09-20T00:00:00Z'}
    class Rag:
        def invoke(self, state, config=None):
            return {'retrieval_status': 'complete', 'search_attempt': 1,
                    'evidence': [evidence], 'ranked_chunk_ids': [evidence['chunk_id']],
                    'ranked_document_ids': ['buyer']}

    received = []
    class FakeLLM:
        def call_native_tools(self, question, items, run_id=''):
            assert items == [evidence]
            return ({'native_get_document_checklist_1': {
                'tool': 'get_document_checklist', 'status': 'success',
                'items': [{'text': 'Az adásvételi szerződés kötelező;',
                           'evidence_ids': [evidence['evidence_id']]}],
                'source_evidence_ids': [evidence['evidence_id']]}},
                {'status': 'completed', 'calls': [{'tool_name': 'get_document_checklist',
                                                 'status': 'executed', 'returned_to_model': True}],
                 'result_returned_to_model': True})

        def answer(self, question, items, tools, **kwargs):
            received.extend(tools)
            return DummyAdapter.answer(question, items, tools, **kwargs)

    monkeypatch.setattr(workflow, 'get_llm', lambda settings, telemetry=None: FakeLLM())
    monkeypatch.setattr(workflow, 'build_rag_graph', lambda *args, **kwargs: Rag())
    settings = replace(Settings(), llm_provider='ollama', embedding_provider='dummy',
                       native_tool_calling_enabled=True, fast_routing=True,
                       data_dir=tmp_path)
    app = workflow.build_workflow(settings)
    assert {'rag_worker', 'execute_tools', 'generate_answer'} <= set(app.nodes)
    state = workflow.initial_state('Vettem egy autót. Milyen dokumentumok kellenek?')
    output = app.invoke(state, config={'configurable': {'thread_id': 'native-graph'},
                                       'recursion_limit': 30})
    assert output['branch_results'] and output['evidence']
    assert output['native_tool_trace']['result_returned_to_model']
    assert any(tool.get('tool') == 'get_document_checklist' for tool in received)
    assert output['final_answer']
