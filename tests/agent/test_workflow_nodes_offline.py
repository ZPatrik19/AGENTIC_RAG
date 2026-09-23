"""Production node integration with graph constructor stub (not a live LangGraph run)."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def test_actual_execute_generate_audit_nodes_consume_native_checklist(monkeypatch, tmp_path):
    from dap_assistant.llm import Claim, Draft
    from dap_assistant.settings import Settings
    import dap_assistant

    graph_pkg = ModuleType('langgraph')
    checkpoint_pkg = ModuleType('langgraph.checkpoint')
    memory_pkg = ModuleType('langgraph.checkpoint.memory')
    graph_mod = ModuleType('langgraph.graph')
    types_mod = ModuleType('langgraph.types')

    class Graph:
        def __init__(self, *_args):
            self.nodes = {}
        def add_node(self, name, fn):
            self.nodes[name] = fn
        def add_edge(self, *_args):
            pass
        def add_conditional_edges(self, *_args):
            pass
        def compile(self, **_kwargs):
            return self

    class FakeMemory:
        pass

    class FakeSend:
        def __init__(self, *_args):
            pass

    graph_mod.StateGraph = Graph
    graph_mod.START = 'START'
    graph_mod.END = 'END'
    memory_pkg.InMemorySaver = FakeMemory
    types_mod.Send = FakeSend
    with monkeypatch.context() as patch:
        for name, obj in (('langgraph', graph_pkg), ('langgraph.graph', graph_mod),
                          ('langgraph.types', types_mod),
                          ('langgraph.checkpoint', checkpoint_pkg),
                          ('langgraph.checkpoint.memory', memory_pkg)):
            patch.setitem(sys.modules, name, obj)
        # Import the REAL node implementations, not a rewritten answer-audit mock.
        path = Path(dap_assistant.__file__).with_name('workflow.py')
        spec = importlib.util.spec_from_file_location('dap_assistant._workflow_p67_nodes', path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        eid = 'E_aaaaaaaaaaaaaaaa'
        doc = 'Az adásvételi szerződést be kell mutatni.'
        deadline = 'Az átírást 15 napon belül kell intézni.'
        evidence = {'evidence_id': eid, 'chunk_id': 'aaaaaaaaaaaaaaaa',
                    'document_id': 'synthetic', 'domain': 'vehicle', 'role': 'buyer',
                    'text': doc + '\n' + deadline, 'source_url': 'https://example.org/test',
                    'document_version': 'synthetic', 'title': 'Teszt'}

        class FakeLLM:
            def call_native_tools(self, question, items, run_id=''):
                return ({'native_get_document_checklist_1': {
                    'tool': 'get_document_checklist', 'status': 'success',
                    'source_evidence_ids': [eid],
                    'items': [{'text': doc, 'evidence_ids': [eid]}]}},
                    {'status': 'completed', 'result_returned_to_model': True,
                     'calls': [{'tool_name': 'get_document_checklist',
                                'status': 'executed', 'returned_to_model': True,
                                'result_evidence_ids': [eid]}]})

            def answer(self, question, chunks, tools, **_kwargs):
                assert any(x.get('tool') == 'get_document_checklist' for x in tools)
                assert _kwargs['context']['native_tool_round_trip'] is True
                return Draft(claims=[Claim(text=deadline, supporting_quote=deadline,
                    evidence_ids=[eid], origin='model_generated', category='steps')],
                    model_context_evidence_ids=[eid])

        patch.setattr(module, 'get_llm', lambda *_args, **_kwargs: FakeLLM())
        patch.setattr(module, 'build_rag_graph', lambda *_args, **_kwargs: None)
        settings = replace(Settings(), llm_provider='ollama', embedding_provider='dummy',
                           native_tool_calling_enabled=True, fast_routing=True,
                           data_dir=tmp_path)
        graph = module.build_workflow(settings)
        state = module.initial_state('Vettem egy autót. Milyen dokumentumok kellenek?')
        state.update(domains=['vehicle'], role='buyer', evidence=[evidence],
                     validation={'status': 'passed'})
        for node in ('execute_tools', 'generate_answer', 'answer_audit'):
            state.update(graph.nodes[node](state))
        assert doc in state['final_answer']
        assert 'eszközből származó' in state['final_answer']
        assert state['answer_validation']['tool_utilization']['counts']['added_verbatim'] == 1
        assert sum(c.get('origin') == 'tool_extract' for c in state['answer_draft']['claims']) == 1
        assert state['answer_validation']['semantic_support']['status'] == 'not_evaluated'
        assert len(state['answer_validation']['claim_provenance']['claims']) == 2
