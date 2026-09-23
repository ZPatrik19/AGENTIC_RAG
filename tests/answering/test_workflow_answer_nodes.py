"""Production node wiring with synthetic text; no local Ollama or legal claims."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _graph(monkeypatch, fake_llm, tmp_path):
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
    graph_mod.StateGraph, graph_mod.START, graph_mod.END = Graph, 'START', 'END'
    memory_pkg.InMemorySaver = type('FakeMemory', (), {})
    types_mod.Send = type('FakeSend', (), {})
    for name, obj in (('langgraph', graph_pkg), ('langgraph.graph', graph_mod),
                      ('langgraph.types', types_mod), ('langgraph.checkpoint', checkpoint_pkg),
                      ('langgraph.checkpoint.memory', memory_pkg)):
        monkeypatch.setitem(sys.modules, name, obj)
    path = Path(dap_assistant.__file__).with_name('workflow.py')
    spec = importlib.util.spec_from_file_location('dap_assistant._workflow_p69_nodes', path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'get_llm', lambda *_args, **_kwargs: fake_llm)
    monkeypatch.setattr(module, 'build_rag_graph', lambda *_args, **_kwargs: None)
    settings = replace(Settings(), llm_provider='ollama', embedding_provider='dummy',
                       native_tool_calling_enabled=True, fast_routing=True, data_dir=tmp_path)
    return module, module.build_workflow(settings)


def test_seller_real_answer_nodes_do_not_promote_pre_sale_checklist(monkeypatch, tmp_path):
    from dap_assistant.llm import Claim, Draft
    from test_life_event_answer_planning import (QUESTION_SALE, SELLER, E1, E2,
                                                     DEADLINE, UPLOAD, PREP, SERVICES)

    class FakeLLM:
        def call_native_tools(self, question, evidence, run_id=''):
            assert question == QUESTION_SALE
            return ({'native_get_document_checklist_1': {
                'tool': 'get_document_checklist', 'status': 'success',
                'source_evidence_ids': [E2],
                'items': [{'text': PREP, 'evidence_ids': [E2]},
                          {'text': UPLOAD, 'evidence_ids': [E2]},
                          {'text': SERVICES, 'evidence_ids': [E2]}]}},
                {'result_returned_to_model': True, 'calls': [{
                    'tool_name': 'get_document_checklist', 'status': 'executed',
                    'returned_to_model': True, 'result_evidence_ids': [E2]}, {
                    'tool_name': 'get_deadline_mentions', 'status': 'executed',
                    'returned_to_model': True, 'result_evidence_ids': [E1]}]})

        def answer(self, question, evidence, tools, **kwargs):
            assert kwargs['context']['response_plan']['stage'] == 'after_event'
            assert kwargs['context']['native_tool_round_trip'] is False
            return Draft(claims=[
                Claim(text=DEADLINE, evidence_ids=[E1], supporting_quote=DEADLINE,
                      category='deadline', origin='model_generated'),
                Claim(text=PREP, evidence_ids=[E2], supporting_quote=PREP,
                      category='documents', origin='model_generated')],
                model_context_evidence_ids=[E1, E2])

    with monkeypatch.context() as patch:
        module, graph = _graph(patch, FakeLLM(), tmp_path)
        state = module.initial_state(QUESTION_SALE)
        state.update(domains=['vehicle'], role='seller', stage='after_event',
                     evidence=SELLER, validation={'status': 'passed'})
        for node in ('execute_tools', 'generate_answer', 'answer_audit'):
            state.update(graph.nodes[node](state))
    answer = state['final_answer']
    assert 'Autóeladás után' in answer and DEADLINE in answer and UPLOAD in answer
    assert PREP not in answer and SERVICES not in answer
    assert state['answer_validation']['tool_utilization']['counts']['added_verbatim'] == 1
    assert state['answer_validation']['tool_utilization']['counts']['irrelevant_to_life_event'] >= 1
    assert not state.get('native_tool_trace')  # Broad seller request does not call another LLM.
    assert state['answer_validation']['text_integrity']['excluded_out_of_situation'] == [PREP]
    assert any(c['origin'] == 'tool_extract' and c['text'] == UPLOAD for c in state['answer_draft']['claims'])
    assert state['answer_validation']['response_plan']['stage'] == 'after_event'
    assert state['answer_validation']['semantic_support']['status'] == 'not_evaluated'


def test_employment_real_answer_nodes_do_not_infer_entitlements(monkeypatch, tmp_path):
    from dap_assistant.llm import Claim, Draft
    from test_life_event_answer_planning import WORK_Q, WORK, REG, E3

    class FakeLLM:
        def call_native_tools(self, question, evidence, run_id=''):
            return ({}, {})
        def answer(self, question, evidence, tools, **kwargs):
            plan = kwargs['context']['response_plan']
            assert plan['domain'] == 'employment' and plan['role'] == ''
            assert 'healthcare' not in plan['source_topic_hints']
            return Draft(claims=[Claim(text=REG, supporting_quote=REG,
                                      evidence_ids=[E3], category='steps', origin='model_generated')],
                         model_context_evidence_ids=[E3])
    with monkeypatch.context() as patch:
        module, graph = _graph(patch, FakeLLM(), tmp_path)
        state = module.initial_state(WORK_Q)
        state.update(domains=['employment'], role='', stage='after_event',
                     evidence=WORK, validation={'status': 'passed'})
        for node in ('execute_tools', 'generate_answer', 'answer_audit'):
            state.update(graph.nodes[node](state))
    assert 'Munkahely elvesztése után' in state['final_answer']
    assert REG in state['final_answer']
    assert 'nem igazolt' in state['final_answer'].lower()
    assert state['answer_validation']['semantic_support']['status'] == 'not_evaluated'
