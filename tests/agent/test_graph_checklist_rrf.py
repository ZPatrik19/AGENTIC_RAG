"""Offline production LangGraph + hybrid fusion + audited tool-to-answer wiring."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from dap_assistant.settings import Settings
from dap_assistant.rag.retrieval import hybrid_search


def test_rrf_preserves_original_ranks_and_does_not_double_count(tmp_path):
    chunks = [
        {'chunk_id': 'c1', 'document_id': 'doc1', 'domain': 'vehicle', 'role': 'buyer',
         'text': 'átírás szerződés szükséges'},
        {'chunk_id': 'c2', 'document_id': 'doc1', 'domain': 'vehicle', 'role': 'general',
         'text': 'szerződés szükséges'},
        {'chunk_id': 'c3', 'document_id': 'doc2', 'domain': 'vehicle', 'role': 'seller',
         'text': 'átírás szerződés szükséges'},
    ]
    processed = tmp_path / 'processed' / 'vehicle'
    processed.mkdir(parents=True)
    (processed / 'doc1.json').write_text(json.dumps({'chunks': chunks}), encoding='utf-8')

    class Dense:
        def search(self, *_args, **_kwargs):
            return [('c2', .94), ('c2', .89), ('c1', .84), ('c3', .95)]

    settings = replace(Settings(), data_dir=tmp_path, embedding_provider='dummy')
    ranked = hybrid_search('átírás', 'vehicle', settings, dense=Dense(), role='buyer')
    by_id = {e['chunk_id']: e for e in ranked}
    assert set(by_id) == {'c1', 'c2'}
    assert by_id['c1']['bm25_rank'] == 1
    assert by_id['c1']['dense_rank'] == 3  # Repeated c2 still occupies dense rank 2.
    assert by_id['c2']['dense_rank'] == 1
    assert by_id['c1']['score'] == pytest.approx(1/61 + 1/63)
    assert by_id['c2']['score'] == pytest.approx(1/61)
    assert by_id['c1']['fusion_method'] == 'rrf_k60_not_probability'
    assert by_id['c1']['bm25_score'] > 0 and by_id['c1']['dense_score'] == .84


def test_production_graph_renders_missing_checklist_from_verified_tool_result(monkeypatch, tmp_path):
    pytest.importorskip('langgraph')
    from dap_assistant import workflow
    from dap_assistant.llm import Claim, Draft

    eid = 'E_aaaaaaaaaaaaaaaa'
    document = 'Az adásvételi szerződést be kell mutatni.'
    deadline = 'Az átírást 15 napon belül kell intézni.'
    evidence = {'evidence_id': eid, 'chunk_id': 'aaaaaaaaaaaaaaaa',
                'document_id': 'buyer', 'domain': 'vehicle', 'role': 'buyer',
                'text': document + '\n' + deadline,
                'source_url': 'https://example.org/synthetic-vehicle', 'title': 'Tesztforrás',
                'document_version': 'synthetic', 'retrieved_at': '2026-09-21T00:00:00Z'}

    class Rag:
        def invoke(self, state, config=None):
            return {'retrieval_status': 'complete', 'search_attempt': 1,
                    'evidence': [evidence], 'ranked_chunk_ids': [evidence['chunk_id']],
                    'ranked_document_ids': ['buyer']}

    class FakeLLM:
        def call_native_tools(self, question, chunks, run_id=''):
            return ({'native_get_document_checklist_1': {
                'tool': 'get_document_checklist', 'status': 'success',
                'source_evidence_ids': [eid],
                'items': [{'text': document, 'evidence_ids': [eid]}]}},
                {'status': 'completed', 'result_returned_to_model': True,
                 'calls': [{'tool_name': 'get_document_checklist', 'status': 'executed',
                            'returned_to_model': True, 'result_evidence_ids': [eid]}]})

        def answer(self, question, chunks, tools, **kwargs):
            assert any(t.get('tool') == 'get_document_checklist' for t in tools)
            return Draft(claims=[Claim(text=deadline, evidence_ids=[eid],
                supporting_quote=deadline, category='steps', origin='model_generated')],
                model_context_evidence_ids=[eid])

    monkeypatch.setattr(workflow, 'get_llm', lambda settings, telemetry=None: FakeLLM())
    monkeypatch.setattr(workflow, 'build_rag_graph', lambda *args, **kwargs: Rag())
    config = replace(Settings(), data_dir=tmp_path, llm_provider='ollama',
                     embedding_provider='dummy', fast_routing=True,
                     native_tool_calling_enabled=True)
    app = workflow.build_workflow(config)
    state = workflow.initial_state('Vettem egy autót. Milyen dokumentumok kellenek?')
    result = app.invoke(state, config={'configurable': {'thread_id': 'tool-utilization'},
                                      'recursion_limit': 30})
    assert {'rag_worker', 'execute_tools', 'generate_answer', 'answer_audit'} <= set(app.nodes)
    assert result['native_tool_trace']['result_returned_to_model']
    assert document in result['final_answer']
    assert '*(eszközből származó, szó szerinti forrásrészlet)*' in result['final_answer']
    assert result['answer_validation']['tool_utilization']['counts']['added_verbatim'] == 1
    tool_claim = next(c for c in result['answer_draft']['claims'] if c.get('origin') == 'tool_extract')
    assert tool_claim['evidence_ids'] == [eid]
    assert any(c['claim_id'] == tool_claim['claim_id'] and c['origin'] == 'tool_extract'
               for c in result['answer_validation']['claim_provenance']['claims'])
    # Neither native tool invocation nor cite validity magically supplies semantic gold.
    from dap_assistant.evaluation.metrics import claim_support_breakdown
    from dap_assistant.evaluation.semantic_review import semantic_review
    assert claim_support_breakdown(result)['citation_integrity_proxy'] == 1.0
    assert semantic_review(result)['faithfulness'] is None
