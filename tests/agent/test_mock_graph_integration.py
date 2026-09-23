"""Real LangGraph + RAG subgraph + HTTP-mocked Ollama, no remote services.

This test MUST run in CI with langgraph installed; locally it is explicitly
skipped when the dependency is absent. All statements are synthetic fixtures.
"""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

pytest.importorskip('langgraph', reason='Install project dependencies to execute real LangGraph')

from dap_assistant import workflow
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import OllamaAdapter
from dap_assistant.rag import rag_graph
from dap_assistant.settings import Settings


def test_real_graph_routes_seller_through_rag_and_grounded_mock_answer(monkeypatch, tmp_path):
    source = [
        dict(chunk_id='aaa1111111111111', document_id='synthetic-seller',
             document_version='synthetic-v1', domain='vehicle', role='seller',
             source_url='https://dap.gov.hu/teszt/synthetic', title='SZINTETIKUS',
             section_path=['Bejelentés'],
             text='Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást. '
                  'A bejelentést a kormányablakban intézheted.', score=0.03),
        dict(chunk_id='bbb2222222222222', document_id='synthetic-seller',
             document_version='synthetic-v1', domain='vehicle', role='seller',
             source_url='https://dap.gov.hu/teszt/synthetic', title='SZINTETIKUS',
             section_path=['Dokumentumok'],
             text='Az adásvételi szerződés adatait őrizd meg a bejelentéshez.', score=0.02),
    ]
    # Only the real retrieval integration boundary is replaced: no index/model
    # download. The real compiled MAIN and RAG graphs execute every node.
    monkeypatch.setattr(rag_graph, 'hybrid_search',
                        lambda *_args, **_kwargs: [dict(item) for item in source])
    sent = []

    def handler(request):
        payload = json.loads(request.content)
        sent.append(payload)
        user = json.loads(payload['messages'][1]['content'])
        if not payload['stream']:
            allowed = {e['evidence_id'] for e in user['available_full_text']}
            ids = list(allowed)
            assert ids, 'RAG evidence must reach the selector'
            selected = {f: ids for f in user['information_needs']}
            return httpx.Response(200, json={'message': {'content': json.dumps(
                {'evidence_by_need': selected, 'missing_needs': []}, ensure_ascii=False)},
                'done_reason': 'stop', 'eval_count': 25})
        evidence = user['evidence']
        assert evidence
        source_claim = next((e for e in evidence if '15 napon' in e['text']), evidence[0])
        claims = [{'evidence_id': source_claim['evidence_id'],
                   'text': 'Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást.',
                   'category': 'deadline'}]
        content = json.dumps({'claims': claims, 'disclaimer': ''}, ensure_ascii=False)
        return httpx.Response(200, content='\n'.join([
            json.dumps({'message': {'content': content}, 'done': False}, ensure_ascii=False),
            json.dumps({'message': {'content': ''}, 'done': True,
                        'done_reason': 'stop', 'eval_count': 47}),
        ]) + '\n')

    settings = replace(Settings(), data_dir=tmp_path, llm_provider='ollama',
                       embedding_provider='dummy', answer_mode='detailed',
                       fast_routing=True, native_tool_calling_enabled=False,
                       ollama_num_ctx=8192, max_subtasks=6)
    telemetry = Telemetry()
    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(handler), telemetry=telemetry)
    monkeypatch.setattr(workflow, 'get_llm', lambda *_a, **_k: adapter)
    try:
        graph = workflow.build_workflow(settings, dense=None, telemetry=telemetry)
        initial = workflow.initial_state('Eladtam az autómat. Mit kell bejelentenem, hol és meddig?')
        initial['run_id'] = 'v10-real-graph-mock-http'
        result = graph.invoke(initial, config={'configurable': {'thread_id': 'v10-real-graph'},
                                               'recursion_limit': 50})
    finally:
        adapter.close()
    assert result['domains'] == ['vehicle'] and result['role'] == 'seller'
    assert set(result['branch_results']) == set(result['subtasks']) == {'t1'}
    assert result['branch_results']['t1']['evidence']
    assert '15 napon' in result['final_answer']
    assert 'https://dap.gov.hu/teszt/synthetic' in result['final_answer']
    assert result['response_status'] in {'partial', 'complete'}
    assert any(p['stream'] for p in sent) and any(not p['stream'] for p in sent)
    assert len(telemetry.snapshot('v10-real-graph-mock-http')['llm_usage']) >= 2
