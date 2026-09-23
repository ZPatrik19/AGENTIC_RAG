"""The native tool cycle has its own bounded context and generation budgets."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx

from dap_assistant.llm import OllamaAdapter
from dap_assistant.settings import Settings

EVIDENCE = {'evidence_id': 'E_aaaaaaaaaaaaaaaa',
            'text': 'Az adásvételi szerződés szükséges, és 15 napon belül át kell íratni.',
            'domain': 'vehicle'}


def test_native_budget_is_independent_of_quick_answer_settings(monkeypatch):
    monkeypatch.setenv('OLLAMA_QUICK_NUM_CTX', '2048')
    monkeypatch.setenv('OLLAMA_QUICK_NUM_PREDICT', '180')
    monkeypatch.setenv('NATIVE_TOOL_NUM_CTX', '4096')
    monkeypatch.setenv('NATIVE_TOOL_NUM_PREDICT', '768')
    settings = replace(Settings(), llm_provider='ollama', ollama_quick_num_predict=180)
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [
                {'type': 'function', 'function': {'name': 'get_deadline_mentions',
                                                 'arguments': {'evidence_id': EVIDENCE['evidence_id']}}},
            ]}, 'done_reason': 'stop', 'eval_count': 24})
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': 'Kész.'},
                                         'done_reason': 'stop', 'eval_count': 15})

    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(respond))
    try:
        _, trace = adapter.call_native_tools('Mennyi a határidő?', [EVIDENCE], run_id='budget')
    finally:
        adapter.close()
    assert trace['status'] == 'completed'
    assert trace['calls'][0]['returned_to_model'] is True
    assert len(requests) == 2
    assert settings.ollama_quick_num_ctx == 2048
    assert settings.ollama_quick_num_predict == 180
    assert all(request['options']['num_ctx'] == 4096 for request in requests)
    assert requests[0]['options']['num_predict'] == 768
    assert requests[1]['options']['num_predict'] == 128  # short result acknowledgment


def test_native_budget_remains_bounded(monkeypatch):
    monkeypatch.setenv('NATIVE_TOOL_NUM_CTX', '999999')
    monkeypatch.setenv('NATIVE_TOOL_NUM_PREDICT', '999999')
    settings = replace(Settings(), llm_provider='ollama')
    def respond(request):
        payload = json.loads(request.content)
        assert payload['options']['num_ctx'] == 8192
        assert payload['options']['num_predict'] == 1024
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': 'No tool needed.'}})
    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(respond))
    try:
        _, trace = adapter.call_native_tools('Ellenőrzés', [EVIDENCE], run_id='bounded')
    finally:
        adapter.close()
    assert trace['status'] == 'no_model_tool_choice'
