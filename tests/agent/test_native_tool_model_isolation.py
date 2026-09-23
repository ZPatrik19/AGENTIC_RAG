"""Dedicated instruct model and accurate length/stream evidence for real tool calling."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx

from dap_assistant.llm import OllamaAdapter
from dap_assistant.settings import Settings

DOC = {'evidence_id': 'E_aaaaaaaaaaaaaaaa',
       'text': 'Az átíráshoz szükséges az adásvételi szerződés.', 'domain': 'vehicle'}
DEADLINE = {'evidence_id': 'E_bbbbbbbbbbbbbbbb',
            'text': 'Az átírás 15 napon belül elvégzendő.', 'domain': 'vehicle'}


def call(name, evidence_id):
    return {'type': 'function', 'function': {
        'name': name, 'arguments': {'evidence_id': evidence_id}}}


def test_dedicated_instruct_model_is_used_for_selection_and_ack(monkeypatch):
    monkeypatch.setenv('NATIVE_TOOL_MODEL', 'qwen3:4b-instruct')
    monkeypatch.setenv('OLLAMA_MODEL', 'qwen3:4b')
    settings = replace(Settings(), llm_provider='ollama')
    payloads = []

    def reply(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        assert payload['model'] == 'qwen3:4b-instruct'
        assert 'think' not in payload  # An instruct-only model need not accept thinking options.
        if len(payloads) == 1:
            return httpx.Response(200, json={'message': {'tool_calls': [
                call('get_document_checklist', DOC['evidence_id']),
                call('get_deadline_mentions', DEADLINE['evidence_id'])]},
                'done': True, 'done_reason': 'stop', 'eval_count': 58})
        return httpx.Response(200, json={'message': {'content': 'Kész.'},
                                         'done_reason': 'stop'})

    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(reply))
    try:
        _, trace = adapter.call_native_tools('Milyen irat és határidő?', [DOC, DEADLINE])
    finally:
        adapter.close()
    assert settings.ollama_model == 'qwen3:4b' and settings.native_tool_model == 'qwen3:4b-instruct'
    assert trace['model'] == 'qwen3:4b-instruct'
    assert trace['status'] == 'completed' and len(trace['calls']) == 2
    assert len(payloads) == 2


def test_truncated_generation_with_complete_native_calls_is_not_falsely_discarded():
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            stream = '\n'.join(json.dumps(frame) for frame in (
                {'message': {'thinking': 'PRIVATE reasoning'}, 'done': False},
                {'message': {'tool_calls': [call('get_deadline_mentions', DEADLINE['evidence_id'])]}, 'done': False},
                {'done': True, 'done_reason': 'length', 'eval_count': 512},
            )) + '\n'
            return httpx.Response(200, text=stream)
        return httpx.Response(200, json={'message': {'content': 'Kész.'}, 'done_reason': 'stop'})

    settings = replace(Settings(), llm_provider='ollama')
    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(reply))
    try:
        _, trace = adapter.call_native_tools('Mennyi a határidő?', [DEADLINE])
    finally:
        adapter.close()
    assert trace['status'] == 'completed'
    assert trace['selection_truncated_after_tool_calls'] is True
    assert trace['selection_stream']['thinking_chars'] == len('PRIVATE reasoning')
    assert trace['selection_stream']['tool_calls_observed'] == 1
    assert trace['calls'][0]['returned_to_model'] is True
    assert 'PRIVATE reasoning' not in json.dumps(trace)


def test_length_without_real_calls_records_reasoning_counts_and_fails_closed():
    stream = '\n'.join(json.dumps(frame) for frame in (
        {'message': {'thinking': 'PRIVATE reasoning'}, 'done': False},
        {'done': True, 'done_reason': 'length', 'eval_count': 512},
    )) + '\n'
    adapter = OllamaAdapter(replace(Settings(), llm_provider='ollama'),
                            transport=httpx.MockTransport(lambda _: httpx.Response(200, text=stream)))
    try:
        outputs, trace = adapter.call_native_tools('Mennyi a határidő?', [DEADLINE])
    finally:
        adapter.close()
    assert outputs == {} and trace['status'] == 'selection_error'
    diagnostic = trace['selection_diagnostic']
    assert diagnostic['failure_stage'] == 'generation_length_limit'
    assert diagnostic['thinking_chars'] == len('PRIVATE reasoning')
    assert diagnostic['tool_calls_observed'] == 0
    assert 'PRIVATE reasoning' not in json.dumps(trace)


def test_thinking_model_retains_false_think_flag_if_explicitly_selected(monkeypatch):
    monkeypatch.setenv('NATIVE_TOOL_MODEL', 'qwen3:4b-thinking')
    seen = []
    def reply(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'message': {'content': 'Nem hívok eszközt.'}, 'done_reason': 'stop'})
    adapter = OllamaAdapter(replace(Settings(), llm_provider='ollama'),
                            transport=httpx.MockTransport(reply))
    try:
        _, trace = adapter.call_native_tools('Határidő?', [DEADLINE])
    finally:
        adapter.close()
    assert trace['status'] == 'no_model_tool_choice'
    assert seen[0]['model'] == 'qwen3:4b-thinking' and seen[0]['think'] is False


def test_smoke_cli_uses_synthetic_chunks_without_loading_corpus(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    import sys
    script_path = Path(__file__).resolve().parents[2] / 'scripts' / 'verify_native_tool_calling.py'
    spec = importlib.util.spec_from_file_location('native_smoke_test_script', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'Settings', lambda: replace(Settings(), llm_provider='ollama'))
    monkeypatch.setattr(module, '_evidence', lambda settings: (_ for _ in ()).throw(
        AssertionError('smoke may not load real corpus')))

    class FakeTelemetry:
        def snapshot(self, run_id):
            return {'llm_usage': []}

    class FakeAdapter:
        def __init__(self, settings, telemetry):
            pass

        def call_native_tools(self, question, evidence, run_id):
            assert all(e['evidence_id'] in (DOC['evidence_id'], DEADLINE['evidence_id'])
                       for e in evidence)
            assert len(evidence) == 2 and 'Teszt:' in question
            return {}, {'status': 'completed', 'result_returned_to_model': True,
                        'calls': [{'tool_name': name, 'status': 'executed',
                                   'returned_to_model': True} for name in (
                            'get_document_checklist', 'get_deadline_mentions')]}

        def close(self):
            pass

    monkeypatch.setattr(module, 'Telemetry', FakeTelemetry)
    monkeypatch.setattr(module, 'OllamaAdapter', FakeAdapter)
    monkeypatch.setattr(sys, 'argv', [str(script_path), '--smoke', '--strict'])
    assert module.main() == 0
    reports = list((tmp_path / 'reports' / 'native_tool_calls').glob('*.json'))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())['evidence_kind'] == 'synthetic_smoke'
