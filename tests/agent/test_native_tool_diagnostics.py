"""Native tool errors must be diagnosable without exporting raw model/source text."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import OllamaAdapter
from dap_assistant.settings import Settings

EVIDENCE = {'evidence_id': 'E_aaaaaaaaaaaaaaaa', 'text': 'Az átírás 15 napon belül történik.',
            'domain': 'vehicle', 'source_url': 'https://example.invalid/vehicle'}
QUESTION = 'Milyen irat szükséges, és mennyi a határidő?'


def invoke(reply):
    def transport(request):
        return reply(request) if callable(reply) else reply

    telemetry = Telemetry()
    settings = replace(Settings(), llm_provider='ollama', answer_mode='quick')
    adapter = OllamaAdapter(settings, telemetry=telemetry, transport=httpx.MockTransport(transport))
    try:
        result, trace = adapter.call_native_tools(QUESTION, [EVIDENCE], run_id='diagnosis')
        return result, trace, telemetry.snapshot('diagnosis')
    finally:
        adapter.close()


@pytest.mark.parametrize('error,status,category', [
    ('error parsing tool call: PRIVATE user data', 500, 'tool_call_parse_error'),
    ('tools are not supported by the model: PRIVATE', 400, 'tool_schema_or_support_error'),
    ('thinking option unsupported: PRIVATE', 400, 'thinking_option_rejected'),
    ('model not found: PRIVATE local model name', 404, 'model_unavailable'),
])
def test_http_error_has_safe_status_and_allowlisted_category(error, status, category):
    result, trace, telemetry = invoke(httpx.Response(status, json={'error': error}))
    assert result == {} and trace['status'] == 'selection_error'
    diagnostic = trace['selection_diagnostic']
    assert diagnostic['http_status'] == status
    assert diagnostic['failure_stage'] == 'http_status'
    assert diagnostic['http_error_category'] == category
    assert diagnostic['eval_count'] is None
    assert diagnostic['requested_num_ctx'] > 0
    assert diagnostic['requested_num_predict'] > 0
    assert telemetry['llm_failures'][0]['http_status'] == status
    assert error not in json.dumps(trace, ensure_ascii=False)
    assert 'PRIVATE' not in json.dumps(telemetry, ensure_ascii=False)


def test_length_error_keeps_ollama_actual_usage_and_reason():
    result, trace, telemetry = invoke(httpx.Response(200, json={
        'done': True, 'done_reason': 'length', 'eval_count': 384, 'prompt_eval_count': 1600,
        'message': {'role': 'assistant', 'content': 'UNTRUSTED user private data'},
    }))
    assert result == {} and trace['status'] == 'selection_error'
    diagnostic = trace['selection_diagnostic']
    assert diagnostic['failure_stage'] == 'generation_length_limit'
    assert diagnostic['done_reason'] == 'length'
    assert diagnostic['eval_count'] == 384 and diagnostic['prompt_eval_count'] == 1600
    assert telemetry['llm_usage'][0]['eval_count'] == 384
    assert telemetry['llm_successes'] == []
    assert 'UNTRUSTED' not in json.dumps(trace)


def test_bad_json_and_invalid_tool_calls_have_distinct_stages():
    _, broken, _ = invoke(httpx.Response(200, text='{bad json'))
    _, wrong_type, _ = invoke(httpx.Response(200, json={
        'message': {'role': 'assistant', 'tool_calls': {'bad': 'UNTRUSTED'}},
        'done_reason': 'stop', 'eval_count': 51}))
    assert broken['selection_diagnostic']['failure_stage'] == 'stream_read'
    assert wrong_type['selection_diagnostic']['failure_stage'] == 'tool_calls_schema'
    assert wrong_type['selection_diagnostic']['eval_count'] == 51
    assert 'UNTRUSTED' not in json.dumps(wrong_type)


def test_transport_timeout_keeps_empty_http_status_and_safe_exception_name():
    def timeout(request):
        raise httpx.ReadTimeout('PRIVATE URL', request=request)
    _, trace, _ = invoke(timeout)
    diagnostic = trace['selection_diagnostic']
    assert diagnostic['failure_stage'] == 'transport'
    assert diagnostic['exception_type'] == 'ReadTimeout'
    assert diagnostic['http_status'] is None
    assert 'PRIVATE' not in json.dumps(trace)


def test_ack_error_keeps_selection_and_ack_diagnostics_separate():
    counter = 0

    def reply(request):
        nonlocal counter
        counter += 1
        if counter == 1:
            return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [{
                'type': 'function', 'function': {'name': 'get_deadline_mentions',
                                                 'arguments': {'evidence_id': EVIDENCE['evidence_id']}}
            }]}, 'eval_count': 37, 'done_reason': 'stop'})
        return httpx.Response(503, json={'error': 'PRIVATE service error'})

    result, trace, telemetry = invoke(reply)
    assert len(result) == 1 and trace['status'] == 'result_ack_error'
    assert trace['ack_diagnostic']['failure_stage'] == 'http_status'
    assert trace['ack_diagnostic']['http_status'] == 503
    assert trace['calls'][0]['status'] == 'executed'
    assert not trace['calls'][0]['returned_to_model']
    assert telemetry['llm_usage'][0]['eval_count'] == 37
    assert 'PRIVATE' not in json.dumps(trace)
