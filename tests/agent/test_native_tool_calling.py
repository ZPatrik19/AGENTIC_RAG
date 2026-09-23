"""A real Ollama tool_calls protocol is required; no fake Python-only tool traces."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import OllamaAdapter
from dap_assistant.tooling.native_tool_calling import TOOL_NAMES, _preview_excerpt, schemas
from dap_assistant.settings import Settings

DOC = {'evidence_id': 'E_aaaaaaaaaaaaaaaa', 'text': 'Az átíráshoz szükséges az adásvételi szerződés.',
       'source_url': 'https://example.test/car', 'domain': 'vehicle'}
DEADLINE = {'evidence_id': 'E_bbbbbbbbbbbbbbbb', 'text': 'A gépjárművet 15 napon belül át kell íratni.',
            'source_url': 'https://example.test/car', 'domain': 'vehicle'}
QUESTION = 'Milyen dokumentumok kellenek az átíráshoz, és milyen határidők vannak?'


def _call(name, evidence_id, **extra):
    return {'type': 'function', 'function': {'name': name, 'arguments': {'evidence_id': evidence_id, **extra}}}


def _adapter(responder, telemetry=None):
    settings = replace(Settings(), llm_provider='ollama', answer_mode='quick')
    return OllamaAdapter(settings, telemetry=telemetry, transport=httpx.MockTransport(responder))


def test_two_declared_functions_with_exact_id_enum():
    declarations = schemas([DOC['evidence_id'], DEADLINE['evidence_id']])
    assert {tool['function']['name'] for tool in declarations} == set(TOOL_NAMES)
    for tool in declarations:
        params = tool['function']['parameters']
        assert params['additionalProperties'] is False
        assert params['required'] == ['evidence_id']
        assert params['properties']['evidence_id']['enum'] == [DOC['evidence_id'], DEADLINE['evidence_id']]


def test_model_selects_two_tools_executes_and_receives_results():
    posts = []
    def mock(request):
        payload = json.loads(request.content)
        posts.append(payload)
        assert request.url.path == '/api/chat'
        if len(posts) == 1:
            assert {t['function']['name'] for t in payload['tools']} == set(TOOL_NAMES)
            assert payload['stream'] is True
            return httpx.Response(200, json={'message': {'role': 'assistant', 'content': '', 'tool_calls': [
                _call('get_document_checklist', DOC['evidence_id']),
                _call('get_deadline_mentions', DEADLINE['evidence_id']),
            ]}, 'done_reason': 'stop', 'eval_count': 45, 'prompt_eval_count': 86})
        assert len(posts) == 2
        assert payload['tools'] == []
        assert payload['messages'][-3]['tool_calls'][0]['function']['name'] == TOOL_NAMES[0]
        tool_msgs = payload['messages'][-2:]
        assert [m['role'] for m in tool_msgs] == ['tool', 'tool']
        assert [m['tool_name'] for m in tool_msgs] == list(TOOL_NAMES)
        first, second = [json.loads(m['content']) for m in tool_msgs]
        assert first['items'][0]['evidence_ids'] == [DOC['evidence_id']]
        assert second['items'][0]['days'] == 15
        assert second['items'][0]['evidence_id'] == DEADLINE['evidence_id']
        assert '2026-09-' not in json.dumps(second)  # No guessed calendar date.
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': 'A két eszköz eredményét feldolgoztam.'},
                                         'done_reason': 'stop', 'eval_count': 25, 'prompt_eval_count': 142})

    telemetry = Telemetry()
    adapter = _adapter(mock, telemetry)
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC, DEADLINE], run_id='trace2')
    finally:
        adapter.close()
    assert trace['status'] == 'completed'
    assert trace['result_returned_to_model'] is True
    assert [c['tool_name'] for c in trace['calls']] == list(TOOL_NAMES)
    assert all(c['status'] == 'executed' and c['returned_to_model'] for c in trace['calls'])
    assert len(outputs) == 2
    assert posts and len(posts) == 2
    for record, result in zip(trace['calls'], outputs.values()):
        from hashlib import sha256
        assert record['result_item_count'] == len(result['items'])
        assert record['result_sha256'] == sha256(json.dumps(
            result, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    saved = telemetry.snapshot('trace2')
    assert saved['native_tool_trace'] == trace
    assert len(saved['llm_usage']) == 2
    assert [e['phase'] for e in saved['llm_successes']] == ['native_tool_selection', 'native_tool_result_ack']
    assert DOC['text'] not in json.dumps(saved, ensure_ascii=False)
    assert DEADLINE['text'] not in json.dumps(saved, ensure_ascii=False)


@pytest.mark.parametrize('call,expected', [
    (_call('get_deadline_mentions', 'E_unknown'), 'unknown_evidence_id'),
    (_call('get_document_checklist', DOC['evidence_id'], injected='x'), 'invalid_arguments'),
    (_call('run_shell_command', DOC['evidence_id']), 'unknown_tool'),
])
def test_rejects_unknown_ids_extra_parameters_and_undeclared_functions(call, expected):
    messages = []
    def mock(request):
        payload = json.loads(request.content)
        messages.append(payload)
        if len(messages) == 1:
            return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [call]},
                                             'done_reason': 'stop'})
        assert json.loads(payload['messages'][-1]['content'])['status'] == expected
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': 'No facts.'}})
    adapter = _adapter(mock)
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC], run_id='blocked')
    finally:
        adapter.close()
    assert outputs == {}
    assert trace['status'] == 'all_calls_rejected'
    assert trace['calls'][0]['status'] == expected
    assert trace['calls'][0]['returned_to_model']
    assert len(messages) == 2


def test_rejects_over_limit_without_running_or_returning_tool_messages():
    seen = []
    def mock(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [
            _call('get_document_checklist', DOC['evidence_id']) for _ in range(3)]}})
    adapter = _adapter(mock)
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC], run_id='limit')
    finally:
        adapter.close()
    assert len(seen) == 1
    assert not outputs and not trace['calls']
    assert trace['status'] == 'tool_call_limit_exceeded'
    assert trace['model_selected_count'] == 3


def test_no_model_choice_is_not_a_tool_success():
    adapter = _adapter(lambda request: httpx.Response(200, json={
        'message': {'role': 'assistant', 'content': 'No tool is necessary.'}}))
    try:
        outputs, trace = adapter.call_native_tools('Mi a helyzet?', [DOC], run_id='none')
    finally:
        adapter.close()
    assert outputs == {}
    assert trace['status'] == 'no_model_tool_choice'
    assert trace['calls'] == [] and not trace['result_returned_to_model']


def test_tool_selection_error_does_not_raise_or_execute_anything():
    telemetry = Telemetry()
    adapter = _adapter(lambda request: httpx.Response(500, text='sensitive error'), telemetry)
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC], run_id='failure')
    finally:
        adapter.close()
    assert outputs == {} and trace['status'] == 'selection_error'
    saved = telemetry.snapshot('failure')
    assert saved['llm_failures'][0]['failure_kind'] == 'native_tool_selection_error'
    assert 'sensitive error' not in json.dumps(saved)


def test_tool_ack_failure_does_not_misreport_return_to_model():
    requests = []
    def mock(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [
                _call('get_document_checklist', DOC['evidence_id'])]}})
        return httpx.Response(503, text='cannot acknowledge')
    adapter = _adapter(mock)
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC], run_id='ackfail')
    finally:
        adapter.close()
    assert outputs['native_get_document_checklist_1']['items']
    assert trace['status'] == 'result_ack_error'
    assert not trace['result_returned_to_model']
    assert not trace['calls'][0]['returned_to_model']


def test_native_call_never_executes_for_no_evidence():
    adapter = _adapter(lambda request: pytest.fail('Unexpected HTTP request'))
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [], run_id='empty')
    finally:
        adapter.close()
    assert not outputs and trace['status'] == 'no_evidence'


def test_prompt_excerpt_centers_relevant_late_deadline():
    text = 'Általános tájékoztató. ' * 30 + 'Az átírást 15 napon belül el kell végezni.'
    excerpt = _preview_excerpt(text)
    assert '15 napon belül' in excerpt
    assert len(excerpt) <= 255


def test_workflow_visualization_shows_only_real_selected_and_acknowledged_tools():
    from dap_assistant.presentation.workflow_visualization import execution_dot
    events = [{'node': 'execute_tools'}, {'node': 'generate_answer'}]
    assert 'native_select' not in execution_dot({}, events)
    graph = execution_dot({'native_tool_trace': {
        'calls': [{'tool_name': 'get_deadline_mentions', 'status': 'executed', 'returned_to_model': True}],
        'result_returned_to_model': True}}, events)
    assert 'native_select' in graph
    assert 'get_deadline_mentions' in graph
    assert 'native_feedback' in graph
    missing_ack = execution_dot({'native_tool_trace': {
        'calls': [{'tool_name': 'get_deadline_mentions', 'status': 'executed', 'returned_to_model': False}],
        'result_returned_to_model': False}}, events)
    assert 'native_feedback' not in missing_ack
