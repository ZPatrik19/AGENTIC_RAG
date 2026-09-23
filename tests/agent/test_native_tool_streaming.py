"""Streaming and two-tool verification without a locally installed Ollama."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx

from dap_assistant.llm import OllamaAdapter
from dap_assistant.settings import Settings

DOC = {'evidence_id': 'E_aaaaaaaaaaaaaaaa',
       'text': 'Az adásvételi szerződés szükséges az átíráshoz.', 'domain': 'vehicle'}
DEADLINE = {'evidence_id': 'E_bbbbbbbbbbbbbbbb',
            'text': 'A gépjárművet 15 napon belül át kell íratni.', 'domain': 'vehicle'}
QUESTION = 'Milyen dokumentumok szükségesek, és milyen határidőket kell betartanom?'


def _call(name, evidence_id):
    return {'type': 'function', 'function': {
        'name': name, 'arguments': {'evidence_id': evidence_id}}}


def _frames(*items):
    return '\n'.join(json.dumps(item, ensure_ascii=False) for item in items) + '\n'


def _adapter(reply, **overrides):
    settings = replace(Settings(), llm_provider='ollama', **overrides)
    return OllamaAdapter(settings, transport=httpx.MockTransport(reply))


def test_streamed_native_tools_are_collected_across_frames_and_acknowledged():
    posts = []

    def reply(request):
        payload = json.loads(request.content)
        posts.append(payload)
        assert payload['stream'] is True
        assert request.extensions['timeout']['read'] == 180.0
        if len(posts) == 1:
            assert len(payload['messages'][1]['content']) < 600
            assert len(json.loads(payload['messages'][1]['content'])['evidence']) == 2
            return httpx.Response(200, content=_frames(
                {'message': {'content': ''}, 'done': False},
                {'message': {'tool_calls': [_call('get_document_checklist', DOC['evidence_id'])]}, 'done': False},
                {'message': {'tool_calls': [_call('get_deadline_mentions', DEADLINE['evidence_id'])]}, 'done': False},
                {'done': True, 'done_reason': 'stop', 'eval_count': 47, 'prompt_eval_count': 350},
            ))
        assert payload['options']['num_predict'] == 128
        assert payload['tools'] == []
        assert [m['role'] for m in payload['messages'][-2:]] == ['tool', 'tool']
        return httpx.Response(200, content=_frames(
            {'message': {'content': 'K'}, 'done': False},
            {'message': {'content': 'ész.'}, 'done': False},
            {'done': True, 'done_reason': 'stop', 'eval_count': 4},
        ))

    adapter = _adapter(reply)
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC, DEADLINE], run_id='streaming')
    finally:
        adapter.close()
    assert trace['status'] == 'completed'
    assert set(c['tool_name'] for c in trace['calls']) == {
        'get_document_checklist', 'get_deadline_mentions'}
    assert len(outputs) == 2 and len(posts) == 2
    assert trace['result_returned_to_model']
    assert trace['selection_stream']['streamed_frames'] == 4
    assert trace['selection_stream']['tool_calls_observed'] == 2
    assert trace['ack_stream']['streamed_frames'] == 3


def test_second_native_round_is_real_model_choice_in_same_conversation():
    posts = []

    def reply(request):
        payload = json.loads(request.content)
        posts.append(payload)
        if len(posts) == 1:
            return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [
                _call('get_document_checklist', DOC['evidence_id'])]}, 'done': True, 'done_reason': 'stop'})
        if len(posts) == 2:
            assert payload['messages'][-1]['role'] == 'tool'
            return httpx.Response(200, json={'message': {'role': 'assistant', 'content': 'Kész.'},
                                             'done': True, 'done_reason': 'stop'})
        if len(posts) == 3:
            assert [t['function']['name'] for t in payload['tools']] == ['get_deadline_mentions']
            assert any(m['role'] == 'tool' and m['tool_name'] == 'get_document_checklist'
                       for m in payload['messages'])
            return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [
                _call('get_deadline_mentions', DEADLINE['evidence_id'])]},
                                             'done': True, 'done_reason': 'stop'})
        assert len(posts) == 4
        assert payload['messages'][-1]['tool_name'] == 'get_deadline_mentions'
        return httpx.Response(200, json={'message': {'role': 'assistant', 'content': 'Kész.'},
                                         'done': True, 'done_reason': 'stop'})

    adapter = _adapter(reply)
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC, DEADLINE], run_id='two-rounds')
    finally:
        adapter.close()
    assert len(posts) == 4
    assert trace['status'] == 'completed' and trace['result_returned_to_model']
    assert trace['selection_rounds'] == 2
    assert trace['second_round_status'] == 'completed'
    assert {c['tool_name'] for c in trace['calls'] if c['returned_to_model']} == {
        'get_document_checklist', 'get_deadline_mentions'}
    assert len(outputs) == 2


def test_second_round_decline_is_never_fabricated_as_a_tool_call():
    n = 0

    def reply(request):
        nonlocal n
        n += 1
        if n == 1:
            return httpx.Response(200, json={'message': {'tool_calls': [
                _call('get_document_checklist', DOC['evidence_id'])]}})
        return httpx.Response(200, json={'message': {'content': 'Nem szükséges eszköz.'}})

    adapter = _adapter(reply)
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC, DEADLINE], run_id='decline')
    finally:
        adapter.close()
    assert n == 3
    assert trace['status'] == 'partial_tool_coverage'
    assert trace['second_round_status'] == 'no_model_tool_choice'
    assert [c['tool_name'] for c in trace['calls']] == ['get_document_checklist']
    assert len(outputs) == 1


class _TimeoutAfterFirstFrame(httpx.SyncByteStream):
    def __iter__(self):
        yield _frames({'message': {'thinking': 'PRIVATE THINKING'}, 'done': False}).encode()
        raise httpx.ReadTimeout('PRIVATE URL')


def test_mid_stream_read_timeout_has_bounded_safe_progress_metadata():
    adapter = _adapter(lambda _: httpx.Response(200, stream=_TimeoutAfterFirstFrame()))
    try:
        outputs, trace = adapter.call_native_tools(QUESTION, [DOC, DEADLINE], run_id='midstream')
    finally:
        adapter.close()
    diagnostic = trace['selection_diagnostic']
    assert outputs == {} and trace['status'] == 'selection_error'
    assert diagnostic['failure_stage'] == 'stream_read'
    assert diagnostic['exception_type'] == 'ReadTimeout'
    assert diagnostic['streamed_frames'] == 1
    assert diagnostic['first_frame_s'] is not None
    assert diagnostic['http_status'] == 200
    assert 'PRIVATE' not in json.dumps(trace)
