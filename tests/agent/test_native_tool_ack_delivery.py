"""Real tool-result receipt must not be conflated with completed assistant prose."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx

from dap_assistant.llm import OllamaAdapter
from dap_assistant.settings import Settings


DOC = {'evidence_id': 'E_aaaaaaaaaaaaaaaa', 'domain': 'vehicle',
       'text': 'Az átíráshoz szükséges az adásvételi szerződés.'}
DEADLINE = {'evidence_id': 'E_bbbbbbbbbbbbbbbb', 'domain': 'vehicle',
            'text': 'A gépjárművet 15 napon belül át kell íratni.'}


def _calls():
    return [{'type': 'function', 'function': {'name': name,
            'arguments': {'evidence_id': item['evidence_id']}}}
            for name, item in (('get_document_checklist', DOC),
                               ('get_deadline_mentions', DEADLINE))]


def _stream(*frames):
    return '\n'.join(json.dumps(frame, ensure_ascii=False) for frame in frames) + '\n'


def _run(ack_frames):
    payloads = []

    def reply(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if len(payloads) == 1:
            return httpx.Response(200, text=_stream(
                {'message': {'tool_calls': _calls()}, 'done': False},
                {'done': True, 'done_reason': 'stop', 'eval_count': 35}))
        return httpx.Response(200, text=_stream(*ack_frames))

    settings = replace(Settings(), llm_provider='ollama', native_tool_model='qwen3:4b-instruct')
    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(reply))
    try:
        _, trace = adapter.call_native_tools('Melyik irat kell és mi a határidő?', [DOC, DEADLINE])
    finally:
        adapter.close()
    return trace, payloads


def test_ack_generation_limit_after_visible_reply_keeps_both_real_tool_deliveries():
    trace, payloads = _run([
        {'message': {'content': 'A forrásból látom a dokumentumokat.'}, 'done': False},
        {'message': {'content': ' Második mondat; már nem fér ki.'}, 'done': False},
        {'done': True, 'done_reason': 'length', 'eval_count': 128,
         'prompt_eval_count': 915},
    ])
    assert trace['status'] == 'completed_with_truncated_ack'
    assert trace['model'] == 'qwen3:4b-instruct'
    assert trace['result_returned_to_model'] is True
    assert trace['ack_generation_complete'] is False
    assert trace['ack_stream']['truncated'] is True
    assert trace['ack_stream']['done_reason'] == 'length'
    assert trace['ack_stream']['content_chars'] > 0
    assert not trace.get('ack_diagnostic')
    assert len(trace['calls']) == 2
    assert all(c['status'] == 'executed' and c['returned_to_model'] for c in trace['calls'])
    assert len(payloads) == 2
    assert [m['role'] for m in payloads[1]['messages'][-3:]] == ['assistant', 'tool', 'tool']
    assert payloads[1]['tools'] == []
    assert all(c['result_item_count'] > 0 for c in trace['calls'])
    assert 'A forrásból látom' not in json.dumps(trace, ensure_ascii=False)


def test_ack_generation_limit_with_no_visible_reply_does_not_claim_receipt():
    trace, _ = _run([
        {'message': {'thinking': 'not visible'}, 'done': False},
        {'done': True, 'done_reason': 'length', 'eval_count': 128,
         'prompt_eval_count': 915},
    ])
    assert trace['status'] == 'result_ack_error'
    assert trace['result_returned_to_model'] is False
    assert all(not c['returned_to_model'] for c in trace['calls'])
    assert trace['ack_diagnostic']['failure_stage'] == 'generation_length_limit'
    assert trace['ack_diagnostic']['tool_calls_observed'] == 0
    assert 'not visible' not in json.dumps(trace)


def test_ack_with_tool_calls_is_not_accepted_even_if_it_has_text():
    trace, _ = _run([
        {'message': {'content': 'Kész.', 'tool_calls': _calls()}, 'done': False},
        {'done': True, 'done_reason': 'length', 'eval_count': 128},
    ])
    assert trace['status'] == 'result_ack_error'
    assert not trace['result_returned_to_model']
    assert trace['ack_diagnostic']['failure_stage'] == 'assistant_message_schema'


def test_strict_cli_accepts_delivered_tools_and_flags_only_truncated_ack(monkeypatch, tmp_path, capsys):
    import sys
    import scripts.verify_native_tool_calling as cli

    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'Settings', lambda: replace(
        Settings(), llm_provider='ollama', native_tool_model='qwen3:4b-instruct'))
    monkeypatch.setattr(cli, '_evidence', lambda _: [DOC, DEADLINE])
    monkeypatch.setattr(sys, 'argv', ['verify_native_tool_calling.py', '--strict'])
    attempts = []

    def reply(request):
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            return httpx.Response(200, text=_stream(
                {'message': {'tool_calls': _calls()}, 'done': False},
                {'done': True, 'done_reason': 'stop', 'eval_count': 20}))
        return httpx.Response(200, text=_stream(
            {'message': {'content': 'A két eszköz eredményét megkaptam, de a szöveg hosszú.'},
             'done': False},
            {'done': True, 'done_reason': 'length', 'eval_count': 128,
             'prompt_eval_count': 915}))

    monkeypatch.setattr(cli, 'OllamaAdapter', lambda settings, telemetry: OllamaAdapter(
        settings, telemetry=telemetry, transport=httpx.MockTransport(reply)))
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert 'visszaigazoló válasz csonkolódott' in captured.err
    assert 'hívás nélkül' not in captured.err
    output = json.loads(captured.out)
    assert output['trace']['status'] == 'completed_with_truncated_ack'
    assert len(output['trace']['calls']) == 2
    assert output['trace']['result_returned_to_model'] is True
    assert (tmp_path / 'reports' / 'native_tool_calls').is_dir()
