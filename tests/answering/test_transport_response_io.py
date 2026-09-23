"""v6 transport protocol regressions: real httpx MockTransport, no paid calls."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.response.models import Classification
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.inference.errors import LLMError
from dap_assistant.inference.ollama_transport import StructuredOllamaTransport
from dap_assistant.settings import Settings


def _frame(content: str = '', **metadata) -> bytes:
    return (json.dumps({'message': {'content': content}, **metadata},
                       ensure_ascii=False) + '\n').encode('utf-8')


def _run_stream(handler, run_id='test'):
    trace = Telemetry()
    adapter = StructuredOllamaTransport(
        replace(Settings(), answer_mode='detailed'),
        transport=httpx.MockTransport(handler), telemetry=trace,
    )
    try:
        result = adapter._ask('system', 'a secret question', Classification,
                              run_id=run_id, phase='answer')
        return result, trace.snapshot(run_id)
    finally:
        adapter.close()


def test_streamed_structured_answer_uses_first_content_and_reports_usage():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, request=request, content=(
            _frame('') + _frame('{"domains":') +
            _frame('["vehicle"]}') +
            _frame('', done=True, done_reason='stop', eval_count=14,
                   prompt_eval_count=21)
        ))

    value, trace = _run_stream(handler)
    assert value.domains == ['vehicle']
    assert len(requests) == 1
    usage = trace['llm_usage'][0]
    assert usage['eval_count'] == 14
    assert usage['streamed_frames'] == 4
    assert usage['ttft_s'] is not None and usage['ttft_s'] >= 0
    assert trace['llm_successes'] == [{'phase': 'answer'}]
    assert not trace['llm_failures']
    assert 'secret question' not in json.dumps(trace)


def test_streamed_length_records_usage_before_failure_and_never_retries():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, request=request, content=(
            _frame('{"domains":') +
            _frame('', done=True, done_reason='length', eval_count=38)
        ))

    telemetry = Telemetry()
    adapter = StructuredOllamaTransport(
        replace(Settings(), answer_mode='detailed'),
        transport=httpx.MockTransport(handler), telemetry=telemetry,
    )
    try:
        with pytest.raises(LLMError, match='LLMError'):
            adapter._ask('system', 'private', Classification,
                         run_id='length', phase='answer')
    finally:
        adapter.close()
    trace = telemetry.snapshot('length')
    assert len(requests) == 1
    assert trace['llm_usage'][0]['eval_count'] == 38
    assert trace['llm_failures'][0]['failure_kind'] == 'output_token_limit'
    assert trace['llm_successes'] == []
    assert 'private' not in json.dumps(trace)


@pytest.mark.parametrize('content', [
    _frame('{"domains":["vehicle"]}'),  # no final done frame
    b'{invalid\n',  # malformed JSON line
    _frame('', done=True, done_reason='stop', eval_count=0),  # empty response
])
def test_incomplete_stream_fails_without_retry_and_without_invented_usage(content):
    count = []
    telemetry = Telemetry()

    def handler(request):
        count.append(request)
        return httpx.Response(200, request=request, content=content)

    adapter = StructuredOllamaTransport(
        replace(Settings(), answer_mode='detailed'),
        transport=httpx.MockTransport(handler), telemetry=telemetry,
    )
    try:
        with pytest.raises(LLMError):
            adapter._ask('system', 'private', Classification,
                         run_id='broken', phase='answer')
    finally:
        adapter.close()
    assert len(count) == 1
    assert telemetry.snapshot('broken')['llm_successes'] == []
    assert 'private' not in json.dumps(telemetry.snapshot('broken'))


def test_nonstream_schema_error_keeps_usage_and_has_no_auto_retry():
    calls = []
    telemetry = Telemetry()

    def handler(request):
        calls.append(request)
        return httpx.Response(200, request=request, json={
            'message': {'content': '{"domains":"invalid-type"}'},
            'eval_count': 7,
        })

    adapter = StructuredOllamaTransport(
        Settings(), transport=httpx.MockTransport(handler), telemetry=telemetry,
    )
    try:
        with pytest.raises(LLMError):
            adapter._ask('system', 'secret', Classification,
                         run_id='schema', phase='classify')
    finally:
        adapter.close()
    trace = telemetry.snapshot('schema')
    assert len(calls) == 1
    assert trace['llm_usage'][0]['eval_count'] == 7
    assert trace['llm_failures'][0]['failure_kind'] == 'schema_validation'
    assert 'invalid-type' not in json.dumps(trace)
    assert 'secret' not in json.dumps(trace)


def test_nonstream_length_records_usage_before_failure():
    telemetry = Telemetry()
    adapter = StructuredOllamaTransport(Settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request, json={
            'message': {'content': '{"domains":'},
            'done_reason': 'length', 'eval_count': 100,
        })), telemetry=telemetry)
    try:
        with pytest.raises(LLMError):
            adapter._ask('system', 'secret', Classification,
                         run_id='nonstream-length', phase='classify')
    finally:
        adapter.close()
    trace = telemetry.snapshot('nonstream-length')
    assert trace['llm_usage'][0]['eval_count'] == 100
    assert trace['llm_failures'][0]['failure_kind'] == 'output_token_limit'
    assert trace['llm_successes'] == []


def test_streamed_response_thinking_is_counted_not_logged():
    trace = Telemetry()

    def handler(request):
        return httpx.Response(200, request=request, content=(
            (json.dumps({'message': {'content': '', 'thinking': 'PRIVATE THINKING'}})+'\n').encode()
            + _frame('{"domains":["vehicle"]}', done=True,
                     done_reason='stop', eval_count=8)
        ))

    adapter = StructuredOllamaTransport(
        replace(Settings(), answer_mode='detailed'),
        transport=httpx.MockTransport(handler), telemetry=trace,
    )
    try:
        result = adapter._ask('system', 'secret', Classification,
                              run_id='thinking', phase='answer')
    finally:
        adapter.close()
    assert result.domains == ['vehicle']
    snapshot = trace.snapshot('thinking')
    assert snapshot['llm_usage'][0]['observed_thinking_chars'] == len('PRIVATE THINKING')
    assert 'PRIVATE THINKING' not in json.dumps(snapshot)
