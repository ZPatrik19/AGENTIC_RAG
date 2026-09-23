"""v5 LLM module boundaries and transport contracts; synthetic inputs only."""
from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from dap_assistant import llm
from dap_assistant.response import grounding
from dap_assistant.prompt_engineering import prompt_payload
from dap_assistant.response.generation import AnswerGenerationMixin
from dap_assistant.response.models import Classification, NaturalResponse
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.inference.errors import LLMError
from dap_assistant.inference.ollama_transport import StructuredOllamaTransport
from dap_assistant.inference.request import build_ollama_request
from dap_assistant.settings import Settings


def test_public_adapter_uses_one_shared_transport_and_answer_implementation():
    assert llm.OllamaAdapter._ask is StructuredOllamaTransport._ask
    assert llm.OllamaAdapter.answer is AnswerGenerationMixin.answer
    assert llm.LLMError is LLMError
    assert llm._attach_source_quotes is grounding._attach_source_quotes
    assert llm.supplement_grounded_facts is grounding.supplement_grounded_facts
    assert llm.answer_context is prompt_payload.answer_context


def test_transport_module_does_not_depend_on_generation_or_ui():
    root = Path(llm.__file__).parent
    transport = ast.parse((root / 'inference' / 'ollama_transport.py').read_text(encoding='utf-8'))
    imports = [part for n in ast.walk(transport) if isinstance(n, ast.ImportFrom)
               for part in [n.module or '']]
    assert not any(name.startswith(('generation', 'grounding', 'streamlit', 'ui')) for name in imports)


@pytest.mark.parametrize('mode,phase,stream', [
    ('quick', 'answer', True),
    ('detailed', 'answer', True),
    ('detailed', 'classify', False),
])
def test_request_builder_keeps_original_mode_and_stream_contract(mode, phase, stream):
    settings = replace(Settings(), answer_mode=mode, ollama_num_ctx=8192,
                       ollama_quick_num_ctx=3072, ollama_answer_num_predict=900,
                       ollama_quick_num_predict=384, ollama_think=False)
    schema = NaturalResponse if phase == 'answer' else Classification
    payload = build_ollama_request(settings, 'qwen3:4b', 'Feladat',
                                   '{"information_needs":["steps","deadline"]}', schema, phase)
    assert payload['stream'] is stream
    assert payload['options']['num_ctx'] == (3072 if mode == 'quick' and stream else 8192)
    assert payload['format'] == schema.model_json_schema()
    assert payload['think'] is False
    assert payload['messages'][0]['role'] == 'system'


def test_instruct_recovery_request_has_separate_budget_and_no_thinking_option():
    settings = replace(Settings(), answer_mode='quick', ollama_think=True)
    payload = build_ollama_request(
        settings, 'qwen3:4b', 'system', '{}', NaturalResponse, 'answer',
        model_override='qwen3:4b-instruct', num_predict_override=640,
    )
    assert payload['model'] == 'qwen3:4b-instruct'
    assert payload['options']['num_predict'] == 640
    assert 'think' not in payload


def test_non_stream_503_only_retries_once_without_prompt_in_trace(monkeypatch):
    monkeypatch.setattr('dap_assistant.inference.ollama_transport.time.sleep', lambda _: None)
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, request=request, json={
            'message': {'content': '{"domains":["vehicle"]}'},
            'eval_count': 11, 'prompt_eval_count': 23,
        })

    telemetry = Telemetry()
    adapter = llm.OllamaAdapter(Settings(), transport=httpx.MockTransport(handler),
                                telemetry=telemetry)
    try:
        assert adapter.classify('PRIVATE classification', run_id='retry').domains == ['vehicle']
    finally:
        adapter.close()
    assert len(requests) == 2
    snapshot = telemetry.snapshot('retry')
    assert len(snapshot['llm_failures']) == 1
    assert snapshot['llm_usage'][0]['eval_count'] == 11
    assert 'PRIVATE classification' not in json.dumps(snapshot, ensure_ascii=False)


def test_non_stream_timeout_does_not_retry_or_echo_secret():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout('SECRET INTERNAL MESSAGE', request=request)

    telemetry = Telemetry()
    adapter = llm.OllamaAdapter(Settings(), transport=httpx.MockTransport(handler),
                                telemetry=telemetry)
    try:
        with pytest.raises(LLMError) as error:
            adapter.classify('SECRET user message', run_id='timeout')
    finally:
        adapter.close()
    assert count == 1
    assert 'SECRET' not in str(error.value)
    trace = telemetry.snapshot('timeout')
    assert trace['llm_failures'][0]['failure_kind'] == 'http_timeout'
    assert 'SECRET' not in json.dumps(trace, ensure_ascii=False)


def test_source_mode_never_calls_ollama_or_selection():
    def handler(request):
        pytest.fail('Source mode must not invoke Ollama')

    adapter = llm.OllamaAdapter(replace(Settings(), answer_mode='source'),
                                transport=httpx.MockTransport(handler))
    try:
        result = adapter.answer('Milyen ügyintézés?', evidence=[], tools=[])
    finally:
        adapter.close()
    assert result.claims == []


def test_facade_keeps_dummy_provider_independent_of_transport():
    settings = replace(Settings(), llm_provider='dummy')
    adapter = llm.get_llm(settings)
    assert isinstance(adapter, llm.DummyAdapter)
    assert adapter.classify('Eladtam az autómat.').domains == ['vehicle']
