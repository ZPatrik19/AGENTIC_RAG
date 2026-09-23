"""Agentic-only bounded output-length recovery; all evidence is synthetic."""
from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.evaluation.comparison import _usage, _model_generated
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import LLMError, OllamaAdapter
from dap_assistant.settings import Settings

EID = 'E_aaaaaaaaaaaaaaaa'
QUOTE = 'Az átíráshoz keresd fel a kormányablakot.'
EVIDENCE = [{'evidence_id': EID, 'chunk_id': 'a' * 32, 'domain': 'vehicle',
             'source_url': 'https://example.org/synthetic', 'text': QUOTE,
             'section_path': ['Átírás']}]
QUESTION = 'Vettem egy használt autót. Hol intézhetem az átírást?'


def stream(content, *, reason='stop', tokens=120):
    return httpx.Response(200, text='\n'.join(json.dumps(frame, ensure_ascii=False) for frame in [
        {'done': False, 'message': {'content': content}},
        {'done': True, 'message': {'content': ''}, 'done_reason': reason,
         'eval_count': tokens, 'prompt_eval_count': 900},
    ]) + '\n')


def adapter_with(handler, **overrides):
    settings = replace(Settings(), llm_provider='ollama', answer_mode='quick',
                       quick_single_pass=True, ollama_quick_num_predict=768,
                       ollama_answer_num_predict=768, native_tool_model='qwen3:4b-instruct',
                       agentic_answer_recovery_enabled=True,
                       agentic_answer_recovery_model='qwen3:4b-instruct',
                       agentic_answer_recovery_num_predict=640, **overrides)
    telemetry = Telemetry()
    return OllamaAdapter(settings, transport=httpx.MockTransport(handler), telemetry=telemetry), telemetry


def test_real_instruct_recovery_after_length_is_a_second_validated_generation():
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            assert payload['model'] == 'qwen3:4b'
            assert 'Minden bizonyítékkal alátámasztott információigényt' in payload['messages'][0]['content']
            assert 'selected_by_need' not in payload['messages'][1]['content']
            return stream('{"claims":[', reason='length', tokens=768)
        assert payload['model'] == 'qwen3:4b-instruct'
        assert 'think' not in payload
        assert payload['options']['num_predict'] == 640
        assert 'evidence' in payload['messages'][1]['content']
        return stream(json.dumps({'claims': [{'text': QUOTE, 'evidence_id': EID,
                                             'category': 'where'}]}, ensure_ascii=False))

    adapter, telemetry = adapter_with(handler)
    try:
        draft = adapter.answer(QUESTION, EVIDENCE, [], run_id='test-recovered',
                               context={'life_events': ['vehicle'], 'role': 'buyer',
                                        'native_tool_round_trip': True})
    finally:
        adapter.close()
    assert len(requests) == 2
    assert any(c.origin == 'model_generated' and c.evidence_ids == [EID] for c in draft.claims)
    assert draft.model_context_evidence_ids == [EID]
    trace = telemetry.snapshot('test-recovered')
    assert trace['selection']['answer_recovery']['status'] == 'grounded_claims'
    assert trace['llm_failures'][0]['failure_kind'] == 'output_token_limit'
    assert _usage(trace)['llm_attempts'] == 2
    assert _usage(trace)['llm_responses_with_usage'] == 2
    assert _usage(trace)['answer_generation_completed'] is True
    assert _model_generated(trace, adapter.settings) is True
    assert QUOTE not in json.dumps(trace, ensure_ascii=False)  # No answer text in persistent diagnostics.


@pytest.mark.parametrize('ctx', [None, {'life_events': ['vehicle'], 'native_tool_round_trip': False}])
def test_no_recovery_without_verified_model_delivered_native_tool_round_trip(ctx):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return stream('{"claims":[', reason='length', tokens=768)
    adapter, telemetry = adapter_with(handler)
    try:
        with pytest.raises(LLMError):
            adapter.answer(QUESTION, EVIDENCE, [], run_id='not-native', context=ctx)
    finally:
        adapter.close()
    assert len(requests) == 1
    assert 'answer_recovery' not in telemetry.snapshot('not-native')['selection']


def test_no_native_tool_round_trip_retains_original_multi_facet_prompt():
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return stream(json.dumps({'claims': [{'text': QUOTE, 'evidence_id': EID,
                                             'category': 'where'}]}, ensure_ascii=False))
    adapter, telemetry = adapter_with(handler)
    try:
        draft = adapter.answer(QUESTION, EVIDENCE, [], run_id='baseline-unmodified',
                               context={'life_events': ['vehicle'], 'native_tool_round_trip': False})
    finally:
        adapter.close()
    assert draft.claims and len(requests) == 1
    assert 'selected_by_need' in requests[0]['messages'][1]['content']
    assert requests[0]['model'] == 'qwen3:4b'
    assert 'answer_recovery' not in telemetry.snapshot('baseline-unmodified')['selection']


def test_invalid_json_with_done_stop_does_not_trigger_recovery():
    requests = []
    def handler(request):
        requests.append(request)
        return stream('{bad json', reason='stop')
    adapter, _ = adapter_with(handler)
    try:
        with pytest.raises(LLMError):
            adapter.answer(QUESTION, EVIDENCE, [], run_id='malformed',
                           context={'life_events': ['vehicle'], 'native_tool_round_trip': True})
    finally:
        adapter.close()
    assert len(requests) == 1


def test_network_timeout_never_triggers_second_inference():
    requests = []
    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout('synthetic timeout')
    adapter, _ = adapter_with(handler)
    try:
        with pytest.raises(LLMError):
            adapter.answer(QUESTION, EVIDENCE, [], run_id='timeout',
                           context={'life_events': ['vehicle'], 'native_tool_round_trip': True})
    finally:
        adapter.close()
    assert len(requests) == 1


def test_failed_recovery_is_not_counted_as_success_and_never_retries_more_than_once():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return stream('{"claims":[', reason='length', tokens=768 if len(requests) == 1 else 640)
    adapter, telemetry = adapter_with(handler)
    try:
        with pytest.raises(LLMError):
            adapter.answer(QUESTION, EVIDENCE, [], run_id='double-truncated',
                           context={'life_events': ['vehicle'], 'native_tool_round_trip': True})
    finally:
        adapter.close()
    assert len(requests) == 2
    trace = telemetry.snapshot('double-truncated')
    assert trace['selection']['answer_recovery']['status'] == 'failed'
    assert trace['llm_successes'] == []
    assert len(trace['llm_failures']) == 2
    assert _usage(trace)['answer_generation_completed'] is False


def test_disabled_switch_never_retries_after_limit():
    requests = []
    def handler(request):
        requests.append(request)
        return stream('{"claims":[', reason='length', tokens=768)
    # replace existing field instead of passing it twice in helper
    adapter, _ = adapter_with(handler)
    adapter.settings = replace(adapter.settings, agentic_answer_recovery_enabled=False)
    try:
        with pytest.raises(LLMError):
            adapter.answer(QUESTION, EVIDENCE, [], run_id='disabled',
                           context={'life_events': ['vehicle'], 'native_tool_round_trip': True})
    finally:
        adapter.close()
    assert len(requests) == 1


def test_recovery_json_with_no_verifiable_claims_remains_a_labeled_fallback():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return stream('{"claims":[', reason='length', tokens=768)
        return stream(json.dumps({'claims': [{'text': 'Nem található alátámasztás.',
                                             'evidence_id': 'E_not_offered',
                                             'category': 'steps'}]}))
    adapter, telemetry = adapter_with(handler)
    try:
        draft = adapter.answer(QUESTION, EVIDENCE, [], run_id='ungrounded',
                               context={'life_events': ['vehicle'], 'native_tool_round_trip': True})
    finally:
        adapter.close()
    assert len(requests) == 2
    assert 'model_no_grounded_complete_claims' in draft.quality_warnings
    assert all(c.origin == 'source_only' for c in draft.claims)
    assert telemetry.snapshot('ungrounded')['selection']['answer_recovery']['status'] == 'no_grounded_claims'
