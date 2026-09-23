"""P6.8.1: verified native tool receipts + zero usable claims recovery.

No local Ollama, Qdrant or legal-source availability required: all evidence is synthetic.
"""
from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.evaluation.comparison import _usage
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import LLMError, OllamaAdapter
from dap_assistant.settings import Settings

EID = 'E_aaaaaaaaaaaaaaaa'
QUOTE = 'Az álláskeresési támogatás igénybevételéhez kérni kell a nyilvántartásba vételt.'
EVIDENCE = [{'evidence_id': EID, 'chunk_id': 'a' * 32, 'domain': 'employment',
             'source_url': 'https://example.org/synthetic', 'text': QUOTE,
             'section_path': ['Álláskeresés']}]
QUESTION = 'Elvesztettem a munkámat. Milyen teendőim vannak?'


def stream(content: str, *, reason='stop', tokens=180):
    return httpx.Response(200, text='\n'.join(json.dumps(frame, ensure_ascii=False) for frame in [
        {'done': False, 'message': {'content': content}},
        {'done': True, 'message': {'content': ''}, 'done_reason': reason,
         'eval_count': tokens, 'prompt_eval_count': 500},
    ]) + '\n')


def make_adapter(handler, **overrides):
    settings = replace(Settings(), llm_provider='ollama', answer_mode='quick',
                       quick_single_pass=True, native_tool_model='qwen3:4b-instruct',
                       agentic_answer_recovery_enabled=True,
                       agentic_answer_recovery_model='qwen3:4b-instruct',
                       agentic_answer_recovery_num_predict=640, **overrides)
    telemetry = Telemetry()
    return OllamaAdapter(settings, transport=httpx.MockTransport(handler), telemetry=telemetry), telemetry


def run(adapter, *, native=True, run_id='zero-claim'):
    try:
        return adapter.answer(QUESTION, EVIDENCE, [], run_id=run_id,
                              context={'life_events': ['employment'], 'role': '',
                                       'native_tool_round_trip': native})
    finally:
        adapter.close()


def natural_claim(text, evidence_id=EID):
    return {'claims': [{'text': text, 'evidence_id': evidence_id, 'category': 'steps'}]}


def test_completed_zero_grounded_claims_uses_one_real_instruct_recovery():
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            assert payload['model'] == 'qwen3:4b'
            return stream(json.dumps(natural_claim('Ezt nem támasztja alá a kiválasztott forrás.')))
        assert payload['model'] == 'qwen3:4b-instruct'
        assert payload['options']['num_predict'] == 640
        assert 'álláskeresési' in payload['messages'][1]['content'].lower()
        assert EID in payload['messages'][1]['content']
        return stream(json.dumps(natural_claim(QUOTE), ensure_ascii=False))
    adapter, telemetry = make_adapter(handler)
    draft = run(adapter)
    assert len(requests) == 2
    assert len(draft.claims) >= 1
    assert draft.claims[0].text == QUOTE
    assert draft.claims[0].origin == 'model_generated'
    assert draft.claims[0].evidence_ids == [EID]
    trace = telemetry.snapshot('zero-claim')
    assert trace['selection']['answer_recovery'] == {
        'attempted': True, 'reason': 'initial_no_grounded_claims',
        'model': 'qwen3:4b-instruct', 'max_attempts': 1, 'status': 'grounded_claims'}
    assert _usage(trace)['llm_attempts'] == 2
    assert _usage(trace)['answer_generation_completed'] is True


def test_no_verified_native_tool_receipt_never_recovers_unusable_first_output():
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return stream(json.dumps(natural_claim('A forrásból nem következő állítás.')))
    adapter, telemetry = make_adapter(handler)
    draft = run(adapter, native=False, run_id='not-native')
    assert len(requests) == 1
    assert 'model_no_grounded_complete_claims' in draft.quality_warnings
    assert all(c.origin == 'source_only' for c in draft.claims)
    assert 'answer_recovery' not in telemetry.snapshot('not-native')['selection']


def test_schema_invalid_first_generation_is_never_replayed():
    requests = []
    def handler(request):
        requests.append(request)
        return stream('{not JSON', reason='stop')
    adapter, telemetry = make_adapter(handler)
    with pytest.raises(LLMError):
        run(adapter, run_id='invalid-json')
    assert len(requests) == 1
    assert 'answer_recovery' not in telemetry.snapshot('invalid-json')['selection']


def test_recovery_that_is_still_ungrounded_is_honest_source_fallback():
    requests = []
    def handler(request):
        requests.append(request)
        return stream(json.dumps(natural_claim('Ez bizonyíthatatlan kitalált tanács.')))
    adapter, telemetry = make_adapter(handler)
    draft = run(adapter, run_id='no-valid-second')
    assert len(requests) == 2
    assert 'model_no_grounded_complete_claims' in draft.quality_warnings
    assert all(c.origin == 'source_only' for c in draft.claims)
    assert telemetry.snapshot('no-valid-second')['selection']['answer_recovery']['status'] == 'no_grounded_claims'


def test_recovery_timeout_does_not_trigger_third_request_or_false_success():
    requests = []
    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return stream('{"claims":[]}')
        raise httpx.ReadTimeout('synthetic offline timeout')
    adapter, telemetry = make_adapter(handler)
    draft = run(adapter, run_id='timeout-second')
    assert len(requests) == 2
    assert 'model_no_grounded_complete_claims' in draft.quality_warnings
    assert all(c.origin == 'source_only' for c in draft.claims)
    trace = telemetry.snapshot('timeout-second')
    assert trace['selection']['answer_recovery']['status'] == 'failed'
    assert trace['llm_failures'][-1]['phase'] == 'answer'


def test_nonempty_valid_claim_does_not_need_recovery():
    requests = []
    def handler(request):
        requests.append(request)
        return stream(json.dumps(natural_claim(QUOTE), ensure_ascii=False))
    adapter, telemetry = make_adapter(handler)
    draft = run(adapter, run_id='already-valid')
    assert len(requests) == 1
    assert draft.claims and draft.claims[0].origin == 'model_generated'
    assert 'answer_recovery' not in telemetry.snapshot('already-valid')['selection']
