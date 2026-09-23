"""Offline request-boundary regression tests; synthetic evidence, no legal gold."""
import json
from dataclasses import replace

import httpx
import pytest

from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.context_engineering.evidence_selection import assemble_selection_context, complete_units
from dap_assistant.context_engineering.context_builder import generation_diagnostics
from dap_assistant.llm import LLMError, NaturalResponse, OllamaAdapter, _attach_source_quotes
from dap_assistant.prompt_engineering.answer_prompt import answer_plan, generation_instruction
from dap_assistant.settings import Settings
from dap_assistant.context_engineering.token_budget import answer_output_budget, fit_answer_payload, request_budget


def evidence(eid, text, facets, **metadata):
    return {'evidence_id': eid, 'text': text, 'facets': facets, **metadata}


def test_request_accounts_for_all_messages_schema_and_output():
    report = request_budget([
        {'role': 'system', 'content': 'System'},
        {'role': 'user', 'content': 'Old question'},
        {'role': 'assistant', 'content': 'Old answer'},
        {'role': 'user', 'content': json.dumps({'question': 'New question',
            'evidence': [evidence('E_1', 'A complete fact.', ['steps'])]})}],
        {'type': 'object'}, 2048, 384)
    assert report['fits']
    assert report['components']['history'] > 0
    assert report['components']['retrieved_context'] > 0
    assert report['input_limit_estimate'] == 2048 - 384 - 128
    assert report['effective_context_verified'] is False


def test_budget_drops_redundant_topic_before_unique_facet_without_mutating_input():
    data = {'question': 'Test', 'information_needs': ['steps', 'costs'],
            'evidence': [evidence('E_1', 'A complete step. ' * 16, ['steps']),
                         evidence('E_2', 'Another step. ' * 16, ['steps']),
                         evidence('E_3', 'A fee is 10 Ft.', ['costs'])]}
    full, original = fit_answer_payload(data, 'Test', {}, window=4096, output=128)
    fitted, report = fit_answer_payload(data, 'Test', {},
        window=original['estimated_input_tokens'] + 256 - 45, output=128)
    assert report['fits']
    assert report['budget_removed_evidence_ids'][0] == 'E_2'
    assert 'E_3' in [e['evidence_id'] for e in fitted['evidence']]
    assert len(data['evidence']) == len(full['evidence']) == 3
    assert fitted['evidence'][0]['text'] == data['evidence'][0]['text']


def test_oversized_question_is_preserved_and_reports_unfit():
    data = {'question': 'Private question ' * 1000, 'information_needs': ['costs'],
            'evidence': [evidence('E_1', '10 Ft.', ['costs'])]}
    fitted, report = fit_answer_payload(data, 'Test', {}, window=1024, output=384)
    assert not report['fits']
    assert fitted['question'] == data['question']
    assert report['missing_facets'] == ['costs']
    assert 'Private' not in str(report)


def test_conflicting_independent_sources_and_dates_survive_when_they_fit():
    data = {'question': 'Test', 'information_needs': ['deadline'], 'evidence': [
        evidence('E_1', 'A határidő 10 nap.', ['deadline'], document_id='a', page_number=2),
        evidence('E_2', 'A határidő 15 nap.', ['deadline'], document_id='b', effective_from='2026-01-01')]}
    fitted, report = fit_answer_payload(data, 'Test', {}, window=2048, output=384)
    assert report['fits'] and fitted['evidence'] == data['evidence']


def test_dedup_aliases_keep_versions_and_exception_units():
    base = {'document_id': 'guide', 'document_version': 'v1', 'domain': 'vehicle',
            'title': 'Guide', 'page_number': 4, 'section_path': ['Section'],
            'text': 'Az átírást 15 napon belül intézd el.\nKivéve, ha a különös feltétel fennáll.'}
    packed = assemble_selection_context([
        {**base, 'evidence_id': 'E_1'}, {**base, 'evidence_id': 'E_2'},
        {**base, 'evidence_id': 'E_3', 'document_version': 'v2'}],
        'Mennyi a határidő az autó átírására?', domain='vehicle', token_budget=1600)
    assert packed.duplicate_aliases == {'E_2': 'E_1'}
    assert len(packed.evidence) == 2
    assert all('Kivéve' in e['text'] and e['page_number'] == 4 for e in packed.evidence)
    assert all('published_at' not in e for e in packed.evidence)


def test_list_intro_and_exception_are_atomic():
    assert complete_units('A szükséges iratok:\n- igazolás;\n- szerződés.\nKivéve a különös esetet.') == [
        'A szükséges iratok:\n- igazolás;\n- szerződés.\nKivéve a különös esetet.']


@pytest.mark.parametrize('think', [False, True])
def test_stream_final_metrics_and_configurable_thinking(think):
    captured = []
    def handler(req):
        captured.append(json.loads(req.content))
        return httpx.Response(200, text='\n'.join(json.dumps(frame) for frame in [
            {'message': {'thinking': 'private reasoning'}, 'done': False},
            {'message': {'content': '{"claims":[]}'}, 'done': False},
            {'done': True, 'done_reason': 'stop', 'eval_count': 24,
             'prompt_eval_count': 80, 'prompt_eval_duration': 10,
             'eval_duration': 20, 'total_duration': 30}]))
    telemetry = Telemetry()
    adapter = OllamaAdapter(replace(Settings(), ollama_model='qwen3:4b', ollama_think=think,
        ollama_top_k=20, ollama_top_p=.95), transport=httpx.MockTransport(handler), telemetry=telemetry)
    try:
        adapter._ask('Test', 'Test', NaturalResponse, phase='answer', run_id='test')
    finally:
        adapter.close()
    assert captured[0]['think'] is think
    assert captured[0]['options']['top_p'] == .95
    usage = telemetry.snapshot('test')['llm_usage'][0]
    assert usage['eval_count'] == 24 and usage['eval_duration'] == 20
    assert usage['observed_thinking_chars'] == len('private reasoning')
    assert 'private reasoning' not in str(telemetry.snapshot('test'))


def test_instruct_main_model_omits_think_and_missing_usage_is_unknown():
    def handler(req):
        assert 'think' not in json.loads(req.content)
        return httpx.Response(200, text=json.dumps({'done': True,
            'message': {'content': '{"claims":[]}'}}))
    telemetry = Telemetry()
    adapter = OllamaAdapter(replace(Settings(), ollama_model='qwen3:4b-instruct', ollama_think=True),
        transport=httpx.MockTransport(handler), telemetry=telemetry)
    try:
        adapter._ask('Test', 'Test', NaturalResponse, phase='answer', run_id='test')
    finally:
        adapter.close()
    assert telemetry.snapshot('test')['llm_usage'][0]['eval_count'] is None


def test_preflight_rejects_without_network_or_retry_and_no_prompt_in_trace():
    def handler(req):
        pytest.fail('Oversized request must not reach Ollama')
    telemetry = Telemetry()
    adapter = OllamaAdapter(Settings(), transport=httpx.MockTransport(handler), telemetry=telemetry)
    try:
        with pytest.raises(LLMError):
            adapter._ask('Test', 'SECRET ' * 10000, NaturalResponse, phase='answer', run_id='test')
    finally:
        adapter.close()
    trace = telemetry.snapshot('test')
    assert not trace['llm_attempts']
    assert trace['llm_failures'][0]['failure_kind'] == 'input_context_budget'
    assert 'SECRET' not in str(trace)


@pytest.mark.parametrize('kind,status', [('input_context_budget', 'input_context_budget_exceeded'),
    ('stream_interrupted_or_incomplete', 'generation_failed_or_interrupted'),
    ('http_timeout', 'timeout')])
def test_failed_generation_is_distinguished(kind, status):
    result = generation_diagnostics({'llm_failures': [{'phase': 'answer', 'failure_kind': kind}]},
        fallback=True, reason='failed', model_claim_count=0, requested_facets=['costs'], draft_facets=[])
    assert result['status'] == status and result['eval_count'] is None


def test_server_length_without_client_failure_is_still_a_limit():
    result = generation_diagnostics({'llm_usage': [{'phase': 'answer', 'done_reason': 'length'}]},
        fallback=True, reason='failed', model_claim_count=0, requested_facets=[], draft_facets=[])
    assert result['status'] == 'output_token_limit'


def test_adaptive_output_is_bounded_and_legacy_cap_is_configurable():
    settings = replace(Settings(), answer_mode='quick', quick_single_pass=True,
                       ollama_quick_num_predict=384, ollama_answer_num_predict=640)
    assert answer_output_budget(settings, 1) == 384
    assert answer_output_budget(settings, 8) == 640
    assert answer_output_budget(replace(settings, answer_adaptive_output=False), 8) == 384


def test_prompt_has_no_fixed_minimum_and_keeps_conflict_and_missing_evidence_rules():
    prompt = generation_instruction(answer_plan('Mennyi a határidő?', 'vehicle'))
    assert '2–4' not in prompt and 'legfeljebb 6' not in prompt
    assert 'Minden bizonyítékkal alátámasztott információigényt' in prompt
    assert 'ne találj ki hozzá adatot' in prompt and 'evidence_id' in prompt


def test_model_disclaimer_cannot_bypass_source_audit():
    draft = _attach_source_quotes(NaturalResponse(disclaimer='A kérelem határideje 987 nap.'), [])
    assert '987' not in draft.disclaimer
    assert draft.quality_warnings == ['unverified_model_disclaimer_removed']
    conflict = _attach_source_quotes(NaturalResponse(disclaimer='forrásellentmondás'), [])
    assert 'lehetséges' in conflict.disclaimer
    assert 'model_reported_source_conflict_unverified' in conflict.quality_warnings
