"""P6.16: per-answer Ollama token audit shows facts, never estimated token counts."""
from __future__ import annotations

from pathlib import Path
import ast
import json

import httpx

from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.response.diagnostics import token_diagnostics
from dap_assistant.llm import OllamaAdapter, Classification
from dap_assistant.settings import Settings


def _entry(phase='answer', **overrides):
    return {'phase': phase, 'model': 'qwen3:4b', 'done': True,
            'requested_num_ctx': 2048, 'requested_num_predict': 768,
            'prompt_eval_count': 1300, 'eval_count': 250,
            'done_reason': 'stop', 'usage_source': 'ollama_eval_count',
            'total_duration': 14_000_000_000, **overrides}


def _prompt(model='qwen3:4b', think=False):
    return {'phase': 'answer', 'model': model, 'think': think,
            'options': {'num_ctx': 2048, 'num_predict': 768}}


def test_completed_but_missing_topic_not_misreported_as_truncation():
    result = token_diagnostics(
        {'llm_usage': [_entry()]}, [_prompt()],
        {'answer_fallback': False, 'answer_context': {'generation_diagnostics': {
            'status': 'completed_with_uncovered_facets', 'model_claim_count': 2,
            'uncovered_facets_proxy': ['documents']}}})
    call = result['calls'][0]
    assert call['requested_num_ctx'] == 2048
    assert call['requested_num_predict'] == 768
    assert call['prompt_eval_count'] == 1300
    assert call['eval_count'] == 250
    assert call['done_reason'] == 'stop'
    assert call['requested_window_ratio'] == round(1300 / 2048, 4)
    assert call['output_budget_ratio'] == round(250 / 768, 4)
    assert call['think'] == 'Kikapcsolva kérve (think=false)'
    assert 'témák lefedettsége' in result['answer_outcome']
    assert 'nem bizonyítja' in result['notes']


def test_truncation_recovered_and_tool_ack_distinct():
    trace = {'llm_usage': [
        _entry('native_tool_selection', model=None, requested_num_ctx=3072,
               requested_num_predict=512, prompt_eval_count=606, eval_count=82),
        _entry('native_tool_result_ack', model=None, requested_num_ctx=3072,
               requested_num_predict=128, prompt_eval_count=744, eval_count=128, done_reason='length'),
        _entry(eval_count=768, done_reason='length'),
        _entry(model='qwen3:4b-instruct', requested_num_predict=640,
               prompt_eval_count=950, eval_count=330),
    ], 'llm_failures': [
        {'phase': 'native_tool_result_ack', 'failure_kind': 'native_tool_ack_truncated_after_receipt',
         'requested_num_predict': 128, 'done_reason': 'length'},
        {'phase': 'answer', 'failure_kind': 'output_token_limit',
         'requested_num_predict': 768, 'done_reason': 'length'},
    ]}
    final = {'answer_fallback': False, 'native_tool_trace': {
        'model': 'qwen3:4b-instruct', 'selection_stream': {'thinking_chars': 0},
        'ack_stream': {'thinking_chars': 30}},
        'answer_context': {'generation_diagnostics': {'status': 'recovered_after_output_limit',
                                                       'model_claim_count': 3}}}
    result = token_diagnostics(trace, [_prompt(), _prompt('qwen3:4b-instruct', None)], final)
    assert len(result['calls']) == 4  # no duplicate synthetic failures
    assert result['calls'][1]['status'] == 'Levágott eszköznyugtázás'
    assert result['calls'][1]['observed_thinking_chars'] == 30
    assert result['calls'][2]['status'] == 'Kimeneti tokenkorlát'
    assert result['calls'][3]['status'].startswith('Természetes leállás')
    assert result['calls'][3]['think'] == 'Nincs think paraméter (modellfüggő)'
    assert 'helyreállító kérés' in result['answer_outcome']
    assert all(row['observed_thinking_tokens'] is None for row in result['calls'])


def test_exact_request_preview_evidence_count_does_not_use_retrieval_candidate_count():
    prompt = _prompt()
    prompt['messages'] = [{'role': 'user', 'content': json.dumps({
        'question': 'Eladtam a kocsit',
        'evidence': [{'evidence_id': 'E_abc', 'text': 'Személyes adat nem naplózható'},
                     {'evidence_id': 'E_def', 'text': 'Hivatkozott adat'},
                     {'evidence_id': 'E_abc', 'text': 'Ismétlés'}]}, ensure_ascii=False)}]
    report = token_diagnostics({'llm_usage': [_entry()]}, [prompt], {'answer_fallback': False})
    assert report['calls'][0]['request_evidence_count'] == 2
    assert report['calls'][0]['request_evidence_ids'] == ['E_abc', 'E_def']
    assert 'Személyes adat' not in json.dumps(report, ensure_ascii=False)


def test_timeout_is_unobserved_not_zero_tokens_or_fake_stop():
    result = token_diagnostics({'llm_usage': [], 'llm_failures': [{
        'phase': 'answer', 'failure_kind': 'total_timeout', 'requested_num_ctx': 2048,
        'requested_num_predict': 768, 'elapsed_s': 45.1,
        'observed_content_bytes': 609,
    }]}, [_prompt()], {'answer_fallback': True})
    call = result['calls'][0]
    assert call['eval_count'] is None and call['prompt_eval_count'] is None
    assert call['done_reason'] is None
    assert call['elapsed_s'] == 45.1
    assert call['status'] == 'Teljes időkorlát'
    assert call['requested_window_ratio'] is None
    assert 'Nincs hiteles Ollama-végkeret' in result['answer_outcome']


def test_no_model_request_and_unknown_effective_window():
    empty = token_diagnostics({'llm_usage': []}, [], {'answer_strategy': 'verified_fee_extract'})
    assert empty['calls'] == []
    assert 'Nincs válaszgeneráló' in empty['answer_outcome']
    over = token_diagnostics({'llm_usage': [_entry(prompt_eval_count=2060)]},
                             [_prompt()], {'answer_fallback': False})
    assert over['calls'][0]['requested_window_ratio'] > 1
    assert 'nem a szerver által igazolt effektív ablak' in over['notes']


def test_streaming_thinking_counter_does_not_store_thinking_text():
    secret = 'NE MENTS EL MAGÁNGONDOLATOT'
    payload = {'done': True, 'done_reason': 'stop', 'eval_count': 30,
               'prompt_eval_count': 60, 'message': {'content': json.dumps(
                   {'domains': ['vehicle'], 'intents': []}), 'thinking': secret}}
    # Test non-stream mode; the same count is collected frame by frame in answer mode.
    telemetry = Telemetry()
    def respond(_request):
        return httpx.Response(200, json=payload)
    model = OllamaAdapter(Settings(), telemetry=telemetry,
                          transport=httpx.MockTransport(respond))
    output = model._ask('Feladat', '{}', Classification, run_id='thinking-check', phase='classify')
    assert output.domains == ['vehicle']
    row = telemetry.snapshot('thinking-check')['llm_usage'][0]
    assert row['observed_thinking_chars'] == len(secret)
    assert secret not in json.dumps(telemetry.snapshot('thinking-check'), ensure_ascii=False)


def test_streaming_answer_thinking_count_is_metadata_only():
    secret = 'THIS THINKING IS PRIVATE'
    response = {'done': True, 'done_reason': 'stop', 'eval_count': 20,
                'prompt_eval_count': 100, 'message': {'content': '{"claims": []}', 'thinking': secret}}
    telemetry = Telemetry()
    adapter = OllamaAdapter(Settings(), telemetry=telemetry,
                            transport=httpx.MockTransport(lambda _: httpx.Response(
                                200, text=json.dumps(response) + '\n')))
    from dap_assistant.llm import NaturalResponse
    adapter._ask('Feladat', '{}', NaturalResponse, run_id='stream-thinking', phase='answer')
    row = telemetry.snapshot('stream-thinking')['llm_usage'][0]
    assert row['observed_thinking_chars'] == len(secret)
    assert secret not in json.dumps(telemetry.snapshot('stream-thinking'), ensure_ascii=False)


def test_ui_integrates_token_tab_per_historical_and_new_answer():
    src = (Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant' / 'ui.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    insight = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'render_insight')
    body = ast.get_source_segment(src, insight)
    assert "'Tokenek és válaszlevágás'" in body
    assert 'with tokens_tab:' in body
    assert 'render_token_diagnostics(report, prompts)' in body
    assert 'st.expander(' in body
    assert "render_insight(message['report'], message.get('prompts', []), f'hist-{index}')" in src
    assert 'render_insight(report, prompts, run_id)' in src
