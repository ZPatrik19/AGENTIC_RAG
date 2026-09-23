from dap_assistant.evaluation.professional import _llm_perf


def test_agentic_context_ratio_is_per_call_not_sum_of_prompts():
    trace = {'llm_usage': [
        {'prompt_eval_count': 1000, 'eval_count': 100, 'eval_duration': 1_000_000_000,
         'requested_num_ctx': 2048, 'requested_num_predict': 640},
        {'prompt_eval_count': 2000, 'eval_count': 20, 'eval_duration': 1_000_000_000,
         'requested_num_ctx': 4096, 'requested_num_predict': 128}]}
    result = _llm_perf(trace, 2048, 640)
    assert result['prompt_tokens'] == 3000
    assert result['context_window_utilization']['ratio'] == 1000 / 2048
    assert result['context_window_utilization']['over_budget'] is False
    assert len(result['per_call_context_windows']) == 2


def test_missing_usage_does_not_manufacture_context_observation():
    result = _llm_perf({'llm_usage': [{}]}, 2048, 640)
    assert result['context_window_utilization']['ratio'] is None
    assert result['prompt_tokens'] is None
