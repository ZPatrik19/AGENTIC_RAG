from scripts.summarize_final_validation import functional_summary, source_checks


def test_source_checks_missing_source_or_answer_is_not_a_pass():
    spec = {'method': 'lexical only', 'cases': {
        'A': {'scope': 'one fact', 'checks': [
            {'id': 'deadline', 'all_patterns': ['15 nap'], 'source_chunk_ids': ['exists']},
            {'id': 'missing', 'all_patterns': ['15 nap'], 'source_chunk_ids': ['absent']}]},
        'B': {'scope': 'missing answer', 'checks': [
            {'id': 'deadline', 'all_patterns': ['15 nap'], 'source_chunk_ids': ['exists']}]}}}
    result = source_checks(spec, [{'question_id': 'A', 'generated_answer': '15 nap'}],
                           [{'chunk_id': 'exists', 'text': '15 nap'}])
    assert result['human_reviewed'] is False
    assert result['cases'][0]['checks'][0]['lexical_match'] is True
    assert result['cases'][0]['checks'][1]['lexical_match'] is None
    assert result['cases'][1]['checks'][0]['lexical_match'] is None


def test_functional_summary_preserves_partial_and_missing_usage():
    result = functional_summary({'summary': {'successful_runs': 1}, 'rows': [
        {'response_status': 'partial', 'node_execution_trace': {'llm_usage': [
            {'prompt_eval_count': 100, 'eval_count': 20, 'done_reason': 'length'}]}}]})
    assert result['response_status_counts'] == {'partial': 1}
    assert result['done_reasons'] == {'length': 1}
    assert result['observed_completion_tokens'] == 20
