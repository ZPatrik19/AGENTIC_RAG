"""P5 regressions: employment healthcare, Ollama attempt accounting, and fallbacks."""
import json
import sys

import httpx
import pytest

from dap_assistant.evaluation.comparison import _mode_summary, _usage
from dap_assistant.evaluation.professional import _full_metrics
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import LLMError, OllamaAdapter, Classification, source_answer
from dap_assistant.settings import Settings


WORK_001 = 'Elvesztettem a munkámat. Milyen ügyintézési teendőim vannak?'
EVIDENCE = [
    {'evidence_id': 'E_healthcare', 'chunk_id': 'healthcare-chunk',
     'document_id': 'neak-post-employment-healthcare', 'domain': 'employment',
     'text': 'A munkaviszony megszűnése után az egészségügyi szolgáltatásra való jogosultságot ellenőrizni kell.',
     'source_url': 'https://neak.gov.hu/pelda', 'title': 'NEAK'},
    {'evidence_id': 'E_work', 'chunk_id': 'work-chunk',
     'document_id': 'dap-employment-overview', 'domain': 'employment',
     'text': 'Az álláskeresőként történő nyilvántartásba vételt kérelmezni kell.',
     'source_url': 'https://dap.gov.hu/pelda', 'title': 'DÁP'},
]


def test_work001_source_fallback_keeps_healthcare_and_other_actions():
    """A healthcare evidence can no longer raise KeyError in source_answer."""
    draft = source_answer(WORK_001, EVIDENCE)
    assert {'healthcare', 'steps'} <= {c.category for c in draft.claims}
    assert all(c.supporting_quote in next(item['text'] for item in EVIDENCE
                                      if item['evidence_id'] == c.evidence_ids[0])
               for c in draft.claims)


def test_timeout_records_attempt_failure_without_pretending_success():
    calls = []

    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout('synthetic timeout', request=request)

    telemetry = Telemetry()
    adapter = OllamaAdapter(Settings(), telemetry=telemetry,
                            transport=httpx.MockTransport(timeout))
    try:
        with pytest.raises(LLMError, match='ReadTimeout'):
            adapter._ask('classification', 'autó', Classification, run_id='timeout', phase='answer')
    finally:
        adapter.close()
    trace = telemetry.snapshot('timeout')
    assert len(calls) == 1  # no replay of a timed-out request
    assert _usage(trace)['llm_calls'] == 0
    assert _usage(trace)['llm_attempts'] == 1
    assert _usage(trace)['llm_failures'] == 1
    assert trace['llm_failures'][0]['type'] == 'ReadTimeout'
    assert not _usage(trace)['answer_generation_completed']
    assert 'synthetic timeout' not in json.dumps(trace)


def test_validated_model_response_records_real_success():
    telemetry = Telemetry()

    def successful(request):
        return httpx.Response(200, request=request, json={
            'message': {'content': json.dumps({'domains': ['vehicle']})},
            'prompt_eval_count': 5, 'eval_count': 4,
        })

    adapter = OllamaAdapter(Settings(), telemetry=telemetry,
                            transport=httpx.MockTransport(successful))
    try:
        assert adapter.classify('Autó', run_id='ok').domains == ['vehicle']
    finally:
        adapter.close()
    trace = telemetry.snapshot('ok')
    assert _usage(trace)['llm_attempts'] == 1
    assert trace['llm_successes'] == [{'phase': 'classify'}]
    assert _usage(trace)['llm_calls'] == 1
    assert _usage(trace)['llm_failures'] == 0


def test_answerable_source_fallback_does_not_count_as_task_completion():
    case = {
        'question_id': 'WORK_001', 'question': WORK_001, 'answerability': 'answerable',
        'expected_domains': ['employment'], 'expected_intents': ['employment_information'],
        'expected_subtasks': [{'domain': 'employment', 'keywords': []}],
        'expected_tool_calls': [], 'expected_facts': [],
        'expected_behavior': 'answer_with_citations',
    }
    output = {
        'domains': ['employment'], 'intents': ['employment_information'],
        'subtasks': {'t1': {'domain': 'employment', 'question': WORK_001, 'status': 'complete'}},
        'branch_results': {'t1': {'ranked_chunk_ids': ['healthcare-chunk']}},
        'evidence': EVIDENCE, 'answer_draft': source_answer(WORK_001, EVIDENCE).model_dump(),
        'final_answer': 'Részleges útmutató', 'response_status': 'partial',
        'answer_fallback': True, 'tool_results': {},
    }
    metrics, _ = _full_metrics(case, output, {}, {}, [], Telemetry(), 'test')
    assert metrics['workflow_success_rate'] == 1.0  # workflow returned without exception
    assert metrics['task_completion_rate'] == 0.0  # Qwen did not complete answer
    assert metrics['answer_completeness'] is None  # no human-verified golden reference


def test_summary_separates_partial_workflow_and_actual_generation():
    rows = [
        {'success': True, 'response_status': 'partial', 'answer_fallback': True,
         'generation_success': False, 'latency_s': 30, 'llm_usage': {
             'llm_calls': 0, 'llm_attempts': 1, 'llm_failures': 1,
             'prompt_tokens': 0, 'generated_tokens': 0}, 'metrics': {}},
        {'success': False, 'response_status': 'error', 'answer_fallback': False,
         'generation_success': False, 'latency_s': 10, 'llm_usage': {
             'llm_calls': 0, 'llm_attempts': 0, 'llm_failures': 0,
             'prompt_tokens': 0, 'generated_tokens': 0}, 'metrics': {}},
    ]
    summary = _mode_summary(rows, ())
    assert summary['successful_runs'] == 1
    assert summary['complete_answers'] == 0
    assert summary['partial_answers'] == 1
    assert summary['fallback_answers'] == 1
    assert summary['generation_successful'] == 0
    assert summary['llm_attempts_mean'] == 0.5


def test_timeout_cli_override_applies_equally_to_every_profile(monkeypatch, tmp_path):
    import scripts.compare_rag_modes as cli
    observed = []

    def compare(settings, **kwargs):
        observed.append((settings.ollama_read_timeout_s, settings.ollama_total_timeout_s))
        return {'kind': 'test'}

    monkeypatch.setattr(cli, 'run_rag_comparison', compare)
    monkeypatch.setattr(cli, 'save_run', lambda result, output: tmp_path)
    monkeypatch.setattr(sys, 'argv', ['compare_rag_modes.py', '--read-timeout', '90',
                                     '--total-timeout', '120'])
    assert cli.main() == 0
    assert observed == [(90, 120)]
    monkeypatch.setattr(sys, 'argv', ['compare_rag_modes.py', '--read-timeout', '120',
                                     '--total-timeout', '90'])
    with pytest.raises(SystemExit, match='2'):
        cli.main()
