
import pytest

from dap_assistant.evaluation.dataset import DATASET, load_dataset
from dap_assistant.evaluation.metrics import (
    abstention_accuracy, citation_coverage, claim_support_breakdown,
    context_metrics, context_window_utilization, generation_speed,
    retrieval_metrics, task_completion_score, tool_call_efficiency, ttft,
)
from dap_assistant.evaluation.professional import (
    available_targets, load_saved_run, save_run, select_cases,
)


def test_exact_twenty_question_dataset_and_required_ids():
    cases = load_dataset(DATASET)
    assert len(cases) == 20
    assert [c['question_id'] for c in cases[:10]] == [f'AUTO_{i:03d}' for i in range(1, 11)]
    assert [c['question_id'] for c in cases[10:]] == [f'WORK_{i:03d}' for i in range(1, 11)]
    assert cases[9]['difficulty'] == 'complex'
    assert cases[19]['difficulty'] == 'complex'


def test_topic_and_custom_case_selection():
    assert len(select_cases(topic='vehicle')) == 10
    assert len(select_cases(topic='employment')) == 10
    selected = select_cases(topic='all', question_ids=['AUTO_001', 'WORK_010'])
    assert [c['question_id'] for c in selected] == ['AUTO_001', 'WORK_010']
    with pytest.raises(ValueError):
        select_cases(topic='vehicle', question_ids=['WORK_001'])


def test_retrieval_precision_recall_mrr_and_empty_denominator_rule():
    result = retrieval_metrics(['x', 'a', 'b'], ['a', 'b'], k=5)
    assert result['recall_at_k'] == 1.0
    assert result['precision_at_k'] == pytest.approx(2 / 3)
    assert result['mrr'] == 0.5
    empty = retrieval_metrics([], ['a'], k=5)
    assert empty['precision_at_k'] == 0.0
    assert empty['recall_at_k'] == 0.0


def test_context_metrics_are_distinct_and_rank_sensitive():
    result = context_metrics(
        ['noise', 'a', 'b'], ['a', 'b'],
        expected_subtasks=[{'domain': 'vehicle'}, {'domain': 'vehicle'}],
        relevant_chunk_ids_by_task={'t1': ['a'], 't2': ['missing']},
    )
    assert result['context_recall'] == 1.0
    assert 0 < result['context_precision'] < 1.0  # noise at rank 1 is penalized
    assert result['context_coverage'] == 0.5
    assert context_metrics(['a'], None)['context_recall'] is None


def test_claim_support_and_citation_coverage_do_not_invent_contradictions():
    output = {
        'evidence': [{'evidence_id': 'E1', 'text': 'Az átírás 15 napon belül szükséges.'}],
        'answer_draft': {'claims': [
            {'text': '15 nap.', 'evidence_ids': ['E1'], 'supporting_quote': 'Az átírás 15 napon belül szükséges.'},
            {'text': 'Másik állítás.', 'evidence_ids': [], 'supporting_quote': ''},
        ]},
        'answer_validation': {'unsupported_claims': ['Másik állítás.']},
    }
    breakdown = claim_support_breakdown(output)
    assert breakdown['counts']['supported'] == 1
    assert breakdown['counts']['unsupported'] == 1
    assert breakdown['counts']['contradicted'] == 0
    assert breakdown['citation_integrity_proxy'] == 0.5
    assert breakdown['faithfulness'] is None  # Citation validity alone is not entailment.
    assert citation_coverage(output) == 0.5


def test_agentic_metrics_and_na_semantics():
    efficiency = tool_call_efficiency(
        {'1': {'tool': 'build_document_checklist', 'status': 'success'}},
        ['build_document_checklist'],
    )
    assert efficiency['score'] == 1.0
    assert abstention_accuracy('partial', 'partial_or_abstain_on_missing_evidence', 'insufficient') == 1.0
    assert abstention_accuracy('complete', 'partial_or_abstain_on_missing_evidence', 'insufficient') == 0.0
    assert task_completion_score(subtask_coverage=None, answer_completeness_score=None,
                                 workflow_success=1.0, answerability='answerable') is None


def test_performance_metric_formulas():
    assert generation_speed(100, 2.0) == 50.0
    assert generation_speed(None, 2.0) is None
    assert ttft(0.2) == 0.2
    util = context_window_utilization(4096, 8192, 512)
    assert util['ratio'] == 0.5
    assert util['over_budget'] is False
    assert context_window_utilization(None, 8192)['ratio'] is None


def test_real_graph_targets_are_valid_and_no_ab_dashboard_target():
    targets = available_targets()
    assert set(targets) == {'single_node', 'subflow', 'full_workflow'}
    assert targets['single_node']['rag/process_query'] == ('process_query',)
    assert targets['subflow']['rag/full_subgraph'][-1] == 'prepare_context'
    assert all('ab' not in key.casefold() for group in targets.values() for key in group)


def test_result_save_and_reload(tmp_path):
    result = {
        'kind': 'functional_v4', 'run_id': 'abc123', 'timestamp_utc': '2026-09-20T00:00:00Z',
        'configuration': {'model': 'qwen3:4b', 'context_window': 8192},
        'summary': {'metrics': {}},
        'rows': [{'question_id': 'AUTO_001', 'success': True, 'latency_s': 0.1, 'metrics': {}}],
    }
    folder = save_run(result, tmp_path)
    loaded = load_saved_run(folder)
    assert loaded['run_id'] == 'abc123'
    assert (folder / 'rows.csv').is_file()
    assert (folder / 'report.md').is_file()
