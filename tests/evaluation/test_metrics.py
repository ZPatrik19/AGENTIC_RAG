from dap_assistant.evaluation.dataset import DATASET, load_dataset, reference_status
from dap_assistant.evaluation.metrics import (fact_coverage, latency_stats, percentile,
                                              retrieval_metrics, task_coverage, tool_accuracy)
from dap_assistant.evaluation.benchmark_summary import _optimization_candidates, _interval_union
from dap_assistant.observability.telemetry import Telemetry


def test_golden_distribution():
    data = load_dataset()
    assert len(data) == 20
    assert sum(c['category'] == 'vehicle' for c in data) == 10
    assert sum(c['category'] == 'employment' for c in data) == 10
    assert all(c['reference_status'] != 'pinned' for c in data)
    # The active dataset can be the version-pinned SILVER file. Missing active
    # index versions must not be misreported as an unversioned reference.
    assert reference_status(data[0], {}) in {'version_mismatch', 'unversioned_reference'}
    assert reference_status(load_dataset(DATASET)[0], {}) == 'unversioned_reference'


def test_percentile_linear_interpolation_and_throughput():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, .50) == 3
    assert percentile(values, .95) == 4.8
    assert percentile(values, .99) == 4.96
    stats = latency_stats(values, elapsed_s=10)
    assert stats['throughput_qps'] == .5
    assert stats['p95_s'] == 4.8
    assert latency_stats([], 0)['p50_s'] is None


def test_retrieval_missing_reference_is_not_zero():
    assert retrieval_metrics(['a'], None)['recall_at_k'] is None
    metrics = retrieval_metrics(['x', 'a', 'b'], ['a', 'b'], k=2)
    assert metrics['recall_at_k'] == .5
    assert metrics['mrr'] == .5
    assert retrieval_metrics(['x'], ['a'])['mrr'] == 0


def test_task_tool_and_fact_coverage():
    assert task_coverage({'t1': {'domain': 'vehicle', 'question': 'autó átírás'}},
                         [{'domain': 'vehicle', 'keywords': ['átírás']}]) == 1
    assert tool_accuracy({'a': {'tool': 'calculate_deadline'}},
                         ['calculate_deadline', 'build_document_checklist']) == 2 / 3
    assert fact_coverage('15 napon belül', [{'answer_pattern': r'15\s+napon'}]) == 1
    assert fact_coverage('anything', [{'answer_pattern': None}]) is None


def test_interval_union_is_wall_coverage_not_sum():
    assert _interval_union([(1, 3), (1, 3), (4, 5)]) == 3
    assert _interval_union([]) == 0


def test_telemetry_distinct_parallel_spans_are_not_wall_time():
    telemetry = Telemetry()
    telemetry.record('one', 'rag/hybrid_retrieval', 2, start=10, end=12)
    telemetry.record('one', 'rag/hybrid_retrieval', 2, start=10, end=12)
    assert sum(s['duration_s'] for s in telemetry.snapshot('one')['spans']) == 4
    assert telemetry.snapshot('other')['spans'] == []
    assert len(_optimization_candidates({'llm_inference': {'total_work_s': 3},
                                         'bm25_retrieval': {'total_work_s': 1}})) == 2
