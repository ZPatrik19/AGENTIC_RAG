"""Regression tests for v7 pure benchmark and comparison metric modules."""
from __future__ import annotations

import ast
from pathlib import Path

from dap_assistant.evaluation import benchmark_summary, comparison, comparison_metrics


def _span(name: str, start: float, end: float) -> dict:
    return {'name': name, 'start_s': start, 'end_s': end, 'duration_s': end - start}


def _row(*, domain: str, latency: float, coverage: float | None, usage: dict | None) -> dict:
    return {
        'success': True, 'latency_s': latency, 'domain': domain, 'error': None,
        'evidence_count': 2, 'response_status': 'complete',
        'request_coverage': {'coverage': coverage} if coverage is not None else None,
        'trace': {
            'spans': [_span('main/plan_tasks', 0.0, latency / 2),
                      _span('main/evidence_gate', latency / 4, latency * .75)],
            'llm_usage': [usage] if usage is not None else [],
        },
    }


def _summary(rows: list[dict]) -> dict:
    return benchmark_summary.summarize_benchmark_rows(
        rows, mode='e2e', count=len(rows), elapsed=4.0,
        concurrency=2, cpu_before=1.0, cpu_after=1.5,
        rss_before=100, rss_after=140, warmup=1,
        warmup_rows=[{'success': True, 'latency_s': 100.0}],
    )


def test_comparison_metric_imports_preserve_public_compatibility():
    for name in ('_simple_metrics', '_model_generated', '_usage', '_mode_summary', '_paired_deltas'):
        assert getattr(comparison, name) is getattr(comparison_metrics, name)


def test_benchmark_summary_is_measured_only_and_excludes_warmup():
    rows = [_row(domain='vehicle', latency=2.0, coverage=.75,
                 usage={'eval_count': 20, 'prompt_eval_count': 100, 'eval_duration': 2_000_000_000}),
            _row(domain='employment', latency=3.0, coverage=1.0, usage=None)]
    summary = _summary(rows)
    assert summary['warmup']['count'] == 1
    assert summary['success_count'] == 2
    assert summary['mean_s'] == 2.5
    assert summary['request_facet_coverage_proxy'] == .875
    assert summary['ollama_usage']['generated_tokens'] == 20
    assert summary['ollama_usage']['generation_tokens_per_second'] == 10
    assert rows[0]['unattributed_wall_s'] == .5  # overlapping intervals form one union
    assert summary['application_rss_delta_bytes'] == 40


def test_missing_coverage_is_unavailable_not_exception_or_zero():
    rows = [_row(domain='vehicle', latency=2.0, coverage=None, usage=None)]
    summary = _summary(rows)
    assert summary['request_facet_coverage_proxy'] is None
    assert summary['ollama_usage']['generation_tokens_per_second'] is None


def test_comparison_missing_provider_usage_stays_unobserved():
    usage = comparison_metrics._usage({'llm_usage': [], 'llm_successes': [], 'llm_attempts': []})
    assert usage['generated_tokens_observed'] is None
    assert usage['token_usage_complete'] is False
    assert usage['answer_generation_completed'] is False


def test_pure_reporting_does_not_load_graph_or_ollama_dependencies():
    package = Path(benchmark_summary.__file__).parent
    for name in ('benchmark_summary.py', 'comparison_metrics.py'):
        module = ast.parse((package / name).read_text(encoding='utf-8'))
        imports = [n.module for n in module.body if isinstance(n, ast.ImportFrom)]
        assert not any(item and ('workflow' in item or 'inference' in item or 'rag_graph' in item)
                       for item in imports)
        # Immutable module-level metric-name constants are fine; executing calls is not.
        assert not any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                       for n in module.body)
