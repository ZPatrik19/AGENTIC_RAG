"""Presentation/export policy for the Evaluation and Performance Streamlit page.

Keep persisted historical reports intact: filtering happens on copied values only.
Do not change shared benchmark, chatbot, or CLI evaluation behavior.
"""
from __future__ import annotations

import csv
from io import StringIO
from typing import Any


HIDDEN_METRICS = frozenset({
    'answer_completeness',
    'faithfulness',
    'unsupported_claim_rate',
    'answer_relevancy',
    'answer_correctness',
    'task_completion_rate',
    'recovery_success_rate',
})


def visible_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Hide withdrawn metrics without mutating their original dictionary."""
    return {key: value for key, value in metrics.items() if key not in HIDDEN_METRICS}


def visible_functional_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove withdrawn fields from a functional report for display and export.

    Non-functional reports, including load/performance reports, are unchanged.
    """
    if result.get('kind') != 'functional_v4':
        return result

    filtered = dict(result)
    summary = dict(result.get('summary') or {})
    summary['metrics'] = visible_metrics(summary.get('metrics') or {})
    filtered['summary'] = summary

    configuration = dict(result.get('configuration') or {})
    configuration.pop('local_judge', None)
    filtered['configuration'] = configuration

    rows = []
    for item in result.get('rows') or []:
        row = dict(item)
        row['metrics'] = visible_metrics(item.get('metrics') or {})
        row.pop('unsupported_claims', None)
        details = dict(item.get('details') or {})
        # Old reports may include the removed judge's full ratings here.
        details.pop('local_judge', None)
        details.pop('semantic_support', None)
        if isinstance(details.get('claim_support'), dict):
            details['claim_support'] = visible_metrics(details['claim_support'])
        row['details'] = details
        rows.append(row)
    filtered['rows'] = rows
    return filtered


def visible_functional_csv(raw: bytes) -> bytes:
    """Filter a previously saved rows.csv without rewriting the saved report."""
    source = StringIO(raw.decode('utf-8-sig'), newline='')
    reader = csv.DictReader(source)
    if reader.fieldnames is None:
        return raw
    excluded = {'metric_' + name for name in HIDDEN_METRICS} | {'unsupported_claims'}
    fieldnames = [name for name in reader.fieldnames if name not in excluded]
    target = StringIO(newline='')
    writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(reader)
    return target.getvalue().encode('utf-8-sig')


# These keys require observable model usage. The load benchmark never substitutes
# zero for unavailable token data. Historic report files remain untouched.
OPTIONAL_LOAD_SUMMARY_METRICS = frozenset({
    'ttft_mean_s', 'ttft_p50_s', 'ttft_p95_s',
    'generation_tokens_per_second_mean', 'context_window_utilization_mean',
    'context_truncation_count',
})


def visible_load_result(result: dict[str, Any]) -> dict[str, Any]:
    """Expose only observed load metrics; never mutate persisted report objects."""
    if result.get('kind') != 'load_v4':
        return result
    filtered = dict(result)
    summary = dict(result.get('summary') or {})
    for key in OPTIONAL_LOAD_SUMMARY_METRICS:
        if summary.get(key) is None:
            summary.pop(key, None)
    # A historic report may have recorded a misleading zero for truncation
    # despite never observing prompt-window usage.
    if summary.get('context_window_utilization_mean') is None:
        summary.pop('context_truncation_count', None)
    filtered['summary'] = summary

    rows = []
    for original in result.get('rows') or []:
        row = dict(original)
        llm = row.get('llm_performance')
        if isinstance(llm, dict):
            trimmed = dict(llm)
            for key in ('ttft_s', 'generation_tokens_per_second'):
                if trimmed.get(key) is None:
                    trimmed.pop(key, None)
            window = trimmed.get('context_window_utilization')
            if isinstance(window, dict) and window.get('ratio') is None:
                trimmed.pop('context_window_utilization', None)
                trimmed.pop('per_call_context_windows', None)
            # Keep usage counters only if observed, not configured constants.
            if not any(trimmed.get(key) is not None for key in (
                    'prompt_tokens', 'generated_tokens', 'ttft_s',
                    'generation_tokens_per_second', 'context_window_utilization')):
                row.pop('llm_performance', None)
            else:
                row['llm_performance'] = trimmed
        rows.append(row)
    filtered['rows'] = rows
    return filtered


def visible_load_csv(raw: bytes) -> bytes:
    """Remove only columns which have no observed values (legacy exports too)."""
    source = StringIO(raw.decode('utf-8-sig'), newline='')
    reader = csv.DictReader(source)
    if reader.fieldnames is None:
        return raw
    rows = list(reader)
    optional = OPTIONAL_LOAD_SUMMARY_METRICS | frozenset({
        'ttft_s', 'generation_tokens_per_second', 'context_window_utilization',
    })
    fieldnames = [name for name in reader.fieldnames
                  if name not in optional or any(row.get(name, '').strip() not in ('', 'None', 'N/A')
                                                     for row in rows)]
    if fieldnames == reader.fieldnames:
        return raw
    target = StringIO(newline='')
    writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode('utf-8-sig')
