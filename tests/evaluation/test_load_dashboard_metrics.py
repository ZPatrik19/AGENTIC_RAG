"""Offline load-dashboard regression tests (no Ollama, Qdrant or network)."""
from __future__ import annotations

import ast
import csv
import json
from datetime import datetime
from io import StringIO
from pathlib import Path

from dap_assistant.evaluation.professional import (
    render_report_markdown, run_report_stem, save_run, summarize_load_llm,
)
from dap_assistant.evaluation.presentation.ui_metric_visibility import (
    visible_load_result, visible_load_csv,
)


PAGE = Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant' / 'pages' / '1_Értékelés_és_teljesítmény.py'


def report():
    return {
        'kind': 'load_v4', 'run_id': 'case-1',
        'timestamp_utc': '2026-09-22T08:35:27+00:00',
        'configuration': {
            'scope': 'single_node', 'target': 'rag/hybrid_retrieval',
            'request_count': 50, 'concurrency': 2, 'seed': 42,
        },
        'summary': {
            'mean_s': 0.6, 'error_rate': 0.0, 'ttft_mean_s': None,
            'ttft_p95_s': None, 'generation_tokens_per_second_mean': None,
            'context_window_utilization_mean': None, 'context_truncation_count': 0,
            'resource_summary': {},
        },
        'rows': [
            {'request_index': 1, 'latency_s': 0.6,
             'llm_performance': {'ttft_s': None, 'generation_tokens_per_second': None,
                                 'context_window_utilization': {'ratio': None},
                                 'context_window': 2048, 'prompt_tokens': None,
                                 'per_call_context_windows': []}},
        ],
    }


def test_absent_llm_usage_has_no_aggregate_fields():
    assert summarize_load_llm([{'latency_s': 0.6}]) == {}
    assert summarize_load_llm([{'llm_performance': {'ttft_s': None,
        'generation_tokens_per_second': None,
        'context_window_utilization': {'ratio': None}}}]) == {}


def test_llm_summary_only_includes_actual_readings():
    rows = [
        {'llm_performance': {'ttft_s': None, 'generation_tokens_per_second': 31.5,
                             'context_window_utilization': {'ratio': None}}},
        {'llm_performance': {'ttft_s': None, 'generation_tokens_per_second': 20.5,
                             'context_window_utilization': {'ratio': None}}},
    ]
    assert summarize_load_llm(rows) == {'generation_tokens_per_second_mean': 26.0}
    rows = [{'llm_performance': {
        'ttft_s': 0.3, 'generation_tokens_per_second': 25.0,
        'context_window_utilization': {'ratio': 0.5, 'over_budget': False},
    }}]
    values = summarize_load_llm(rows)
    assert values['ttft_mean_s'] == 0.3
    assert values['ttft_p50_s'] == 0.3
    assert values['context_truncation_count'] == 0
    assert values['context_window_utilization_mean'] == 0.5


def test_legacy_missing_values_removed_without_mutating_original():
    old = report()
    cleaned = visible_load_result(old)
    assert cleaned['summary'] == {'mean_s': 0.6, 'error_rate': 0.0, 'resource_summary': {}}
    assert 'llm_performance' not in cleaned['rows'][0]
    assert old['summary']['ttft_mean_s'] is None
    assert 'llm_performance' in old['rows'][0]
    assert visible_load_result(cleaned) == cleaned


def test_legacy_mixed_observations_keep_measured_metrics_only():
    old = report()
    old['rows'][0]['llm_performance']['generation_tokens_per_second'] = 14.0
    old['summary']['generation_tokens_per_second_mean'] = 14.0
    cleaned = visible_load_result(old)
    assert cleaned['rows'][0]['llm_performance']['generation_tokens_per_second'] == 14.0
    assert cleaned['summary']['generation_tokens_per_second_mean'] == 14.0
    assert 'ttft_mean_s' not in cleaned['summary']
    assert 'context_truncation_count' not in cleaned['summary']


def test_legacy_csv_optional_empty_columns_removed():
    out = StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['request_index', 'latency_s', 'ttft_s',
                                             'generation_tokens_per_second'])
    writer.writeheader()
    writer.writerow({'request_index': 1, 'latency_s': 0.6, 'ttft_s': '',
                     'generation_tokens_per_second': 32})
    original = out.getvalue().encode('utf-8-sig')
    clean = visible_load_csv(original)
    assert clean.startswith(b'\xef\xbb\xbf')
    parsed = csv.DictReader(StringIO(clean.decode('utf-8-sig'), newline=''))
    assert parsed.fieldnames == ['request_index', 'latency_s', 'generation_tokens_per_second']
    assert next(parsed)['generation_tokens_per_second'] == '32'


def test_load_filenames_include_test_type_scope_and_target():
    result = report()
    at = datetime(2026, 9, 22, 10, 35, 27)
    stem = run_report_stem(result, at=at)
    assert stem == '2026-09-22_10-35-27_terheleses_teszt_single_node_rag_hybrid_retrieval'
    result['configuration'].update(scope='full_workflow', target='agentic/full')
    assert run_report_stem(result, at=at) == '2026-09-22_10-35-27_terheleses_teszt_full_workflow'
    assert not set(stem).intersection('\\/:*?"<>|')


def test_new_saved_load_report_has_no_unmeasured_token_metrics(tmp_path):
    result = visible_load_result(report())
    folder = save_run(result, tmp_path)
    saved = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    assert 'ttft_mean_s' not in saved['summary']
    assert 'context_truncation_count' not in saved['summary']
    assert 'llm_performance' not in saved['rows'][0]
    assert 'ttft_mean_s' not in (folder / 'report.md').read_text(encoding='utf-8')
    assert not folder.name.startswith('20260922T')
    assert 'ttft_mean_s' not in render_report_markdown(result)


def test_streamlit_load_inputs_have_explanations_and_ui_skips_unobserved_metrics():
    source = PAGE.read_text(encoding='utf-8')
    ast.parse(source)
    for label in ('Lekérdezések száma', 'Párhuzamosság', 'Timeout / kérés (s)',
                  'Véletlen seed', 'Warm-up kérések'):
        assert label in source
    assert source.count('help=') >= 12
    assert 'if s.get(key) is not None' in source
    assert "(row.get('llm_performance') or {})" in source
    assert 'visible_load_result(result)' in source
    assert 'visible_load_csv(csv_content)' in source
    assert 'render_report_markdown(result)' in source
    assert 'st.columns(len(group)' in source


def test_card_rows_have_no_gaps_and_balance_four_item_tail():
    source = PAGE.read_text(encoding='utf-8')
    parsed = ast.parse(source)
    fn = next(node for node in parsed.body
              if isinstance(node, ast.FunctionDef) and node.name == '_balanced_groups')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(PAGE), 'exec'), namespace)
    group = namespace['_balanced_groups']
    assert [len(row) for row in group(list(range(7)), 3)] == [3, 2, 2]
    assert [len(row) for row in group(list(range(4)), 2)] == [2, 2]
    assert group([], 3) == []
    assert sum(group(list(range(8)), 3), []) == list(range(8))
