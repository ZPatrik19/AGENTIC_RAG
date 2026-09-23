"""No-network regression tests for the simplified dashboard and saved-run names."""
from __future__ import annotations

import ast
import json
import re

import pytest
from datetime import datetime
from pathlib import Path

from dap_assistant.evaluation.professional import (
    list_saved_runs,
    load_saved_run,
    run_report_stem,
    save_run,
)


DASHBOARD = (Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant'
             / 'pages' / '1_Értékelés_és_teljesítmény.py')


def minimal_report(scope: str, target: str) -> dict:
    return {
        'kind': 'functional_v4',
        'run_id': 'example-123',
        'timestamp_utc': '2026-09-22T08:35:27+00:00',
        'configuration': {'scope': scope, 'target': target, 'model': 'dummy'},
        'summary': {'metrics': {}, 'evaluated_cases': 0},
        'rows': [],
    }


@pytest.mark.parametrize(
    ('scope', 'target', 'expected'),
    [
        ('full_workflow', 'agentic/full',
         '2026-09-22_10-35-27_funkcionalis_meres_full_workflow'),
        ('single_node', 'rag/hybrid_retrieval',
         '2026-09-22_10-35-27_funkcionalis_meres_single_node_rag_hybrid_retrieval'),
    ],
)
def test_functional_report_name_contains_scope_and_optional_node(scope, target, expected):
    result = minimal_report(scope, target)
    assert run_report_stem(result, at=datetime(2026, 9, 22, 10, 35, 27)) == expected


def test_subflow_name_is_windows_safe():
    result = minimal_report('subflow', 'rag/query_to_rerank')
    actual = run_report_stem(result, at=datetime(2026, 9, 22, 10, 35, 27))
    assert actual == '2026-09-22_10-35-27_funkcionalis_meres_subflow_rag_query_to_rerank'
    assert not set(actual).intersection('\\/:*?"<>|')


def test_saved_reports_use_dated_folder_and_preserve_existing_filenames(tmp_path):
    report = minimal_report('single_node', 'rag/prepare_context')
    first = save_run(report, tmp_path)
    second = save_run(report, tmp_path)
    assert first != second
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_funkcionalis_meres_single_node_rag_prepare_context',
                        first.name)
    assert second.name == first.name + '_02' or re.fullmatch(
        r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_funkcionalis_meres_single_node_rag_prepare_context', second.name)
    assert (first / 'result.json').exists()
    assert (first / 'rows.csv').exists()
    assert (first / 'report.md').exists()
    assert load_saved_run(first)['run_id'] == report['run_id']
    assert first in list_saved_runs(tmp_path)
    assert json.loads((first / 'result.json').read_text(encoding='utf-8'))['configuration'] == report['configuration']


def test_legacy_saved_run_can_be_loaded(tmp_path):
    old = tmp_path / '20260922T083527Z_12345678'
    old.mkdir()
    report = minimal_report('full_workflow', 'agentic/full')
    (old / 'result.json').write_text(json.dumps(report), encoding='utf-8')
    assert old in list_saved_runs(tmp_path)
    assert load_saved_run(old) == report
    assert re.match(r'^\d{4}-\d{2}-\d{2}_', run_report_stem(report))


def test_dashboard_does_not_expose_golden_panels_or_removed_warning():
    source = DASHBOARD.read_text(encoding='utf-8')
    ast.parse(source)
    for title in (
        'Golden készenlét és hiányzó források',
        'Automatikus referencia – 20 kérdés, bizonyítékok és hiányok',
        'Referenciaállapot kérdésenként',
        'Az N/A nem 0%. A technikailag érvényes hivatkozás',
    ):
        assert title not in source
    assert 'ensure_auto_reference(settings.data_dir, chunks=indexed)' in source
    assert 'reference_readiness(' not in source
    assert 'run_reference_diagnostics(' not in source
    assert 'file_name=f\'{export_stem}.json\'' in source
    assert 'file_name=f\'{export_stem}.csv\'' in source
    assert 'file_name=f\'{export_stem}.md\'' in source
    assert 'local_judge=False' in source
