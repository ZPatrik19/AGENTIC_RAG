"""No-Ollama regression coverage for withdrawn evaluation-dashboard metrics."""
from __future__ import annotations

import ast
import csv
from io import StringIO
from pathlib import Path

from dap_assistant.evaluation.presentation.ui_metric_visibility import (
    HIDDEN_METRICS,
    visible_functional_csv,
    visible_functional_result,
    visible_metrics,
)


DASHBOARD = (
    Path(__file__).resolve().parents[2]
    / 'src' / 'dap_assistant' / 'pages' / '1_Értékelés_és_teljesítmény.py'
)


def sample_report():
    metrics = {name: 0.75 for name in HIDDEN_METRICS}
    metrics.update({'retrieval_recall_at_5': 0.5, 'citation_accuracy': 1.0})
    return {
        'kind': 'functional_v4',
        'configuration': {'local_judge': True, 'scope': 'full_workflow'},
        'summary': {'metrics': dict(metrics), 'evaluated_cases': 1},
        'rows': [{
            'question_id': 'AUTO_001',
            'metrics': dict(metrics),
            'unsupported_claims': 4,
            'details': {
                'local_judge': {'faithfulness': 0.75},
                'semantic_support': {'unsupported_claim_rate': 0.25},
                'retrieval': {'recall_at_5': 0.5},
                'claim_support': {'faithfulness': None, 'unsupported_claim_rate': None,
                                  'citation_integrity_proxy': 1.0},
            },
        }],
    }


def test_visible_metrics_hide_only_requested_five():
    original = {'answer_completeness': 1, 'faithfulness': 0.5, 'unsupported_claim_rate': 0.3,
                'answer_relevancy': 0.9, 'answer_correctness': 0.6,
                'retrieval_mrr': 0.5, 'context_recall': 0.7}
    assert visible_metrics(original) == {'retrieval_mrr': 0.5, 'context_recall': 0.7}
    assert 'answer_completeness' in original


def test_historical_report_filtered_without_touching_original():
    original = sample_report()
    filtered = visible_functional_result(original)
    for name in HIDDEN_METRICS:
        assert name not in filtered['rows'][0]['metrics']
        assert name not in filtered['summary']['metrics']
    assert filtered['rows'][0]['metrics']['retrieval_recall_at_5'] == 0.5
    assert filtered['rows'][0]['metrics']['citation_accuracy'] == 1.0
    assert 'unsupported_claims' not in filtered['rows'][0]
    assert 'local_judge' not in filtered['rows'][0]['details']
    assert 'semantic_support' not in filtered['rows'][0]['details']
    assert filtered['rows'][0]['details']['retrieval'] == {'recall_at_5': 0.5}
    assert filtered['rows'][0]['details']['claim_support'] == {'citation_integrity_proxy': 1.0}
    assert 'local_judge' not in filtered['configuration']
    assert 'local_judge' in original['configuration']
    assert 'unsupported_claims' in original['rows'][0]
    assert original['rows'][0]['details']['semantic_support']['unsupported_claim_rate'] == 0.25


def test_load_report_unmodified():
    result = {'kind': 'load_v4', 'summary': {'metrics': {'faithfulness': 1}}}
    assert visible_functional_result(result) is result


def test_csv_export_legacy_columns_removed_and_bom_preserved():
    fields = ['question_id', 'metric_answer_correctness', 'metric_answer_completeness',
              'metric_faithfulness', 'metric_answer_relevancy',
              'metric_unsupported_claim_rate', 'unsupported_claims',
              'metric_retrieval_recall_at_5', 'latency_s']
    output = StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerow(dict.fromkeys(fields, '0.5'))
    raw = output.getvalue().encode('utf-8-sig')
    exported = visible_functional_csv(raw)
    assert exported.startswith(b'\xef\xbb\xbf')
    reader = csv.DictReader(StringIO(exported.decode('utf-8-sig'), newline=''))
    assert reader.fieldnames == ['question_id', 'metric_retrieval_recall_at_5', 'latency_s']
    assert next(reader)['metric_retrieval_recall_at_5'] == '0.5'
    assert raw != exported


def test_empty_csv_remains_empty():
    assert visible_functional_csv(b'') == b''


def test_ui_does_not_offer_judge_or_display_hidden_metrics():
    source = DASHBOARD.read_text(encoding='utf-8')
    tree = ast.parse(source)
    assert 'Helyi Qwen szemantikai bíráló' not in source
    assert '**Unsupported Claims**' not in source
    assert not any(name in source for name in ('answer_completeness', 'faithfulness',
                                               'answer_correctness', 'answer_relevancy',
                                               'unsupported_claim_rate'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    run_calls = [call for call in calls if isinstance(call.func, ast.Name)
                 and call.func.id == 'run_functional']
    assert len(run_calls) == 1
    assert any(keyword.arg == 'local_judge' and isinstance(keyword.value, ast.Constant)
               and keyword.value.value is False for keyword in run_calls[0].keywords)
    assert 'visible_functional_result(result)' in source
    assert 'visible_functional_csv(csv_content)' in source


def test_two_additional_dashboard_metrics_hidden():
    original = {'task_completion_rate': 0.0, 'recovery_success_rate': None,
                'workflow_success_rate': 1.0, 'retrieval_mrr': 0.5}
    assert visible_metrics(original) == {'workflow_success_rate': 1.0, 'retrieval_mrr': 0.5}
    assert {'task_completion_rate', 'recovery_success_rate'} <= HIDDEN_METRICS
