import csv
import json

from dap_assistant.evaluation.reporting import save_run


def test_export_creates_three_formats(tmp_path):
    report = {
        'kind': 'load_v4', 'timestamp_utc': '2026-09-20T00:00:00+00:00',
        'configuration': {'scope': 'subflow', 'target': 'rag/full_subgraph',
                          'model': 'qwen3:4b', 'context_window': 2048},
        'execution_environment': {'index_sha256': 'abc'},
        'summary': {'throughput_qps': .5},
        'rows': [{'question_id': 'V01', 'latency_s': 2, 'success': True,
                  'node_execution_trace': {'spans': []}}],
    }
    folder = save_run(report, tmp_path)
    assert json.loads((folder / 'result.json').read_text())['summary']['throughput_qps'] == .5
    with (folder / 'rows.csv').open(encoding='utf-8-sig') as handle:
        assert next(csv.DictReader(handle))['question_id'] == 'V01'
    assert 'qwen3:4b' in (folder / 'report.md').read_text()
