"""v6 metric boundary: orchestration imports pure metric assembly, not vice versa."""
from __future__ import annotations

import ast
from pathlib import Path

from dap_assistant.evaluation import functional_metrics, professional


def test_professional_retains_legacy_metric_import_contract():
    assert professional._full_metrics is functional_metrics._full_metrics
    assert professional._rag_metrics is functional_metrics._rag_metrics
    assert professional._context_chunk_ids is functional_metrics._context_chunk_ids
    assert professional._FUNCTIONAL_METRIC_NAMES is functional_metrics._FUNCTIONAL_METRIC_NAMES


def test_functional_metrics_does_not_run_workflows_or_call_models():
    tree = ast.parse(Path(functional_metrics.__file__).read_text(encoding='utf-8'))
    imports = {node.module for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module}
    assert not imports.intersection({'professional', 'runner', 'load_testing'})
    assert not any(name.startswith(('langgraph', 'httpx', 'streamlit')) for name in imports)
    code = Path(functional_metrics.__file__).read_text(encoding='utf-8')
    assert 'app.invoke(' not in code
    assert 'ensure_auto_reference(' not in code
