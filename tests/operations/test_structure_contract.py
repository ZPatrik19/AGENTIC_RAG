"""Structural checks should not require Streamlit, Ollama, LangGraph or Qdrant."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_project import audit
from dap_assistant.rag import graph_contract
from dap_assistant.evaluation import professional, reporting
from dap_assistant.evaluation.presentation import ui_metric_layout


ROOT = Path(__file__).resolve().parents[2]


def test_node_order_is_shared_across_graph_evaluation_and_presentation():
    assert graph_contract.RAG_NODE_ORDER == professional.RAG_NODES
    assert graph_contract.RAG_SUBFLOWS == professional.SUBFLOW_TARGETS
    assert graph_contract.RAG_SUBFLOWS == ui_metric_layout.RAG_SUBFLOWS
    assert graph_contract.STANDALONE_RAG_TARGETS == professional.STANDALONE_TARGETS


def test_report_api_remains_available_from_legacy_import_path():
    assert professional.save_run is reporting.save_run
    assert professional.run_report_stem is reporting.run_report_stem
    assert professional.load_saved_run is reporting.load_saved_run


def test_context_and_presentation_modules_are_not_duplicated_at_root():
    assert (ROOT / 'src/dap_assistant/context_engineering/evidence_selection.py').is_file()
    assert (ROOT / 'src/dap_assistant/prompt_engineering/answer_prompt.py').is_file()
    assert (ROOT / 'src/dap_assistant/response/generation.py').is_file()
    assert not (ROOT / 'src/dap_assistant/evidence_selection.py').exists()
    assert (ROOT / 'src/dap_assistant/evaluation/presentation/ui_metric_layout.py').is_file()
    assert not (ROOT / 'src/dap_assistant/evaluation/ui_metric_layout.py').exists()


def test_audit_detects_duplicates_and_never_modifies_files(tmp_path):
    folder = tmp_path / 'tests'
    folder.mkdir()
    source = 'def test_one():\n    x = 1\n    y = 2\n    assert x < y\n'
    a = folder / 'test_first.py'
    b = folder / 'test_second.py'
    a.write_text(source, encoding='utf-8')
    b.write_text(source.replace('test_one', 'test_two'), encoding='utf-8')
    before = (a.read_bytes(), b.read_bytes())
    result = audit(tmp_path, large_module_lines=3)
    assert result['python_files'] == 2
    assert result['test_functions'] == 2
    assert len(result['exact_function_candidates']) == 1
    assert (a.read_bytes(), b.read_bytes()) == before


def test_audit_rejects_non_positive_threshold(tmp_path):
    with pytest.raises(ValueError, match='positive'):
        audit(tmp_path, large_module_lines=0)
