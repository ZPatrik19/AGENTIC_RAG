"""v8: configuration snapshots, UI/application separation and import boundaries."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dap_assistant.settings import Settings
from dap_assistant.orchestration import runtime
from dap_assistant.evaluation import professional, runtime_environment
from scripts.audit_project import audit, inspect_import_boundaries


ROOT = Path(__file__).resolve().parents[2]


def test_env_values_are_read_per_settings_instance_not_at_module_import(monkeypatch):
    initial = Settings()
    monkeypatch.setenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11555')
    monkeypatch.setenv('OLLAMA_NUM_CTX', '3072')
    monkeypatch.setenv('OLLAMA_ANSWER_NUM_PREDICT', '1200')
    monkeypatch.setenv('OLLAMA_READ_TIMEOUT_S', '95')
    monkeypatch.setenv('MAX_SUBTASKS', '6')
    updated = Settings()
    assert updated.ollama_base_url == 'http://127.0.0.1:11555'
    assert (updated.ollama_num_ctx, updated.ollama_answer_num_predict) == (3072, 1200)
    assert updated.ollama_read_timeout_s == 95
    assert updated.max_subtasks == 6
    # An existing instance is an immutable snapshot; replacing an explicit
    # field does not silently change the other settings to the new .env.
    assert replace(initial, answer_mode='detailed').ollama_num_ctx == initial.ollama_num_ctx


def test_settings_resolves_data_path_and_optional_sampler_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('OLLAMA_TOP_P', '')
    monkeypatch.setenv('OLLAMA_TOP_K', '')
    monkeypatch.setenv('OLLAMA_THINK', 'yes')
    selected = Settings()
    assert selected.data_dir == tmp_path
    assert selected.ollama_top_p is None and selected.ollama_top_k is None
    assert selected.ollama_think is True


def test_settings_rejects_nonnumeric_context_at_creation(monkeypatch):
    monkeypatch.setenv('OLLAMA_NUM_CTX', 'not-an-integer')
    with pytest.raises(ValueError):
        Settings()


def test_ui_runtime_skips_missing_index_without_creating_embedding(monkeypatch, tmp_path):
    def unexpected(_settings):
        raise AssertionError('an absent index must not load an embedding model')
    monkeypatch.setattr(runtime, 'acquire_dense', unexpected)
    assert runtime.open_chat_index(str(tmp_path), 'model', 'sentence_transformers') is None
    assert runtime.open_chat_index(str(tmp_path), 'model', 'dummy') is None


def test_ui_runtime_passes_settings_dense_and_telemetry_to_graph(monkeypatch, tmp_path):
    captured = {}
    sentinel_dense = object()
    sentinel_graph = object()

    def make_graph(settings, *, dense, telemetry):
        captured.update(settings=settings, dense=dense, telemetry=telemetry)
        return sentinel_graph

    graph, telemetry = runtime.create_chat_workflow(
        provider='dummy', fast=True, answer_mode='detailed',
        data_dir=str(tmp_path), embedding_model='example-model',
        embedding_provider='dummy', read_timeout_s=120, total_timeout_s=240, dense=sentinel_dense, workflow_builder=make_graph,
    )
    assert graph is sentinel_graph
    assert captured['dense'] is sentinel_dense
    assert captured['telemetry'] is telemetry
    assert captured['settings'].data_dir == tmp_path
    assert captured['settings'].answer_mode == 'detailed'
    assert captured['settings'].ollama_total_timeout_s == 240
    assert captured['settings'].ollama_read_timeout_s == 120


def test_functional_evaluation_uses_canonical_environment_implementation():
    assert professional._environment(Settings(), [])['index_sha256'] == runtime_environment.environment(
        Settings(), [],
    )['index_sha256']


def test_real_project_has_no_import_time_cycles_or_streamlit_in_domain_layer():
    result = inspect_import_boundaries(ROOT)
    assert result['import_cycles'] == []
    assert result['presentation_boundary_violations'] == []


def test_architecture_audit_detects_actual_top_level_cycle(tmp_path):
    package = tmp_path / 'src' / 'dap_assistant' / 'rag'
    package.mkdir(parents=True)
    (package / 'first.py').write_text('from .second import helper\n', encoding='utf-8')
    (package / 'second.py').write_text('from .first import helper\n', encoding='utf-8')
    result = audit(tmp_path)
    assert result['import_cycles'] == [['rag.first', 'rag.second']]


def test_architecture_audit_distinguishes_lazy_imports_from_import_time_cycle(tmp_path):
    package = tmp_path / 'src' / 'dap_assistant' / 'rag'
    package.mkdir(parents=True)
    (package / 'first.py').write_text('def get():\n    from .second import helper\n', encoding='utf-8')
    (package / 'second.py').write_text('from .first import get\n', encoding='utf-8')
    assert inspect_import_boundaries(tmp_path)['import_cycles'] == []


def test_architecture_audit_catches_streamlit_import_in_domain_layer(tmp_path):
    package = tmp_path / 'src' / 'dap_assistant' / 'documents'
    package.mkdir(parents=True)
    (package / 'ingestion.py').write_text('import streamlit as st\n', encoding='utf-8')
    result = inspect_import_boundaries(tmp_path)
    assert result['presentation_boundary_violations'] == ['documents.ingestion:1: streamlit']


def test_telemetry_is_owned_by_observability():
    from dap_assistant.observability import telemetry

    assert telemetry.Telemetry is not None
    assert not (ROOT / "src/dap_assistant/evaluation/telemetry.py").exists()
