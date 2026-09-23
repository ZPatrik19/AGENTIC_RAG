"""Early configuration validation and DI boundaries, without an LLM or network."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import get_type_hints

import pytest

from dap_assistant.orchestration import runtime
from dap_assistant.orchestration.state import (
    AssistantState, BranchResult, EvidenceChunk, TaskRecord, merge_dict,
)
from dap_assistant.settings import Settings


@pytest.mark.parametrize('variable,value,expected', [
    ('LLM_PROVIDER', 'unknown', 'LLM_PROVIDER'),
    ('EMBEDDING_PROVIDER', 'unknown', 'EMBEDDING_PROVIDER'),
    ('ANSWER_MODE', 'verbose', 'ANSWER_MODE'),
    ('OLLAMA_BASE_URL', 'ftp://localhost:11434', 'OLLAMA_BASE_URL'),
    ('OLLAMA_BASE_URL', 'http://localhost:not-a-port', 'OLLAMA_BASE_URL'),
    ('OLLAMA_BASE_URL', 'http://localhost:11434?token=x', 'OLLAMA_BASE_URL'),
    ('OLLAMA_BASE_URL', 'http://admin:secret@localhost:11434', 'OLLAMA_BASE_URL'),
    ('OLLAMA_NUM_CTX', '0', 'OLLAMA_NUM_CTX'),
    ('OLLAMA_ANSWER_NUM_PREDICT', '-3', 'OLLAMA_ANSWER_NUM_PREDICT'),
    ('OLLAMA_TOTAL_TIMEOUT_S', '-2', 'OLLAMA_TOTAL_TIMEOUT_S'),
    ('OLLAMA_TEMPERATURE', '3', 'OLLAMA_TEMPERATURE'),
    ('OLLAMA_TOP_P', '1.1', 'OLLAMA_TOP_P'),
    ('OLLAMA_TOP_K', '0', 'OLLAMA_TOP_K'),
    ('MAX_SUBTASKS', '7', 'MAX_SUBTASKS'),
    ('MAX_RAG_ATTEMPTS', '0', 'MAX_RAG_ATTEMPTS'),
    ('OLLAMA_THINK', 'sometimes', 'OLLAMA_THINK'),
])
def test_invalid_env_rejected_before_side_effects(monkeypatch, variable, value, expected):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=expected):
        Settings()


def test_settings_are_frozen_and_replace_also_validates():
    config = Settings()
    with pytest.raises(FrozenInstanceError):
        config.max_subtasks = 7
    with pytest.raises(ValueError, match='MAX_SUBTASKS'):
        replace(config, max_subtasks=7)
    assert replace(config, max_subtasks=6).max_subtasks == 6


def test_source_strategy_is_internal_only(monkeypatch):
    monkeypatch.setenv('ANSWER_MODE', 'source')
    with pytest.raises(ValueError, match='ANSWER_MODE'):
        Settings()
    monkeypatch.setenv('ANSWER_MODE', 'quick')
    assert replace(Settings(), answer_mode='source').answer_mode == 'source'


def test_runtime_accepts_injected_dense_acquirer_with_no_real_index(monkeypatch, tmp_path):
    index = tmp_path / 'vectorstore' / 'qdrant'
    index.mkdir(parents=True)
    captured = []
    sentinel = object()

    def fake_acquire(settings):
        captured.append(settings)
        return sentinel

    result = runtime.open_chat_index(
        str(tmp_path), 'tiny', 'sentence_transformers',
        base_settings=replace(Settings(), llm_provider='dummy'),
        dense_acquirer=fake_acquire,
    )
    assert result is sentinel
    assert len(captured) == 1
    assert captured[0].data_dir == tmp_path
    assert captured[0].embedding_model == 'tiny'


def test_missing_index_never_initializes_injected_dense_model(tmp_path):
    def fail(_settings):
        pytest.fail('Embedding model loaded despite absent index')

    assert runtime.open_chat_index(
        str(tmp_path), 'tiny', 'sentence_transformers', dense_acquirer=fail,
    ) is None


def test_workflow_injection_preserves_explicit_settings_and_telemetry(tmp_path):
    original = replace(Settings(), llm_provider='dummy', data_dir=tmp_path, answer_mode='quick')
    sentinel_graph = object()
    sentinel_dense = object()
    fake_telemetry = object()
    seen = []

    def builder(settings, *, dense, telemetry):
        seen.append((settings, dense, telemetry))
        return sentinel_graph

    graph, telemetry = runtime.create_chat_workflow(
        provider='dummy', fast=True, answer_mode='detailed',
        data_dir=str(tmp_path), embedding_model='test', embedding_provider='dummy',
        read_timeout_s=180, total_timeout_s=480, dense=sentinel_dense, workflow_builder=builder,
        base_settings=original, telemetry_factory=lambda: fake_telemetry,
    )
    assert graph is sentinel_graph
    assert telemetry is fake_telemetry
    config, dense, trace = seen.pop()
    assert dense is sentinel_dense and trace is fake_telemetry
    assert config.ollama_read_timeout_s == 180
    assert config.ollama_total_timeout_s == 480
    assert config.data_dir == tmp_path
    assert config.answer_mode == 'detailed'
    assert original.answer_mode != config.answer_mode  # snapshot remains unchanged


def test_invalid_ui_override_rejected_before_graph_or_telemetry(tmp_path):
    def fail(*args, **kwargs):
        pytest.fail('Invalid config must not allocate resources')

    with pytest.raises(ValueError, match='ANSWER_MODE'):
        runtime.create_chat_workflow(
            provider='dummy', fast=True, answer_mode='unknown',
            data_dir=str(tmp_path), embedding_model='test', embedding_provider='dummy',
            read_timeout_s=120, total_timeout_s=240, workflow_builder=fail, telemetry_factory=fail,
        )


def test_parallel_worker_reducer_is_copy_on_write_and_retry_updates_one_key():
    older = {'t1': {'retrieval_status': 'insufficient'}, 't2': {'retrieval_status': 'passed'}}
    update = {'t1': {'retrieval_status': 'passed'}}
    result = merge_dict(older, update)
    assert result == {'t1': {'retrieval_status': 'passed'}, 't2': {'retrieval_status': 'passed'}}
    assert older['t1']['retrieval_status'] == 'insufficient'
    assert result is not older and result is not update
    assert merge_dict(None, update) == update


def test_state_contracts_document_tasks_and_versioned_evidence():
    fields = get_type_hints(AssistantState, include_extras=True)
    assert 'branch_results' in fields and 'tool_results' in fields
    assert {'task_id', 'depends_on', 'worker_retries'} <= TaskRecord.__optional_keys__
    assert {'chunk_id', 'document_id', 'document_version', 'source_url'} <= EvidenceChunk.__optional_keys__
    assert {'ranked_chunk_ids', 'retrieval_status', 'evidence'} <= BranchResult.__optional_keys__


def test_rag_state_is_explicit_without_importing_langgraph():
    from dap_assistant.rag.state import RAGState, RAGChunk

    assert {'task_id', 'domain', 'query', 'evidence', 'search_attempt'} <= RAGState.__optional_keys__
    assert {'chunk_id', 'document_id', 'document_version'} <= RAGChunk.__optional_keys__
