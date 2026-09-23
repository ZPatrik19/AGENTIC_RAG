from dap_assistant.evaluation.comparison import (
    COMPARISON_PROFILES,
    _as_evidence,
    _audit_simple_draft,
    _paired_deltas,
)
from dap_assistant.rag import dense_resources
from types import SimpleNamespace

import pytest


def test_comparison_profiles_disable_features_incrementally():
    baseline = COMPARISON_PROFILES['baseline_dense']
    hybrid = COMPARISON_PROFILES['hybrid_rrf']
    agentic = COMPARISON_PROFILES['agentic']
    assert baseline['bm25'] is False and baseline['agentic_routing'] is False
    assert hybrid['bm25'] is True and hybrid['rrf'] is True and hybrid['retry'] is False
    assert agentic['agentic_routing'] is True and agentic['retry'] is True and agentic['tools'] is True


def test_as_evidence_adds_stable_id_without_mutating_input():
    original = {'chunk_id': 'abcdef0123456789XYZ', 'text': 'forrás', 'document_id': 'doc'}
    out = _as_evidence([original])
    assert out[0]['evidence_id'] == 'E_abcdef0123456789'
    assert 'evidence_id' not in original


def test_simple_audit_rejects_number_not_in_quote():
    evidence = _as_evidence([{
        'chunk_id': 'abcdef0123456789XYZ', 'document_id': 'doc',
        'text': 'Az ügyintézési határidő 15 nap.', 'source_url': 'https://example.invalid',
    }])
    eid = evidence[0]['evidence_id']
    draft = {
        'claims': [{
            'text': 'Az ügyintézési határidő 30 nap.',
            'evidence_ids': [eid],
            'supporting_quote': 'Az ügyintézési határidő 15 nap.',
            'category': 'deadline',
        }],
        'disclaimer': '',
    }
    audited, rejected = _audit_simple_draft(draft, evidence)
    assert audited['claims'] == []
    assert rejected


def test_paired_deltas_are_explicitly_against_baseline():
    summary = {
        'baseline_dense': {'latency_mean_s': 2.0, 'llm_calls_mean': 1.0,
                           'metrics': {'faithfulness': {'value': .8}}},
        'hybrid_rrf': {'latency_mean_s': 2.5, 'llm_calls_mean': 1.0,
                       'metrics': {'faithfulness': {'value': .9}}},
        'agentic': {'latency_mean_s': 4.0, 'llm_calls_mean': 2.0,
                    'metrics': {'faithfulness': {'value': None}}},
    }
    out = _paired_deltas(summary)
    assert out['hybrid_rrf_minus_baseline_dense']['latency_mean_s'] == .5
    assert abs(out['hybrid_rrf_minus_baseline_dense']['metrics']['faithfulness'] - .1) < 1e-12
    assert out['agentic_minus_baseline_dense']['llm_calls_mean'] == 1.0
    assert out['agentic_minus_baseline_dense']['metrics']['faithfulness'] is None


def test_qdrant_external_lock_explains_how_to_release_it_without_deleting_data(tmp_path, monkeypatch):
    """A second process should produce actionable diagnostics and no stale lease."""
    directory = tmp_path / 'vectorstore' / 'qdrant'
    directory.mkdir(parents=True)
    settings = SimpleNamespace(data_dir=tmp_path, embedding_provider='sentence_transformers',
                               embedding_model='test-model')

    def locked_client(*_args):
        raise RuntimeError(f'Storage folder {directory} is already accessed by another instance of Qdrant client.')

    import dap_assistant.rag.retrieval as retrieval
    monkeypatch.setattr(retrieval, 'LocalEmbeddings', lambda *_args: object())
    monkeypatch.setattr(retrieval, 'LocalQdrant', locked_client)

    with pytest.raises(RuntimeError, match='Streamlit') as error:
        dense_resources.acquire_dense(settings)

    assert 'Ctrl+C' in str(error.value)
    assert 'Ne töröld' in str(error.value)
    assert directory.resolve() not in dense_resources._CLIENTS


def test_qdrant_lock_does_not_hide_other_initialization_errors(tmp_path, monkeypatch):
    directory = tmp_path / 'vectorstore' / 'qdrant'
    directory.mkdir(parents=True)
    settings = SimpleNamespace(data_dir=tmp_path, embedding_provider='sentence_transformers',
                               embedding_model='test-model')

    def broken_client(*_args):
        raise RuntimeError('Embedding dimension mismatch')

    import dap_assistant.rag.retrieval as retrieval
    monkeypatch.setattr(retrieval, 'LocalEmbeddings', lambda *_args: object())
    monkeypatch.setattr(retrieval, 'LocalQdrant', broken_client)

    with pytest.raises(RuntimeError, match='Embedding dimension mismatch'):
        dense_resources.acquire_dense(settings)
    assert directory.resolve() not in dense_resources._CLIENTS


def test_qdrant_lease_is_released_if_llm_initialization_fails(tmp_path, monkeypatch):
    """A benchmark must never retain the index after a failed LLM setup."""
    import dap_assistant.evaluation.comparison as comparison

    fake_case = {'question_id': 'TEST_001', 'category': 'vehicle', 'question': 'Teszt?'}
    monkeypatch.setattr(comparison, 'select_cases', lambda **_kwargs: [fake_case])
    monkeypatch.setattr(comparison, 'load_chunks', lambda _path: [{'document_id': 'd1'}])
    monkeypatch.setattr(comparison, 'index_versions', lambda _chunks: {})
    # This test isolates Qdrant lease cleanup; automatic reference generation
    # has independent coverage and requires real chunk IDs and text.
    import dap_assistant.evaluation.automatic_reference as automatic_reference
    monkeypatch.setattr(automatic_reference, 'ensure_auto_reference', lambda *_args, **_kwargs: None)
    acquired = object()
    released = []
    monkeypatch.setattr(comparison, 'acquire_dense', lambda _settings: acquired)
    monkeypatch.setattr(comparison, 'release_dense', lambda dense: released.append(dense))

    def fail_llm(*_args, **_kwargs):
        raise RuntimeError('Model is unavailable')

    monkeypatch.setattr(comparison, 'get_llm', fail_llm)

    with pytest.raises(RuntimeError, match='Model is unavailable'):
        comparison.run_rag_comparison(SimpleNamespace(data_dir=tmp_path))
    assert released == [acquired]
