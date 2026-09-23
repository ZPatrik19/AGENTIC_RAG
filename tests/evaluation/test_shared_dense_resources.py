"""Regression: UI + evaluation lease one embedded Qdrant instance, no network/model."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from dap_assistant.rag import dense_resources
from dap_assistant.evaluation.preflight import acquire_verified_dense, BenchmarkPrerequisiteError
from dap_assistant.settings import Settings


class FakeEmbeddings:
    def __init__(self, settings):
        self.dimension = 3


class FakeQdrant:
    instances = []

    def __init__(self, settings, embeddings):
        self.closed = False
        self.settings = settings
        self.embeddings = embeddings
        self.__class__.instances.append(self)

    def close(self):
        assert not self.closed, 'double close'
        self.closed = True


@pytest.fixture
def fake_qdrant(monkeypatch, tmp_path):
    from dap_assistant.rag import retrieval

    FakeQdrant.instances.clear()
    monkeypatch.setattr(retrieval, 'LocalEmbeddings', FakeEmbeddings)
    monkeypatch.setattr(retrieval, 'LocalQdrant', FakeQdrant)
    (tmp_path / 'vectorstore' / 'qdrant').mkdir(parents=True)
    settings = replace(Settings(), data_dir=tmp_path,
                       embedding_provider='sentence_transformers', embedding_model='fake-e5')
    yield settings
    # Each test must release all leases. No real Qdrant filesystem is involved.
    assert not dense_resources._CLIENTS


def test_evaluation_reuses_ui_client_without_closing_it(fake_qdrant):
    ui_client = dense_resources.acquire_dense(fake_qdrant)
    assert acquire_verified_dense(fake_qdrant, require_inference=False, real_only=False) is ui_client
    assert len(FakeQdrant.instances) == 1
    dense_resources.release_dense(ui_client)  # Evaluation exits, UI still holds lease.
    assert not ui_client.closed
    assert dense_resources.acquire_dense(fake_qdrant) is ui_client
    dense_resources.release_dense(ui_client)  # Extra UI lease.
    dense_resources.release_dense(ui_client)  # Original UI lease.
    assert ui_client.closed


def test_threaded_acquisition_opens_exactly_one_client(fake_qdrant):
    barrier = Barrier(4)

    def obtain(_):
        barrier.wait(timeout=5)
        return dense_resources.acquire_dense(fake_qdrant)

    with ThreadPoolExecutor(max_workers=4) as pool:
        clients = list(pool.map(obtain, range(4)))
    assert all(client is clients[0] for client in clients)
    assert len(FakeQdrant.instances) == 1
    for client in clients:
        dense_resources.release_dense(client)
    assert clients[0].closed


def test_embedding_change_is_rejected_before_second_qdrant_open(fake_qdrant):
    client = dense_resources.acquire_dense(fake_qdrant)
    incompatible = replace(fake_qdrant, embedding_model='different-e5')
    with pytest.raises(RuntimeError, match='embeddingmodell'):
        dense_resources.acquire_dense(incompatible)
    assert len(FakeQdrant.instances) == 1
    dense_resources.release_dense(client)


def test_release_unknown_client_is_not_silent(fake_qdrant):
    with pytest.raises(ValueError, match='not acquired'):
        dense_resources.release_dense(object())


def test_benchmark_preflight_failure_releases_only_its_lease(fake_qdrant, monkeypatch):
    """Failed point-count validation must not close the chat's active index."""
    import hashlib
    import json
    from threading import RLock
    from types import SimpleNamespace

    from dap_assistant.evaluation import preflight

    ui_client = dense_resources.acquire_dense(fake_qdrant)
    ui_client._client_lock = RLock()
    ui_client.client = SimpleNamespace(count=lambda **kwargs: SimpleNamespace(count=0))
    ui_client.COLLECTION = 'official_documents'
    processed = [{'chunk_id': 'fixture-1', 'text': 'ellenőrzött tesztdokumentum',
                  'document_id': 'test-doc', 'document_version': 'sha256-fixture'}]
    expected_hash = hashlib.sha256(json.dumps(
        [('fixture-1', 'ellenőrzött tesztdokumentum')], ensure_ascii=False,
    ).encode('utf-8')).hexdigest()
    (fake_qdrant.data_dir / 'vectorstore' / 'index_meta.json').write_text(
        json.dumps({'chunk_content_sha256': expected_hash, 'chunk_count': 1,
                    'embedding_model': 'fake-e5', 'embedding_dimension': 3,
                    'document_versions': {'test-doc': 'sha256-fixture'},
                    'embedding_text_template': 'title / section_path + text (v2)',
                        'active_source_ids': sorted(s.id for s in __import__('dap_assistant.documents.sources', fromlist=['load_manifest']).load_manifest(fake_qdrant.manifest).sources if s.processing_status != 'disabled'),
                    'embedding_provider': 'sentence_transformers'}),
        encoding='utf-8',
    )
    monkeypatch.setattr(preflight, 'load_chunks', lambda _: processed)
    from dap_assistant.evaluation import index_health
    monkeypatch.setattr(index_health, 'load_chunks', lambda _: processed)
    monkeypatch.setattr(preflight, 'ollama_health', lambda _: {'configured_model_found': True})
    settings = replace(fake_qdrant, llm_provider='ollama', answer_mode='quick')
    with pytest.raises(BenchmarkPrerequisiteError, match='Qdrant:'):
        acquire_verified_dense(settings, require_inference=True, real_only=True)
    assert not ui_client.closed
    assert dense_resources._CLIENTS[fake_qdrant.data_dir / 'vectorstore' / 'qdrant'].references == 1
    dense_resources.release_dense(ui_client)
    assert ui_client.closed
