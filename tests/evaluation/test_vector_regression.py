"""Regression: a 2D query was interpreted by Qdrant as a multivector."""
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from threading import RLock
import sys

import pytest

from dap_assistant.rag.retrieval import LocalEmbeddings, LocalQdrant
from dap_assistant.settings import Settings


class VectorRow:
    def tolist(self):
        return [0.1, 0.2, 0.3]


class VectorBatch:
    def __getitem__(self, index):
        assert index == 0
        return VectorRow()


class FakeSentenceTransformer:
    def __init__(self, model, *, device=None):
        self.model = model
        assert device == "cpu"  # Leave constrained VRAM available to Ollama.

    def get_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        assert len(texts) == 1
        assert texts[0].startswith('query: ')
        return VectorBatch()


def test_query_embedding_is_one_dimensional(monkeypatch):
    module = ModuleType('sentence_transformers')
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, 'sentence_transformers', module)
    embed = LocalEmbeddings(replace(Settings(), embedding_provider='sentence_transformers'))
    assert embed.query('autó átírás') == [0.1, 0.2, 0.3]


def test_qdrant_receives_dense_not_multivector(monkeypatch):
    http = ModuleType('qdrant_client.http')
    models = ModuleType('qdrant_client.http.models')
    models.FieldCondition = lambda **kwargs: kwargs
    models.MatchValue = lambda **kwargs: kwargs
    models.Filter = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, 'qdrant_client.http', http)
    monkeypatch.setitem(sys.modules, 'qdrant_client.http.models', models)

    class FakeClient:
        def query_points(self, *, query, **kwargs):
            assert query == [0.1, 0.2, 0.3]
            assert not isinstance(query[0], list)
            return SimpleNamespace(points=[SimpleNamespace(payload={'chunk_id': 'c1'}, score=.91)])

    local = LocalQdrant.__new__(LocalQdrant)
    local._client_lock = RLock()
    local.client = FakeClient()
    local.embeddings = SimpleNamespace(dimension=3, query=lambda q: [0.1, 0.2, 0.3])
    assert local.search('autó', 'vehicle') == [('c1', .91)]
    local.embeddings = SimpleNamespace(dimension=3, query=lambda q: [[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match='flat dense vector'):
        local.search('autó', 'vehicle')


def test_shared_local_qdrant_operations_are_serialized(monkeypatch):
    """Streamlit background worker must not use QdrantLocal SQLite concurrently."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import RLock
    from time import sleep

    models = ModuleType('qdrant_client.http.models')
    models.FieldCondition = lambda **kwargs: kwargs
    models.MatchValue = lambda **kwargs: kwargs
    models.Filter = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, 'qdrant_client.http.models', models)

    class FakeClient:
        active = 0
        maximum = 0

        def query_points(self, *, query, **kwargs):
            self.active += 1
            try:
                self.maximum = max(self.maximum, self.active)
                sleep(.002)
                return SimpleNamespace(points=[SimpleNamespace(payload={'chunk_id': 'c1'}, score=.9)])
            finally:
                self.active -= 1

    local = LocalQdrant.__new__(LocalQdrant)
    local._client_lock = RLock()
    local.client = FakeClient()
    local.embeddings = SimpleNamespace(dimension=3, query=lambda _: [0.1, 0.2, 0.3])
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: local.search('autó', 'vehicle'), range(8)))
    assert len(results) == 8
    assert local.client.maximum == 1


def test_direct_dense_search_excludes_opposite_vehicle_role(monkeypatch):
    """Dense-only baseline cannot bypass the role guard in hybrid fusion."""
    from types import ModuleType, SimpleNamespace
    from threading import RLock
    import sys

    models = ModuleType('qdrant_client.http.models')
    models.FieldCondition = lambda **kw: kw
    models.MatchValue = lambda **kw: kw
    models.Filter = lambda **kw: kw
    monkeypatch.setitem(sys.modules, 'qdrant_client.http.models', models)

    class FakeClient:
        def query_points(self, *, query_filter, **kwargs):
            assert len(query_filter['must']) == 1
            assert {condition['match']['value'] for condition in query_filter['should']} == {
                'buyer', 'general', '',
            }
            # Deliberately return an invalid opposite-role payload: a safety
            # check must still reject it if a backend ignores its query filter.
            return SimpleNamespace(points=[
                SimpleNamespace(payload={'chunk_id': 'seller', 'role': 'seller'}, score=.9),
                SimpleNamespace(payload={'chunk_id': 'general', 'role': 'general'}, score=.8),
                SimpleNamespace(payload={'chunk_id': 'buyer', 'role': 'buyer'}, score=.7),
            ])

    local = LocalQdrant.__new__(LocalQdrant)
    local._client_lock = RLock()
    local.client = FakeClient()
    local.embeddings = SimpleNamespace(dimension=3, query=lambda q: [0.1, 0.2, 0.3])
    assert local.search('autó átírás', 'vehicle', role='buyer') == [
        ('general', .8), ('buyer', .7),
    ]
