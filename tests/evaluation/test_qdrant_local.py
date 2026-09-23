"""Optional real Qdrant local regression, no model downloads or external network."""
from dataclasses import replace

import pytest

pytest.importorskip('qdrant_client')

from dap_assistant.rag.retrieval import LocalQdrant
from dap_assistant.settings import Settings


class FixedEmbeddings:
    dimension = 3

    def passages(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def query(self, text):
        return [1.0, 0.0, 0.0]


def test_actual_local_qdrant_dense_query_with_single_vector(tmp_path):
    settings = replace(Settings(), data_dir=tmp_path)
    db = LocalQdrant(settings, FixedEmbeddings())
    try:
        db.index_document([{
            'chunk_id': 'test-0001', 'document_id': 'test-doc', 'domain': 'vehicle',
            'role': 'buyer', 'text': 'Adásvételi szerződés és átírás',
        }])
        result = db.search('autó átírás', 'vehicle', role='buyer')
        assert result and result[0][0] == 'test-0001'
        assert db.search('autó átírás', 'vehicle', role='seller') == []
    finally:
        db.close()
