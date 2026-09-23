"""Verify local Qdrant metadata and repair stale indexes by re-embedding, never by inventing metadata."""
from __future__ import annotations

import json

from dap_assistant.documents.ingestion import load_chunks
from ..settings import Settings
from dap_assistant.documents.index_metadata import corpus_fingerprint as corpus_fingerprint, index_metadata_path, metadata_problems


def repair_missing_metadata(settings: Settings, dense: object) -> dict:
    """Reindex with the leased shared Qdrant client after verifying model and source versions."""
    path = index_metadata_path(settings)
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    chunks = load_chunks(settings.data_dir)
    if not chunks:
        raise RuntimeError('Nincsenek feldolgozott dokumentumok; előbb töltsd le és indexeld őket.')
    from dap_assistant.rag.retrieval import index_all
    index_all(settings, dense=dense)
    if not path.exists():
        raise RuntimeError('Az index újraépült, de a verziómetaadat nem jött létre.')
    return json.loads(path.read_text(encoding='utf-8'))


def check_index(settings: Settings, dense: object, *, repair_missing: bool = False) -> dict:
    """Verify metadata and actual point count, optionally reindexing legacy data."""
    path = index_metadata_path(settings)
    if not path.exists():
        if not repair_missing:
            raise RuntimeError('Hiányzik az index verziómetaadata; javítás szükséges.')
        repair_missing_metadata(settings, dense)
    metadata = json.loads(path.read_text(encoding='utf-8'))
    chunks = load_chunks(settings.data_dir)
    issues = metadata_problems(metadata, settings, chunks)
    if metadata.get('embedding_dimension') != dense.embeddings.dimension:
        issues.append('embedding_dimension')
    if issues:
        raise RuntimeError('Az index és a feldolgozott dokumentumok eltérnek: '
                           + ', '.join(issues) + '. Állítsd le a többi folyamatot, majd indexelj újra.')
    with dense._client_lock:
        count = dense.client.count(collection_name=dense.COLLECTION, exact=True).count
    if count != len(chunks):
        raise RuntimeError(f'Qdrant: {count} vektor, feldolgozott dokumentumrészek: {len(chunks)}. '
                           'Az index újraépítése szükséges.')
    return metadata
