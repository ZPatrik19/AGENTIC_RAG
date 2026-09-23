"""Hybrid BM25 + locally embedded Qdrant retrieval. No hosted services."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import json
import hashlib
import os
import math
import re
import uuid
import time
from threading import RLock

from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.documents.sources import load_manifest
from dap_assistant.settings import Settings
from ..documents.index_metadata import corpus_fingerprint


def tokenize(text: str) -> list[str]:
    return re.findall(r'[\wáéíóöőúüű]+', text.lower(), flags=re.UNICODE)


@lru_cache(maxsize=2048)
def _term_counts(chunk_id: str, text: str) -> Counter:
    """Cache corpus tokenization; reindexing changes text and invalidates the key."""
    return Counter(tokenize(text))


@lru_cache(maxsize=3)
def _load_cached_chunks(data_dir: str, signature: tuple) -> tuple[dict, ...]:
    return tuple(load_chunks(Path(data_dir)))


def _chunk_snapshot(data_dir: Path) -> list[dict]:
    # Detect changed files without parsing every JSON document for each query.
    manifest_path = data_dir.parent / 'config' / 'document_sources.yaml'
    active_ids = ({item.id for item in load_manifest(manifest_path).sources
                   if item.processing_status != 'disabled'} if manifest_path.exists() else None)
    files = sorted(path for path in (data_dir / 'processed').glob('*/*.json')
                   if active_ids is None or path.stem in active_ids)
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                      for path in files)
    return list(_load_cached_chunks(str(data_dir), signature))


def bm25_search(query: str, chunks: list[dict], domain: str, limit: int = 12, role: str = "") -> list[tuple[str, float]]:
    """Small-corpus Okapi BM25 (no model download)."""
    filtered = [c for c in chunks if c['domain'] == domain and (not role or c.get('role', 'general') in ('general', role))]
    if not filtered:
        return []
    docs = [_term_counts(c['chunk_id'], c.get('retrieval_text') or c['text']) for c in filtered]
    query_terms = set(tokenize(query))
    df = Counter(term for doc in docs for term in doc)
    avgdl = sum(sum(doc.values()) for doc in docs) / len(docs)
    scored = []
    for chunk, counts in zip(filtered, docs):
        dl = sum(counts.values())
        score = 0.0
        for term in query_terms:
            if not counts[term]:
                continue
            idf = math.log(1.0 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (counts[term] * 2.2) / (counts[term] + 1.2 * (0.25 + 0.75 * dl / max(avgdl, 1)))
        if score:
            scored.append((chunk['chunk_id'], score))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:limit]


class LocalEmbeddings:
    def __init__(self, settings: Settings):
        if settings.embedding_provider != 'sentence_transformers':
            raise ValueError('Only sentence_transformers is a production embedding provider')
        from sentence_transformers import SentenceTransformer

        # Keep the small embedding model on CPU by default; leave limited VRAM for Ollama.
        self.model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
        self.dimension = (self.model.get_embedding_dimension() if hasattr(self.model, 'get_embedding_dimension') else self.model.get_sentence_embedding_dimension())

    def passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(['passage: ' + t for t in texts], normalize_embeddings=True, show_progress_bar=False).tolist()

    @lru_cache(maxsize=128)
    def _query_cached(self, text: str) -> list[float]:
        return self.model.encode(['query: ' + text], normalize_embeddings=True)[0].tolist()

    def query(self, text: str) -> list[float]:
        return self._query_cached(text)


class LocalQdrant:
    COLLECTION = 'official_documents'

    def __init__(self, settings: Settings, embeddings: LocalEmbeddings):
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        path = settings.data_dir / 'vectorstore' / 'qdrant'
        path.mkdir(parents=True, exist_ok=True)
        # A cached QdrantLocal client is shared by the Streamlit main thread and
        # its graph worker. SQLite's default check_same_thread would otherwise
        # crash. All local client operations are serialized by this explicit lock.
        self._client_lock = RLock()
        self.client = QdrantClient(path=str(path), force_disable_check_same_thread=True)
        self.embeddings = embeddings
        if not self.client.collection_exists(self.COLLECTION):
            self.client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=embeddings.dimension, distance=Distance.COSINE),
            )
        elif self.client.get_collection(self.COLLECTION).config.params.vectors.size != embeddings.dimension:
            raise RuntimeError('Embedding dimension mismatch; rebuild data/vectorstore after model change')
        metadata_path = settings.data_dir / 'vectorstore' / 'index_meta.json'
        if metadata_path.exists():
            stored = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (stored.get('embedding_model') != settings.embedding_model
                    or stored.get('embedding_dimension') != embeddings.dimension):
                raise RuntimeError('Embedding model/index mismatch: stop Streamlit, remove data/vectorstore/qdrant and data/vectorstore/index_meta.json, then run python -m dap_assistant.cli index')

    def index_document(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct

        doc_id = chunks[0]['document_id']
        vectors = self.embeddings.passages([
            c.get('retrieval_text') or f"{c.get('title') or c.get('document_id', '')} / {' / '.join(c.get('section_path', []))}\n{c['text']}"
            for c in chunks
        ])
        # Remove all previous versions, then insert the new stable IDs.
        selector = FilterSelector(filter=Filter(must=[FieldCondition(key='document_id', match=MatchValue(value=doc_id))]))
        points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c['chunk_id'])), vector=v, payload=c)
                  for c, v in zip(chunks, vectors)]
        with self._client_lock:
            self.client.delete(self.COLLECTION, points_selector=selector, wait=True)
            self.client.upsert(self.COLLECTION, points=points, wait=True)

    def search(self, query: str, domain: str, limit: int = 12, role: str = "", telemetry=None, run_id: str = "") -> list[tuple[str, float]]:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        must = [FieldCondition(key='domain', match=MatchValue(value=domain))]
        # Role-filter at the database as well as during hybrid fusion. Direct
        # dense-only callers must never receive the opposite vehicle role,
        # while general NAV/NJT documents remain eligible.
        role_conditions = ([FieldCondition(key='role', match=MatchValue(value=value))
                            for value in dict.fromkeys((role, 'general', ''))]
                           if role in ('buyer', 'seller') else [])
        started = time.perf_counter()
        vector = self.embeddings.query(query)
        if telemetry is not None:
            telemetry.record(run_id, 'embedding_query', time.perf_counter() - started)
        if len(vector) != self.embeddings.dimension or any(not isinstance(v, (float, int)) for v in vector):
            raise ValueError('Query embedding must be a flat dense vector of the configured dimension')
        started = time.perf_counter()
        with self._client_lock:
            results = self.client.query_points(
                collection_name=self.COLLECTION,
                query=vector,
                query_filter=Filter(must=must, should=role_conditions or None),
                limit=limit,
                with_payload=True,
            ).points
        if telemetry is not None:
            telemetry.record(run_id, 'dense_retrieval', time.perf_counter() - started)
        return [(p.payload['chunk_id'], float(p.score)) for p in results
                if (not role_conditions or (p.payload or {}).get('role', 'general')
                    in ('general', '', role))]


    def remove_inactive_documents(self, active_ids: set[str]) -> None:
        """Purge old domains from the embedded collection during an explicit reindex."""
        from qdrant_client.http.models import PointIdsList
        offset = None
        old_ids = []
        with self._client_lock:
            while True:
                points, offset = self.client.scroll(self.COLLECTION, limit=256, offset=offset,
                                                    with_payload=['document_id'], with_vectors=False)
                old_ids.extend(point.id for point in points
                               if (point.payload or {}).get('document_id') not in active_ids)
                if offset is None:
                    break
            if old_ids:
                self.client.delete(self.COLLECTION, points_selector=PointIdsList(points=old_ids), wait=True)

    def close(self):
        with self._client_lock:
            self.client.close()


def hybrid_search(query: str, domain: str, settings: Settings, *, limit: int = 5, dense: LocalQdrant | None = None, role: str = "", telemetry=None, run_id: str = "") -> list[dict]:
    chunks = _chunk_snapshot(settings.data_dir)
    # Defense in depth: stale/unfiltered dense IDs cannot introduce the opposite role.
    by_id = {chunk['chunk_id']: chunk for chunk in chunks
             if chunk['domain'] == domain and (not role or chunk.get('role', 'general') in ('general', role))}
    started = time.perf_counter()
    bm25 = bm25_search(query, chunks, domain, limit=12, role=role)
    if telemetry is not None:
        telemetry.record(run_id, 'bm25_retrieval', time.perf_counter() - started)
    dense_results = dense.search(query, domain, limit=12, role=role, telemetry=telemetry, run_id=run_id) if dense is not None else []
    fusion_started = time.perf_counter()
    # Stable deduplication by chunk_id, with each original rank/score retained.
    provenance: dict[str, dict] = {}
    for engine, ranking in (('bm25', bm25), ('dense', dense_results)):
        for rank, (chunk_id, raw_score) in enumerate(ranking, start=1):
            if chunk_id not in by_id:
                continue
            info = provenance.setdefault(chunk_id, {
                'bm25_rank': None, 'bm25_score': None,
                'dense_rank': None, 'dense_score': None,
                'rrf_components': {},
            })
            if info[f'{engine}_rank'] is not None:
                continue  # Duplicate candidates must not accumulate another vote.
            info[f'{engine}_rank'] = rank
            info[f'{engine}_score'] = float(raw_score)
            info['rrf_components'][engine] = 1.0 / (60 + rank)
    ids = sorted(provenance, key=lambda cid: (-sum(provenance[cid]['rrf_components'].values()), cid))[:limit]
    if telemetry is not None:
        telemetry.record(run_id, 'retrieval_fusion', time.perf_counter() - fusion_started)
    return [{**by_id[cid], **provenance[cid],
             'score': sum(provenance[cid]['rrf_components'].values()),
             'fusion_method': 'rrf_k60_not_probability'} for cid in ids]



def diversify_results(ranked: list[dict], limit: int = 5) -> list[dict]:
    """Include the best result from each retrieved document before filling remaining slots.

    This improves cross-document coverage without additional model calls. The
    returned score remains the original search/rerank score, NOT probability.
    """
    selected: list[dict] = []
    seen_documents: set[str] = set()
    for item in ranked:
        doc_id = item.get('document_id', item['chunk_id'])
        if doc_id not in seen_documents:
            selected.append(item)
            seen_documents.add(doc_id)
        if len(selected) == limit:
            return selected
    selected_ids = {item['chunk_id'] for item in selected}
    selected.extend(item for item in ranked if item['chunk_id'] not in selected_ids)
    return selected[:limit]


def index_all(settings: Settings, *, dense: LocalQdrant | None = None) -> dict:
    chunks = load_chunks(settings.data_dir)
    if settings.embedding_provider == 'dummy':
        return {'status': 'lexical_only_dummy', 'chunks': len(chunks), 'documents': len({c['document_id'] for c in chunks})}
    if not chunks:
        raise RuntimeError('No processed documents. Run download and ingestion first.')
    embeddings = dense.embeddings if dense is not None else LocalEmbeddings(settings)
    db = dense if dense is not None else LocalQdrant(settings, embeddings)
    try:
        by_doc: dict[str, list[dict]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk['document_id'], []).append(chunk)
        # Remove stale points belonging to documents removed from the manifest.
        # A user may still have old housing/business files or vectors locally.
        if hasattr(db, 'remove_inactive_documents'):
            db.remove_inactive_documents(set(by_doc))
        for group in by_doc.values():
            db.index_document(group)
        meta = {
            'embedding_model': settings.embedding_model,
            'embedding_text_template': 'title / section_path + text (v2)',
            'active_source_ids': sorted({source.id for source in load_manifest(settings.manifest).sources if source.processing_status != 'disabled'}),
            'embedding_provider': settings.embedding_provider,
            'embedding_dimension': embeddings.dimension,
            'document_versions': {doc: group[0]['document_version'] for doc, group in by_doc.items()},
            'chunk_count': len(chunks),
            'chunk_content_sha256': corpus_fingerprint(chunks),
        }
        metadata_path = settings.data_dir / 'vectorstore' / 'index_meta.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temporary, metadata_path)
        return {'status': 'indexed', 'chunks': len(chunks), 'documents': len(by_doc), 'active_domains': sorted({c['domain'] for c in chunks})}
    finally:
        if dense is None:
            db.close()
