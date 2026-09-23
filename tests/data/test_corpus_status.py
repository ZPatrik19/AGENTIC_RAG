"""Corpus status: offline evidence, no network, embeddings, or human-gold invention."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import sys
from types import SimpleNamespace

import yaml

from dap_assistant.documents.corpus_status import (
    build_corpus_report, compare_index_payloads, corpus_ready,
)
from dap_assistant.evaluation.index_health import corpus_fingerprint
from dap_assistant.documents.ingestion import make_chunks
from dap_assistant.settings import Settings
from dap_assistant.documents.sources import Source


def _fixture(tmp_path):
    root = tmp_path
    data = root / 'data'
    (root / 'config').mkdir()
    source = Source(id='vehicle-test-doc', title='Jarmu hivatalos tajekoztato',
                    url='https://dap.gov.hu/test', domain='vehicle', type='html',
                    destination='vehicle', required=True)
    optional = Source(id='optional-test-doc', title='Optional document',
                      url='https://dap.gov.hu/optional', domain='vehicle', type='html',
                      destination='vehicle')
    disabled = Source(id='disabled-test-doc', title='Disabled document',
                      url='https://dap.gov.hu/disabled', domain='vehicle', type='html',
                      destination='vehicle', processing_status='disabled')
    (root / 'config' / 'document_sources.yaml').write_text(
        yaml.safe_dump({'sources': [s.model_dump() for s in (source, optional, disabled)]}),
        encoding='utf-8')
    settings = replace(Settings(), root=root, data_dir=data)
    return settings, source, optional


def _create_verified(settings, source, *, backslashes=False):
    raw = b'<html><body><main><h1>Dokumentum</h1><p>' + b'teszt ' * 40 + b'</p></main></body></html>'
    digest = sha256(raw).hexdigest()
    rel = f'raw/{source.domain}/{source.id}.html'
    path = settings.data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    meta = {'id': source.id, 'url': source.url, 'domain': source.domain,
            'type': source.type, 'file': rel.replace('/', '\\') if backslashes else rel,
            'sha256': digest, 'retrieved_at': '2026-09-21T09:00:00+02:00'}
    metadata_path = settings.data_dir / 'interim' / 'downloads' / f'{source.id}.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta), encoding='utf-8')
    chunks = make_chunks(source, meta, [
        {'section': ['Eljárás'], 'text': 'Az ugyintezeshez iratok szuksegesek. ' * 8, 'page': None},
    ])
    processed = settings.data_dir / 'processed' / source.domain / f'{source.id}.json'
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_text(json.dumps({'document_id': source.id,
                                      'document_version': digest, 'chunks': chunks}),
                         encoding='utf-8')
    return path, processed, chunks


def _index_meta(settings, chunks):
    from dap_assistant.documents.sources import load_manifest
    path = settings.data_dir / 'vectorstore' / 'index_meta.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'embedding_model': settings.embedding_model,
        'embedding_provider': settings.embedding_provider,
        'embedding_dimension': 384,
        'embedding_text_template': 'title / section_path + text (v2)',
        'active_source_ids': sorted(s.id for s in load_manifest(settings.manifest).sources
                                    if s.processing_status != 'disabled'),
        'chunk_count': len(chunks),
        'chunk_content_sha256': corpus_fingerprint(chunks),
        'document_versions': {c['document_id']: c['document_version'] for c in chunks},
    }), encoding='utf-8')
    return path


def test_empty_zip_distinguishes_configured_and_downloaded(tmp_path):
    settings, _, _ = _fixture(tmp_path)
    report = build_corpus_report(settings)
    assert report['summary']['configured'] == 3
    assert report['summary']['enabled'] == 2
    assert report['summary']['disabled'] == 1
    assert report['summary']['download_verified_cached'] == 0
    assert report['summary']['processed_verified_cached'] == 0
    assert report['summary']['required_ready'] is False
    assert report['index']['status'] == 'missing'
    assert not corpus_ready(report)


def test_corpus_versions_and_windows_paths(tmp_path):
    settings, source, _ = _fixture(tmp_path)
    _, _, chunks = _create_verified(settings, source, backslashes=True)
    report = build_corpus_report(settings)
    row = report['sources'][0]
    assert row['download'] == 'verified_cached'
    assert row['processed'] == 'verified_cached'
    assert row['chunks'] == len(chunks)
    assert report['summary']['required_ready']
    assert report['index']['status'] == 'missing'
    assert not corpus_ready(report)


def test_corrupt_raw_and_outdated_processed_detected(tmp_path):
    settings, source, _ = _fixture(tmp_path)
    raw, processed, _ = _create_verified(settings, source)
    raw.write_bytes(raw.read_bytes() + b'changed')
    row = build_corpus_report(settings)['sources'][0]
    assert row['download'] == 'hash_mismatch'
    assert row['processed'] == 'raw_unverified'
    # A separately changed processed version must not be counted as indexed-ready.
    raw.write_bytes(raw.read_bytes()[:-7])
    payload = json.loads(processed.read_text(encoding='utf-8'))
    payload['document_version'] = '0' * 64
    processed.write_text(json.dumps(payload), encoding='utf-8')
    row = build_corpus_report(settings)['sources'][0]
    assert row['download'] == 'verified_cached'
    assert row['processed'] == 'invalid'  # stale envelope cannot match its chunks


def test_consistent_old_processed_version_is_marked_stale(tmp_path):
    settings, source, _ = _fixture(tmp_path)
    _, processed, _ = _create_verified(settings, source)
    payload = json.loads(processed.read_text(encoding='utf-8'))
    old_version = '0' * 64
    payload['document_version'] = old_version
    for number, chunk in enumerate(payload['chunks']):
        chunk['document_version'] = old_version
        chunk['chunk_id'] = sha256(f'{source.id}:{old_version}:{number}'.encode()).hexdigest()[:32]
    processed.write_text(json.dumps(payload), encoding='utf-8')
    report = build_corpus_report(settings)
    assert report['sources'][0]['processed'] == 'stale'
    assert report['summary']['processed_verified_cached'] == 0
    assert not corpus_ready(report)


def test_missing_raw_never_proves_processed_version(tmp_path):
    settings, source, _ = _fixture(tmp_path)
    raw, _, _ = _create_verified(settings, source)
    raw.unlink()
    report = build_corpus_report(settings)
    assert report['sources'][0]['download'] == 'missing_file'
    assert report['sources'][0]['processed'] == 'raw_unverified'
    assert report['index']['status'] == 'missing'


def test_metadata_is_not_actual_qdrant_verification(tmp_path):
    settings, source, _ = _fixture(tmp_path)
    _, _, chunks = _create_verified(settings, source)
    _index_meta(settings, chunks)
    report = build_corpus_report(settings)
    assert report['index']['status'] == 'metadata_consistent_unverified'
    assert not corpus_ready(report)
    meta_path = settings.data_dir / 'vectorstore' / 'index_meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta['chunk_count'] = 999
    meta_path.write_text(json.dumps(meta), encoding='utf-8')
    report = build_corpus_report(settings)
    assert report['index']['status'] == 'metadata_mismatch'
    assert 'chunk_count' in report['index']['issues']


def test_qdrant_payload_compare_detects_same_count_different_chunks():
    chunk = {'chunk_id': 'a', 'document_id': 'doc', 'document_version': 'v', 'text': 'truth'}
    class Client:
        def __init__(self, payload):
            self.payload = payload
        def scroll(self, **kwargs):
            assert kwargs['with_vectors'] is False
            return [SimpleNamespace(payload=self.payload)], None
    assert compare_index_payloads(Client(chunk), 'test', [chunk])['status'] == 'payloads_match'
    assert compare_index_payloads(Client({**chunk, 'text': 'changed'}), 'test', [chunk])['status'] == 'mismatch'
    assert compare_index_payloads(Client({**chunk, 'chunk_id': 'other'}), 'test', [chunk])['status'] == 'mismatch'


def test_explicit_qdrant_mode_and_strict(tmp_path, monkeypatch):
    settings, source, _ = _fixture(tmp_path)
    _, _, chunks = _create_verified(settings, source)
    _index_meta(settings, chunks)
    folder = settings.data_dir / 'vectorstore' / 'qdrant'
    folder.mkdir(parents=True)
    (folder / 'local.db').write_bytes(b'stub')
    class Client:
        def __init__(self, **kwargs):
            self.closed = False
        def collection_exists(self, name):
            return True
        def scroll(self, **kwargs):
            return [SimpleNamespace(payload=c) for c in chunks], None
        def close(self):
            self.closed = True
    monkeypatch.setitem(sys.modules, 'qdrant_client', SimpleNamespace(QdrantClient=Client))
    report = build_corpus_report(settings, verify_qdrant=True)
    assert report['index']['status'] == 'payloads_match'
    assert corpus_ready(report)


def test_stale_or_missing_qdrant_does_not_get_auto_repaired(tmp_path):
    settings, source, _ = _fixture(tmp_path)
    _create_verified(settings, source)
    report = build_corpus_report(settings, verify_qdrant=True)
    assert report['index']['status'] == 'missing'
    assert not (settings.data_dir / 'vectorstore' / 'qdrant').exists()


def test_original_zip_contains_no_cache_and_status_is_read_only(tmp_path):
    settings, _, _ = _fixture(tmp_path)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    build_corpus_report(settings, verify_qdrant=False)
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    assert before == after
