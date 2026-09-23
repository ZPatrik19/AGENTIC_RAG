"""Retired optional source, no invented gold, immutable local snapshots, safe metadata sync."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dap_assistant.documents.corpus_status import synchronize_retired_source_metadata
from dap_assistant.evaluation.corpus_quality import extraction_report, golden_diagnostics
from dap_assistant.evaluation.dataset import DATASET, load_dataset
from dap_assistant.evaluation.index_health import corpus_fingerprint
from dap_assistant.documents.ingestion import make_chunks
from dap_assistant.settings import Settings
from dap_assistant.documents.sources import Source, load_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_removed_source_not_in_active_manifest_or_gold():
    manifest = load_manifest(ROOT / 'config/document_sources.yaml')
    ids = {src.id for src in manifest.sources}
    assert 'nyilvantarto-jszp' not in ids
    assert len(ids) == 26
    assert len([s for s in manifest.sources if s.processing_status != 'disabled']) == 24
    assert sum(s.required for s in manifest.sources) == 4
    for case in load_dataset(DATASET):
        assert not set(case['expected_source_ids']) - ids
        assert all(s in ids and next(x for x in manifest.sources if x.id == s).processing_status != 'disabled'
                   for s in case['expected_source_ids'])
        for fact in case['expected_facts']:
            assert set(fact.get('source_ids', [])) <= ids
    assert sum(c['human_reviewed'] for c in load_dataset(DATASET)) == 0


def _local_fixture(tmp_path):
    (tmp_path / 'config').mkdir()
    src = Source(id='dap-vehicle-buyer', title='DÁP vevői tájékoztató',
                 url='https://dap.gov.hu/vehicle', domain='vehicle', type='html',
                 destination='vehicle', required=True)
    (tmp_path / 'config' / 'document_sources.yaml').write_text(
        yaml.safe_dump({'sources': [src.model_dump()]}), encoding='utf8')
    settings = replace(Settings(), root=tmp_path, data_dir=tmp_path / 'data')
    raw = ('<html><main><h1>Autó adásvétel</h1><p>' +
           'A gépjármű ügyintézéshez dokumentumokat kell előkészíteni. ' * 10 +
           '</p></main></html>').encode('utf8')
    digest = sha256(raw).hexdigest()
    raw_file = settings.data_dir / 'raw' / 'vehicle' / (src.id + '.html')
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(raw)
    meta = {'id': src.id, 'file': f'raw/vehicle/{src.id}.html', 'type': 'html',
            'url': src.url, 'domain': src.domain, 'sha256': digest,
            'retrieved_at': '2026-09-21T08:00:00+02:00'}
    mpath = settings.data_dir / 'interim' / 'downloads' / (src.id + '.json')
    mpath.parent.mkdir(parents=True)
    mpath.write_text(json.dumps(meta), encoding='utf8')
    from dap_assistant.documents.ingestion import parse_html
    chunks = make_chunks(src, meta, parse_html(raw))
    out = settings.data_dir / 'processed' / 'vehicle' / (src.id + '.json')
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({'document_id': src.id, 'document_version': digest,
                               'chunks': chunks}), encoding='utf8')
    index_path = settings.data_dir / 'vectorstore' / 'index_meta.json'
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps({
        'active_source_ids': [src.id, 'nyilvantarto-jszp'],
        'embedding_model': settings.embedding_model,
        'embedding_provider': settings.embedding_provider,
        'embedding_text_template': 'title / section_path + text (v2)',
        'embedding_dimension': 384,
        'document_versions': {src.id: digest}, 'chunk_count': len(chunks),
        'chunk_content_sha256': corpus_fingerprint(chunks),
    }), encoding='utf8')
    class Client:
        def scroll(self, **kwargs):
            return [SimpleNamespace(payload=dict(c)) for c in chunks], None
    return settings, chunks, index_path, Client()


def test_metadata_only_migration_preserves_chunks_and_index(tmp_path):
    settings, chunks, path, client = _local_fixture(tmp_path)
    old_chunks = (settings.data_dir / 'processed' / 'vehicle' / 'dap-vehicle-buyer.json').read_bytes()
    result = synchronize_retired_source_metadata(settings, client)
    assert result['status'] == 'metadata_synchronized'
    assert result['vectors_recomputed'] is False
    assert result['points_verified'] == len(chunks)
    assert json.loads(path.read_text())['active_source_ids'] == ['dap-vehicle-buyer']
    assert synchronize_retired_source_metadata(settings, client)['status'] == 'already_current'
    assert (settings.data_dir / 'processed' / 'vehicle' / 'dap-vehicle-buyer.json').read_bytes() == old_chunks


def test_migration_refuses_changed_payload(tmp_path):
    settings, chunks, path, client = _local_fixture(tmp_path)
    original = path.read_bytes()
    class Altered:
        def scroll(self, **kwargs):
            return [SimpleNamespace(payload={**c, 'text': 'WRONG'}) for c in chunks], None
    with pytest.raises(ValueError, match='Qdrant payloads differ'):
        synchronize_retired_source_metadata(settings, Altered())
    assert path.read_bytes() == original


def test_real_extraction_roundtrip_and_no_gold_fabrication(tmp_path, monkeypatch):
    import dap_assistant.evaluation.corpus_quality as quality
    settings, chunks, _, _ = _local_fixture(tmp_path)
    monkeypatch.setattr(quality, 'REVIEWED_DATASET', tmp_path / 'no_human_review.json')
    extracted = extraction_report(settings)
    assert extracted['summary']['reparsed_roundtrip_match'] == 1
    assert extracted['documents'][0]['legal_effect_status'] == 'not_independently_verified'
    report = golden_diagnostics(settings, chunks)
    assert report['question_count'] == 20
    assert report['dense_executed'] is False
    assert report['pinned_reviewed_cases'] == 0
    assert all(c['bm25']['metrics']['recall_at_k'] is None for c in report['cases'])
    assert all(c['bm25']['metrics']['mrr'] is None for c in report['cases'])
    assert all(c['dense']['status'] == 'not_run' for c in report['cases'])


def test_extraction_detects_changed_processed_text(tmp_path):
    settings, chunks, _, _ = _local_fixture(tmp_path)
    path = settings.data_dir / 'processed' / 'vehicle' / 'dap-vehicle-buyer.json'
    document = json.loads(path.read_text())
    document['chunks'][0]['text'] = 'megváltoztatott, de nem eredeti szöveg'
    path.write_text(json.dumps(document), encoding='utf8')
    report = extraction_report(settings)
    assert report['documents'][0]['extraction_status'] == 'roundtrip_mismatch'


def test_hybrid_20_cases_requires_real_gold_even_with_fake_dense(tmp_path, monkeypatch):
    import dap_assistant.evaluation.corpus_quality as quality
    settings, chunks, _, _ = _local_fixture(tmp_path)
    monkeypatch.setattr(quality, 'REVIEWED_DATASET', tmp_path / 'missing.json')
    class Dense:
        def search(self, query, domain, *, limit=12, **kwargs):
            return [(chunks[0]['chunk_id'], 0.88)] if domain == 'vehicle' else []
    report = golden_diagnostics(settings, chunks, dense=Dense())
    assert report['dense_executed']
    assert report['pinned_reviewed_cases'] == 0
    assert all(c['hybrid']['metrics']['recall_at_k'] is None for c in report['cases'])
    assert all(c['dense']['metrics']['mrr'] is None for c in report['cases'])
    assert report['cases'][0]['hybrid']['results'][0]['chunk_id'] == chunks[0]['chunk_id']


def test_embedding_numeric_smoke_cannot_be_mistaken_for_semantic_quality():
    from dap_assistant.evaluation.corpus_quality import embedding_runtime_sanity
    class Embeddings:
        dimension = 2
        def query(self, text):
            return [1.0, 0.0] if 'autó' in text else [0.0, 1.0]
        def passages(self, texts):
            return [[0.6, 0.8]]
    result = embedding_runtime_sanity(Embeddings(), [{'text': 'Autó dokumentum'}])
    assert result['status'] == 'numeric_smoke_pass'
    assert 'NOT' not in result['status']
    assert 'Does not establish' in result['limitations']
    class WrongEmbeddings(Embeddings):
        def passages(self, texts):
            return [[float('nan'), 0.0]]
    assert embedding_runtime_sanity(WrongEmbeddings(), [{'text': 'Autó'}])['status'] == 'numeric_smoke_fail'
