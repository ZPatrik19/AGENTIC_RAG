"""Two-domain migration, source robustness and bounded legal retrieval (offline)."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from dap_assistant import cli

from dap_assistant.rag import retrieval
from dap_assistant.evaluation.dataset import DATASET, load_dataset
from dap_assistant.documents.ingestion import load_chunks, make_chunks, parse_html, select_legal_sections
from dap_assistant.settings import Settings
from dap_assistant.documents.sources import load_manifest


def test_source_manifest_is_two_domain_with_four_mandatory_pages():
    manifest = load_manifest(Settings().manifest).sources
    assert len(manifest) >= 15  # expanded official corpus is intentional
    assert set(Counter(s.domain for s in manifest)) == {'vehicle', 'employment'}
    assert {s.id for s in manifest if s.required} == {
        'dap-vehicle-buyer', 'dap-vehicle-seller',
        'dap-employment-overview', 'dap-employment-benefit',
    }
    assert sum(s.type == 'pdf' for s in manifest) >= 1
    assert any(s.url.startswith('https://njt.jog.gov.hu/') and s.legal_sections for s in manifest)
    assert all(s.destination == s.domain for s in manifest)


def test_legal_provisions_span_html_headings_and_exclude_unrequested():
    html = b'''<main>
        <h1>1991 IV</h1><p>24. &sect;</p>
        <p>Relevant 24 paragraph with twenty five words to simulate an official legal rule on unemployment allowances and eligibility.</p>
        <p>25. &sect;</p>
        <p>Relevant 25 paragraph with twenty five words to simulate another official legal rule and record its exact conditions.</p>
        <p>26. &sect;</p>
        <p>Unrelated section with twenty five words to show the scanner excludes all other statutory provisions and text.</p>
    </main>'''
    selected = select_legal_sections(parse_html(html), ['24', '25'])
    assert len(selected) == 2
    assert [s['section'][-1] for s in selected] == ['24. §', '25. §']
    assert 'Relevant 24 paragraph' in selected[0]['text']
    assert 'Relevant 25 paragraph' in selected[1]['text']
    assert all('Unrelated section' not in s['text'] for s in selected)
    with pytest.raises(ValueError, match='None of the selected'):
        select_legal_sections(parse_html(html), ['99'])


def test_hierarchical_chunks_are_role_labeled_versioned_and_bounded():
    src = next(s for s in load_manifest(Settings().manifest).sources if s.id == 'dap-vehicle-buyer')
    text = '\n'.join('Eredetiségvizsgálat és átírás részletes feltételei ' + str(i) for i in range(110))
    sections = [{'section': ['Gépjármű', 'Átírás'], 'text': text, 'page': None}]
    meta = {'sha256': '1' * 64, 'retrieved_at': '2026-09-20T15:00:00Z'}
    first = make_chunks(src, meta, sections)
    assert len(first) > 1
    assert all(len(c['text']) <= 1500 for c in first)
    assert all(c['section_path'] == ['Gépjármű', 'Átírás'] and c['role'] == 'buyer' for c in first)
    assert all(c['source_url'] == src.url and c['document_version'] == meta['sha256'] for c in first)
    assert [c['chunk_id'] for c in first] == [c['chunk_id'] for c in make_chunks(src, meta, sections)]


def test_stale_housing_business_files_are_not_loaded(tmp_path):
    source_manifest = Settings().manifest.read_bytes()
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'document_sources.yaml').write_bytes(source_manifest)
    folder = tmp_path / 'data' / 'processed'
    for domain, document in [('vehicle', 'dap-vehicle-buyer'), ('housing', 'dap-housing-after'),
                             ('business', 'dap-business-sole')]:
        dest = folder / domain
        dest.mkdir(parents=True)
        (dest / f'{document}.json').write_text(json.dumps({
            'chunks': [{'document_id': document, 'domain': domain,
                        'chunk_id': document, 'text': document}]
        }))
    assert [c['document_id'] for c in load_chunks(tmp_path / 'data')] == ['dap-vehicle-buyer']
    assert [c['document_id'] for c in retrieval._chunk_snapshot(tmp_path / 'data')] == ['dap-vehicle-buyer']


def test_optional_download_errors_do_not_abort_if_mandatory_sources_are_indexed(monkeypatch, tmp_path):
    settings = replace(Settings(), data_dir=tmp_path)
    mandatory = {s.id for s in load_manifest(settings.manifest).sources if s.required}
    optional = next(s.id for s in load_manifest(settings.manifest).sources if not s.required)
    monkeypatch.setattr(cli, 'Settings', lambda: settings)
    monkeypatch.setattr(cli, 'download_all', lambda _: [{'id': optional, 'status': 'error', 'error': '404'}])
    monkeypatch.setattr(cli, 'ingest_all', lambda _: [
        *({'id': doc_id, 'status': 'processed', 'chunks': 1} for doc_id in sorted(mandatory)),
        {'id': optional, 'status': 'missing_download'},
    ])
    monkeypatch.setattr(retrieval, 'index_all', lambda _: {'status': 'indexed', 'documents': 4, 'chunks': 4})
    assert cli.rebuild(download=True, index=True) == 0


def test_missing_mandatory_source_must_fail_setup(monkeypatch, tmp_path):
    settings = replace(Settings(), data_dir=tmp_path)
    mandatory = sorted(s.id for s in load_manifest(settings.manifest).sources if s.required)
    monkeypatch.setattr(cli, 'Settings', lambda: settings)
    monkeypatch.setattr(cli, 'download_all', lambda _: [{'id': mandatory[0], 'status': 'error'}])
    monkeypatch.setattr(cli, 'ingest_all', lambda _: [
        {'id': doc_id, 'status': 'processed', 'chunks': 1} for doc_id in mandatory[1:]
    ])
    monkeypatch.setattr(retrieval, 'index_all', lambda _: {'status': 'indexed', 'chunks': 3})
    assert cli.rebuild(download=True, index=True) != 0


def test_dense_retrieval_keeps_general_legal_documents_for_buyer(monkeypatch):
    from threading import RLock
    from types import ModuleType
    import sys
    http_models = ModuleType('qdrant_client.http.models')
    http_models.FieldCondition = lambda **kw: kw
    http_models.MatchValue = lambda **kw: kw
    http_models.Filter = lambda **kw: kw
    monkeypatch.setitem(sys.modules, 'qdrant_client.http.models', http_models)
    class FakeClient:
        def query_points(self, *, query_filter, **kw):
            assert len(query_filter['must']) == 1  # Domain only, role checked during fusion.
            return SimpleNamespace(points=[SimpleNamespace(payload={'chunk_id': 'nav-rate'}, score=.9)])
    store = retrieval.LocalQdrant.__new__(retrieval.LocalQdrant)
    store._client_lock = RLock()
    store.embeddings = SimpleNamespace(query=lambda q: [0.1, 0.2, 0.3], dimension=3)
    store.client = FakeClient()
    assert store.search('illeték', 'vehicle', role='buyer') == [('nav-rate', .9)]


def test_ollama_8192_and_cpu_embedding_default_match_docker():
    settings = Settings()
    assert settings.ollama_num_ctx == 8192
    assert settings.embedding_device == 'cpu'
    compose = (settings.root / 'docker-compose.yml').read_text()
    assert 'OLLAMA_NUM_CTX: ${OLLAMA_NUM_CTX:-8192}' in compose
    assert 'EMBEDDING_DEVICE: ${EMBEDDING_DEVICE:-cpu}' in compose
    assert 'OLLAMA_NUM_CTX=8192' in (settings.root / '.env.example').read_text()


def test_golden_scoped_and_unreviewed_results_not_fabricated():
    cases = load_dataset(DATASET)
    assert len(cases) == 20
    assert Counter(c['category'] for c in cases) == {'vehicle': 10, 'employment': 10}
    assert not any(c.get('human_reviewed') or c.get('relevant_chunk_ids') for c in cases)
    assert all(set(c['expected_domains']) <= {'vehicle', 'employment'} for c in cases)
