"""Offline extraction, provenance and golden-question diagnostics.

No model inference, external requests, automatic gold labels or source-date guesses.
Optional dense rankings are measured only when a real indexed model is available.
"""
from __future__ import annotations

from collections import Counter
import json
import re

from dap_assistant.documents.corpus_status import _processed_status, _raw_status
from dap_assistant.documents.ingestion import make_chunks, parse_html, parse_pdf, select_legal_sections
from dap_assistant.rag.retrieval import bm25_search, hybrid_search
from ..settings import Settings
from dap_assistant.documents.sources import load_manifest
from .dataset import DATASET, REVIEWED_DATASET, index_versions, load_dataset, reference_status, validate_relevant_chunks
from .metrics import retrieval_metrics


def extraction_report(settings: Settings) -> dict:
    """Reparse verified raw bytes and compare ALL derived chunk texts and IDs.

    Exact roundtrip establishes deterministic extraction consistency, not that the
    external page is complete, readable, current or legally applicable.
    """
    rows = []
    for source in load_manifest(settings.manifest).sources:
        if source.processing_status == 'disabled':
            continue
        raw_status, _, digest = _raw_status(source, settings.data_dir)
        processed_status, _, stored = _processed_status(source, settings.data_dir, digest)
        row = {'document_id': source.id, 'source_url': source.url,
               'legal_or_period_sensitive': bool(source.legal_sections or '2026' in source.id),
               'raw_status': raw_status,
               'processed_status': processed_status, 'chunks': len(stored),
               'extraction_status': 'not_verified', 'quality_flags': [],
               'legal_effect_status': 'not_independently_verified'}
        if raw_status != 'verified_cached' or processed_status != 'verified_cached':
            row['quality_flags'].append('raw_or_processed_not_verified')
            rows.append(row)
            continue
        try:
            meta = json.loads((settings.data_dir / 'interim' / 'downloads' /
                               f'{source.id}.json').read_text(encoding='utf-8'))
            path = settings.data_dir / 'raw' / source.domain / f'{source.id}.{source.type}'
            body = path.read_bytes()
            sections = parse_pdf(body) if source.type == 'pdf' else parse_html(body)
            if source.legal_sections:
                sections = select_legal_sections(sections, source.legal_sections)
            rebuilt = make_chunks(source, meta, sections)
            keys = ('chunk_id', 'text', 'retrieval_text', 'section_path', 'page_number',
                    'document_version', 'source_url')
            parity = (len(rebuilt) == len(stored) and all(
                all(a.get(key) == b.get(key) for key in keys)
                for a, b in zip(rebuilt, stored)))
            row['extraction_status'] = 'roundtrip_match' if parity else 'roundtrip_mismatch'
            row['downloaded_at_not_effective_date'] = meta.get('retrieved_at')
            row['published_at'] = stored[0].get('published_at') if stored else None
            row['effective_from'] = stored[0].get('effective_from') if stored else None
            row['effective_to'] = stored[0].get('effective_to') if stored else None
            row['extracted_characters'] = sum(len(sec['text']) for sec in sections)
            row['page_count_extracted'] = len({sec['page'] for sec in sections
                                                if sec.get('page') is not None}) if source.type == 'pdf' else None
            row['chunk_max_chars'] = max((len(c['text']) for c in stored), default=0)
            row['replacement_character_count'] = sum(c['text'].count('\ufffd') for c in stored)
            if not parity:
                row['quality_flags'].append('stored_chunks_differ_from_current_parser')
            if row['replacement_character_count']:
                row['quality_flags'].append('unicode_replacement_characters')
            if row['extracted_characters'] < 150:
                row['quality_flags'].append('very_short_extraction_manual_review')
            if source.type == 'pdf' and not row['page_count_extracted']:
                row['quality_flags'].append('pdf_page_provenance_missing')
            if any(re.search(r'<(?:script|style|div|html)\b', c['text'], re.I) for c in stored):
                row['quality_flags'].append('possible_html_markup_leak')
            if any(not c.get('section_path') for c in stored):
                row['quality_flags'].append('missing_section_heading_review')
            if source.legal_sections:
                found = {path[-1].replace('. §', '') for path in
                         (c.get('section_path', []) for c in stored) if path}
                row['requested_legal_sections'] = list(source.legal_sections)
                row['missing_legal_sections'] = sorted(set(source.legal_sections) - found)
                if row['missing_legal_sections']:
                    row['quality_flags'].append('requested_legal_sections_missing_review')
            if row['legal_or_period_sensitive'] and not all(c.get('effective_from') or c.get('effective_to')
                                                 for c in stored):
                row['quality_flags'].append('legal_effective_dates_unverified')
            if not all(c.get('published_at') for c in stored):
                row['quality_flags'].append('publication_date_unverified')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row['extraction_status'] = 'reparse_error'
            row['quality_flags'].append(f'reparse_error:{type(exc).__name__}')
        rows.append(row)
    return {'documents': rows, 'summary': {
        'reparsed_roundtrip_match': sum(r['extraction_status'] == 'roundtrip_match' for r in rows),
        'reparsed_roundtrip_mismatch': sum(r['extraction_status'] == 'roundtrip_mismatch' for r in rows),
        'unverified_or_missing': sum(r['extraction_status'] == 'not_verified' for r in rows),
        'quality_flags': dict(Counter(flag for row in rows for flag in row['quality_flags'])),
    }, 'limitations': ('Exact parse/chunk roundtrip does not establish that an official HTML page '
                       'is complete or a provision applies to a particular event date. '
                       'Date fields are not inferred from retrieval timestamps.')}


def _preview(ranking: list[tuple[str, float]], by_id: dict[str, dict], k: int) -> list[dict]:
    return [{'rank': rank, 'chunk_id': cid, 'document_id': by_id[cid]['document_id'],
             'title': by_id[cid].get('title'), 'document_version': by_id[cid]['document_version'],
             'score': score, 'text_excerpt': by_id[cid].get('text', '')[:350]}
            for rank, (cid, score) in enumerate(ranking[:k], 1) if cid in by_id]


def golden_diagnostics(settings: Settings, chunks: list[dict], *, dense=None, k: int = 5) -> dict:
    """Diagnose actual BM25/dense rankings without fabricating Recall or MRR absent approved labels."""
    if k < 1:
        raise ValueError('k must be >= 1')
    if not chunks:
        raise ValueError('Missing local processed corpus; no synthetic documents are substituted')
    cases = load_dataset(DATASET)
    from dap_assistant.documents.sources import load_manifest
    manifest = {src.id: src for src in load_manifest(settings.manifest).sources}
    by_id = {c['chunk_id']: c for c in chunks}
    if len(by_id) != len(chunks):
        raise ValueError('Duplicate chunk IDs in local processed corpus')
    versions = index_versions(chunks)
    reviewed = {c['question_id']: c for c in load_dataset(REVIEWED_DATASET)} if REVIEWED_DATASET.exists() else {}
    rows = []
    for case in cases:
        domain = case['category']
        hint_ids = case['expected_source_ids']
        active_hints = [sid for sid in hint_ids if sid in manifest
                        and manifest[sid].processing_status != 'disabled']
        issues = []
        if len(active_hints) != len(hint_ids):
            issues.append('inactive_or_unknown_candidate_sources')
        missing = sorted(set(active_hints) - set(versions))
        if missing:
            issues.append('candidate_sources_not_indexed')
        lexical = bm25_search(case['question'], chunks, domain, limit=max(12, k))
        semantic = dense.search(case['question'], domain, limit=max(12, k)) if dense else None
        human_case = reviewed.get(case['question_id'])
        pinned = bool(human_case and reference_status(human_case, versions) == 'pinned'
                      and validate_relevant_chunks(human_case, chunks))
        scored = human_case['relevant_chunk_ids'] if pinned else None
        fused = hybrid_search(case['question'], domain, settings, limit=k, dense=dense) if dense else None
        lexical_metrics = retrieval_metrics([cid for cid, _ in lexical], scored, k=k)
        semantic_metrics = (retrieval_metrics([cid for cid, _ in semantic], scored, k=k)
                            if semantic is not None else None)
        rows.append({
            'question_id': case['question_id'], 'question': case['question'],
            'candidate_source_ids_not_gold': hint_ids,
            'missing_indexed_candidate_sources': missing,
            'review_status': 'pinned' if pinned else (
                reference_status(human_case, versions) if human_case else 'awaiting_manual_review'),
            'human_reviewed_and_version_pinned': pinned, 'issues': issues,
            'bm25': {'results': _preview(lexical, by_id, k), 'metrics': lexical_metrics},
            'dense': ({'results': _preview(semantic, by_id, k), 'metrics': semantic_metrics}
                      if semantic is not None else {'status': 'not_run'}),
            'hybrid': ({'results': [{'rank': pos, 'chunk_id': item['chunk_id'],
                                    'document_id': item['document_id']}
                                   for pos, item in enumerate(fused, 1)],
                        'metrics': retrieval_metrics([item['chunk_id'] for item in fused], scored, k=k)}
                       if fused is not None else {'status': 'not_run'}),
        })
    return {'question_count': len(rows), 'indexed_documents': len(versions),
            'indexed_chunks': len(chunks), 'dense_executed': dense is not None,
            'pinned_reviewed_cases': sum(r['human_reviewed_and_version_pinned'] for r in rows),
            'cases': rows,
            'note': ('Candidate source matches are NOT recall or MRR. Only explicitly human-reviewed '
                     'and source-version-pinned chunk labels enable metrics. '
                     'BM25/dense rankings are produced from the full same-domain corpus.')}


def embedding_runtime_sanity(embeddings, chunks: list[dict]) -> dict:
    """Numerical smoke test for REAL Hungarian E5 embeddings; NOT semantic quality."""
    from math import isfinite, sqrt

    if not chunks:
        raise ValueError('No passages available for embedding smoke test')
    queries = ('Milyen dokumentum kell használt autó átírásához?',
               'Hogyan igényelhetek álláskeresési járadékot?')
    vectors = [embeddings.query(query) for query in queries]
    passages = embeddings.passages([chunks[0].get('retrieval_text') or chunks[0]['text']])
    dimension = embeddings.dimension
    all_vectors = vectors + passages
    norms = [sqrt(sum(float(value) ** 2 for value in vector)) for vector in all_vectors]
    valid = (all(len(vector) == dimension and dimension > 0 for vector in all_vectors)
             and all(all(isfinite(value) for value in vector) for vector in all_vectors)
             and all(0.95 <= norm <= 1.05 for norm in norms)
             and vectors[0] != vectors[1])
    return {'status': 'numeric_smoke_pass' if valid else 'numeric_smoke_fail',
            'embedding_dimension': dimension, 'query_count': len(queries),
            'passage_count': len(passages), 'l2_norms': norms,
            'limitations': 'Does not establish Hungarian semantic retrieval relevance or legal accuracy'}
