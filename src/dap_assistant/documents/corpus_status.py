"""Verify local raw, processed and indexed document versions before claiming corpus availability."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from dap_assistant.documents.index_metadata import metadata_problems
from dap_assistant.settings import Settings
from dap_assistant.documents.sources import Source, load_manifest


def _json_file(path: Path) -> dict:
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('JSON object expected')
    return result


def _raw_status(source: Source, data_dir: Path) -> tuple[str, str, str | None]:
    """Return (status, safe diagnostic, digest); never trust metadata by itself."""
    metadata_path = data_dir / 'interim' / 'downloads' / f'{source.id}.json'
    if not metadata_path.is_file():
        return 'missing', 'No local download metadata', None
    try:
        meta = _json_file(metadata_path)
        expected = Path('raw') / source.domain / f'{source.id}.{source.type}'
        # Never follow an arbitrary path supplied by a downloaded metadata file.
        if (str(meta.get('file', '')).replace('\\', '/') != expected.as_posix() or meta.get('id') != source.id
                or meta.get('url') != source.url or meta.get('domain') != source.domain
                or meta.get('type') != source.type):
            return 'invalid_metadata', 'Download metadata differs from the active manifest', None
        raw_path = data_dir / expected
        if not raw_path.is_file():
            return 'missing_file', 'Download metadata exists but the raw file is missing', None
        content = raw_path.read_bytes()
        digest = sha256(content).hexdigest()
        if digest != meta.get('sha256'):
            return 'hash_mismatch', 'Raw SHA-256 differs from download metadata', None
        if source.type == 'pdf' and not content.lstrip().startswith(b'%PDF-'):
            return 'invalid_format', 'Raw file does not start with PDF signature', None
        if source.type == 'html' and not any(tag in content[:4096].lower()
                                                  for tag in (b'<html', b'<!doctype')):
            return 'invalid_format', 'Raw file does not appear to be HTML', None
        return 'verified_cached', 'Local file and download metadata match; live freshness unverified', digest
    except (ValueError, OSError, UnicodeError) as exc:
        return 'invalid_metadata', f'Cannot inspect raw snapshot: {type(exc).__name__}', None


def _processed_status(source: Source, data_dir: Path, raw_digest: str | None
                      ) -> tuple[str, str, list[dict]]:
    path = data_dir / 'processed' / source.domain / f'{source.id}.json'
    if not path.is_file():
        return 'missing', 'No local processed chunks', []
    try:
        payload = _json_file(path)
        chunks = payload.get('chunks')
        version = payload.get('document_version')
        if (payload.get('document_id') != source.id or not isinstance(version, str)
                or len(version) != 64 or not isinstance(chunks, list) or not chunks):
            return 'invalid', 'Invalid processed-document envelope or empty chunks', []
        ids: set[str] = set()
        for i, chunk in enumerate(chunks):
            expected_id = sha256(f'{source.id}:{version}:{i}'.encode()).hexdigest()[:32]
            if (not isinstance(chunk, dict) or chunk.get('chunk_id') != expected_id
                    or chunk.get('document_id') != source.id
                    or chunk.get('document_version') != version
                    or chunk.get('domain') != source.domain
                    or chunk.get('source_url') != source.url
                    or not isinstance(chunk.get('text'), str) or not chunk['text'].strip()
                    or chunk['chunk_id'] in ids):
                return 'invalid', f'Invalid chunk metadata, ID, or content at index {i}', []
            ids.add(chunk['chunk_id'])
        if raw_digest is None:
            return 'raw_unverified', 'Processed chunks exist but raw snapshot is not verified', []
        if version != raw_digest:
            return 'stale', 'Processed version does not match verified raw SHA-256', []
        return 'verified_cached', 'Chunks match the verified raw document version', chunks
    except (ValueError, OSError, UnicodeError, TypeError) as exc:
        return 'invalid', f'Cannot inspect processed chunks: {type(exc).__name__}', []


def compare_index_payloads(client: Any, collection: str, chunks: list[dict]) -> dict:
    """Compare EVERY Qdrant payload, not just collection size or metadata.

    A test double may implement the same ``scroll`` protocol. No vectors are loaded.
    """
    expected = {c['chunk_id']: c for c in chunks}
    if len(expected) != len(chunks):
        return {'status': 'mismatch', 'reason': 'Duplicate processed chunk IDs'}
    observed: set[str] = set()
    offset = None
    count = 0
    while True:
        points, next_offset = client.scroll(collection_name=collection, limit=256,
                                            offset=offset, with_payload=True,
                                            with_vectors=False)
        for point in points:
            count += 1
            payload = point.payload or {}
            chunk_id = payload.get('chunk_id')
            if chunk_id not in expected or chunk_id in observed or payload != expected[chunk_id]:
                return {'status': 'mismatch', 'reason': 'Unexpected, duplicate, or changed Qdrant payload',
                        'points_examined': count}
            observed.add(chunk_id)
        if next_offset is None:
            break
        if next_offset == offset:
            return {'status': 'mismatch', 'reason': 'Qdrant pagination did not advance'}
        offset = next_offset
    if observed != set(expected):
        return {'status': 'mismatch', 'reason': 'Some processed chunks are missing from Qdrant',
                'points_examined': count}
    return {'status': 'payloads_match', 'points_examined': count,
            'note': 'Payload parity does not independently verify vector values or embedding quality'}


def _index_status(settings: Settings, chunks: list[dict],
                  *, verify_qdrant: bool, processed_complete: bool) -> dict:
    meta_path = settings.data_dir / 'vectorstore' / 'index_meta.json'
    if not meta_path.is_file():
        return {'status': 'missing', 'reason': 'No index metadata; vectors not verified'}
    if not processed_complete or not chunks:
        return {'status': 'blocked', 'reason': 'Processed corpus is incomplete or unverified'}
    try:
        metadata = _json_file(meta_path)
        issues = metadata_problems(metadata, settings, chunks)
        if issues:
            return {'status': 'metadata_mismatch', 'issues': issues}
        if metadata.get('embedding_dimension', 0) <= 0:
            return {'status': 'metadata_mismatch', 'issues': ['embedding_dimension']}
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return {'status': 'invalid_metadata', 'reason': type(exc).__name__}
    if not verify_qdrant:
        return {'status': 'metadata_consistent_unverified',
                'reason': 'Metadata matches processed chunks; actual Qdrant payloads not inspected'}
    directory = settings.data_dir / 'vectorstore' / 'qdrant'
    if not directory.is_dir() or not any(directory.iterdir()):
        return {'status': 'missing', 'reason': 'No local Qdrant files'}
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return {'status': 'unavailable', 'reason': 'qdrant-client is not installed'}
    client = None
    try:
        # Open only by explicit request, with all other Qdrant processes stopped.
        client = QdrantClient(path=str(directory), force_disable_check_same_thread=True)
        collection = 'official_documents'
        if not client.collection_exists(collection):
            return {'status': 'missing', 'reason': f'Missing Qdrant collection: {collection}'}
        return compare_index_payloads(client, collection, chunks)
    except Exception as exc:
        # A running UI or file lock must not be misreported as missing data.
        return {'status': 'unavailable', 'reason': f'Qdrant inspection failed: {type(exc).__name__}: {exc}'}
    finally:
        if client is not None:
            client.close()


def build_corpus_report(settings: Settings, *, verify_qdrant: bool = False) -> dict:
    """Report separately: configured, cached, processed, index metadata, actual index.

    Never call the downloader, embedder or LLM. Paths and stale documents stay intact.
    """
    manifest = load_manifest(settings.manifest)
    rows = []
    verified_chunks: list[dict] = []
    all_active_processed = True
    for source in manifest.sources:
        row = {'id': source.id, 'domain': source.domain, 'type': source.type,
               'required': source.required, 'configured_status': source.processing_status}
        if source.processing_status == 'disabled':
            row.update({'download': 'disabled', 'processed': 'disabled', 'chunks': 0})
        else:
            raw_status, raw_reason, digest = _raw_status(source, settings.data_dir)
            processed_status, processed_reason, chunks = _processed_status(source, settings.data_dir,
                                                                           digest)
            row.update({'download': raw_status, 'download_detail': raw_reason,
                        'processed': processed_status, 'processed_detail': processed_reason,
                        'chunks': len(chunks), 'document_version': digest if chunks else None})
            if processed_status != 'verified_cached':
                all_active_processed = False
            verified_chunks.extend(chunks)
        rows.append(row)
    active_rows = [r for r in rows if r['configured_status'] != 'disabled']
    required_rows = [r for r in rows if r['required']]
    # Optional sources may be unavailable. Index parity is evaluated against
    # verified processed files only when there are no unverified processed files.
    unsafe_processed = any(r['processed'] not in ('missing', 'verified_cached') for r in active_rows)
    ready_for_index_check = bool(verified_chunks) and not unsafe_processed
    index = _index_status(settings, verified_chunks,
                          verify_qdrant=verify_qdrant, processed_complete=ready_for_index_check)
    return {
        'verification_mode': 'qdrant_payloads' if verify_qdrant else 'offline_metadata_only',
        'summary': {
            'configured': len(rows), 'enabled': len(active_rows),
            'configured_by_domain': dict(Counter(r['domain'] for r in rows)),
            'configured_by_type': dict(Counter(r['type'] for r in rows)),
            'disabled': len(rows) - len(active_rows), 'required': len(required_rows),
            'download_verified_cached': sum(r['download'] == 'verified_cached' for r in active_rows),
            'processed_verified_cached': sum(r['processed'] == 'verified_cached' for r in active_rows),
            'verified_processed_chunks': len(verified_chunks),
            'qdrant_verified_documents': (len({c['document_id'] for c in verified_chunks})
                                           if index['status'] == 'payloads_match' else None),
            'qdrant_verified_chunks': (len(verified_chunks)
                                       if index['status'] == 'payloads_match' else None),
            'required_ready': bool(required_rows) and all(
                r['configured_status'] != 'disabled' and r['download'] == 'verified_cached'
                and r['processed'] == 'verified_cached' for r in required_rows),
            'all_enabled_processed': all_active_processed,
            'download_statuses': dict(Counter(r['download'] for r in active_rows)),
            'processed_statuses': dict(Counter(r['processed'] for r in active_rows)),
        },
        'index': index,
        'sources': rows,
        'note': ('A manifest entry is NOT a downloaded document. Cached verification does not '
                 'prove online freshness or legal validity; metadata alone does not prove Qdrant contents.'),
    }


def corpus_ready(report: dict) -> bool:
    """Strict mode requires all mandatory sources and actual index payload parity."""
    return bool(report['summary']['required_ready'] and
                report['index']['status'] == 'payloads_match')


def synchronize_retired_source_metadata(settings: Settings, client: Any,
                                        *, retired_id: str = 'nyilvantarto-jszp') -> dict:
    """Remove only a verified retired source from index metadata; do not change Qdrant vectors."""
    from dap_assistant.documents.sources import load_manifest
    metadata_path = settings.data_dir / 'vectorstore' / 'index_meta.json'
    metadata = _json_file(metadata_path)
    active = sorted(s.id for s in load_manifest(settings.manifest).sources
                    if s.processing_status != 'disabled')
    original = metadata.get('active_source_ids')
    if not isinstance(original, list) or len(original) != len(set(original)):
        raise ValueError('Index metadata source list is invalid')
    if retired_id in active:
        raise ValueError('Retired source is still active in the manifest')
    if original == active:
        return {'status': 'already_current', 'active_sources': len(active)}
    if sorted(set(original) - set(active)) != [retired_id] or not set(active) <= set(original):
        raise ValueError('Metadata differs by more than the single retired source; reindex explicitly')
    report = build_corpus_report(settings, verify_qdrant=False)
    if (not report['summary']['required_ready']
            or report['summary']['processed_verified_cached'] != len(active)
            or report['index'].get('status') != 'metadata_mismatch'
            or report['index'].get('issues') != ['active_source_ids']):
        raise ValueError('Raw/processed corpus or other index metadata is not fully verified')
    from dap_assistant.documents.ingestion import load_chunks
    chunks = load_chunks(settings.data_dir)
    if (len(chunks) != report['summary']['verified_processed_chunks']
            or {c['document_id'] for c in chunks} != set(active)):
        raise ValueError('Loaded chunks differ from the verified corpus')
    parity = compare_index_payloads(client, 'official_documents', chunks)
    if parity['status'] != 'payloads_match':
        raise ValueError('Qdrant payloads differ; migration blocked: ' + parity.get('reason', 'unknown'))
    updated = {**metadata, 'active_source_ids': active}
    temp = metadata_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    import os
    os.replace(temp, metadata_path)
    return {'status': 'metadata_synchronized', 'retired_source': retired_id,
            'active_sources': len(active), 'points_verified': parity['points_examined'],
            'vectors_recomputed': False}
