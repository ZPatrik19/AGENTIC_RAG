"""Index metadata invariants shared by corpus inventory and evaluation.

Read-only checks; never infer index parity from metadata presence alone.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

from dap_assistant.settings import Settings


def corpus_fingerprint(chunks: list[dict]) -> str:
    return sha256(json.dumps(
        sorted((c['chunk_id'], c['text']) for c in chunks),
        ensure_ascii=False,
    ).encode('utf-8')).hexdigest()

def index_metadata_path(settings: Settings) -> Path:
    return settings.data_dir / 'vectorstore' / 'index_meta.json'

def metadata_problems(metadata: dict, settings: Settings, chunks: list[dict]) -> list[str]:
    """Check independently verifiable index metadata; do not infer model identity."""
    issues = []
    if metadata.get('embedding_text_template') != 'title / section_path + text (v2)':
        issues.append('embedding_text_template')
    from dap_assistant.documents.sources import load_manifest
    expected_ids = sorted(s.id for s in load_manifest(settings.manifest).sources if s.processing_status != 'disabled')
    if metadata.get('active_source_ids') != expected_ids:
        issues.append('active_source_ids')
    if metadata.get('embedding_model') != settings.embedding_model:
        issues.append('embedding_model')
    if metadata.get('embedding_provider', 'sentence_transformers') != settings.embedding_provider:
        issues.append('embedding_provider')
    if metadata.get('chunk_count') != len(chunks):
        issues.append('chunk_count')
    if metadata.get('chunk_content_sha256') != corpus_fingerprint(chunks):
        issues.append('chunk_content_sha256')
    versions = {c['document_id']: c['document_version'] for c in chunks}
    if metadata.get('document_versions') != versions:
        issues.append('document_versions')
    return issues
