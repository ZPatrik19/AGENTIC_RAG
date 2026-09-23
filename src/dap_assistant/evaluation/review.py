"""Explicit human review of local document/chunk gold labels.

Suggestions are NOT gold. A reviewer must inspect the actual chunks and
explicitly approve each case and any answer patterns before scoring them.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re

from .dataset import DATASET, REVIEWED_DATASET, index_versions, load_dataset

from dap_assistant.rag.retrieval import bm25_search


def candidates(case: dict, chunks: list[dict], limit: int = 8) -> list[dict]:
    expected = set(case['expected_source_ids'])
    by_id = {c['chunk_id']: c for c in chunks if c['document_id'] in expected}
    if not by_id:
        return []
    ranks: dict[str, float] = {}
    for fact in case.get('expected_facts', []):
        query = fact.get('statement', '')
        for domain in case['expected_domains']:
            for pos, (chunk_id, _) in enumerate(bm25_search(query, list(by_id.values()), domain,
                                                               limit=limit), 1):
                ranks[chunk_id] = ranks.get(chunk_id, 0.0) + 1 / (60 + pos)
    ids = sorted(ranks, key=ranks.get, reverse=True)[:limit]
    return [by_id[chunk_id] for chunk_id in ids]


def review_case(case_id: str, *, selected_chunks: list[str], answer_patterns: dict[str, str],
                confirm_facts: bool, confirm_chunks: bool, chunks: list[dict],
                destination: Path = REVIEWED_DATASET,
                task_chunk_ids: dict[str, list[str]] | None = None,
                additional_source_ids: list[str] | None = None) -> dict:
    """Record explicit review; never infer review from selected candidates."""
    if not (confirm_facts and confirm_chunks):
        raise ValueError('A dokumentumokat és a referenciaállításokat külön jóvá kell hagyni.')
    if not selected_chunks:
        raise ValueError('Legalább egy releváns, ellenőrzött chunkazonosító szükséges.')
    if len(set(selected_chunks)) != len(selected_chunks):
        raise ValueError('A kiválasztott chunkazonosítók nem ismétlődhetnek.')
    lookup = {c['chunk_id']: c for c in chunks}
    if len(lookup) != len(chunks):
        raise ValueError('Nem egyedi chunkazonosítók az indexben; előbb javítsd az indexelést.')
    source_cases = load_dataset(DATASET)
    original = next((c for c in source_cases if c['question_id'] == case_id), None)
    if original is None:
        raise ValueError('Ismeretlen golden kérdésazonosító.')
    # New official documents can be added only by a human reviewer, not by
    # the retriever suggesting them. The 20-case baseline remains immutable.
    extra = set(additional_source_ids or [])
    from dap_assistant.documents.sources import ALLOWED_HOSTS, load_manifest
    from ..settings import Settings
    from urllib.parse import urlparse
    known = {source.id: source for source in load_manifest(Settings().manifest).sources}
    if any(doc_id not in known or known[doc_id].domain not in original['expected_domains']
           or known[doc_id].processing_status == 'disabled' for doc_id in extra):
        raise ValueError('A kiegészítő referenciaforrás nincs az adott élethelyzet hivatalos manifestjében.')
    if any(not any(lookup[cid]['document_id'] == doc_id for cid in selected_chunks if cid in lookup)
           for doc_id in extra):
        raise ValueError('Minden új dokumentumból ki kell választani egy ellenőrzött chunkot.')
    candidate_sources = set(original['expected_source_ids']) | extra
    if any(cid not in lookup or lookup[cid]['document_id'] not in candidate_sources
           or (lookup[cid]['document_id'] not in known
               or known[lookup[cid]['document_id']].processing_status == 'disabled')
           or urlparse(lookup[cid].get('source_url', '')).hostname not in ALLOWED_HOSTS
           for cid in selected_chunks):
        raise ValueError('A kiválasztott chunk nem tartozik az ellenőrzött forrásokhoz.')
    # A baseline source list is a review candidate set, not automatically gold.
    # Pin only sources for which the reviewer selected at least one concrete chunk.
    allowed = {lookup[cid]['document_id'] for cid in selected_chunks}
    if len(original['expected_subtasks']) > 1 and task_chunk_ids is None:
        raise ValueError('Több részfeladatnál kötelező a feladatonként ellenőrzött chunkhozzárendelés.')
    if task_chunk_ids is not None:
        task_ids = {f't{i}' for i in range(1, len(original['expected_subtasks']) + 1)}
        if set(task_chunk_ids) != task_ids or any(not ids for ids in task_chunk_ids.values()):
            raise ValueError('Minden referencia-részfeladathoz ellenőrzött chunkokat kell rendelni.')
        if set().union(*(set(ids) for ids in task_chunk_ids.values())) != set(selected_chunks):
            raise ValueError('Minden kiválasztott chunkot legalább egy részfeladathoz hozzá kell rendelni.')
        for task_id, ids in task_chunk_ids.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f'Ismétlődő chunkazonosító a(z) {task_id} részfeladatnál.')
            domain = original['expected_subtasks'][int(task_id[1:]) - 1]['domain']
            if any(cid not in selected_chunks or lookup[cid].get('domain') != domain for cid in ids):
                raise ValueError(f'Hibás vagy más domainhez tartozó referencia: {task_id}')
    for fact in original['expected_facts']:
        source_ids = set(fact.get('source_ids') or ([fact['source_id']] if fact.get('source_id') else []))
        if source_ids and not any(lookup[cid]['document_id'] in source_ids for cid in selected_chunks):
            raise ValueError(f"A(z) {fact['fact_id']} állítás forrásához nem jelöltél releváns chunkot.")
    expected_fact_ids = {f['fact_id'] for f in original['expected_facts']}
    if set(answer_patterns) != expected_fact_ids:
        raise ValueError('Minden referenciaállításhoz külön válaszminta szükséges.')
    for pattern in answer_patterns.values():
        if not pattern.strip():
            raise ValueError('Üres válaszminta nem fogadható el.')
        if len(pattern) > 180:
            raise ValueError('A válaszminta legfeljebb 180 karakteres lehet.')
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f'Hibás válaszminta: {exc}') from exc
    # Review approvals refer to exact currently indexed document versions.
    versions = index_versions(chunks)
    if not allowed <= versions.keys() or any(not versions.get(doc) for doc in allowed):
        raise ValueError('Hiányzik a dokumentum vagy a verzióazonosító az indexből.')
    payload = (json.loads(destination.read_text(encoding='utf-8')) if destination.exists()
               else json.loads(DATASET.read_text(encoding='utf-8')))
    if payload.get('version') != '4.0':
        raise ValueError('Nem kompatibilis golden dataset verzió.')
    entry = next(c for c in payload['cases'] if c['question_id'] == case_id)
    # Always start from the unmodified baseline; stale previous reviews cannot survive.
    entry.update(deepcopy(original))
    entry['expected_source_ids'] = sorted(allowed)
    entry['expected_source_versions'] = {doc: versions[doc] for doc in sorted(allowed)}
    entry['relevant_chunk_ids'] = list(dict.fromkeys(selected_chunks))
    if task_chunk_ids is not None:
        entry['relevant_chunk_ids_by_task'] = {key: list(dict.fromkeys(value))
                                               for key, value in task_chunk_ids.items()}
    entry['human_reviewed'] = True
    entry['reference_status'] = 'human_reviewed_pinned'
    entry['reviewed_at_utc'] = datetime.now(timezone.utc).isoformat()
    for fact in entry['expected_facts']:
        fact['answer_pattern'] = answer_patterns[fact['fact_id']]
    payload['reference_policy'] = ('Human-selected chunk IDs and approved source facts; '
                                   'each source version must match the current index.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, destination)
    return entry


def prepare_review_report(cases: list[dict], chunks: list[dict], *, limit: int = 12) -> dict:
    """Prepare inspectable *candidates*, never mark suggested snippets as human gold.

    This output is local and contains complete source text. The annotator must
    read the original source, verify the facts, and approve labels separately.
    """
    if not chunks:
        raise ValueError('Nincs feldolgozott dokumentum. Előbb készítsd el a helyi korpuszt.')
    if limit < 1:
        raise ValueError('A jelöltek maximális száma legalább 1 legyen.')
    from dap_assistant.documents.sources import load_manifest
    from ..settings import Settings

    manifest = {s.id: s for s in load_manifest(Settings().manifest).sources}
    versions = index_versions(chunks)
    if any(not version for version in versions.values()):
        raise ValueError('Hiányzó dokumentumverzió; előbb dolgozd fel újra az érintett forrást.')
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Nem egyedi chunkazonosítók; az index nem alkalmas referenciaellenőrzésre.')
    rows = []
    for case in cases:
        hints = case['expected_source_ids']
        eligible_ids = {doc for doc in hints if doc in manifest
                        and manifest[doc].processing_status != 'disabled'}
        eligible_chunks = [c for c in chunks if c['document_id'] in eligible_ids]
        suggested = candidates(case, eligible_chunks, limit=limit)
        rows.append({
            'question_id': case['question_id'],
            'question': case['question'],
            'expected_subtasks': case['expected_subtasks'],
            'reference_facts_to_verify': case['expected_facts'],
            'candidate_source_ids_not_gold': hints,
            'missing_indexed_source_ids': sorted(eligible_ids - set(versions)),
            'disabled_source_ids': sorted(set(hints) - eligible_ids),
            'review_status': 'awaiting_manual_review',
            'human_reviewed': False,
            'candidate_chunks_not_gold': [{
                key: chunk.get(key) for key in (
                    'chunk_id', 'document_id', 'document_version', 'domain',
                    'title', 'source_url', 'page_number', 'section_path', 'text')
            } for chunk in suggested],
        })
    return {
        'version': '4.0',
        'reference_policy': ('Candidate chunks are NOT retrieval gold. A human must '
                             'verify original documents, approve facts and task-level '
                             'chunk IDs in the existing Streamlit review panel.'),
        'indexed_document_versions': versions,
        'cases': rows,
    }
