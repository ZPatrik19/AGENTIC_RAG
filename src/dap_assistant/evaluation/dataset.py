"""Versioned 20-case evaluation dataset and reviewed-reference handling."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

DATASET = Path(__file__).resolve().parents[3] / 'evaluation' / 'golden_v4.json'
REVIEWED_DATASET = DATASET.with_name('golden_reviewed_v4.json')
AUTO_DATASET = DATASET.with_name('golden_auto_v4.json')

REQUIRED = {
    'question_id', 'question', 'expected_domains', 'expected_intents',
    'expected_subtasks', 'expected_source_ids', 'expected_facts',
    'expected_tool_calls', 'expected_behavior', 'reference_answer', 'reference_date',
    'answerability', 'difficulty', 'category',
}
DOMAINS = {'vehicle', 'employment'}


def active_dataset() -> Path:
    # Never overwrite human gold; the automatic proxy is an independent dataset.
    if AUTO_DATASET.exists():
        return AUTO_DATASET
    return REVIEWED_DATASET if REVIEWED_DATASET.exists() else DATASET


def load_dataset(path: Path | None = None) -> list[dict]:
    path = path or active_dataset()
    payload = json.loads(path.read_text(encoding='utf-8'))
    cases = payload['cases']
    if payload.get('version') != '4.0' or len(cases) != 20:
        raise ValueError('Golden v4 requires precisely 20 cases and version 4.0')
    ids = [c['question_id'] for c in cases]
    if len(set(ids)) != 20:
        raise ValueError('Golden v4 requires 20 unique question IDs')
    if sum(i.startswith('AUTO_') for i in ids) != 10 or sum(i.startswith('WORK_') for i in ids) != 10:
        raise ValueError('Golden v4 requires exactly 10 AUTO and 10 WORK cases')
    for case in cases:
        missing = REQUIRED - case.keys()
        if missing or not set(case['expected_domains']) <= DOMAINS:
            raise ValueError(f'Invalid golden case {case.get("question_id")}: {missing}')
        if case['category'] not in DOMAINS:
            raise ValueError(f'Invalid category in {case["question_id"]}')
        if case['answerability'] not in ('answerable', 'partial', 'insufficient'):
            raise ValueError(f'Invalid answerability in {case["question_id"]}')
        if case.get('human_reviewed') and not case.get('expected_source_versions'):
            raise ValueError('Human-reviewed cases must pin source versions')
    return cases


def index_versions(chunks: list[dict]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for chunk in chunks:
        doc_id, version = chunk['document_id'], chunk.get('document_version', '')
        if doc_id in versions and versions[doc_id] != version:
            raise ValueError(f'Mixed document versions for {doc_id}')
        versions[doc_id] = version
    return versions


def reference_status(case: dict, versions: dict[str, str]) -> str:
    if case.get('reference_status') == 'synthetic':
        return 'synthetic'
    if case.get('automatic_reference') and case.get('human_reviewed'):
        return 'invalid_reference_type'
    if case.get('automatic_reference') and case.get('reference_status') != 'automatic_proxy_pinned':
        return 'automatic_proxy_incomplete'
    expected = case.get('expected_source_versions', {})
    if not expected:
        return 'unversioned_reference'
    if (set(expected) != set(case.get('expected_source_ids', []))
            or any(not isinstance(version, str) or not version for version in expected.values())):
        return 'incomplete_source_versions'
    if any(versions.get(doc_id) != version for doc_id, version in expected.items()):
        return 'version_mismatch'
    if not case.get('human_reviewed') and not case.get('automatic_reference'):
        return 'unreviewed_reference'
    relevant = case.get('relevant_chunk_ids')
    if not relevant or len(relevant) != len(set(relevant)):
        return 'pinned_source_only'
    # A global gold list is not enough for several retrieval branches:
    # each independently ranked branch needs its own reviewed gold labels.
    subtasks = case.get('expected_subtasks') or []
    task_gold = case.get('relevant_chunk_ids_by_task') or {}
    if task_gold or len(subtasks) > 1:
        required = {f't{i}' for i in range(1, len(subtasks) + 1)}
        if (set(task_gold) != required or any(not task_gold[key] for key in required)
                or set().union(*(set(ids) for ids in task_gold.values())) != set(relevant)):
            return 'task_mapping_missing'
    return 'automatic_proxy_pinned' if case.get('automatic_reference') else 'pinned'


def validate_relevant_chunks(case: dict, chunks: list[dict]) -> bool:
    wanted = case.get('relevant_chunk_ids')
    if not wanted or len(wanted) != len(set(wanted)):
        return False
    lookup = {c['chunk_id']: c for c in chunks}
    # An ambiguous chunk ID cannot be safely tied to one source/version.
    if len(lookup) != len(chunks):
        return False
    versions = case.get('expected_source_versions') or {}
    task_gold = case.get('relevant_chunk_ids_by_task') or {}
    subtasks = case.get('expected_subtasks') or []
    if task_gold:
        expected_tasks = {f't{i}' for i in range(1, len(subtasks) + 1)}
        if (set(task_gold) != expected_tasks or any(not ids for ids in task_gold.values())
                or set().union(*(set(ids) for ids in task_gold.values())) != set(wanted)):
            return False
        for i, task in enumerate(subtasks, 1):
            if any(lookup.get(cid, {}).get('domain') != task.get('domain')
                   for cid in task_gold[f't{i}']):
                return False
    elif len(subtasks) > 1:
        return False
    hashes = case.get('reference_chunk_sha256') or {}
    if case.get('automatic_reference') and (set(hashes) != set(wanted) or any(
        hashes[cid] != sha256(lookup[cid].get('text', '').encode('utf-8')).hexdigest()
        for cid in wanted if cid in lookup
    )):
        return False
    return all(
        cid in lookup
        and lookup[cid]['document_id'] in case.get('expected_source_ids', [])
        and versions.get(lookup[cid]['document_id']) == lookup[cid].get('document_version')
        for cid in wanted
    )
