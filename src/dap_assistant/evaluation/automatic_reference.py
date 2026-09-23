"""Build independent SILVER retrieval proxies from the active corpus, not model answers or retriever Top-K."""
from __future__ import annotations


from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import unicodedata
from urllib.parse import urlparse

from dap_assistant.documents.sources import ALLOWED_HOSTS

from . import dataset
from .dataset import index_versions, validate_relevant_chunks

AUTO_DATASET = dataset.DATASET.with_name('golden_auto_v4.json')
# No model/API calls. Tokens here are solely evidence-discovery signals, NOT
# automatic semantic/entailment verification.
_STOP = frozenset('a az és vagy hogy amely milyen mikor hogyan mennyi melyik milyen kell lehet van vannak egy ezt azon annak erre arra illetve is de akkor azzal előtt után során esetén közül magyarországon magyarország'.split())
_WORD = re.compile(r'\w+', re.UNICODE)


def _tokens(value: object) -> set[str]:
    normalized = unicodedata.normalize('NFKD', str(value).casefold())
    ascii_text = ''.join(c for c in normalized if not unicodedata.combining(c))
    return {token for token in _WORD.findall(ascii_text) if (len(token) >= 3 or token.isdigit()) and token not in _STOP}


def _matching_tokens(statement: str, text: str) -> tuple[int, int]:
    words = _tokens(statement)
    text_words = _tokens(text)
    # A constrained prefix allows Hungarian inflection, but numeric values must
    # match exactly. The prefix approximation is explicitly *not* gold evidence.
    matches = sum(any((word == other if word.isdigit() else
                       len(word) >= 5 and len(other) >= 5 and word[:5] == other[:5])
                      for other in text_words) for word in words)
    return matches, len(words)


def corpus_fingerprint(chunks: list[dict]) -> str:
    """Bind labels to every indexed chunk's source, version and exact text."""
    rows = sorted((str(c['chunk_id']), str(c.get('document_id') or ''), str(c.get('document_version') or ''),
                   sha256(str(c.get('text') or '').encode('utf-8')).hexdigest()) for c in chunks)
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError('Nem egyedi chunkazonosítók az aktuális korpuszban.')
    return sha256(json.dumps(rows, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def _fact_sources(case: dict, fact: dict) -> set[str]:
    return set(fact.get('source_ids') or ([fact['source_id']] if fact.get('source_id') else [])
               or case.get('expected_source_ids') or [])


def _rank(statement: str, chunks: list[dict], source_ids: set[str], domain: str | None,
          *, max_hits: int = 2) -> list[dict]:
    """Fact-anchored full-corpus lexical candidates, not retriever predictions."""
    matches = []
    for chunk in chunks:
        if chunk.get('document_id') not in source_ids or (domain and chunk.get('domain') != domain):
            continue
        if not chunk.get('text') or urlparse(chunk.get('source_url') or '').hostname not in ALLOWED_HOSTS:
            continue
        # Avoid a wrong deadline/amount being selected based on surrounding
        # nouns alone. This is a necessary condition, NOT entailment proof.
        numeric = {w for w in _tokens(statement) if w.isdigit()}
        if numeric and not numeric <= _tokens(chunk.get('text', '')):
            continue
        hit, total = _matching_tokens(statement, chunk.get('text', ''))
        # Two distinctive lexical matches: a bare source ID does not pass.
        if total < 2 or hit < 2 or hit / total < 0.40:
            continue
        matches.append((hit / total, hit, chunk))
    matches.sort(key=lambda row: (-row[0], -row[1], row[2]['chunk_id']))
    return [{**chunk, '_lexical_overlap': round(score, 4)} for score, _, chunk in matches[:max_hits]]


def create_auto_reference(cases: list[dict], chunks: list[dict], *, baseline_bytes: bytes) -> dict:
    """Always produce an audit entry for every original case; do not invent gaps."""
    if len(cases) != 20 or len({c['question_id'] for c in cases}) != 20:
        raise ValueError('Az automatikus referencia pontosan 20 különálló tesztesetet igényel.')
    if not chunks:
        raise ValueError('Nincs feldolgozott helyi korpusz, így referencia nem készíthető.')
    versions = index_versions(chunks)
    if any(not version for version in versions.values()):
        raise ValueError('Az indexből hiányzik dokumentumverzió; előbb javítsd az indexelést.')
    fingerprint = corpus_fingerprint(chunks)
    corpus_ids = {c['chunk_id'] for c in chunks}
    results: list[dict] = []
    for original in cases:
        case = deepcopy(original)
        all_selected: dict[str, dict] = {}
        evidence_facts = []
        missing_facts = []
        for fact in original.get('expected_facts') or []:
            source_ids = _fact_sources(original, fact)
            suggestions = _rank(fact.get('statement', ''), chunks, source_ids, None)
            if not suggestions:
                missing_facts.append(fact.get('fact_id', 'unknown'))
            for chunk in suggestions:
                all_selected[chunk['chunk_id']] = chunk
            evidence_facts.append({'fact_id': fact.get('fact_id'), 'statement': fact.get('statement'),
                                   'candidate_chunk_ids': [c['chunk_id'] for c in suggestions],
                                   'matching_method': 'lexical_overlap_not_semantic_verification'})
        tasks = original.get('expected_subtasks') or []
        by_task: dict[str, list[str]] = {}
        missing_tasks = []
        for index, task in enumerate(tasks, 1):
            task_id = f't{index}'
            domain = task.get('domain')
            task_query = ' '.join((str(task.get('label') or ''), ' '.join(task.get('keywords') or []),
                                   str(original.get('question') or '')))
            task_sources = set(original.get('expected_source_ids') or [])
            source_candidates = {c['document_id'] for c in chunks
                                 if c.get('domain') == domain and c['document_id'] in task_sources}
            candidates = [chunk for chunk in all_selected.values() if chunk.get('domain') == domain]
            # Only fact-backed chunks can be used as labels. For an uncovered
            # subtask, use a separate source-anchored task-query heuristic,
            # explicitly flagging the absence of fact-level corroboration.
            if not candidates and not original.get('expected_facts'):
                candidates = _rank(task_query, chunks, source_candidates, domain)
                for chunk in candidates:
                    all_selected[chunk['chunk_id']] = chunk
            # Domain alone is not enough for two subtasks in the same domain.
            if len(tasks) > 1 and len({t.get('domain') for t in tasks}) < len(tasks):
                filtered = [chunk for chunk in candidates
                            if _matching_tokens(task_query, chunk.get('text', ''))[0] >= 2]
                candidates = filtered
            by_task[task_id] = sorted({chunk['chunk_id'] for chunk in candidates})
            if not by_task[task_id]:
                missing_tasks.append(task_id)
        if not tasks:
            missing_tasks.append('t1')
        # Do not allow extra unassigned chunks into global labels.
        assigned = set().union(*(set(ids) for ids in by_task.values())) if by_task else set()
        relevant = sorted(assigned)
        selected_sources = sorted({all_selected[cid]['document_id'] for cid in relevant})
        case['expected_source_ids'] = selected_sources
        case['expected_source_versions'] = {doc: versions[doc] for doc in selected_sources}
        case['relevant_chunk_ids'] = relevant
        case['relevant_chunk_ids_by_task'] = by_task
        case['reference_chunk_sha256'] = {
            cid: sha256(all_selected[cid].get('text', '').encode('utf-8')).hexdigest()
            for cid in relevant
        }
        case['human_reviewed'] = False
        case['automatic_reference'] = True
        case['reference_status'] = 'automatic_proxy_pinned' if (
            relevant and not missing_facts and not missing_tasks
        ) else 'automatic_proxy_incomplete'
        case['reference_method'] = 'independent_baseline_fact_to_official_corpus_lexical_proxy'
        case['reference_audit'] = {
            'fact_matches': evidence_facts,
            'missing_fact_ids': missing_facts,
            'missing_task_ids': missing_tasks,
            'unavailable_baseline_sources': sorted(set(original.get('expected_source_ids') or []) - set(versions)),
            'candidate_chunk_ids': relevant,
            'fact_coverage': (len(evidence_facts) - len(missing_facts)) / len(evidence_facts)
                             if evidence_facts else None,
            'method_limits': 'Lexical/source-anchored matching is an automatic silver/proxy label, not human-verified relevance.',
        }
        if case['reference_status'] == 'automatic_proxy_pinned' and (
            not validate_relevant_chunks(case, chunks) or not set(relevant) <= corpus_ids
        ):
            case['reference_status'] = 'automatic_proxy_incomplete'
        results.append(case)
    return {
        'version': '4.0',
        'reference_policy': 'AUTOMATIC SILVER/PROXY: no human review; recall/precision/MRR against approximate, version-pinned independent corpus labels.',
        'reference_type': 'automatic_silver_proxy',
        'human_reviewed': False,
        'baseline_sha256': sha256(baseline_bytes).hexdigest(),
        'corpus_fingerprint': fingerprint,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'reference_ready_cases': sum(c['reference_status'] == 'automatic_proxy_pinned' for c in results),
        'cases': results,
    }


def ensure_auto_reference(data_dir: Path, *, chunks: list[dict] | None = None,
                          baseline_path: Path | None = None, destination: Path | None = None,
                          force: bool = False) -> dict:
    """Idempotent. Never overwrite the human-reviewed dataset or baseline."""
    from dap_assistant.documents.ingestion import load_chunks

    baseline_path = baseline_path or dataset.DATASET
    destination = destination or baseline_path.with_name('golden_auto_v4.json')
    baseline_bytes = baseline_path.read_bytes()  # fail clearly if baseline absent
    chunks = chunks if chunks is not None else load_chunks(data_dir)
    if not chunks:
        raise ValueError('Hiányoznak a feldolgozott dokumentumok (data/processed).')
    fingerprint = corpus_fingerprint(chunks)
    digest = sha256(baseline_bytes).hexdigest()
    reviewed_path = baseline_path.with_name('golden_reviewed_v4.json')
    reviewed_bytes = reviewed_path.read_bytes() if reviewed_path.exists() else b''
    reviewed_digest = sha256(reviewed_bytes).hexdigest() if reviewed_bytes else None
    if destination.exists() and not force:
        try:
            previous = json.loads(destination.read_text(encoding='utf-8'))
            if (previous.get('reference_type') == 'automatic_silver_proxy'
                    and previous.get('baseline_sha256') == digest
                    and previous.get('corpus_fingerprint') == fingerprint
                    and previous.get('human_gold_sha256') == reviewed_digest
                    and len(previous.get('cases') or []) == 20):
                return previous
        except (ValueError, OSError, TypeError):
            pass
    cases = dataset.load_dataset(baseline_path)
    payload = create_auto_reference(cases, chunks, baseline_bytes=baseline_bytes)
    payload['human_gold_sha256'] = reviewed_digest
    # Preserve stronger existing human-approved cases wherever their source
    # versions and exact chunk IDs are still valid. Never relabel them silver.
    if reviewed_bytes:
        versions = index_versions(chunks)
        try:
            reviewed = {c['question_id']: c for c in dataset.load_dataset(reviewed_path)}
        except (ValueError, OSError, KeyError):
            reviewed = {}
        for index, candidate in enumerate(payload['cases']):
            original = reviewed.get(candidate['question_id'])
            if (original and dataset.reference_status(original, versions) == 'pinned'
                    and validate_relevant_chunks(original, chunks)):
                payload['cases'][index] = deepcopy(original)
        human_count = sum(bool(c.get('human_reviewed')) for c in payload['cases'])
        payload['human_gold_ready_cases'] = human_count
        payload['reference_ready_cases'] = human_count + sum(
            c.get('reference_status') == 'automatic_proxy_pinned'
            for c in payload['cases']
        )
        if human_count:
            payload['reference_type'] = 'mixed_human_gold_and_automatic_silver_proxy'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, destination)
    return payload
