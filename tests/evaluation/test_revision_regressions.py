"""Focused offline regressions for initialization, legacy-index repair and gold review."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import json

import pytest

from dap_assistant.evaluation.dataset import (
    index_versions, load_dataset, reference_status, validate_relevant_chunks,
)
from dap_assistant.evaluation.index_health import (
    check_index, corpus_fingerprint, index_metadata_path,
)
from dap_assistant.evaluation.review import review_case
from dap_assistant.evaluation.functional_metrics import _full_metrics
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.presentation.runtime_view import available_life_events
from dap_assistant.settings import Settings


class FakeDense:
    COLLECTION = 'official_documents'

    def __init__(self):
        self.embeddings = SimpleNamespace(dimension=3)
        self._client_lock = RLock()
        self.points: dict[str, dict] = {}
        self.index_calls = 0
        self.client = SimpleNamespace(count=lambda **kwargs: SimpleNamespace(count=len(self.points)))

    def index_document(self, chunks):
        self.index_calls += 1
        for chunk in chunks:
            self.points[chunk['chunk_id']] = chunk


def fixture_chunk(*, document='dap-vehicle-buyer', text='Az átírás 15 napon belül szükséges.'):
    return {'chunk_id': 'vehicle-001', 'document_id': document,
            'document_version': 'reviewed-document-sha', 'domain': 'vehicle',
            'text': text, 'title': 'Autót veszek',
            'source_url': 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek'}


def test_domain_labels_once_and_in_stable_order():
    chunks = ([{'domain': 'vehicle'}] * 30 + [{'domain': 'housing'}] * 20
              + [{'domain': 'employment'}] * 13 + [{'domain': 'business'}] * 100)
    assert available_life_events(chunks) == [
        'Autóvásárlás vagy -eladás', 'Munkahely elvesztése',
    ]


def test_navigation_is_explicit_and_workflow_is_prepared_before_chat_submission():
    root = Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant'
    launcher = (root / 'Chatbot.py').read_text(encoding='utf-8')
    page = (root / 'ui.py').read_text(encoding='utf-8')
    assert 'st.navigation(' in launcher
    assert launcher.count('st.Page(') == 3
    assert "'workflow_prepared' not in st.session_state" in page
    assert page.index("if chunks and 'workflow_prepared'") < page.index('question = st.chat_input(')
    assert 'Agentic workflow inicializálása…' not in page


def test_legacy_metadata_is_rebuilt_using_existing_dense_client(tmp_path, monkeypatch):
    from dap_assistant.rag import retrieval
    from dap_assistant.evaluation import index_health

    settings = replace(Settings(), data_dir=tmp_path,
                       embedding_model='intfloat/multilingual-e5-small',
                       embedding_provider='sentence_transformers')
    chunk = fixture_chunk()
    dense = FakeDense()
    monkeypatch.setattr(index_health, 'load_chunks', lambda _: [chunk])
    monkeypatch.setattr(retrieval, 'load_chunks', lambda _: [chunk])
    metadata = check_index(settings, dense, repair_missing=True)
    assert dense.index_calls == 1
    assert metadata['embedding_model'] == settings.embedding_model
    assert metadata['embedding_dimension'] == 3
    assert metadata['chunk_content_sha256'] == corpus_fingerprint([chunk])
    assert metadata['document_versions'] == {'dap-vehicle-buyer': 'reviewed-document-sha'}
    assert index_metadata_path(settings).exists()
    assert check_index(settings, dense, repair_missing=True) == metadata
    assert dense.index_calls == 1  # no repeated embedding/rebuild


def test_existing_metadata_mismatch_is_rejected_not_relabelled(tmp_path, monkeypatch):
    from dap_assistant.evaluation import index_health

    settings = replace(Settings(), data_dir=tmp_path)
    dense = FakeDense()
    chunk = fixture_chunk()
    dense.points[chunk['chunk_id']] = chunk
    monkeypatch.setattr(index_health, 'load_chunks', lambda _: [chunk])
    path = index_metadata_path(settings)
    path.parent.mkdir(parents=True)
    original = {'embedding_model': 'wrong-embedding-model',
                'embedding_dimension': 3, 'document_versions': {'dap-vehicle-buyer': 'reviewed-document-sha'},
                'chunk_count': 1, 'chunk_content_sha256': corpus_fingerprint([chunk])}
    path.write_text(json.dumps(original), encoding='utf-8')
    with pytest.raises(RuntimeError, match='embedding_model'):
        check_index(settings, dense, repair_missing=True)
    assert json.loads(path.read_text()) == original
    assert dense.index_calls == 0


def test_rag_benchmark_preflight_repairs_legacy_metadata_without_ollama(tmp_path, monkeypatch):
    """RAG-only measures real embeddings, not LLM inference or dummy vectors."""
    from dap_assistant.rag import retrieval
    from dap_assistant.evaluation import index_health, preflight

    settings = replace(Settings(), data_dir=tmp_path,
                       embedding_provider='sentence_transformers',
                       llm_provider='dummy', answer_mode='source')
    (tmp_path / 'vectorstore' / 'qdrant').mkdir(parents=True)
    chunk = fixture_chunk()
    dense = FakeDense()
    monkeypatch.setattr(retrieval, 'load_chunks', lambda _: [chunk])
    monkeypatch.setattr(index_health, 'load_chunks', lambda _: [chunk])
    monkeypatch.setattr(preflight, 'acquire_dense', lambda _: dense)
    monkeypatch.setattr(preflight, 'load_chunks', lambda _: [chunk])
    monkeypatch.setattr(preflight, 'ollama_health',
                        lambda _: pytest.fail('RAG-only preflight must not call Ollama'))
    returned = preflight.acquire_verified_dense(
        settings, require_inference=False, real_only=True
    )
    assert returned is dense
    assert dense.index_calls == 1
    assert index_metadata_path(settings).exists()


def test_review_requires_two_explicit_approvals(tmp_path):
    case = load_dataset()[1]
    chunk = fixture_chunk()
    patterns = {fact['fact_id']: r'15\s+nap' for fact in case['expected_facts']}
    with pytest.raises(ValueError, match='külön jóvá'):
        review_case(case['question_id'], selected_chunks=[chunk['chunk_id']],
                    answer_patterns=patterns, confirm_facts=True, confirm_chunks=False,
                    chunks=[chunk], destination=tmp_path / 'review.json')
    assert not (tmp_path / 'review.json').exists()


def test_review_pins_real_chunks_and_detects_document_updates(tmp_path):
    case = load_dataset()[1]  # vehicle buyer deadline
    chunk = fixture_chunk()
    destination = tmp_path / 'reviewed.json'
    patterns = {fact['fact_id']: r'15\s+nap' for fact in case['expected_facts']}
    reviewed = review_case(case['question_id'], selected_chunks=[chunk['chunk_id']],
                           answer_patterns=patterns, confirm_facts=True, confirm_chunks=True,
                           chunks=[chunk], destination=destination)
    assert reviewed['human_reviewed'] is True
    assert reviewed['relevant_chunk_ids'] == [chunk['chunk_id']]
    assert reference_status(reviewed, index_versions([chunk])) == 'pinned'
    assert validate_relevant_chunks(reviewed, [chunk]) is True
    assert reference_status(reviewed, {chunk['document_id']: 'new-version'}) == 'version_mismatch'
    assert validate_relevant_chunks(reviewed, [{**chunk, 'document_version': 'new-version'}]) is False
    assert len(load_dataset(destination)) == 20
    assert reference_status(load_dataset(destination)[0], index_versions([chunk])) == 'unversioned_reference'


def test_review_rejects_cross_document_and_invalid_regex(tmp_path):
    case = load_dataset()[1]
    chunk = fixture_chunk(document='dap-vehicle-seller')
    patterns = {fact['fact_id']: '15' for fact in case['expected_facts']}
    with pytest.raises(ValueError, match='nem tartozik'):
        review_case(case['question_id'], selected_chunks=[chunk['chunk_id']],
                    answer_patterns=patterns, confirm_facts=True, confirm_chunks=True,
                    chunks=[chunk], destination=tmp_path / 'reviewed.json')
    chunk = fixture_chunk()
    patterns = {fact['fact_id']: '[' for fact in case['expected_facts']}
    with pytest.raises(ValueError, match='Hibás válaszminta'):
        review_case(case['question_id'], selected_chunks=[chunk['chunk_id']],
                    answer_patterns=patterns, confirm_facts=True, confirm_chunks=True,
                    chunks=[chunk], destination=tmp_path / 'reviewed.json')


def test_pinned_retrieval_is_measured_and_false_gold_is_not():
    case = load_dataset()[1]
    chunk = fixture_chunk()
    # Keep the fixture internally complete: all expected sources must be pinned.
    case['expected_source_ids'] = [chunk['document_id']]
    case['expected_source_versions'] = {chunk['document_id']: chunk['document_version']}
    # Replace any auto-reference fields inherited from the active SILVER dataset:
    # this fixture explicitly models a separately human-reviewed, pinned case.
    case.pop('automatic_reference', None)
    case.pop('reference_chunk_sha256', None)
    case.pop('reference_audit', None)
    case['human_reviewed'] = True
    case['reference_status'] = 'pinned'
    case['relevant_chunk_ids'] = [chunk['chunk_id']]
    case['relevant_chunk_ids_by_task'] = {'t1': [chunk['chunk_id']]}
    case['expected_facts'][0]['answer_pattern'] = r'15\s+nap'
    evidence = [{**chunk, 'evidence_id': 'E_one', 'score': 0.5}]
    output = {'domains': case['expected_domains'], 'intents': case['expected_intents'],
              'subtasks': {'t1': {'domain': 'vehicle', 'question': case['question'], 'status': 'complete'}},
              'branch_results': {'t1': {'retrieval_status': 'complete', 'evidence': evidence,
                                       'ranked_chunk_ids': [chunk['chunk_id']]}},
              'evidence': evidence, 'final_answer': 'Az átírás 15 napon belül szükséges.',
              'tool_results': {}, 'answer_draft': {'claims': []}}
    metrics, details = _full_metrics(
        case, output, {'spans': []},
        {chunk['document_id']: chunk['document_version']}, [chunk], Telemetry(), 'test-run',
    )
    assert metrics['retrieval_recall_at_5'] == 1.0
    assert metrics['retrieval_mrr'] == 1.0
    assert metrics['source_recall_at_5'] == 1.0
    assert metrics['answer_completeness'] == 1.0
    assert metrics['answer_correctness'] is None  # NOT invented semantic accuracy
    invalid_metrics, invalid_details = _full_metrics(
        case, output, {'spans': []},
        {chunk['document_id']: chunk['document_version']}, [], Telemetry(), 'invalid-run',
    )
    assert invalid_details['reference_status'] == 'invalid_chunk_reference'
    assert invalid_metrics['retrieval_mrr'] is None
