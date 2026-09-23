"""P1 regression: only reviewed, source-versioned, per-task chunk IDs earn gold scores."""
from copy import deepcopy

import pytest

from dap_assistant.evaluation.dataset import (
    DATASET, index_versions, load_dataset, reference_status, validate_relevant_chunks,
)
from dap_assistant.evaluation.review import prepare_review_report, review_case


def _chunk(chunk_id='buyer-1', document='dap-vehicle-buyer', domain='vehicle'):
    return {
        'chunk_id': chunk_id, 'document_id': document, 'document_version': 'sha-123',
        'domain': domain, 'text': 'Az átírásra a dokumentum szerint 15 nap áll rendelkezésre.',
        'title': 'Hivatalos tájékoztató',
        'source_url': 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek',
        'section_path': ['Ügyintézés'], 'page_number': None,
    }


def test_candidate_report_does_not_invent_human_review_or_gold():
    case = load_dataset(DATASET)[1]
    chunk = _chunk()
    report = prepare_review_report([case], [chunk], limit=5)
    row = report['cases'][0]
    assert row['review_status'] == 'awaiting_manual_review'
    assert row['human_reviewed'] is False
    assert row['candidate_chunks_not_gold'][0]['text'] == chunk['text']
    assert row['candidate_chunks_not_gold'][0]['document_version'] == chunk['document_version']
    assert 'relevant_chunk_ids' not in row
    assert reference_status(case, index_versions([chunk])) == 'unversioned_reference'


def test_candidate_report_blocks_missing_corpus_and_ambiguous_chunk_ids():
    case = load_dataset(DATASET)[1]
    with pytest.raises(ValueError, match='Nincs feldolgozott'):
        prepare_review_report([case], [])
    with pytest.raises(ValueError, match='Nem egyedi chunkazonosítók'):
        prepare_review_report([case], [_chunk(), _chunk()])


def test_reference_status_requires_mapping_for_multiple_tasks():
    case = deepcopy(load_dataset(DATASET)[9])
    assert len(case['expected_subtasks']) > 1
    case.update(expected_source_ids=['dap-vehicle-buyer'],
                expected_source_versions={'dap-vehicle-buyer': 'sha-123'},
                relevant_chunk_ids=['buyer-1'], human_reviewed=True,
                relevant_chunk_ids_by_task={})
    chunks = [_chunk()]
    assert reference_status(case, index_versions(chunks)) == 'task_mapping_missing'
    assert not validate_relevant_chunks(case, chunks)
    case['relevant_chunk_ids_by_task'] = {
        f't{i}': ['buyer-1'] for i in range(1, len(case['expected_subtasks']) + 1)
    }
    assert reference_status(case, index_versions(chunks)) == 'pinned'
    assert validate_relevant_chunks(case, chunks)
    case['relevant_chunk_ids_by_task']['t1'] = ['different']
    assert reference_status(case, index_versions(chunks)) == 'task_mapping_missing'
    assert not validate_relevant_chunks(case, chunks)


def test_review_does_not_approve_multi_task_case_without_task_mapping(tmp_path):
    case = load_dataset(DATASET)[9]
    chunks = [_chunk()]
    patterns = {fact['fact_id']: r'15\\s*nap' for fact in case['expected_facts']}
    target = tmp_path / 'reviewed.json'
    with pytest.raises(ValueError, match='kötelező'):
        review_case(case['question_id'], selected_chunks=['buyer-1'],
                    answer_patterns=patterns, confirm_facts=True, confirm_chunks=True,
                    chunks=chunks, destination=target)
    assert not target.exists()


def test_review_requires_every_selected_chunk_to_have_task(tmp_path):
    case = load_dataset(DATASET)[1]
    chunks = [_chunk(), _chunk('buyer-2')]
    patterns = {fact['fact_id']: r'15\\s*nap' for fact in case['expected_facts']}
    with pytest.raises(ValueError, match='Minden kiválasztott'):
        review_case(case['question_id'], selected_chunks=['buyer-1', 'buyer-2'],
                    answer_patterns=patterns, confirm_facts=True, confirm_chunks=True,
                    chunks=chunks, destination=tmp_path / 'reviewed.json',
                    task_chunk_ids={'t1': ['buyer-1']})


def test_reviewed_document_update_invalidates_metrics():
    case = deepcopy(load_dataset(DATASET)[1])
    case.update(expected_source_ids=['dap-vehicle-buyer'],
                expected_source_versions={'dap-vehicle-buyer': 'sha-123'},
                relevant_chunk_ids=['buyer-1'], human_reviewed=True)
    assert reference_status(case, {'dap-vehicle-buyer': 'sha-123'}) == 'pinned'
    assert validate_relevant_chunks(case, [_chunk()])
    assert reference_status(case, {'dap-vehicle-buyer': 'sha-changed'}) == 'version_mismatch'
    assert not validate_relevant_chunks(case, [{**_chunk(), 'document_version': 'sha-changed'}])
