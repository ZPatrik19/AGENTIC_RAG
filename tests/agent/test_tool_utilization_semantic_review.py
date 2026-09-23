"""P6.7 synthetic adversarial facts: no claims about real-world legal correctness."""
from __future__ import annotations


import pytest

from dap_assistant.response.quality import claim_identifier
from dap_assistant.evaluation.metrics import claim_support_breakdown
from dap_assistant.evaluation.semantic_review import semantic_review, source_fingerprint
from dap_assistant.tooling.result_integration import integrate_checklist


EID = 'E_aaaaaaaaaaaaaaaa'
EVIDENCE = {'evidence_id': EID, 'text': (
    'Az adásvételi szerződést be kell mutatni.\n'
    'Az átírást 15 napon belül kell intézni.\n'
    'A bejelentkezéshez használható az elektronikus ügyintézési felület.'),
    'source_url': 'https://example.org/frozen-synthetic', 'domain': 'vehicle'}
DOC = 'Az adásvételi szerződést be kell mutatni.'


def native_result(text=DOC, eid=EID):
    return {'native_get_document_checklist_1': {
        'tool': 'get_document_checklist', 'status': 'success',
        'source_evidence_ids': [eid],
        'items': [{'text': text, 'evidence_ids': [eid]}]}}


def delivered(eid=EID, yes=True):
    return {'calls': [{'tool_name': 'get_document_checklist',
                       'status': 'executed', 'returned_to_model': yes,
                       'result_evidence_ids': [eid]}]}


def run_tool(*, evidence=None, existing=None, native=None, trace=None, ordinary=None):
    return integrate_checklist(
        question='Vettem egy autót. Milyen dokumentumok kellenek?',
        domain='vehicle', role='buyer', evidence=evidence or [EVIDENCE],
        existing_claims=existing or [], tool_results=ordinary or {},
        native_tool_results=native or native_result(),
        native_tool_trace=trace or delivered())


def test_real_delivered_tool_item_is_renderable_with_source_and_provenance():
    added, report = run_tool()
    assert len(added) == 1
    assert added[0]['text'] == DOC
    assert added[0]['supporting_quote'] == DOC
    assert added[0]['origin'] == 'tool_extract'
    assert added[0]['evidence_ids'] == [EID]
    assert added[0]['claim_id'] == claim_identifier(added[0])
    assert report['counts']['added_verbatim'] == 1


def test_native_result_without_model_delivery_is_not_promoted():
    added, report = run_tool(trace=delivered(yes=False))
    assert not added
    assert report['counts']['not_delivered_to_model'] == 1


@pytest.mark.parametrize('text,status', [
    ('A bejelentkezéshez használható az elektronikus ügyintézési felület.', 'not_a_document_requirement'),
    ('A szerződést csak holnap.', 'source_mismatch'),
    ('Az átírást 15 napon belül kell intézni.', 'not_a_document_requirement'),
    ('Az adásvételi szerződést be kell mutatni, de', 'source_mismatch'),
])
def test_tool_result_never_invents_paperwork(text, status):
    added, report = run_tool(native=native_result(text))
    assert not added
    assert report['items'][0]['status'] == status


def test_tool_item_already_in_model_answer_is_reported_not_duplicated():
    model_claim = {'text': DOC, 'supporting_quote': DOC,
                   'evidence_ids': [EID], 'category': 'documents',
                   'origin': 'model_generated'}
    added, report = run_tool(existing=[model_claim])
    assert not added
    assert report['counts']['covered_by_answer'] == 1


def test_native_and_deterministic_same_item_dont_duplicate():
    added, report = run_tool(ordinary={'checklist': {
        'tool': 'build_document_checklist', 'status': 'success',
        'items': [{'text': DOC, 'evidence_ids': [EID]}]}})
    assert len(added) == 1
    assert report['items'][0]['origin'] == 'native_tool'
    assert report['counts']['duplicate_tool_item'] == 1


def test_valid_citation_does_not_claim_semantic_entailment():
    source = {'evidence_id': EID, 'text': 'A díj 10 egység.'}
    claim = {'text': 'A díj 10 egység, és mindenki mentes.',
             'supporting_quote': 'A díj 10 egység.', 'evidence_ids': [EID]}
    output = {'evidence': [source], 'answer_draft': {'claims': [claim]}}
    struct = claim_support_breakdown(output)
    assert struct['citation_integrity_proxy'] == 1.0
    assert struct['faithfulness'] is None
    assert semantic_review(output)['faithfulness'] is None
    # Synthetic, intentionally adversarial annotation. NOT a legal golden reference.
    approved = {'human_reviewed': True, 'approved_semantic_reviews': [{
        'claim_id': claim_identifier(claim), 'claim_text': claim['text'],
        'supporting_quote': claim['supporting_quote'], 'evidence_ids': [EID],
        'source_sha256': {EID: source_fingerprint(source['text'])},
        'label': 'not_supported', 'reviewer': 'synthetic-test-annotator',
        'reviewed_at': '2026-09-21', 'approved': True}]}
    result = semantic_review(output, approved)
    assert result['faithfulness'] == 0.0
    assert result['unsupported_claim_rate'] == 1.0


def test_stale_source_fingerprint_and_partial_reviews_mean_not_evaluated():
    source = {'evidence_id': EID, 'text': DOC}
    first = {'text': DOC, 'supporting_quote': DOC, 'evidence_ids': [EID]}
    second = {'text': 'Másik állítás.', 'supporting_quote': DOC, 'evidence_ids': [EID]}
    output = {'evidence': [source], 'answer_draft': {'claims': [first, second]}}
    reviewed = {'human_reviewed': True, 'approved_semantic_reviews': [{
        'claim_id': claim_identifier(first), 'claim_text': DOC,
        'supporting_quote': DOC, 'evidence_ids': [EID],
        'source_sha256': {EID: source_fingerprint(DOC)}, 'label': 'entailed',
        'reviewer': 'synthetic-test-annotator', 'reviewed_at': '2026-09-21',
        'approved': True}]}
    partial = semantic_review(output, reviewed)
    assert partial['review_coverage'] == 0.5
    assert partial['faithfulness'] is None
    reviewed['approved_semantic_reviews'][0]['source_sha256'][EID] = '0' * 64
    stale = semantic_review(output, reviewed)
    assert stale['reviewed_claims'] == 0
    assert stale['faithfulness'] is None


def test_review_export_never_preapproves_any_claim():
    from scripts.export_semantic_review import review_template
    worksheet = review_template({'run_id': 'test', 'rows': [{
        'question_id': 'SYN_001', 'mode': 'agentic',
        'details': {'claim_support': {'claims': [{
            'claim_id': 'C_fake', 'claim': DOC, 'supporting_quote': DOC,
            'source_sha256': {EID: source_fingerprint(DOC)},
            'evidence_ids': [EID]}]}}}]})
    claim = worksheet['cases'][0]['claims'][0]
    assert worksheet['status'] == 'unreviewed_template'
    assert claim['approved'] is False and claim['label'] is None
    assert claim['source_sha256'][EID] == source_fingerprint(DOC)


def test_review_export_rejects_old_benchmark_without_original_quote_or_digest():
    from scripts.export_semantic_review import review_template
    with pytest.raises(ValueError, match='rerun the comparison with P6.7'):
        review_template({'rows': [{'details': {'claim_support': {'claims': [
            {'claim': 'Régi eredmény', 'evidence_ids': [EID]}]}}}]})
