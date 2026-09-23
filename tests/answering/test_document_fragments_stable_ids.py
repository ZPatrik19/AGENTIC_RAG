"""P6.5 regressions from AUTO_001 / WORK_001 benchmark (synthetic fixtures).

These test structural consistency and provenance; they are not legal gold labels.
"""
from __future__ import annotations


from dap_assistant.response.filters import low_value_procedural_unit
from dap_assistant.response.quality import (
    claim_facets, claim_identifier, claim_provenance_report,
    clean_claims, non_independent_source_fragment, render_claim_groups,
    structural_facet_coverage,
)
from dap_assistant.evaluation.comparison import _audit_simple_draft, _render_simple_answer
from dap_assistant.llm import Draft, source_answer, supplement_grounded_facts

WORK = 'Elvesztettem a munkámat. Milyen ügyintézési teendőim vannak?'
AUTO = 'Vettem egy használt autót. Milyen ügyintézési teendőim vannak?'
DOCUMENT = 'A munkáltató foglalkoztatói igazolást állít ki.'
WHERE = 'Nyilvántartásba vételét az elektronikus ügyintézési felületen is indíthatja.'
FRAGMENT = ('társadalombiztosítási kifizetőhellyel rendelkező munkaadó esetén a '
            'biztosítási jogviszony megszüntetésekor a biztosítási jogviszony megszűnését '
            'közvetlenül megelőző két éven belül folyósított táppénz, baleseti táppénz, '
            'csecsemőgondozási díj, örökbefogadói díj és gyermekgondozási díj időtartamát.')


def claim(text=DOCUMENT, *, category='documents', eid='E_DOC', quote=None, origin='model_generated'):
    return {'text': text, 'supporting_quote': quote or text, 'category': category,
            'evidence_ids': [eid], 'origin': origin, 'in_model_prompt': origin == 'model_generated'}


def evidence(eid, text, domain='employment'):
    return {'evidence_id': eid, 'chunk_id': eid, 'document_id': 'synthetic',
            'domain': domain, 'role': 'general' if domain == 'employment' else 'buyer',
            'text': text, 'title': 'Synthetic test', 'section_path': [],
            'source_url': 'https://example.invalid/synthetic'}


def test_registration_channel_without_named_document_does_not_cover_documents():
    item = claim(WHERE, category='documents')
    assert 'documents' not in claim_facets(item, 'employment')
    cleaned, _ = clean_claims([item], question=WORK, domain='employment')
    assert cleaned[0]['category'] == 'where'
    cov = structural_facet_coverage(('documents', 'where'), cleaned, domain='employment')
    assert cov['missing'] == ['documents']


def test_named_issued_document_with_explicit_action_covers_documents():
    item = claim()
    assert {'documents', 'steps'} <= claim_facets(item, 'employment')
    cleaned, _ = clean_claims([item], question=WORK, domain='employment')
    assert cleaned[0]['category'] == 'documents'
    assert structural_facet_coverage(('documents',), cleaned, domain='employment')['coverage'] == 1.0


def test_bare_employment_form_name_does_not_prove_it_is_required():
    item = claim('Foglalkoztatói igazolás', category='documents')
    assert 'documents' not in claim_facets(item, 'employment')


def test_automobile_contract_in_transfer_deadline_is_not_document_checklist():
    text = 'Az adásvételi szerződés megkötésétől számított 15 napon belül át kell íratnod a gépjárművet.'
    assert 'documents' not in claim_facets(claim(text, eid='E_VEHICLE'), 'vehicle')
    assert 'deadline' in claim_facets(claim(text, eid='E_VEHICLE'), 'vehicle')


def test_short_literal_vehicle_contract_list_item_is_document_not_deadline():
    text = 'Adásvételi szerződés'
    assert 'documents' in claim_facets(claim(text), 'vehicle')


def test_subordinate_legal_enumeration_with_terminal_period_is_rejected():
    assert non_independent_source_fragment(FRAGMENT)
    cleaned, report = clean_claims([claim(FRAGMENT, eid='E_LAW')], question=WORK, domain='employment')
    assert cleaned == []
    assert report['incomplete_claims']


def test_real_imperative_ending_in_accusative_is_not_rejected():
    text = ('Ha a munkaviszony megszűnik, ellenőrizd a biztosítási jogviszony '
            'megszűnésének időpontját.')
    assert not non_independent_source_fragment(text)
    kept, _ = clean_claims([claim(text)], question=WORK, domain='employment')
    assert len(kept) == 1


def test_source_extraction_never_offers_fragment_as_document_or_step():
    extracted = source_answer(WORK, [evidence('E_LAW', FRAGMENT), evidence('E_DOC', DOCUMENT)])
    assert extracted.claims
    assert all(FRAGMENT != c.text for c in extracted.claims)
    assert any(DOCUMENT == c.text for c in extracted.claims)


def test_post_assembly_does_not_resurrect_fragment_or_mislabel_where_as_document():
    draft = Draft(claims=[], model_context_evidence_ids=['E_WHERE'])
    source = [evidence('E_LAW', FRAGMENT), evidence('E_WHERE', WHERE),
              evidence('E_DOC', DOCUMENT)]
    result = supplement_grounded_facts(draft, WORK, source, ('documents',),
                                      model_prompt_evidence=[source[1]])
    assert all(FRAGMENT not in fact.text for fact in result.claims)
    assert all('documents' in claim_facets(fact.model_dump(), 'employment') for fact in result.claims)
    assert result.claims and result.claims[0].text == DOCUMENT
    assert result.claims[0].origin == 'source_supplement'
    assert result.claims[0].in_model_prompt is False


def test_source_only_document_channel_without_document_does_not_mark_full_coverage():
    item = claim(WHERE, category='documents', eid='E_WHERE')
    audited, rejected = _audit_simple_draft({'claims': [item]}, [evidence('E_WHERE', WHERE)],
                                            question=WORK, domain='employment')
    assert not rejected
    assert 'documents' not in structural_facet_coverage(('documents',), audited['claims'],
                                                       domain='employment')['covered']


def test_stable_identifier_ignores_order_and_heading_but_changes_when_evidence_changes():
    first = claim(DOCUMENT, origin='model_generated')
    second = claim(DOCUMENT, category='steps', origin='source_supplement')
    assert claim_identifier(first) == claim_identifier(second)
    assert claim_identifier(first) != claim_identifier(claim(DOCUMENT, eid='E_OTHER'))
    assert claim_identifier(first) != claim_identifier(claim(DOCUMENT + ' A munkáltató adja ki.'))
    cleaned, _ = clean_claims([first], question=WORK, domain='employment')
    assert cleaned[0]['claim_id'] == claim_identifier(first)
    cov = structural_facet_coverage(('documents',), cleaned, domain='employment')
    assert cov['claim_evidence'][0]['claim_id'] == cleaned[0]['claim_id']
    prov = claim_provenance_report(cleaned, ['E_DOC'])
    assert prov['claims'][0]['claim_id'] == cleaned[0]['claim_id']


def test_final_answer_and_report_have_exact_same_stable_id():
    cleaned, _ = clean_claims([claim(DOCUMENT)], question=WORK, domain='employment')
    identifier = cleaned[0]['claim_id']
    by_id = {'E_DOC': evidence('E_DOC', DOCUMENT)}
    rendered = '\n'.join(render_claim_groups(cleaned, by_id, {'documents': 'Dokumentumok'}))
    baseline = _render_simple_answer({'claims': cleaned, 'text_integrity': {}}, list(by_id.values()))
    provenance = claim_provenance_report(cleaned, ['E_DOC'])
    assert f'[{identifier}]' in rendered and f'[{identifier}]' in baseline
    assert provenance['claims'][0]['claim_id'] == identifier
    assert identifier.startswith('C_') and len(identifier) == 22


def test_older_avdh_authentication_guide_not_added_to_generic_work_question():
    historic = ('Nyilvántartási kérelem nyomtatványt az e-Papír szolgáltatáson '
                'keresztül, AVDH hitelesítéssel aláírva csatold.')
    assert low_value_procedural_unit(historic, WORK, 'employment')
    assert not low_value_procedural_unit(historic, WORK + ' Hogyan működik az AVDH?', 'employment')
