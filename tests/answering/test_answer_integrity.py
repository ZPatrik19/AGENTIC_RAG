"""P6 offline regressions for the two actual AUTO_001 / WORK_001 defects."""
from dap_assistant.response.quality import (
    claim_identifier, clean_claims, is_complete_statement, render_claim_groups, structural_facet_coverage,
)
from dap_assistant.context_engineering.evidence_selection import assemble_selection_context
from dap_assistant.llm import NaturalClaim, NaturalResponse, _attach_source_quotes, source_answer


def _claim(text: str, category: str = 'steps', eid: str = 'E_ONE') -> dict:
    return {'text': text, 'supporting_quote': text, 'evidence_ids': [eid], 'category': category}


def test_real_auto_001_trailing_fragment_not_accepted_as_an_obligation():
    fragment = ('Ha a vásárlás előtt nem történt eredetiségvizsgálat vagy 60 napnál régebben, '
                'az adásvételi szerződés megkötésétől számított 15 napon belül el kell vége')
    good = ('Ha a vásárlás előtt nem történt eredetiségvizsgálat, '
            'az adásvételi szerződés megkötésétől számított 15 napon belül el kell végeztetni.')
    clean, report = clean_claims([_claim(fragment), _claim(good)])
    assert [item['text'] for item in clean] == [good]
    assert report['incomplete_claims'] == [fragment[:120]]


def test_work_001_conditionless_reference_and_cropped_date_are_not_legal_claims():
    assert not is_complete_statement('Ez azt jelenti, hogy nem kell TB járulékot fizetned.')
    assert not is_complete_statement('január 1-jét követő munkaviszony megszűnése esetén igazolást kell kiállítani.')
    assert is_complete_statement('Amíg álláskeresési járadékot kapsz, megmarad a biztosítási jogviszonyod.')
    assert is_complete_statement('Az álláskeresőként történő nyilvántartásba vételt kérelmezni kell.')


def test_repeated_insurance_and_transfer_subclaims_are_deduplicated_across_headings():
    combined = ('Az átíráshoz keresd fel bármelyik kormányablakot, vagy foglalj időpontot. '
                'Az átírással egy időben az illetéket is befizetheted.')
    short = 'Az átíráshoz keresd fel bármelyik kormányablakot, vagy foglalj időpontot.'
    insurance = 'Tulajdonosváltáskor a korábbi biztosítás megszűnik, ezért azonnal újat kell kötnöd.'
    claims, report = clean_claims([_claim(combined, 'steps'), _claim(insurance, 'insurance'),
                                   _claim(short, 'where'), _claim(insurance, 'steps')])
    assert len(claims) == 2
    assert len(report['duplicate_claims']) == 2
    groups = render_claim_groups(claims, {'E_ONE': {'source_url': 'https://dap.gov.hu/teszt'}},
                                 {'steps': 'Teendők', 'insurance': 'Biztosítás'})
    assert groups.count('**Teendők**') == 1
    assert groups.count('**Biztosítás**') == 1


def test_interleaved_categories_render_one_section_per_category():
    claims = [_claim('Az átírást intézd el.', 'steps'),
              _claim('A biztosítást kösd meg.', 'insurance'),
              _claim('A hivatalhoz fordulj.', 'steps')]
    lines = render_claim_groups(claims, {'E_ONE': {'source_url': 'https://dap.gov.hu/teszt'}},
                                {'steps': 'Teendők', 'insurance': 'Biztosítás'})
    assert lines.count('**Teendők**') == 1
    assert lines.count('**Biztosítás**') == 1
    assert lines.index(f'1. [{claim_identifier(claims[0])}] Az átírást intézd el. [Forrás E_ONE](<https://dap.gov.hu/teszt>)') < lines.index(f'2. [{claim_identifier(claims[2])}] A hivatalhoz fordulj. [Forrás E_ONE](<https://dap.gov.hu/teszt>)')


def test_source_units_do_not_offer_a_truncated_legal_rule_to_model():
    evidence = [{'evidence_id': 'E_ONE', 'title': 'Gépjármű', 'source_url': 'https://dap.gov.hu/teszt',
                 'text': ('Ha a vásárlás előtt nem történt eredetiségvizsgálat, az adásvételi szerződés '
                          'megkötésétől számított 15 napon belül el kell vége\n'
                          'Az adásvételi szerződés megkötésétől számított 15 napon belül át kell íratni az autót.'),
                 'document_id': 'dap-vehicle-buyer', 'domain': 'vehicle'}]
    packed = assemble_selection_context(evidence, 'Vettem egy autót, mik a teendők?',
                                        domain='vehicle', role='buyer', token_budget=300)
    assert packed.evidence
    assert 'el kell vége' not in packed.evidence[0]['text']
    assert 'át kell íratni az autót.' in packed.evidence[0]['text']


def test_model_fragment_is_reported_and_not_cited_even_if_original_quote_is_full():
    original = 'Az adásvételi szerződés megkötésétől számított 15 napon belül át kell íratnod a gépjárművet.'
    response = NaturalResponse(claims=[NaturalClaim(
        evidence_id='E_ONE', category='deadline', text='A szerződéstől számított 15 napon belül el kell vége')])
    draft = _attach_source_quotes(response, [{'evidence_id': 'E_ONE', 'text': original}])
    assert draft.claims == []
    assert draft.quality_warnings == ['model_incomplete_statement']


def test_missing_healthcare_facet_is_structural_warning_not_human_gold_score():
    coverage = structural_facet_coverage(('steps', 'healthcare'),
                                         [_claim('Kérd a nyilvántartásba vételt.')])
    assert coverage['missing'] == ['healthcare']
    assert 'not_semantic' in coverage['measurement']


def test_existing_complete_work_source_still_extracts_both_categories():
    question = 'Elvesztettem a munkámat. Milyen teendőim vannak?'
    sources = [
        {'evidence_id': 'E_H', 'chunk_id': 'H', 'document_id': 'neak-post-employment-healthcare',
         'domain': 'employment', 'title': 'NEAK', 'source_url': 'https://neak.gov.hu/pelda',
         'text': 'A munkaviszony megszűnése után az egészségügyi szolgáltatásra való jogosultságot ellenőrizni kell.'},
        {'evidence_id': 'E_S', 'chunk_id': 'S', 'document_id': 'dap-employment-overview',
         'domain': 'employment', 'title': 'DÁP', 'source_url': 'https://dap.gov.hu/pelda',
         'text': 'Az álláskeresőként történő nyilvántartásba vételt kérelmezni kell.'},
    ]
    result = source_answer(question, sources)
    assert {'healthcare', 'steps'} <= {claim.category for claim in result.claims}
