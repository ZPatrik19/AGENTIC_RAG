from dap_assistant.response.quality import (
    claim_category,
    claim_facets,
    claim_provenance_report,
    clean_claims,
    structural_facet_coverage,
)


def _claim(text, quote=None, *, category='steps', evidence_id='E1', origin='unknown', in_prompt=None):
    return {
        'text': text,
        'supporting_quote': quote or text,
        'category': category,
        'evidence_ids': [evidence_id],
        'origin': origin,
        'in_model_prompt': in_prompt,
    }


def test_deadline_is_covered_from_claim_and_quote_even_when_model_category_is_steps():
    text = ('Az adásvételi szerződés megkötésétől számított 15 napon belül '
            'át kell íratnod a gépjárművet.')
    claim = _claim(text, category='steps')
    assert 'deadline' in claim_facets(claim, 'vehicle')
    coverage = structural_facet_coverage(('steps', 'deadline'), [claim], domain='vehicle')
    assert coverage['covered'] == ['steps', 'deadline']
    assert coverage['missing'] == []


def test_generic_authority_processing_deadline_does_not_count_as_vehicle_transfer_deadline():
    text = ('Ha jogszabály eltérően nem rendelkezik, az e rendelet hatálya alá '
            'tartozó eljárásokban az ügyintézési határidő tíz nap.')
    claim = _claim(text, category='deadline')
    assert 'deadline' not in claim_facets(claim, 'vehicle')
    assert claim_category(claim, 'vehicle') == 'steps'


def test_healthcare_document_does_not_pose_as_entitlement():
    text = ('TB kiskönyv/OEP igazolvány a biztosítási jogviszonyról és az '
            'egészségbiztosítási ellátásokról, amelyet a munkáltató ad ki.')
    claim = _claim(text, category='healthcare')
    assert 'documents' in claim_facets(claim, 'employment')
    assert 'healthcare' not in claim_facets(claim, 'employment')
    assert claim_category(claim, 'employment') == 'documents'


def test_explicit_jobseeker_allowance_healthcare_condition_is_healthcare():
    text = ('Amíg kapod az álláskeresési járadékot, megmarad a biztosítási '
            'jogviszonyod, ezért nem kell TB járulékot fizetned.')
    claim = _claim(text, category='steps')
    assert 'healthcare' in claim_facets(claim, 'employment')
    assert claim_category(claim, 'employment') == 'healthcare'
    coverage = structural_facet_coverage(('healthcare',), [claim], domain='employment')
    assert coverage['coverage'] == 1.0


def test_clean_claims_relabels_healthcare_heading_without_rewriting_text():
    text = ('TB kiskönyv/OEP igazolvány a biztosítási jogviszonyról és az '
            'egészségbiztosítási ellátásokról, amelyet a munkáltató ad ki.')
    cleaned, integrity = clean_claims([_claim(text, category='healthcare')], domain='employment')
    assert integrity['healthcare_condition_issues'] == []
    assert len(cleaned) == 1
    assert cleaned[0]['text'] == text
    assert cleaned[0]['category'] == 'documents'


def test_provenance_distinguishes_model_claim_from_post_assembly_supplement():
    claims = [
        _claim('Át kell íratnod a járművet.', evidence_id='E_MODEL',
               origin='model_generated', in_prompt=True),
        _claim('Az eredetiségvizsgálatot vizsgálóállomáson intézd.', evidence_id='E_EXTRA',
               origin='source_supplement', in_prompt=False),
    ]
    report = claim_provenance_report(claims, ['E_MODEL'])
    assert report['counts']['model_generated'] == 1
    assert report['counts']['source_supplement'] == 1
    assert report['claims'][0]['quoted_unit_in_model_prompt'] is True
    assert report['claims'][1]['quoted_unit_in_model_prompt'] is False


def test_provenance_does_not_claim_prompt_presence_from_id_mismatch():
    claim = _claim('Át kell íratnod a járművet.', evidence_id='E_NOT_IN_PROMPT',
                   origin='model_generated', in_prompt=True)
    report = claim_provenance_report([claim], ['E_OTHER'])
    assert report['claims'][0]['quoted_unit_in_model_prompt'] is None
