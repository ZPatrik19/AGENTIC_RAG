"""Regression: combined user needs must not be collapsed into a single facet.

All amounts and texts below are SYNTHETIC TEST FIXTURES, not live legal facts.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from dap_assistant.context_engineering.information_needs import question_needs, answer_facets
from dap_assistant.fast_path import classify_explicit, plan_explicit
from dap_assistant.llm import source_answer
from dap_assistant.settings import Settings


QUESTION = 'Autót vásároltam milyen teendőim vannak és milyen kötségekkel jár?'


def item(key: str, text: str) -> dict:
    return {
        'evidence_id': f'E_{key}', 'chunk_id': key, 'document_id': 'synthetic-buyer',
        'domain': 'vehicle', 'role': 'buyer', 'text': text,
        'section_path': ['Teszt'], 'source_url': 'https://dap.gov.hu/synthetic-test-only',
    }


SYNTHETIC = [
    item('deadline', 'A szerződést követően 15 napon belül át kell írni a gépjárművet.'),
    item('insurance', 'Tulajdonosváltáskor új biztosítást kell kötni.'),
    item('inspection', 'Az eredetiségvizsgálat díja 17 000 - 20 000 Ft.'),
    item('registration', 'Forgalmi engedély díja: 6 000 Ft\nTörzskönyv díja: 6 000 Ft'),
]


@pytest.mark.parametrize('q', [
    QUESTION,
    'Gépjárművet vettem: mi a teendő és mennyi a költsége?',
    'Eladtam a kocsim, mit kell intéznem, és milyen díjakra számítsak?',
])
def test_combined_request_has_two_independent_search_tasks(q):
    needs = question_needs(q)
    assert needs.steps and needs.costs
    assert classify_explicit(q).domains == ['vehicle']
    planned = plan_explicit(q, ['vehicle']).tasks
    assert len(planned) == 2
    assert all(task.domain == 'vehicle' and not task.depends_on for task in planned)
    assert 'költségek' in planned[1].question


def test_mixed_request_preserves_procedural_and_multiple_distinct_fee_sources():
    # The quick Qwen selection is intentionally bad: only the deadline ID.
    draft = source_answer(QUESTION, SYNTHETIC, selected_ids=['E_deadline'])
    assert {'deadline', 'insurance', 'cost'} <= {c.category for c in draft.claims}
    prices = [c.text for c in draft.claims if c.category == 'cost']
    assert any('17 000 - 20 000 Ft' in text for text in prices)
    assert any('Forgalmi engedély' in text for text in prices)
    assert any('Törzskönyv' in text for text in prices)
    assert answer_facets(QUESTION, [c.model_dump() for c in draft.claims])['coverage'] == 1
    for claim in draft.claims:
        assert claim.supporting_quote in next(e['text'] for e in SYNTHETIC if e['evidence_id'] in claim.evidence_ids)


def test_no_price_is_partial_coverage_not_invented_number():
    draft = source_answer(QUESTION, SYNTHETIC[:2])
    facets = answer_facets(QUESTION, [c.model_dump() for c in draft.claims])
    assert facets['missing'] == ['costs']
    assert facets['coverage'] == .5
    assert not any(c.category == 'cost' for c in draft.claims)


def test_standalone_inspection_fee_does_not_confuse_registration_fee():
    draft = source_answer('Mennyi az eredetvizsga díja?', SYNTHETIC)
    assert all(c.category == 'cost' for c in draft.claims)
    assert len(draft.claims) == 1
    assert '17 000 - 20 000 Ft' in draft.claims[0].text


def test_retrieval_cost_sweep_is_role_filtered_and_only_uses_real_price(monkeypatch, tmp_path):
    pytest.importorskip('langgraph')
    from dap_assistant.rag import rag_graph
    monkeypatch.setattr(rag_graph, 'hybrid_search', lambda *a, **kw: [
        {**SYNTHETIC[0], 'score': .03, 'title': 'Fixture', 'document_version': 'test'},
    ])
    priced = [
        {**SYNTHETIC[2], 'score': .03, 'title': 'Fixture', 'document_version': 'test'},
        {**item('seller_fee', 'Az eladói tesztdíj 44 444 Ft.'), 'role': 'seller', 'score': .03,
         'title': 'Fixture', 'document_version': 'test'},
    ]
    monkeypatch.setattr('dap_assistant.rag.retrieval._chunk_snapshot', lambda _: priced)
    settings = replace(Settings(), data_dir=tmp_path, embedding_provider='dummy')
    graph = rag_graph.build_rag_graph(settings)
    result = graph.invoke({'task_id': 't2', 'domain': 'vehicle', 'role': 'buyer',
                           'query': QUESTION, 'search_attempt': 0})
    ids = {e['evidence_id'] for e in result['evidence']}
    assert 'E_inspection' in ids
    assert 'E_seller_fee' not in ids


def test_broad_cost_only_query_lists_separate_sourced_fee_items():
    q = 'Autóvásárlásnál milyen költségekkel jár az ügyintézés?'
    assert question_needs(q).costs is True
    assert question_needs(q).steps is False
    assert len(plan_explicit(q, ['vehicle']).tasks) == 1
    draft = source_answer(q, SYNTHETIC)
    assert len(draft.claims) >= 3
    assert all(c.category == 'cost' for c in draft.claims)
    assert all('Ft' in c.supporting_quote for c in draft.claims)


def test_evaluation_uses_only_audited_claims_for_facets():
    from dap_assistant.evaluation.functional_metrics import request_coverage
    bad = {'text': 'A díj 999 Ft.', 'category': 'cost', 'supporting_quote': 'A díj 999 Ft.'}
    good = {'text': 'Át kell írni.', 'category': 'steps', 'supporting_quote': 'Át kell írni.'}
    output = {'answer_draft': {'claims': [bad, good]},
              'answer_validation': {'status': 'partial', 'unsupported_claims': ['A díj 999 Ft.']}}
    assert request_coverage(QUESTION, output)['missing'] == ['costs']
    output['answer_validation'] = {'status': 'failed'}
    assert request_coverage(QUESTION, output)['coverage'] == 0.0
