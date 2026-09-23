"""Offline acceptance cases: original retrieval, evidence coverage, and two Qwen stages."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from dap_assistant.rag import retrieval
from dap_assistant.context_engineering.evidence_selection import (
    assemble_selection_context, complete_units, facet_queries, requested_facets,
    validate_selection,
)
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import (
    LLMError, NaturalClaim, NaturalResponse, OllamaAdapter, _attach_source_quotes,
    source_answer,
)
from dap_assistant.settings import Settings


@pytest.fixture
def vehicle_evidence():
    url = 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek'
    return [
        {'evidence_id': 'E_INS', 'chunk_id': 'insurance', 'document_id': 'dap-vehicle-buyer',
         'domain': 'vehicle', 'title': 'Kötelező biztosítás', 'source_url': url,
         'facet_matches': ['insurance'],
         'text': 'Tulajdonosváltáskor új kötelező gépjármű-felelősségbiztosítást kell kötni.'},
        {'evidence_id': 'E_DEAD', 'chunk_id': 'deadline', 'document_id': 'dap-vehicle-buyer',
         'domain': 'vehicle', 'title': 'Átírás', 'source_url': url,
         'facet_matches': ['deadline', 'steps'],
         'text': 'Az átírást az adásvételi szerződéstől számított 15 napon belül intézd.'},
        {'evidence_id': 'E_DOC', 'chunk_id': 'documents', 'document_id': 'dap-vehicle-buyer',
         'domain': 'vehicle', 'title': 'Dokumentumok', 'source_url': url,
         'facet_matches': ['documents'],
         'text': 'Az átíráshoz szükséges az adásvételi szerződés és a törzskönyv.'},
        {'evidence_id': 'E_WHERE', 'chunk_id': 'office', 'document_id': 'dap-vehicle-buyer',
         'domain': 'vehicle', 'title': 'Ügyintézés', 'source_url': url,
         'facet_matches': ['where'],
         'text': 'Az átírást a kormányablakban is intézheted.'},
    ]


def test_bm25_dense_origins_and_stable_dedup(monkeypatch, tmp_path):
    chunks = [{'chunk_id': cid, 'document_id': 'official', 'domain': 'vehicle',
               'role': 'buyer', 'text': 'A vevő átírja az autót.'} for cid in ('A', 'B', 'C')]
    monkeypatch.setattr(retrieval, '_chunk_snapshot', lambda _: chunks)
    monkeypatch.setattr(retrieval, 'bm25_search', lambda *a, **kw: [('A', 8.), ('A', 8.), ('B', 4.)])

    class Dense:
        def search(self, *args, **kwargs):
            return [('B', .95), ('A', .86), ('C', .5)]

    got = retrieval.hybrid_search('gépjármű átírás', 'vehicle',
                                  replace(Settings(), data_dir=tmp_path), limit=5, dense=Dense(), role='buyer')
    assert {row['chunk_id'] for row in got} == {'A', 'B', 'C'}
    by_id = {row['chunk_id']: row for row in got}
    assert by_id['A']['bm25_rank'] == 1
    assert by_id['A']['dense_rank'] == 2
    assert by_id['B']['bm25_rank'] == 3  # original position, despite duplicate A
    assert by_id['B']['dense_rank'] == 1
    assert by_id['A']['score'] == pytest.approx(1/61 + 1/62)
    assert sum(by_id['A']['rrf_components'].values()) == pytest.approx(by_id['A']['score'])
    assert by_id['A']['fusion_method'] == 'rrf_k60_not_probability'


def test_multifaceted_buyer_query_preserves_insurance_documents_deadline():
    facets = requested_facets('Vettem egy autót, milyen teendőim vannak?', 'vehicle', role='buyer')
    assert {'steps', 'insurance', 'deadline', 'documents', 'where'} <= set(facets)
    queries = facet_queries('Vettem egy autót, milyen teendőim vannak?', 'vehicle', role='buyer')
    assert set(queries) == set(facets)
    assert len(queries) <= 8
    assert 'kötelező' in queries['insurance'] and 'kormányablak' in queries['where']


def test_selection_reserves_complete_units_for_each_information_need(vehicle_evidence):
    context = assemble_selection_context(vehicle_evidence,
        'Vettem egy autót, milyen teendőim vannak?', domain='vehicle', role='buyer', token_budget=1200)
    assert {'insurance', 'deadline', 'documents', 'where'} <= set(context.facet_coverage)
    assert not context.rejected_ids
    for row in context.evidence:
        original = next(x for x in vehicle_evidence if x['evidence_id'] == row['evidence_id'])
        assert row['text'] in original['text']
        assert all(unit in complete_units(original['text']) for unit in complete_units(row['text']))
    assert context.estimated_input_tokens <= context.token_budget_estimate


def test_no_middle_of_table_row_or_legal_condition():
    text = '2026. évi tájékoztató\nIlleték: 750 Ft/kW\nKivétel: csak bizonyított mentesség esetén.'
    packed = assemble_selection_context([{
        'evidence_id': 'E_RULE', 'title': 'Díjtábla', 'source_url': 'https://nav.gov.hu/',
        'text': text, 'facet_matches': ['costs']}],
        'Mennyi az illeték?', domain='vehicle', token_budget=140)
    assert packed.evidence
    assert 'Illeték: 750 Ft/kW' in packed.evidence[0]['text']
    assert not any(line.endswith('Ft/k') for line in packed.evidence[0]['text'].splitlines())


def test_fake_ids_and_unknown_facets_rejected():
    with pytest.raises(ValueError, match='Invalid selected evidence IDs'):
        validate_selection({'steps': ['FAKE_ID']}, {'E1'}, ('steps',))
    with pytest.raises(ValueError, match='unrequested'):
        validate_selection({'fake': ['E1']}, {'E1'}, ('steps',))
    assert validate_selection({'steps': ['E1', 'E1']}, {'E1'}, ('steps',)) == {'steps': ['E1']}


def test_generated_numeric_claim_is_not_allowed_without_same_source_number(vehicle_evidence):
    unsupported = NaturalResponse(claims=[NaturalClaim(
        evidence_id='E_DEAD', category='deadline',
        text='Az autót az adásvétel után 30 napon belül kell átírni.')])
    assert _attach_source_quotes(unsupported, vehicle_evidence).claims == []
    supported = NaturalResponse(claims=[NaturalClaim(
        evidence_id='E_DEAD', category='deadline',
        text='Az átírásra az adásvételtől számítva 15 nap áll rendelkezésre.'),
        NaturalClaim(evidence_id='E_DEAD', category='deadline',
        text='Az átírásra az adásvételtől számítva 15 nap áll rendelkezésre.')])
    assert len(_attach_source_quotes(supported, vehicle_evidence).claims) == 1


def test_two_stage_generation_not_just_quoted_python_bullets(vehicle_evidence):
    before = source_answer('Vettem egy autót, milyen teendőim vannak?', vehicle_evidence)
    assert before.claims
    sent = []
    def handler(req):
        data = json.loads(req.content)
        sent.append(data)
        if not data['stream']:
            return httpx.Response(200, json={
                'done': True, 'message': {'content': json.dumps({
                    'evidence_by_need': {'insurance': ['E_INS'], 'deadline': ['E_DEAD'],
                                         'documents': ['E_DOC'], 'where': ['E_WHERE'],
                                         'steps': ['E_DEAD']}, 'missing_needs': []})},
                'prompt_eval_count': 210, 'eval_count': 41})
        response = {'claims': [
            {'text': 'A gépjármű-átírást az adásvételtől számított 15 napon belül intézd.',
             'evidence_id': 'E_DEAD', 'category': 'deadline'},
            {'text': 'Tulajdonosváltás esetén új kötelező gépjármű-felelősségbiztosítást kell kötnöd.',
             'evidence_id': 'E_INS', 'category': 'insurance'},
            {'text': 'A szükséges iratok közé az adásvételi szerződés és a törzskönyv tartozik.',
             'evidence_id': 'E_DOC', 'category': 'documents'}], 'disclaimer': ''}
        frames = [{'done': False, 'message': {'content': json.dumps(response, ensure_ascii=False)}},
                  {'done': True, 'message': {'content': ''}, 'prompt_eval_count': 100,
                   'eval_count': 73, 'eval_duration': 1_000_000_000}]
        return httpx.Response(200, content='\n'.join(json.dumps(f, ensure_ascii=False) for f in frames)+'\n')
    telemetry = Telemetry()
    adapter = OllamaAdapter(replace(Settings(), answer_mode='quick', quick_single_pass=False),
        telemetry=telemetry, transport=httpx.MockTransport(handler))
    try:
        after = adapter.answer('Vettem egy autót, milyen teendőim vannak?', vehicle_evidence, [],
                               run_id='synthetic-demo', context={'life_events': ['vehicle'], 'role': 'buyer'})
    finally:
        adapter.close()
    assert len(sent) == 2 and sent[0]['stream'] is False and sent[1]['stream'] is True
    assert len(after.claims) >= 3  # source-backed coverage can add missing office step
    assert any('kell kötnöd' in claim.text for claim in after.claims)
    assert all(c.supporting_quote in next(e['text'] for e in vehicle_evidence
               if c.evidence_ids == [e['evidence_id']]) for c in after.claims)
    assert after.claims[1].text != before.claims[0].text
    assert [item['phase'] for item in telemetry.snapshot('synthetic-demo')['llm_usage']] == ['selection', 'answer']


def test_truncated_selection_raises_for_workflow_partial_fallback(vehicle_evidence):
    def handler(request):
        data = json.loads(request.content)
        if not data['stream']:
            return httpx.Response(200, json={'done_reason': 'length', 'message': {
                'content': '{"evidence_by_need": {'}})
        raise AssertionError('Generation must not run after truncated evidence selection')
    adapter = OllamaAdapter(replace(Settings(), answer_mode='quick', quick_single_pass=False), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(LLMError, match='LLMError'):
            adapter.answer('Vettem egy autót', vehicle_evidence, [], context={'life_events': ['vehicle']})
    finally:
        adapter.close()


def test_automatic_silver_cannot_impersonate_human_gold():
    """The shipped automatic reference must never claim human approval."""
    path = Path(__file__).resolve().parents[2] / 'evaluation' / 'golden_auto_v4.json'
    if not path.is_file():
        # A clean clone contains only the baseline; synthetic SILVER generation
        # is exercised in tests/evaluation/test_automatic_reference_all20.py.
        baseline = json.loads((path.parent / 'golden_v4.json').read_text(encoding='utf-8'))
        assert len(baseline['cases']) == 20
        assert all(case['human_reviewed'] is False for case in baseline['cases'])
        return
    silver = json.loads(path.read_text(encoding='utf-8'))
    assert len(silver['cases']) == 20
    assert silver['human_reviewed'] is False
    assert all(case.get('human_reviewed') is False and case.get('automatic_reference')
               for case in silver['cases'])


def test_missing_facet_triggers_bounded_targeted_retry(monkeypatch):
    """Run real RAG node closures with a tiny graph stub, not a fake LLM/index."""
    import importlib
    import sys
    from types import ModuleType

    class RecordingGraph:
        def __init__(self, *args):
            self.nodes = {}
            self.routes = {}
        def add_node(self, name, func):
            self.nodes[name] = func
        def add_edge(self, *args):
            pass
        def add_conditional_edges(self, name, route):
            self.routes[name] = route
        def compile(self):
            return self

    graph_module = ModuleType('langgraph.graph')
    graph_module.StateGraph = RecordingGraph
    graph_module.START, graph_module.END = 'start', 'end'
    pkg = ModuleType('langgraph')
    pkg.graph = graph_module
    monkeypatch.setitem(sys.modules, 'langgraph', pkg)
    monkeypatch.setitem(sys.modules, 'langgraph.graph', graph_module)
    import dap_assistant
    rag_path = Path(dap_assistant.__file__).resolve().parent / 'rag' / 'rag_graph.py'
    spec = importlib.util.spec_from_file_location('dap_assistant._rag_graph_test_only', rag_path)
    rag_graph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_graph)
    calls = []
    url = 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek'
    rows = {
        'steps': {'chunk_id': 'step', 'text': 'A gépjármű átírás ügyintézési lépés szükséges.', 'score': .02},
        'insurance': {'chunk_id': 'insurance', 'text': 'Az új kötelező gépjármű-felelősségbiztosítás szükséges.', 'score': .02},
        'deadline': {'chunk_id': 'deadline', 'text': 'Az átírás 15 napon belül szükséges.', 'score': .02},
        'documents': {'chunk_id': 'documents', 'text': 'Az átírás adásvételi szerződés és törzskönyv szükséges.', 'score': .02},
        'where': {'chunk_id': 'where', 'text': 'Az ügyintézés a kormányablakban történik.', 'score': .02},
    }
    cycle = [1]
    def fake_search(query, *args, **kwargs):
        calls.append(query)
        if 'hivatalos' in query:
            cycle[0] = 2
        if cycle[0] == 1:
            return [{**rows['steps'], 'document_id': 'dap-vehicle-buyer', 'domain': 'vehicle',
                     'role': 'buyer', 'source_url': url, 'section_path': ['Ügyintézés']}]
        facet_query = query.split(' hivatalos ', 1)[0]
        facet = next((f for f in ('insurance', 'deadline', 'documents', 'where')
                      if any(token in facet_query for token in {'insurance': ('felelősségbiztosítás',),
                      'deadline': ('határidő',), 'documents': ('okmányok',),
                      'where': ('kormányablak',)}[f])), 'steps')
        return [{**rows[facet], 'document_id': 'dap-vehicle-buyer', 'domain': 'vehicle',
                 'role': 'buyer', 'source_url': url, 'section_path': ['Ügyintézés']}]
    monkeypatch.setattr(rag_graph, 'hybrid_search', fake_search)
    settings = replace(Settings(), max_rag_attempts=2)
    graph = rag_graph.build_rag_graph(settings)
    state = {'task_id': 't1', 'domain': 'vehicle', 'role': 'buyer',
             'query': 'Vettem autót, milyen teendőim vannak?', 'search_attempt': 0}
    def run_pass(current):
        for node in rag_graph.RAG_NODE_ORDER[:4]:
            current.update(graph.nodes[node](current))
        return current
    try:
        run_pass(state)
        assert state['retrieval_status'] == 'partial'
        assert 'insurance' in state['missing_facets']
        assert graph.routes['evaluate_evidence'](state) == 'process_query'
        initial_count = len(calls)
        run_pass(state)
        assert len(calls) > initial_count
        assert any('hivatalos' in q for q in calls[initial_count:])
        assert {'step', 'insurance'} <= {c['chunk_id'] for c in state['candidates']}
        assert state['search_attempt'] == 2
        assert graph.routes['evaluate_evidence'](state) == 'prepare_context'
    finally:
        # The graph stub is limited to this temporary module and the test's
        # monkeypatch context; production module globals remain unchanged.
        assert rag_graph.__name__ == 'dap_assistant._rag_graph_test_only'
