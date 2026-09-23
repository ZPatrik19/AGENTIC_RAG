"""Offline regressions: legal data never inferred, real retrieval ranks only."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from dap_assistant.response.administrative_steps import extract_steps
from dap_assistant.tooling.calculators import (
    LoanInput, VehicleDutyInput, calculate_illustrative_loan,
    calculate_vehicle_acquisition_duty, parse_loan_request,
    parse_vehicle_duty_request,
)
from dap_assistant.evaluation.layers import per_task_retrieval, verified_task_execution
from dap_assistant.evaluation.functional_metrics import _full_metrics
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.documents.ingestion import make_chunks, parse_html
from dap_assistant.llm import source_answer
from dap_assistant.documents.sources import load_manifest
from dap_assistant.presentation.workflow_visualization import execution_dot


NAV_TABLE = """Gépjármű visszterhes vagyonszerzési illeték 2026.
Teljesítmény (kW) | 0–3 év | 4–8 év | 8 év felett
0–40 | 550 Ft/kW | 450 Ft/kW | 300 Ft/kW
41–80 | 750 Ft/kW | 550 Ft/kW | 450 Ft/kW
81–120 | 850 Ft/kW | 750 Ft/kW | 550 Ft/kW
120 felett | 950 Ft/kW | 850 Ft/kW | 750 Ft/kW"""


def nav_evidence(text=NAV_TABLE, version='sha-version'):
    return [{'evidence_id': 'E_NAV', 'document_id': 'nav-vehicle-duty-2026',
             'source_url': 'https://nav.gov.hu/ugyfeliranytu/adokulcsok_jarulekmertekek/valorizalt-adomertekek/gepjarmu-visszterhes-vagyonszerzesi-illetek/gepjarmu-visszterhes-vagyonszerzesi-illetek',
             'title': 'Gépjármű illeték 2026', 'text': text, 'document_version': version,
             'section_path': ['Gépjármű illeték 2026'], 'domain': 'vehicle'}]


def test_manifest_four_real_domain_collections_and_document_types():
    path = Path(__file__).resolve().parents[2] / 'config' / 'document_sources.yaml'
    sources = load_manifest(path).sources
    by_domain = {domain: [s for s in sources if s.domain == domain]
                 for domain in ('vehicle', 'employment')}
    assert all(len(items) >= 4 for items in by_domain.values())
    assert len({s.id for s in sources}) == len(sources)
    assert any(s.type == 'pdf' for s in by_domain['employment'])
    # Official JS-only portal pages remain in the manifest for provenance, but
    # cannot be indexed as ordinary HTML until an extractable source exists.
    assert all(s.url.startswith('https://') and s.processing_status in ('pending', 'disabled') for s in sources)


def test_html_tariff_table_retains_header_and_all_rates_in_one_chunk():
    from dap_assistant.documents.sources import Source
    html = ('<main><h1>Gépjármű illeték 2026</h1><p>Bevezető</p><table>' +
            ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row.split('|')) + '</tr>'
                    for row in NAV_TABLE.splitlines()[1:]) + '</table></main>')
    sections = parse_html(html.encode())
    assert any('0–3 év' in s['text'] and '950 Ft/kW' in s['text'] for s in sections)
    src = Source(id='nav-vehicle-duty-2026', title='Gépjármű illeték 2026',
                 url='https://nav.gov.hu/ugyfeliranytu/', domain='vehicle',
                 destination='vehicle', type='html')
    chunks = make_chunks(src, {'sha256': 'a' * 64, 'retrieved_at': '2026-09-20'}, sections)
    assert any('0–3 év' in c['text'] and '950 Ft/kW' in c['text'] for c in chunks)


def test_vehicle_duty_exact_source_checked_and_tied_to_document():
    result = calculate_vehicle_acquisition_duty(
        VehicleDutyInput(manufacturing_year=2015, registered_kw=155), nav_evidence())
    assert result['status'] == 'calculated'
    assert result['amount_huf'] == 116250
    assert result['rate_huf_per_kw'] == 750
    assert result['source_evidence_ids'] == ['E_NAV']
    assert result['document_version'] == 'sha-version'


@pytest.mark.parametrize('evidence,year', [
    ([], 2026), (nav_evidence(NAV_TABLE.replace('750 Ft/kW', '999 Ft/kW')), 2026),
    (nav_evidence(NAV_TABLE.replace('2026', '2025')), 2026),
    (nav_evidence(), 2025),
])
def test_vehicle_duty_fails_closed_on_missing_or_wrong_tariff(evidence, year):
    result = calculate_vehicle_acquisition_duty(
        VehicleDutyInput(manufacturing_year=2015, registered_kw=155, assessment_year=year), evidence)
    assert result['status'] == 'incomplete'
    assert 'amount_huf' not in result


def test_hp_approximation_is_not_an_official_kw_or_payable_amount():
    req = parse_vehicle_duty_request('Névreírás díja egy 2015-ös 211 lóerős autónál?', 2026)
    assert req and req.manufacturing_year == 2015 and req.horsepower == 211
    result = calculate_vehicle_acquisition_duty(req, nav_evidence())
    assert result['status'] == 'incomplete' and result['approximation_only']
    assert 'amount_huf' not in result


def test_housing_loan_annuity_is_clearly_illustrative():
    req = parse_loan_request('Lakáshitel 20 millió Ft, 6% kamat, 20 év futamidő')
    assert req and req.months == 240
    result = calculate_illustrative_loan(req)
    assert result['status'] == 'calculated' and result['monthly_payment_huf'] > 100_000
    assert 'THM' in ' '.join(result['assumptions'])
    zero = calculate_illustrative_loan(LoanInput(principal_huf=1_200_000,
                                                  annual_interest_percent=Decimal('0'), months=12))
    assert zero['monthly_payment_huf'] == 100_000


def test_bare_source_bullets_are_kept_verbatim_and_cited():
    evidence = [{**nav_evidence()[0], 'text': 'A vagyonszerzési illeték: 750 Ft/kW\n'
                 'A foglalkoztatási osztályon intézheted az álláskeresőként nyilvántartásba vételt.'}]
    steps = extract_steps(evidence)
    assert any('750 Ft/kW' in s.description for s in steps)
    assert any('foglalkoztatási osztály' in s.description for s in steps)
    draft = source_answer('Mennyi a vagyonszerzési illeték?', evidence)
    assert any('750 Ft/kW' in c.text and c.evidence_ids == ['E_NAV'] for c in draft.claims)


def test_retrieval_scoring_uses_true_task_rank_and_requires_review():
    output = {'branch_results': {
        't1': {'ranked_chunk_ids': ['offtopic', 'expected_vehicle'], 'evidence': [{'chunk_id': 'offtopic'}]},
        't2': {'ranked_chunk_ids': ['expected_business'], 'evidence': [{'chunk_id': 'offtopic'}]},
    }}
    case = {'relevant_chunk_ids_by_task': {'t1': ['expected_vehicle'], 't2': ['expected_business']}}
    result = per_task_retrieval(output, case, pinned=True)
    assert result['recall_at_5'] == 1
    assert result['mrr'] == 0.75
    assert per_task_retrieval(output, case, pinned=False)['mrr'] is None
    assert per_task_retrieval(output, {'relevant_chunk_ids': ['expected_vehicle']}, pinned=True)['mrr'] is None


def test_completed_task_requires_retrieval_status_not_merely_one_chunk():
    result = verified_task_execution({
        'subtasks': {'t1': {'status': 'complete'}, 't2': {'status': 'complete'}},
        'branch_results': {'t1': {'retrieval_status': 'insufficient', 'evidence': [{'chunk_id': 'x'}]},
                           't2': {'retrieval_status': 'complete', 'evidence': [{'chunk_id': 'y'}]}},
    })
    assert result == 0.5


def test_real_workflow_graph_only_draws_executed_nodes_and_tool():
    graph = execution_dot(
        {'subtasks': {'t1': {'domain': 'vehicle', 'status': 'complete', 'depends_on': []}},
         'tool_results': {'vehicle_duty': {'tool': 'calculate_vehicle_acquisition_duty', 'status': 'calculated'}}},
        [{'node': node} for node in ('classify_intent', 'plan_tasks', 'rag_worker', 'execute_tools')],
    )
    assert 'calculate_vehicle_acquisition_duty' in graph
    assert 'task_t1' in graph
    assert 'answer_audit' not in graph


def test_source_recall_uses_raw_rank_not_reordered_final_evidence():
    case = {'question_id': 't', 'question': 'autó', 'expected_domains': ['vehicle'],
            'expected_intents': ['vehicle_information'], 'expected_subtasks': [
                {'domain': 'vehicle', 'keywords': []}], 'expected_source_ids': ['target'],
            'expected_tool_calls': [], 'expected_behavior': 'normal', 'expected_facts': [],
            'human_reviewed': True, 'expected_source_versions': {'target': 'ver'},
            'relevant_chunk_ids': ['gold'], 'relevant_chunk_ids_by_task': {'t1': ['gold']}}
    chunks = [{'chunk_id': 'gold', 'document_id': 'target', 'document_version': 'ver',
               'domain': 'vehicle', 'text': 'text'}]
    output = {'domains': ['vehicle'], 'intents': ['vehicle_information'],
              'subtasks': {'t1': {'domain': 'vehicle', 'question': 'autó', 'status': 'complete', 'depends_on': []}},
              'branch_results': {'t1': {'retrieval_status': 'complete', 'ranked_chunk_ids': ['gold'],
                                       'evidence': [{'chunk_id': 'not-the-gold', 'document_id': 'wrong'}]}},
              'evidence': [], 'answer_draft': {'claims': []}, 'answer_validation': {'status': 'passed'},
              'response_status': 'complete', 'final_answer': 'test'}
    metrics, _ = _full_metrics(
        case, output, {'spans': []}, {'target': 'ver'}, chunks, Telemetry(), 'source-rank-test'
    )
    assert metrics['source_recall_at_5'] == 1.0
    assert metrics['retrieval_recall_at_5'] == 1.0
    assert metrics['task_completion_rate'] == 1.0


def test_extra_official_reference_requires_explicit_review_and_pins_new_version(tmp_path):
    from dap_assistant.evaluation.dataset import load_dataset, reference_status
    from dap_assistant.evaluation.review import review_case
    case = load_dataset()[1]
    initial_source_ids = list(case['expected_source_ids'])
    original = {'chunk_id': 'buyer-001', 'document_id': 'dap-vehicle-buyer',
                'document_version': 'buyer-v1', 'domain': 'vehicle',
                'source_url': 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek',
                'text': 'Az átírást 15 napon belül végezd el.'}
    extra = {'chunk_id': 'nav-001', 'document_id': 'nav-vehicle-duty-2026',
             'document_version': 'nav-v1', 'domain': 'vehicle',
             'source_url': nav_evidence()[0]['source_url'], 'text': NAV_TABLE}
    patterns = {fact['fact_id']: r'15\s+nap' for fact in case['expected_facts']}
    kwargs = dict(case_id=case['question_id'], selected_chunks=['buyer-001', 'nav-001'],
                  answer_patterns=patterns, confirm_facts=True, confirm_chunks=True,
                  chunks=[original, extra], destination=tmp_path / 'reviewed.json',
                  task_chunk_ids={'t1': ['buyer-001', 'nav-001']})
    with pytest.raises(ValueError, match='nem tartozik'):
        review_case(**kwargs)
    reviewed = review_case(**kwargs, additional_source_ids=['nav-vehicle-duty-2026'])
    assert set(reviewed['expected_source_ids']) == {'dap-vehicle-buyer', 'nav-vehicle-duty-2026'}
    assert reference_status(reviewed, {'dap-vehicle-buyer': 'buyer-v1',
                                       'nav-vehicle-duty-2026': 'nav-v1'}) == 'pinned'
    assert reference_status(reviewed, {'dap-vehicle-buyer': 'buyer-v1',
                                       'nav-vehicle-duty-2026': 'nav-v2'}) == 'version_mismatch'
    assert load_dataset()[1]['expected_source_ids'] == initial_source_ids  # baseline not rewritten


def test_nonofficial_source_cannot_be_pinned_as_additional_gold(tmp_path):
    from dap_assistant.evaluation.dataset import load_dataset
    from dap_assistant.evaluation.review import review_case
    case = load_dataset()[1]
    with pytest.raises(ValueError, match='hivatalos manifestjében'):
        review_case(case['question_id'], selected_chunks=['anything'],
                    answer_patterns={}, confirm_facts=True, confirm_chunks=True,
                    chunks=[], destination=tmp_path / 'reviewed.json',
                    additional_source_ids=['example-unofficial'])


def test_main_workflow_routes_personalized_calculations_and_preserves_results(monkeypatch, tmp_path):
    """Graph-node integration without pretending that LangGraph/Ollama is installed."""
    import importlib
    import sys
    import types
    from dap_assistant.settings import Settings

    class GraphStub:
        def __init__(self, _state):
            self.nodes = {}
            self.routes = {}
        def add_node(self, name, func):
            self.nodes[name] = func
        def add_edge(self, *_args):
            pass
        def add_conditional_edges(self, name, route):
            self.routes[name] = route
        def compile(self, **_kwargs):
            return self

    modules = {'langgraph': types.ModuleType('langgraph'),
               'langgraph.graph': types.ModuleType('langgraph.graph'),
               'langgraph.types': types.ModuleType('langgraph.types'),
               'langgraph.checkpoint': types.ModuleType('langgraph.checkpoint'),
               'langgraph.checkpoint.memory': types.ModuleType('langgraph.checkpoint.memory')}
    modules['langgraph.graph'].START = 'START'
    modules['langgraph.graph'].END = 'END'
    modules['langgraph.graph'].StateGraph = GraphStub
    modules['langgraph.types'].Send = lambda *args, **kwargs: (args, kwargs)
    modules['langgraph.checkpoint.memory'].InMemorySaver = lambda: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Restore imported modules as well as monkeypatched LangGraph stubs.
    # Otherwise a later pytest file may silently import the fake graph runtime.
    package = importlib.import_module('dap_assistant')
    rag_package = importlib.import_module('dap_assistant.rag')
    owners = {'workflow': package, 'rag_graph': rag_package}
    fullnames = {'workflow': 'dap_assistant.workflow', 'rag_graph': 'dap_assistant.rag.rag_graph'}
    previous = {name: sys.modules.get(fullnames[name]) for name in owners}
    previous_attributes = {name: getattr(owner, name, None) for name, owner in owners.items()}
    try:
        # Other optional integration tests still skip without real LangGraph.
        monkeypatch.delitem(sys.modules, 'dap_assistant.workflow', raising=False)
        monkeypatch.delitem(sys.modules, 'dap_assistant.rag.rag_graph', raising=False)
        # `from package import submodule` otherwise reuses a cached attribute,
        # even after sys.modules was cleared. Keep this graph stub isolated.
        monkeypatch.delattr(package, 'workflow', raising=False)
        monkeypatch.delattr(rag_package, 'rag_graph', raising=False)
        from dap_assistant.rag import rag_graph
        import dap_assistant.workflow as workflow

        settings = replace(Settings(), data_dir=tmp_path, llm_provider='dummy',
                           embedding_provider='dummy', answer_mode='source')
        graph = workflow.build_workflow(settings)
        assert len(graph.nodes) == 10
        assert len(rag_graph.build_rag_graph(settings).nodes) == 5

        # A follow-up can have a valid office statement but no evidence for
        # unrelated requested facets. After the bounded retry, do not drop it.
        office = {
            'evidence_id': 'E_OFFICE', 'chunk_id': 'office',
            'document_id': 'dap-vehicle-buyer', 'domain': 'vehicle',
            'source_url': 'https://dap.gov.hu/teszt', 'title': 'DÁP',
            'document_version': 'fixture',
            'text': 'Az átíráshoz keresd fel bármelyik kormányablakot, vagy foglalj időpontot.',
        }
        incomplete = workflow.initial_state('Hol tudom ezeket az ügyeket intézni?')
        incomplete.update(domains=['vehicle'], retry_count=1,
                          subtasks={'t1': {'status': 'running', 'depends_on': []}},
                          branch_results={'t1': {'retrieval_status': 'partial', 'evidence': [office]}})
        update = graph.nodes['evidence_gate'](incomplete)
        assert update['subtasks']['t1']['status'] == 'partial'
        assert update['validation']['status'] == 'partial'
        assert update['evidence'] == [office]
        assert not update['pending_task_ids']
        no_evidence = {**incomplete, 'branch_results': {
            't1': {'retrieval_status': 'insufficient', 'evidence': []}}}
        rejected = graph.nodes['evidence_gate'](no_evidence)
        assert rejected['subtasks']['t1']['status'] == 'failed'
        assert rejected['validation']['status'] == 'failed'
        assert not rejected['evidence']

        # The retained statement must be surfaced with a PARTIAL status, not
        # silently reclassified as a complete legal answer.
        incomplete.update(update)
        incomplete.update(graph.nodes['generate_answer'](incomplete))
        audited = graph.nodes['answer_audit'](incomplete)
        assert 'kormányablak' in audited['final_answer']
        assert audited['response_status'] == 'partial'

        # Retrieval metadata is not a user request for an unrelated checklist.
        support_question = 'Milyen támogatások járhatnak, ha megszűnt a munkaviszonyom?'
        support = workflow.initial_state(support_question)
        support.update(domains=['employment'],
                       resolved_question='Korábbi ügyintézési téma: munkaviszony megszűnéséhez kapcsolódó ügyintézés. Aktuális kérdés: ' + support_question,
                       subtasks={'t1': {'status': 'partial'}},
                       validation={'status': 'partial'})
        assert graph.routes['evidence_gate'](support) == 'engineer_context'
        assert graph.routes['engineer_context'](support) == 'generate_answer'
        assert graph.nodes['execute_tools'](support)['tool_results'] == {}

        state = workflow.initial_state('Mennyi a névreírás díja egy 2015-ös 155 kW-os autónak?',
                                        reference_date=__import__('datetime').date(2026, 9, 20))
        state.update(domains=['vehicle'], role='buyer', validation={'status': 'passed'},
                     subtasks={'t1': {'status': 'complete'}}, evidence=nav_evidence())
        assert graph.routes['evidence_gate'](state) == 'engineer_context'
        assert graph.routes['engineer_context'](state) == 'execute_tools'
        tool_update = graph.nodes['execute_tools'](state)
        assert tool_update['tool_results']['vehicle_duty']['amount_huf'] == 116250
        state.update(tool_update)
        state.update(graph.nodes['generate_answer'](state))
        audit = graph.nodes['answer_audit'](state)
        assert '116 250 Ft' in audit['final_answer']
        assert 'NAV-forrás' in audit['final_answer']

        loan_state = workflow.initial_state('Lakáshitel 20 millió Ft 6% kamat 20 év',
                                            reference_date=__import__('datetime').date(2026, 9, 20))
        loan_state.update(domains=['housing'], validation={'status': 'passed'},
                          subtasks={'t1': {'status': 'complete'}}, evidence=nav_evidence())
        assert graph.routes['evidence_gate'](loan_state) == 'engineer_context'
        assert graph.routes['engineer_context'](loan_state) == 'execute_tools'
        assert graph.nodes['execute_tools'](loan_state)['tool_results']['illustrative_loan']['status'] == 'calculated'
    finally:
        for name in ('workflow', 'rag_graph'):
            fullname = fullnames[name]
            sys.modules.pop(fullname, None)
            if previous[name] is not None:
                sys.modules[fullname] = previous[name]
            if previous_attributes[name] is not None:
                setattr(owners[name], name, previous_attributes[name])
            elif hasattr(owners[name], name):
                delattr(owners[name], name)
