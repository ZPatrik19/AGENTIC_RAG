"""Bounded retrieval and inspectable provenance regressions; never contacts Ollama."""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from dap_assistant.context_engineering.evidence_selection import (
    assemble_selection_context, facet_queries, validate_selection,
)
from dap_assistant.rag.retrieval import hybrid_search
from dap_assistant.settings import Settings


def test_rrf_keeps_raw_rank_and_does_not_inflate_score(monkeypatch, tmp_path):
    from dap_assistant.rag import retrieval

    chunks = [
        {'chunk_id': 'overlap', 'domain': 'vehicle', 'role': 'buyer',
         'document_id': 'dap-vehicle-buyer', 'text': 'Ellenőrzött szerződés.'},
        {'chunk_id': 'dense_only', 'domain': 'vehicle', 'role': 'buyer',
         'document_id': 'dap-vehicle-buyer', 'text': 'Biztosítás.'},
    ]
    monkeypatch.setattr(retrieval, '_chunk_snapshot', lambda _path: chunks)
    monkeypatch.setattr(retrieval, 'bm25_search', lambda *a, **k: [('overlap', 9.0)])

    class Dense:
        def search(self, *a, **k):
            return [('dense_only', 0.9), ('overlap', 0.8)]

    got = hybrid_search('átírás', 'vehicle', replace(Settings(), data_dir=tmp_path), dense=Dense())
    overlap = next(item for item in got if item['chunk_id'] == 'overlap')
    assert overlap['bm25_rank'] == 1
    assert overlap['dense_rank'] == 2
    assert overlap['rrf_components'] == {'bm25': 1 / 61, 'dense': 1 / 62}
    assert math.isclose(overlap['score'], 1 / 61 + 1 / 62)
    assert len(got) == 2


def test_no_silent_facet_limit_truncation():
    broad = ('Elvesztettem a munkámat; milyen teendők, dokumentumok, határidők, '
             'támogatások, jogosultság, járadék összege, költségek, hol intézem?')
    queries = facet_queries(broad, 'employment')
    assert {'steps', 'documents', 'deadline', 'supports', 'eligibility', 'benefit_amount',
            'costs', 'where'} <= set(queries)
    with pytest.raises(ValueError, match='discard'):
        facet_queries(broad, 'employment', max_queries=1)


def test_selection_keeps_whole_units_and_reports_excluded():
    evidence = [
        {'evidence_id': 'E_INS', 'title': 'Biztosítás', 'source_url': 'https://dap.gov.hu/a',
         'text': 'Tulajdonosváltáskor új kötelező gépjármű-felelősségbiztosítást kell kötni.'},
        {'evidence_id': 'E_DEAD', 'title': 'Határidő', 'source_url': 'https://dap.gov.hu/b',
         'text': 'Az átírást 15 napon belül szükséges intézni.'},
        {'evidence_id': 'E_LONG', 'title': 'Hosszú szöveg', 'source_url': 'https://dap.gov.hu/c',
         'text': 'Nagyon hosszú, irreleváns háttér. ' * 500},
    ]
    packed = assemble_selection_context(evidence, 'Vettem egy autót. Milyen teendőim vannak?',
                                         domain='vehicle', role='buyer', token_budget=300)
    assert packed.evidence
    assert any('kell kötni.' in item['text'] for item in packed.evidence)
    assert packed.estimated_input_tokens <= 300
    assert 'E_LONG' in packed.rejected_ids
    with pytest.raises(ValueError, match='Invalid selected evidence'):
        validate_selection({'insurance': ['E_INVENTED']}, {'E_INS'}, ('insurance',))


def test_facet_retrieval_exposes_every_query_rank_and_separate_rerank(monkeypatch):
    """The selected graph nodes are real closures; only LangGraph's import is stubbed."""
    class Graph:
        def __init__(self, *_a):
            self.nodes = {}
        def add_node(self, name, fn):
            self.nodes[name] = fn
        def add_edge(self, *_a):
            pass
        def add_conditional_edges(self, *_a):
            pass
        def compile(self):
            return self

    graph_mod = ModuleType('langgraph.graph')
    graph_mod.StateGraph, graph_mod.START, graph_mod.END = Graph, 'START', 'END'
    pkg = ModuleType('langgraph')
    pkg.graph = graph_mod
    monkeypatch.setitem(sys.modules, 'langgraph', pkg)
    monkeypatch.setitem(sys.modules, 'langgraph.graph', graph_mod)
    import dap_assistant

    path = Path(dap_assistant.__file__).parent / 'rag' / 'rag_graph.py'
    spec = importlib.util.spec_from_file_location('dap_assistant._rag_graph_v6_test', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    def search(query, *_a, **_kwargs):
        is_insurance = 'felelősségbiztosítás' in query
        rank = 3 if is_insurance else 1
        return [dict(chunk_id='same', domain='vehicle', role='buyer',
                     document_id='dap-vehicle-buyer', source_url='https://dap.gov.hu/example',
                     text='Az átírás 15 napon belül történik. Kötelező gépjármű-felelősségbiztosítást kell kötni.',
                     section_path=['Ügyintézés'], score=1/(60+rank),
                     bm25_rank=rank, dense_rank=rank, bm25_score=4.0, dense_score=0.5,
                     rrf_components={'bm25': 1/(60+rank)})]
    monkeypatch.setattr(module, 'hybrid_search', search)
    graph = module.build_rag_graph(replace(Settings(), max_rag_attempts=2))
    state = {'query': 'Vettem autót, milyen teendőim vannak?', 'domain': 'vehicle',
             'role': 'buyer', 'search_attempt': 0}
    state.update(graph.nodes['process_query'](state))
    state.update(graph.nodes['hybrid_retrieval'](state))
    assert len(state['candidates']) == 1
    traces = state['candidates'][0]['retrieval_traces']
    labels = {item['query_label'] for item in traces}
    assert 'main' in labels and 'facet:insurance:attempt:1' in labels
    assert next(item for item in traces if item['query_label'] == 'main')['bm25_rank'] == 1
    assert next(item for item in traces if item['query_label'] == 'facet:insurance:attempt:1')['bm25_rank'] == 3
    state.update(graph.nodes['rerank_results'](state))
    assert state['ranked'][0]['score'] <= 1/61  # Original RRF unchanged by rerank bonus.
    assert state['ranked'][0]['rerank_score'] > state['ranked'][0]['score']


def test_natural_generation_deduplicates_rephrased_same_fact():
    from dap_assistant.llm import NaturalClaim, NaturalResponse, _attach_source_quotes

    source = {'evidence_id': 'E_DOC', 'text': 'Az átíráshoz szükséges az adásvételi szerződés és a törzskönyv.'}
    response = NaturalResponse(claims=[
        NaturalClaim(evidence_id='E_DOC', text='Az átíráshoz az adásvételi szerződés szükséges.', category='documents'),
        NaturalClaim(evidence_id='E_DOC', text='A szerződés szükséges az átíráshoz.', category='documents'),
    ])
    result = _attach_source_quotes(response, [source])
    assert len(result.claims) == 1
    assert result.claims[0].supporting_quote == source['text']
