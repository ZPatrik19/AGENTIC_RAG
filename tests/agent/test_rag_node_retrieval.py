"""Exercise real retrieval node logic without installing LangGraph or embeddings."""
from __future__ import annotations

from dataclasses import replace

from dap_assistant.rag.nodes import RAGNodes
from dap_assistant.settings import Settings


def _chunk(chunk_id: str, score: float) -> dict:
    return {
        'chunk_id': chunk_id, 'document_id': 'doc', 'document_version': 'v1',
        'text': 'Ügyintézés a kormányablakban', 'domain': 'vehicle',
        'role': 'buyer', 'score': score,
        'bm25_rank': 1, 'dense_rank': None,
    }


def test_domain_coverage_retains_main_rank_provenance_and_merges_duplicates(tmp_path):
    calls = []

    def search(query, domain, settings, **kwargs):
        calls.append((query, kwargs['limit']))
        if len(calls) == 1:
            return [_chunk('one', 0.01)]
        return [_chunk('one', 0.03), _chunk('two', 0.02)]

    settings = replace(Settings(), data_dir=tmp_path, embedding_provider='dummy')
    nodes = RAGNodes(settings, dense=None, telemetry=None,
                     hybrid_search_fn=search, diversify_results_fn=lambda xs, **kw: xs)
    result = nodes.retrieve({
        'query': 'Hol intézem az átírást?', 'original_query': 'Hol intézem az átírást?',
        'domain': 'vehicle', 'role': 'buyer', 'facet': 'where',
        'search_attempt': 1,
    })
    assert len(calls) == 2  # main + one bounded domain coverage search
    assert result['requested_facets'] == ['where']
    assert result['pre_rerank_chunk_ids'] == ['one', 'two']
    by_id = {item['chunk_id']: item for item in result['candidates']}
    assert by_id['one']['score'] == 0.03
    assert [t['query_label'] for t in by_id['one']['retrieval_traces']] == [
        'main', 'vehicle_coverage',
    ]


def test_missing_facet_retry_reuses_previous_evidence(tmp_path):
    calls = []

    def search(query, domain, settings, **kwargs):
        calls.append(query)
        return [_chunk('fresh', 0.02)]

    nodes = RAGNodes(replace(Settings(), data_dir=tmp_path, embedding_provider='dummy'),
                     dense=None, telemetry=None, hybrid_search_fn=search,
                     diversify_results_fn=lambda xs, **kw: xs)
    previous = _chunk('previous', 0.04)
    result = nodes.retrieve({
        'query': 'határidő új keresés', 'original_query': 'Meddig kell intézni?',
        'domain': 'vehicle', 'role': 'buyer', 'facet': 'deadline',
        'missing_facets': ['deadline'], 'search_attempt': 2,
        'candidates': [previous],
    })
    assert len(calls) == 2  # main + the missing deadline query, not other facets
    assert result['requested_facets'] == ['deadline']
    assert result['pre_rerank_chunk_ids'] == ['previous', 'fresh']
