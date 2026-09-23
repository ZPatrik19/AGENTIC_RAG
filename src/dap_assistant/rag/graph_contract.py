"""Canonical RAG node order and measurable subflows; no graph dependencies."""
from __future__ import annotations

RAG_NODE_ORDER: tuple[str, ...] = (
    'process_query', 'hybrid_retrieval', 'rerank_results',
    'evaluate_evidence', 'prepare_context',
)

RAG_SUBFLOWS: dict[str, tuple[str, ...]] = {
    'rag/query_to_retrieval': RAG_NODE_ORDER[:2],
    'rag/query_to_rerank': RAG_NODE_ORDER[:3],
    'rag/retrieval_to_rerank': RAG_NODE_ORDER[1:3],
    'rag/full_subgraph': RAG_NODE_ORDER,
}

STANDALONE_RAG_TARGETS: dict[str, tuple[str, ...]] = {
    f'rag/{node}': (node,) for node in RAG_NODE_ORDER
}
