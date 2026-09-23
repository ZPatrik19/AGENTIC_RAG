"""Typed data contract for the independently executable RAG subgraph.

TypedDict describes the boundary without requiring synthetic fields in partial
node fixtures. Validated source versions are still checked by retrieval/evaluation.
"""
from __future__ import annotations

from typing import TypedDict


class RAGChunk(TypedDict, total=False):
    chunk_id: str
    document_id: str
    document_version: str
    source_url: str
    text: str
    domain: str
    score: float
    bm25_rank: int
    dense_rank: int


class RAGState(TypedDict, total=False):
    task_id: str
    run_id: str
    domain: str
    role: str
    facet: str
    query: str
    original_query: str
    search_attempt: int
    candidates: list[RAGChunk]
    ranked: list[RAGChunk]
    pre_rerank_chunk_ids: list[str]
    ranked_chunk_ids: list[str]
    ranked_document_ids: list[str]
    evidence: list[RAGChunk]
    retrieval_status: str
    missing_information: list[str]
    requested_facets: list[str]
    selected_facet_evidence: dict[str, list[str]]
    missing_facets: list[str]
    rejected_chunk_ids: list[str]
