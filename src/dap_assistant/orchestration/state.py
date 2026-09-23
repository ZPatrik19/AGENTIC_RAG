"""Explicit LangGraph state contracts and parallel-branch reducers."""
from __future__ import annotations
from typing import Annotated, Any, Literal, TypedDict


class TaskRecord(TypedDict, total=False):
    """A planned task's mutable execution metadata (optional for old checkpoints)."""

    task_id: str
    domain: str
    question: str
    facet: str
    status: Literal['pending', 'running', 'complete', 'partial', 'failed']
    depends_on: list[str]
    worker_retries: int
    role: str


class EvidenceChunk(TypedDict, total=False):
    """Versioned source reference; richer retrieval metadata is also permitted."""

    evidence_id: str
    chunk_id: str
    document_id: str
    document_version: str
    source_url: str
    text: str
    domain: str


class BranchResult(TypedDict, total=False):
    """One RAG worker's findings, keyed by task_id in AssistantState."""

    task_id: str
    evidence: list[EvidenceChunk]
    ranked_chunk_ids: list[str]
    retrieval_status: str
    missing_information: list[str]


class ToolResult(TypedDict, total=False):
    status: str
    result: Any
    error: str


def merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Merge parallel worker updates without mutating either input.

    Right-hand keys take precedence, including legitimate retries."""
    return {**(left or {}), **(right or {})}


class AssistantState(TypedDict, total=False):
    user_question: str
    resolved_question: str
    previous_turn: dict
    context_resolution: dict
    question_analysis: dict
    focus: str
    unsupported_reason: str
    run_id: str
    conversation_summary: str
    reference_date: str
    user_context: dict
    classification_status: str
    domains: list[str]
    intents: list[str]
    role: str
    stage: str
    clarification_question: str
    subtasks: dict[str, TaskRecord]
    pending_task_ids: list[str]
    branch_results: Annotated[dict[str, BranchResult], merge_dict]
    evidence: list[EvidenceChunk]
    context_evidence: list[EvidenceChunk]
    context_engineering: dict
    sources: dict[str, dict]
    tool_results: Annotated[dict[str, ToolResult], merge_dict]
    native_tool_results: dict[str, dict]
    native_tool_trace: dict
    administrative_steps: list[dict]
    validation: dict
    answer_draft: dict
    answer_context: dict
    answer_fallback: bool
    answer_fallback_reason: str
    answer_strategy: str
    answer_validation: dict
    final_answer: str
    response_status: str
    retry_count: int
    execution_round: int
    errors: list[str]


class WorkerState(TypedDict):
    task: TaskRecord
    search_attempt: int
    run_id: str
