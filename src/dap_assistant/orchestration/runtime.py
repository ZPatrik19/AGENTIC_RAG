"""Framework-independent runtime assembly for chat and its Streamlit adapter.

Streamlit owns caching and widgets; this module owns configuration and resource
construction. It does not start inference or open a model merely on import.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol

from ..observability.telemetry import Telemetry
from ..rag.dense_resources import acquire_dense
from ..settings import Settings


class WorkflowBuilder(Protocol):
    def __call__(self, settings: Settings, *, dense: object, telemetry: Telemetry) -> object: ...


def open_chat_index(data_dir: str, embedding_model: str, provider: str, *,
                    base_settings: Settings | None = None,
                    dense_acquirer: Callable[[Settings], object] | None = None):
    """Acquire an existing dense index; the UI cache keeps the lease alive."""
    settings = replace(
        base_settings if base_settings is not None else Settings(),
        data_dir=Path(data_dir), embedding_model=embedding_model,
        embedding_provider=provider,
    )
    if provider == 'dummy' or not (settings.data_dir / 'vectorstore' / 'qdrant').exists():
        return None
    acquire = dense_acquirer if dense_acquirer is not None else acquire_dense
    return acquire(settings)


def create_chat_workflow(
    *, provider: str, fast: bool, answer_mode: str, data_dir: str,
    embedding_model: str, embedding_provider: str, read_timeout_s: float, total_timeout_s: float, dense=None,
    workflow_builder: WorkflowBuilder | None = None,
    base_settings: Settings | None = None,
    telemetry_factory: Callable[[], Telemetry] | None = None,
):
    """Build one configured graph and its run telemetry without Streamlit imports."""
    settings = replace(
        base_settings if base_settings is not None else Settings(),
        llm_provider=provider, fast_routing=fast, answer_mode=answer_mode,
        data_dir=Path(data_dir), embedding_model=embedding_model,
        embedding_provider=embedding_provider, ollama_read_timeout_s=read_timeout_s,
        ollama_total_timeout_s=total_timeout_s,
    )
    telemetry = telemetry_factory() if telemetry_factory is not None else Telemetry()
    if workflow_builder is None:
        # A dummy or structural test must be able to import the application
        # layer without the optional LangGraph integration being installed.
        from ..workflow import build_workflow as workflow_builder
    return workflow_builder(settings, dense=dense, telemetry=telemetry), telemetry
