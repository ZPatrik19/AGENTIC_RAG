"""Validation and shared-resource acquisition for real local benchmarks."""
from __future__ import annotations

from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.llm import ollama_health
from dap_assistant.rag.dense_resources import acquire_dense, release_dense
from dap_assistant.settings import Settings

from .index_health import check_index


class BenchmarkPrerequisiteError(RuntimeError):
    """Raised when a requested real benchmark cannot be measured faithfully."""


def acquire_verified_dense(
    settings: Settings,
    *,
    require_inference: bool = True,
    real_only: bool = True,
):
    """Lease and validate the shared dense index used by benchmark/evaluation runs.

    The helper never opens a second embedded Qdrant instance in-process.  When
    validation fails, only the lease acquired by this call is released.
    """
    if real_only:
        if settings.embedding_provider != 'sentence_transformers':
            raise BenchmarkPrerequisiteError(
                'Benchmark requires real sentence-transformer embeddings'
            )
        if require_inference and settings.llm_provider != 'ollama':
            raise BenchmarkPrerequisiteError(
                'E2E benchmark requires local Ollama answer generation'
            )
        if require_inference and not ollama_health(settings).get('configured_model_found'):
            raise BenchmarkPrerequisiteError(
                f'Local Ollama / model {settings.ollama_model} unavailable'
            )
        if not (settings.data_dir / 'vectorstore' / 'qdrant').exists():
            raise BenchmarkPrerequisiteError(
                'Dense index missing; run python scripts/download_documents.py --index'
            )

    if settings.embedding_provider != 'sentence_transformers':
        return None
    if not (settings.data_dir / 'vectorstore' / 'qdrant').exists():
        if real_only:
            raise BenchmarkPrerequisiteError('Qdrant index missing')
        return None
    if real_only and not load_chunks(settings.data_dir):
        raise BenchmarkPrerequisiteError('Processed official documents missing')

    dense = acquire_dense(settings)
    if dense is None:
        if real_only:
            raise BenchmarkPrerequisiteError('Qdrant index unavailable')
        return None
    if real_only:
        try:
            check_index(settings, dense, repair_missing=True)
        except Exception as exc:
            release_dense(dense)
            raise BenchmarkPrerequisiteError(str(exc)) from exc
    return dense
