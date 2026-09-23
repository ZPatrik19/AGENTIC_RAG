"""Pure construction of bounded, schema-constrained Ollama chat requests.

Request payloads are returned to the transport for in-memory prompt inspection;
they must never be persisted in diagnostic logs.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from ..settings import Settings
from ..response.models import Draft, FacetSelection, NaturalResponse
from ..context_engineering.token_budget import SYSTEM_SUFFIX, answer_output_budget


def build_ollama_request(
    settings: Settings, default_model: str, instruction: str, question: str,
    schema: type[BaseModel], phase: str, *, model_override: str | None = None,
    num_predict_override: int | None = None,
) -> dict:
    """Build the same request for quick, detailed, and recovery generations."""
    streaming = phase == 'answer'
    need_count = 0
    if schema is NaturalResponse:
        try:
            parsed = json.loads(question)
            if isinstance(parsed, dict):
                need_count = len(parsed.get('information_needs', []))
        except (ValueError, TypeError):
            pass
    payload = {
        'model': model_override or default_model,
        'messages': [
            {'role': 'system', 'content': instruction + SYSTEM_SUFFIX},
            {'role': 'user', 'content': question},
        ],
        'think': settings.ollama_think,
        'stream': streaming,
        'keep_alive': settings.ollama_keep_alive,
        'format': schema.model_json_schema(),
        'options': {
            'temperature': settings.ollama_temperature,
            'num_ctx': (min(settings.ollama_num_ctx, settings.ollama_quick_num_ctx)
                        if settings.answer_mode == 'quick' and settings.quick_single_pass
                        and phase == 'answer' else settings.ollama_num_ctx),
            'num_predict': (max(384, settings.ollama_selection_num_predict) if schema is FacetSelection
                            else answer_output_budget(settings, need_count) if schema is NaturalResponse
                            else max(640, settings.ollama_answer_num_predict) if schema is Draft
                            else min(192, settings.ollama_num_predict)),
        },
    }
    if 'instruct' in payload['model'].casefold():
        payload.pop('think', None)  # Instruct-only models need no thinking flag.
    if num_predict_override is not None:
        payload['options']['num_predict'] = max(128, min(1024, num_predict_override))
    for name in ('top_p', 'top_k'):
        value = getattr(settings, 'ollama_' + name)
        if value is not None:
            payload['options'][name] = value
    return payload
