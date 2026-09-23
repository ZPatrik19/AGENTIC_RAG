"""Decode Ollama chat responses without controlling retries or building prompts.

Both paths report server usage before JSON/schema validation, including responses
that reached num_predict. Never persist response content or thinking text.
"""
from __future__ import annotations

import json
import time
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from ..settings import Settings
from .errors import LLMError, LLMGenerationLengthError


class ResponseClient(Protocol):
    """Minimal transport state required by the two response readers."""

    client: httpx.Client
    base_url: str
    settings: Settings
    telemetry: Any


def read_stream_response(
    owner: ResponseClient, payload: dict, schema: type[BaseModel],
    run_id: str, phase: str, started: float, diagnostic: dict,
) -> BaseModel:
    """Read a streaming response; an interrupted stream is never replayed."""
    fragments: list[str] = []
    final_payload: dict = {}
    first_content_s: float | None = None
    with owner.client.stream(
        'POST', f'{owner.base_url}/api/chat', json=payload,
        timeout=httpx.Timeout(
            connect=5.0,
            read=min(owner.settings.ollama_read_timeout_s,
                     owner.settings.ollama_total_timeout_s),
            write=15.0, pool=5.0,
        ),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            frame = json.loads(line)
            diagnostic['streamed_frames'] += 1
            if not isinstance(frame, dict):
                raise ValueError('Ollama stream frame is not a JSON object')
            message = frame.get('message') or {}
            if not isinstance(message, dict):
                raise ValueError('Ollama message is not a JSON object')
            content = message.get('content', '')
            if not isinstance(content, str):
                raise ValueError('Ollama content is not text')
            if content and first_content_s is None:
                first_content_s = time.perf_counter() - started
            # Count thinking characters only: never persist private reasoning.
            thinking = message.get('thinking')
            if isinstance(thinking, str):
                diagnostic['observed_thinking_chars'] += len(thinking)
            fragments.append(content)
            diagnostic['observed_content_bytes'] += len(content.encode('utf-8'))
            diagnostic['observed_content_chars'] += len(content)
            if diagnostic['observed_content_bytes'] > 65536:
                raise LLMError('Ollama output exceeded safe size limit')
            if owner.telemetry is not None:
                owner.telemetry.progress(
                    run_id, phase, diagnostic['observed_content_bytes'],
                    time.perf_counter() - started,
                )
            if time.perf_counter() - started > owner.settings.ollama_total_timeout_s:
                raise LLMError('Ollama total generation time limit exceeded')
            if frame.get('done'):
                final_payload = frame
                break
    if not final_payload:
        raise LLMError('Ollama stream ended without a final done frame')
    diagnostic.update({k: final_payload.get(k) for k in
                       ('done_reason', 'eval_count', 'prompt_eval_count')})
    diagnostic['usage_source'] = (
        'ollama_eval_count' if isinstance(final_payload.get('eval_count'), int)
        else 'unavailable'
    )
    # A length-truncated or invalid-JSON response still consumed actual tokens.
    if owner.telemetry is not None:
        owner.telemetry.llm_usage(run_id, {
            **final_payload,
            'ttft_s': first_content_s,
            'requested_num_ctx': diagnostic['requested_num_ctx'],
            'requested_num_predict': diagnostic['requested_num_predict'],
            'observed_content_bytes': diagnostic['observed_content_bytes'],
            'observed_content_chars': diagnostic['observed_content_chars'],
            'streamed_frames': diagnostic['streamed_frames'],
            'observed_thinking_chars': diagnostic['observed_thinking_chars'],
            'usage_source': diagnostic['usage_source'],
        }, phase=phase)
    if final_payload.get('done_reason') == 'length':
        raise LLMGenerationLengthError(
            'Ollama reached num_predict before completing the JSON answer'
        )
    text = ''.join(fragments)
    if not text:
        raise ValueError('Ollama returned no response content')
    return schema.model_validate_json(text)


def read_single_response(
    owner: ResponseClient, payload: dict, schema: type[BaseModel],
    run_id: str, phase: str, diagnostic: dict,
) -> BaseModel:
    """Decode one nonstreamed response, retaining its usage on invalid output."""
    response = owner.client.post(f'{owner.base_url}/api/chat', json=payload)
    response.raise_for_status()
    result = response.json()
    message = result.get('message') or {}
    if not isinstance(message, dict):
        raise ValueError('Ollama message is not a JSON object')
    content = message.get('content', '')
    if not isinstance(content, str):
        raise ValueError('Ollama content is not text')
    diagnostic.update({k: result.get(k) for k in
                       ('done_reason', 'eval_count', 'prompt_eval_count')})
    diagnostic['observed_content_chars'] = len(content)
    thinking = message.get('thinking')
    diagnostic['observed_thinking_chars'] = len(thinking) if isinstance(thinking, str) else 0
    diagnostic['observed_content_bytes'] = len(content.encode('utf-8'))
    diagnostic['usage_source'] = (
        'ollama_eval_count' if isinstance(result.get('eval_count'), int) else 'unavailable'
    )
    if owner.telemetry is not None:
        owner.telemetry.llm_usage(run_id, {
            **result,
            'requested_num_ctx': diagnostic['requested_num_ctx'],
            'requested_num_predict': diagnostic['requested_num_predict'],
            'observed_content_bytes': diagnostic['observed_content_bytes'],
            'observed_content_chars': diagnostic['observed_content_chars'],
            'observed_thinking_chars': diagnostic['observed_thinking_chars'],
            'usage_source': diagnostic['usage_source'],
        }, phase=phase)
    if result.get('done_reason') == 'length':
        raise LLMGenerationLengthError('Ollama reached num_predict before completing JSON')
    return schema.model_validate_json(content)
