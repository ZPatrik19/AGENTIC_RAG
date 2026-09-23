"""Bounded Ollama native-tool transport and privacy-safe diagnostics."""
from __future__ import annotations

from typing import Any

import json
import httpx


def _safe_http_category(response: httpx.Response | None) -> str | None:
    """Categorize server errors WITHOUT storing Ollama's free-form error text.

    Ollama errors can contain arbitrary source text, prompts or local paths.  Only
    the allowlisted category and numeric HTTP status are persisted.
    """
    if response is None or response.status_code < 400:
        return None
    try:
        body = response.json()
        message = body.get('error', '') if isinstance(body, dict) else ''
        message = message.casefold() if isinstance(message, str) else ''
    except (ValueError, TypeError):
        return 'non_json_http_error'
    if 'think' in message and any(word in message for word in ('support', 'invalid', 'unsupported')):
        return 'thinking_option_rejected'
    if 'tool' in message and any(word in message for word in ('support', 'schema', 'invalid', 'unsupported')):
        return 'tool_schema_or_support_error'
    if any(word in message for word in ('tool call', 'tool_calls', 'tool_call', 'parse tool')):
        return 'tool_call_parse_error'
    if any(word in message for word in ('model not found', 'model \'', 'pull a model')):
        return 'model_unavailable'
    if any(word in message for word in ('memory', 'out of memory', 'cuda', 'vram')):
        return 'resource_error'
    return 'other_http_error'

def _safe_diagnostic(*, stage: str, exc: Exception,
                     response: httpx.Response | None, data: dict | None,
                     requested_num_ctx: int, requested_num_predict: int,
                     progress: dict | None = None) -> dict:
    """Stable, bounded metadata for a failed /api/chat request; no raw body."""
    return {
        'failure_stage': stage,
        'exception_type': type(exc).__name__,
        'http_status': response.status_code if response is not None else None,
        'http_error_category': _safe_http_category(response),
        'done_reason': data.get('done_reason') if isinstance(data, dict) and isinstance(data.get('done_reason'), str) else None,
        'done': data.get('done') if isinstance(data, dict) and isinstance(data.get('done'), bool) else None,
        'eval_count': data.get('eval_count') if isinstance(data, dict) and isinstance(data.get('eval_count'), int) else None,
        'prompt_eval_count': data.get('prompt_eval_count') if isinstance(data, dict) and isinstance(data.get('prompt_eval_count'), int) else None,
        'response_bytes': progress.get('response_bytes') if progress else (
            len(response.content) if response is not None and response.is_stream_consumed else None),
        'requested_num_ctx': requested_num_ctx,
        'requested_num_predict': requested_num_predict,
        'elapsed_s': round(progress['elapsed_s'], 2) if progress and 'elapsed_s' in progress else None,
        'streamed_frames': progress.get('streamed_frames') if progress else None,
        'first_frame_s': progress.get('first_frame_s') if progress else None,
        'streaming': progress.get('streaming') if progress else None,
        'thinking_chars': progress.get('thinking_chars') if progress else None,
        'content_chars': progress.get('content_chars') if progress else None,
        'thinking_frames': progress.get('thinking_frames') if progress else None,
        'tool_calls_observed': progress.get('tool_calls_observed') if progress else None,
    }

def _request_chat(*, client: httpx.Client, settings: Any, payload: dict,
                  progress: dict) -> tuple[httpx.Response, dict]:
    """Read bounded Ollama NDJSON, with separate idle and total deadlines.

    Record metadata without logging prompts, responses or source text."""
    from time import perf_counter

    started = perf_counter()
    total = max(10.0, min(600.0, float(getattr(settings, 'native_tool_total_timeout_s', 300))))
    idle = max(10.0, min(total, float(getattr(settings, 'native_tool_read_timeout_s', 180))))
    progress.update(streaming=bool(payload.get('stream')), streamed_frames=0,
                    response_bytes=0, thinking_chars=0, content_chars=0,
                    thinking_frames=0, content_frames=0, tool_calls_observed=0,
                    failure_stage='transport')
    try:
        with client.stream('POST', settings.ollama_base_url.rstrip('/') + '/api/chat',
                           json=payload, timeout=httpx.Timeout(connect=8, read=idle,
                                                                write=20, pool=8)) as response:
            progress['response'] = response
            progress['failure_stage'] = 'http_status'
            if response.status_code >= 400:
                # The categorizer only persists an allowlisted class, not the body.
                response.read()
                response.raise_for_status()
            progress['failure_stage'] = 'stream_read' if payload.get('stream') else 'json_decode'
            content_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list = []
            merged: dict = {}
            for line in response.iter_lines():
                if not line.strip():
                    continue
                progress['streamed_frames'] += 1
                if 'first_frame_s' not in progress:
                    progress['first_frame_s'] = round(perf_counter() - started, 2)
                progress['response_bytes'] += len(line.encode('utf-8'))
                if progress['response_bytes'] > 65536 or progress['streamed_frames'] > 2048:
                    progress['failure_stage'] = 'response_size_limit'
                    raise ValueError('Native tool response too large')
                frame = json.loads(line)
                if not isinstance(frame, dict):
                    progress['failure_stage'] = 'response_schema'
                    raise ValueError('Invalid Ollama frame')
                # Final metadata appears on the final frame, while tool calls can
                # appear in earlier frames. Never throw earlier calls away.
                for field in ('done', 'done_reason', 'eval_count', 'prompt_eval_count',
                              'total_duration', 'eval_duration', 'prompt_eval_duration'):
                    if field in frame:
                        merged[field] = frame[field]
                # Preserve non-sensitive final counters even if later schema
                # validation fails (or a following stream frame times out).
                progress['data'] = merged
                message = frame.get('message')
                if isinstance(message, dict):
                    for key, parts, max_chars in (('content', content_parts, 4096),
                                                   ('thinking', thinking_parts, 8192)):
                        text = message.get(key)
                        if isinstance(text, str):
                            if sum(map(len, parts)) + len(text) > max_chars:
                                progress['failure_stage'] = 'response_size_limit'
                                raise ValueError('Native tool message too large')
                            parts.append(text)
                            if text:
                                progress[f'{key}_chars'] += len(text)
                                progress[f'{key}_frames'] += 1
                    if 'tool_calls' in message and message['tool_calls'] is not None:
                        if not isinstance(message['tool_calls'], list):
                            merged['message'] = {'tool_calls': message['tool_calls']}
                            progress['failure_stage'] = 'tool_calls_schema'
                            raise ValueError('Invalid tool_calls type')
                        tool_calls.extend(message['tool_calls'])
                        progress['tool_calls_observed'] = len(tool_calls)
                if perf_counter() - started > total:
                    progress['failure_stage'] = 'total_timeout'
                    raise TimeoutError('Native tool request exceeded total deadline')
                if frame.get('done') is True:
                    break
            if not progress['streamed_frames']:
                progress['failure_stage'] = 'stream_read'
                raise ValueError('Empty Ollama response')
            merged['message'] = {'role': 'assistant', 'content': ''.join(content_parts),
                                 'thinking': ''.join(thinking_parts), 'tool_calls': tool_calls}
            progress['data'] = merged
            progress['failure_stage'] = 'response_schema'
            return response, merged
    finally:
        progress['elapsed_s'] = perf_counter() - started
