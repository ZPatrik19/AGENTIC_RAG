"""Bounded Ollama HTTP transport for structured Pydantic responses.

Retries only transient non-streamed HTTP 429/503 once; preserves stream usage telemetry.
No answer planning, selection, grounding, or UI decisions belong in this module.
"""
from __future__ import annotations

import json
import time

import httpx
from pydantic import BaseModel, ValidationError

from ..settings import Settings
from ..context_engineering.token_budget import request_budget, ContextBudgetError
from .request import build_ollama_request
from .errors import LLMError, LLMGenerationLengthError
from .response_io import read_stream_response, read_single_response


class StructuredOllamaTransport:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None, telemetry=None):
        if settings.answer_mode not in ('quick', 'source', 'detailed'):
            raise ValueError('ANSWER_MODE must be quick, source or detailed')
        self.base_url = settings.ollama_base_url.rstrip('/')
        self.model = settings.ollama_model
        self.settings = settings
        self.telemetry = telemetry
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=settings.ollama_read_timeout_s,
                                  write=15.0, pool=5.0),
            transport=transport, trust_env=False,
        )

    def close(self) -> None:
        self.client.close()

    def _ask(self, instruction: str, question: str, schema: type[BaseModel],
             run_id: str = '', phase: str = 'reasoning', *,
             model_override: str | None = None, num_predict_override: int | None = None):
        payload = build_ollama_request(
            self.settings, self.model, instruction, question, schema, phase,
            model_override=model_override, num_predict_override=num_predict_override,
        )
        streaming = payload['stream']
        budget_report = request_budget(payload['messages'], payload['format'],
            payload['options']['num_ctx'], payload['options']['num_predict'],
            self.settings.context_safety_tokens)
        if self.telemetry is not None:
            self.telemetry.selection(run_id, {phase + '_request_budget': budget_report})
        if not budget_report['fits']:
            if self.telemetry is not None:
                self.telemetry.llm_failure(run_id, phase, 'ContextBudgetError', 0,
                    diagnostic={'failure_kind': 'input_context_budget', **budget_report})
            raise LLMError('Request exceeds the conservative input context budget') from ContextBudgetError()
        if self.telemetry is not None:
            self.telemetry.prompt(run_id, phase, payload)
        error = None
        for attempt in range(2):
            started = time.perf_counter()
            if self.telemetry is not None:
                self.telemetry.llm_attempt(run_id, phase)
            # Only counters and server metadata survive; never persist streamed JSON,
            # Pydantic's input_value, prompt text, or original exception messages.
            diagnostic = {'done_reason': None, 'eval_count': None,
                          'prompt_eval_count': None, 'observed_content_bytes': 0,
                          'observed_content_chars': 0, 'streamed_frames': 0,
                          'requested_num_predict': payload['options']['num_predict'],
                          'requested_num_ctx': payload['options']['num_ctx'],
                          'usage_source': 'unavailable', 'observed_thinking_chars': 0}
            try:
                if streaming:
                    validated = read_stream_response(
                        self, payload, schema, run_id, phase, started, diagnostic,
                    )
                else:
                    validated = read_single_response(
                        self, payload, schema, run_id, phase, diagnostic,
                    )
                if self.telemetry is not None:
                    self.telemetry.llm_success(run_id, phase)
                return validated
            except (httpx.HTTPError, KeyError, ValueError, LLMError) as exc:
                error = exc
                if isinstance(exc, ValidationError):
                    diagnostic['validation_issue_types'] = sorted({str(item['type'])
                        for item in exc.errors(include_input=False)})[:8]
                    invalid_json = 'json_invalid' in diagnostic['validation_issue_types']
                    diagnostic['failure_kind'] = 'json_syntax' if invalid_json else 'schema_validation'
                    diagnostic['validation_error_type'] = ('json_invalid' if invalid_json else
                        diagnostic['validation_issue_types'][0] if diagnostic['validation_issue_types']
                        else 'unknown_schema_error')
                elif isinstance(exc, json.JSONDecodeError):
                    diagnostic['failure_kind'] = 'json_syntax'
                    diagnostic['validation_error_type'] = 'json_syntax'
                    diagnostic['json_error_line'] = exc.lineno
                    diagnostic['json_error_column'] = exc.colno
                elif isinstance(exc, httpx.TimeoutException):
                    diagnostic['failure_kind'] = 'http_timeout'
                    diagnostic['validation_error_type'] = None
                elif isinstance(exc, httpx.HTTPStatusError):
                    diagnostic['failure_kind'] = 'http_status'
                    diagnostic['http_status_code'] = exc.response.status_code
                    diagnostic['validation_error_type'] = None
                elif isinstance(exc, LLMError) and diagnostic['done_reason'] == 'length':
                    diagnostic['failure_kind'] = 'output_token_limit'
                    diagnostic['validation_error_type'] = None
                elif isinstance(exc, LLMError) and 'total generation time limit' in str(exc):
                    diagnostic['failure_kind'] = 'total_timeout'
                    diagnostic['validation_error_type'] = None
                elif isinstance(exc, LLMError) and 'output exceeded safe size' in str(exc):
                    diagnostic['failure_kind'] = 'output_size_limit'
                    diagnostic['validation_error_type'] = None
                elif isinstance(exc, LLMError) and diagnostic['streamed_frames'] > 0 and diagnostic['done_reason'] is None:
                    diagnostic['failure_kind'] = 'stream_interrupted_or_incomplete'
                    diagnostic['validation_error_type'] = None
                elif isinstance(exc, ValueError):
                    diagnostic['failure_kind'] = 'empty_or_invalid_response'
                    diagnostic['validation_error_type'] = 'value_error'
                else:
                    diagnostic['failure_kind'] = 'transport_or_response_error'
                    diagnostic['validation_error_type'] = None
                if self.telemetry is not None:
                    self.telemetry.llm_failure(run_id, phase, type(exc).__name__,
                                               time.perf_counter() - started,
                                               diagnostic=diagnostic)
                # Never replay timeouts or partial streams: Ollama may still be generating.
                # Only transient non-streamed HTTP 429/503 may retry once.
                # Invalid JSON / schema failures must not generate a second paid call.
                retryable_status = (
                    not streaming
                    and isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in (429, 503)
                )
                if not retryable_status or attempt != 0:
                    break
                time.sleep(0.2)
            finally:
                if self.telemetry is not None:
                    self.telemetry.record(run_id, 'llm_inference', time.perf_counter() - started)
        # Keep the existing public error label stable for clients/tests.
        error_label = 'LLMError' if isinstance(error, LLMGenerationLengthError) else type(error).__name__
        raise LLMError(f'Ollama response unavailable or invalid ({error_label})') from error
