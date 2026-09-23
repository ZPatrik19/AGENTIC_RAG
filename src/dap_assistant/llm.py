"""Public LLM facade: stable imports, Ollama adapter, and explicit dummy provider.

Transport, prompt context, grounding, and answer orchestration have distinct owners.
The public API remains compatible with existing workflow, evaluation, and tests.
"""
from __future__ import annotations

import json

import httpx

from .settings import Settings
from .response.models import (
    Classification, PlannedTask, Plan, Claim, Draft, FastSelection,
    FacetSelection, NaturalClaim, NaturalResponse,
)
from .response.fallback import select_excerpt, source_answer
from .prompt_engineering.prompt_payload import _verified_tool_hints as _verified_tool_hints, answer_context
from .response.grounding import _attach_source_quotes as _attach_source_quotes, supplement_grounded_facts
from .response.generation import AnswerGenerationMixin
from .inference.errors import LLMError, LLMGenerationLengthError
from .inference.ollama_transport import StructuredOllamaTransport


__all__ = (
    'Classification', 'PlannedTask', 'Plan', 'Claim', 'Draft',
    'FastSelection', 'FacetSelection', 'NaturalClaim', 'NaturalResponse',
    'select_excerpt', 'source_answer', 'answer_context', 'supplement_grounded_facts',
    'LLMError', 'LLMGenerationLengthError', 'OllamaAdapter', 'DummyAdapter',
    'get_llm', 'ollama_health',
)


class OllamaAdapter(AnswerGenerationMixin, StructuredOllamaTransport):
    """Public model adapter with shared transport and answer orchestration."""

    def call_native_tools(self, question: str, evidence: list[dict], run_id: str = '') -> tuple[dict, dict]:
        """Actual Ollama tool_calls + validated Python tools + role=tool response."""
        from .tooling.native_tool_calling import run_native_tool_cycle
        return run_native_tool_cycle(client=self.client, settings=self.settings,
                                     question=question, evidence=evidence,
                                     run_id=run_id, telemetry=self.telemetry)

    def classify(self, question: str, run_id: str = '') -> Classification:
        return self._ask(
            'Classify Hungarian life event questions into vehicle or employment only. Multiple domains allowed. Never invent a domain. Include one normalized intent per detected domain: vehicle_information, employment_information. Do not add unrelated intents.',
            question, Classification, run_id=run_id, phase='classify',
        )

    def plan(self, question: str, domains: list[str], run_id: str = '') -> Plan:
        return self._ask(
            'Create 1-6 independent or dependent document-retrieval tasks. IDs t1 to t6; use only given domains. No circular dependencies. Split comprehensive life-event questions by independently searchable information need, not by benchmark examples.',
            json.dumps({'question': question, 'domains': domains}, ensure_ascii=False),
            Plan, run_id=run_id, phase='plan',
        )


class DummyAdapter:
    """Deterministic workflow fixture; no LLM inference."""
    @staticmethod
    def classify(question: str, run_id: str = '') -> Classification:
        q = question.lower()
        hints = {
            'vehicle': ('autó', 'autot', 'kocsi', 'gépjármű', 'jármű', 'átírás'),
            'employment': ('munka', 'állás', 'járadék', 'kilépő', 'foglalkoztatás'),
        }
        domains = [domain for domain, words in hints.items() if any(w in q for w in words)]
        role = 'seller' if 'eladtam' in q or 'eladok' in q else 'buyer' if 'vettem' in q or 'vásárol' in q else ''
        return Classification(domains=domains, intents=[f'{d}_information' for d in domains], role=role, stage='after_event' if 'vettem' in q or 'megszűnt' in q else 'unknown')

    @staticmethod
    def plan(question: str, domains: list[str], run_id: str = '') -> Plan:
        tasks = [PlannedTask(task_id=f't{i}', domain=domain, question=f'{question} ({domain}: hivatalos teendők és dokumentumok)')
                 for i, domain in enumerate(domains[:3], start=1)]
        return Plan(tasks=tasks[:6])

    @staticmethod
    def answer(question: str, evidence: list[dict], tools: list[dict], run_id: str = '',
               context: dict | None = None) -> Draft:
        draft = source_answer(question, evidence)
        draft.disclaimer = 'DUMMY MÓD: determinisztikus forráskivonat, nem valódi LLM-következtetés.'
        return draft


def get_llm(settings: Settings, telemetry=None):
    if settings.llm_provider == 'dummy':
        return DummyAdapter()
    if settings.llm_provider == 'ollama':
        return OllamaAdapter(settings, telemetry=telemetry)
    raise ValueError('LLM_PROVIDER must be ollama or dummy')


def ollama_health(settings: Settings) -> dict:
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            response = client.get(settings.ollama_base_url.rstrip('/') + '/api/tags')
            response.raise_for_status()
            names = [item.get('name', '') for item in response.json().get('models', [])]
            return {'available': True, 'models': names, 'configured_model_found': settings.ollama_model in names}
    except (httpx.HTTPError, ValueError) as exc:
        return {'available': False, 'error': type(exc).__name__, 'models': []}
