"""Offline answer-pipeline integration: real selection, HTTP adapter, grounding, audit.

All sources are SYNTHETIC. HTTP uses MockTransport; results are engineering
regressions, not legal fact checks, human gold, or observed Qwen quality.
"""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.context_engineering.evidence_selection import (
    validate_selection_response, requested_facets,
)
from dap_assistant.response.models import Draft
from dap_assistant.response.audit import audit_answer
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import OllamaAdapter
from dap_assistant.settings import Settings

QUESTION = 'Eladtam az autómat. Mit kell bejelentenem, hol és meddig?'
SOURCES = [
    {
        'evidence_id': 'E_DEADLINE', 'chunk_id': 'synthetic-deadline',
        'document_id': 'synthetic-seller', 'document_version': 'synthetic-v1',
        'domain': 'vehicle', 'role': 'seller', 'source_url': 'https://example.org/synthetic',
        'title': 'SZINTETIKUS ELADÁSI PÉLDA',
        'text': ('Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást. '
                 'A bejelentést online, a Webes Ügysegéden vagy személyesen, '
                 'bármelyik kormányablakban megteheted.'),
    },
    {
        'evidence_id': 'E_DOCUMENTS', 'chunk_id': 'synthetic-documents',
        'document_id': 'synthetic-seller', 'document_version': 'synthetic-v1',
        'domain': 'vehicle', 'role': 'seller', 'source_url': 'https://example.org/synthetic',
        'title': 'SZINTETIKUS ELADÁSI PÉLDA',
        'text': ('Az eladás után jelezd a biztosítód felé az autó eladását. '
                 'Az adásvételi szerződés adatait őrizd meg a bejelentéshez.'),
    },
]


def _mocked_answer(*, contradictory_selection: bool = False,
                   invented_number: bool = False, failed_answer: bool = False):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if not payload['stream']:
            question = json.loads(payload['messages'][1]['content'])
            ids = {item['evidence_id'] for item in question['available_full_text']}
            assert ids == {'E_DEADLINE', 'E_DOCUMENTS'}
            by_need = {
                'steps': ['E_DEADLINE'], 'deadline': ['E_DEADLINE'],
                'where': ['E_DEADLINE'],
                'documents': [] if contradictory_selection else ['E_DOCUMENTS'],
            }
            response = {'evidence_by_need': by_need, 'missing_needs': []}
            return httpx.Response(200, json={'message': {
                'content': json.dumps(response, ensure_ascii=False)},
                'done_reason': 'stop', 'eval_count': 48, 'prompt_eval_count': 220})
        if failed_answer:
            raise httpx.ReadTimeout('SYNTHETIC_ERROR_PRIVATE', request=request)
        question = json.loads(payload['messages'][1]['content'])
        ids = {item['evidence_id'] for item in question['evidence']}
        assert ids == {'E_DEADLINE', 'E_DOCUMENTS'}
        claims = [
            {'evidence_id': 'E_DEADLINE',
             'text': 'Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást.',
             'category': 'deadline'},
            {'evidence_id': 'E_DOCUMENTS',
             'text': 'Az adásvételi szerződés adatait őrizd meg a bejelentéshez.',
             'category': 'documents'},
        ]
        if invented_number:
            claims.append({'evidence_id': 'E_DEADLINE',
                           'text': 'A tulajdonosváltást 99 napon belül be kell jelentened.',
                           'category': 'deadline'})
        content = json.dumps({'claims': claims, 'disclaimer': ''}, ensure_ascii=False)
        frames = [
            {'message': {'content': content}, 'done': False},
            {'message': {'content': ''}, 'done': True, 'done_reason': 'stop',
             'eval_count': 120, 'prompt_eval_count': 410},
        ]
        return httpx.Response(200, content='\n'.join(json.dumps(f, ensure_ascii=False) for f in frames) + '\n')

    telemetry = Telemetry()
    settings = replace(Settings(), answer_mode='detailed', native_tool_calling_enabled=False,
                       ollama_num_ctx=8192, ollama_answer_num_predict=900)
    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(handler), telemetry=telemetry)
    return adapter, telemetry, sent


def test_selection_requires_declared_missing_facets():
    facets = requested_facets(QUESTION, 'vehicle', role='seller')
    assert facets == ('steps', 'deadline', 'where', 'documents')
    with pytest.raises(ValueError, match='Inconsistent'):
        validate_selection_response(
            {'steps': ['E_DEADLINE'], 'deadline': ['E_DEADLINE'],
             'where': ['E_DEADLINE']}, [],
            {'E_DEADLINE', 'E_DOCUMENTS'}, facets,
        )
    assert validate_selection_response(
        {'steps': ['E_DEADLINE'], 'deadline': ['E_DEADLINE'],
         'where': ['E_DEADLINE'], 'documents': []}, ['documents'],
        {'E_DEADLINE', 'E_DOCUMENTS'}, facets,
    )['documents'] == []
    with pytest.raises(ValueError, match='Duplicate'):
        validate_selection_response({}, ['documents', 'documents'], set(), facets)


@pytest.mark.parametrize('contradictory,invented', [
    (False, False), (True, False), (False, True),
])
def test_mock_ollama_selection_generation_grounding_and_audit(contradictory, invented):
    adapter, telemetry, sent = _mocked_answer(
        contradictory_selection=contradictory, invented_number=invented)
    try:
        draft: Draft = adapter.answer(
            QUESTION, SOURCES, [], run_id='synthetic-v10',
            context={'life_events': ['vehicle'], 'role': 'seller', 'stage': 'after_event'},
        )
    finally:
        adapter.close()
    assert len(sent) == 2  # exactly one selection and one answer, no hidden paid calls
    assert all(claim.evidence_ids[0] in {'E_DEADLINE', 'E_DOCUMENTS'} for claim in draft.claims)
    assert len(draft.claims) >= 2
    assert all('99' not in claim.text for claim in draft.claims)
    assert any('szerződés' in claim.text.casefold() for claim in draft.claims)
    state = {
        'user_question': QUESTION, 'domains': ['vehicle'], 'role': 'seller',
        'stage': 'after_event', 'evidence': SOURCES, 'tool_results': {},
        'answer_draft': draft.model_dump(), 'answer_context': {},
        'context_engineering': {},
    }
    result = audit_answer(state)
    assert result['final_answer']
    assert '99 napon' not in result['final_answer']
    assert result['answer_validation']['status'] != 'failed'
    snapshot = telemetry.snapshot('synthetic-v10')
    assert len(snapshot['llm_usage']) == 2
    assert len(snapshot['llm_attempts']) == 2
    if contradictory:
        assert snapshot['selection']['selection_validation_error'] == 'ValueError'
        assert snapshot['selection']['recovered_with'] == 'deterministic_in_budget_sources'


def test_mock_timeout_does_not_replay_or_emit_an_uncited_answer():
    adapter, telemetry, sent = _mocked_answer(failed_answer=True)
    try:
        with pytest.raises(Exception) as caught:
            adapter.answer(QUESTION, SOURCES, [], run_id='synthetic-timeout',
                           context={'life_events': ['vehicle'], 'role': 'seller'})
    finally:
        adapter.close()
    assert len(sent) == 2
    assert 'SYNTHETIC_ERROR_PRIVATE' not in str(caught.value)
    failure = telemetry.snapshot('synthetic-timeout')['llm_failures']
    assert len(failure) == 1
    assert 'SYNTHETIC_ERROR_PRIVATE' not in json.dumps(failure)
