"""Regression tests for real-model prompt transparency and seller/buyer isolation."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.fast_path import classify_explicit
from dap_assistant.llm import OllamaAdapter, answer_context, source_answer
from dap_assistant.presentation.runtime_view import final_prompt_view, ollama_status
from dap_assistant.rag.retrieval import hybrid_search
from dap_assistant.settings import Settings
from dap_assistant.observability.telemetry import Telemetry


@pytest.mark.parametrize('question,role,stage', [
    ('Vásároltam egy autót milyen teendőim vannak?', 'buyer', 'after_event'),
    ('Megvettem a használt autót.', 'buyer', 'after_event'),
    ('Eladtam az autómat.', 'seller', 'after_event'),
    ('Szeretnék egy autót vásárolni.', 'buyer', 'planning'),
    ('Autót veszek és eladok.', '', 'unknown'),
])
def test_vehicle_roles(question, role, stage):
    result = classify_explicit(question)
    assert result is not None
    assert result.role == role
    assert result.stage == stage


def test_hybrid_rejects_opposite_role_even_if_dense_returns_it(tmp_path):
    folder = tmp_path / 'processed' / 'vehicle'
    folder.mkdir(parents=True)
    chunks = [
        {'chunk_id': 'BUY', 'document_id': 'buyer', 'domain': 'vehicle', 'role': 'buyer',
         'text': 'A vevő átíratja a gépjárművet.', 'title': 'Vevő',
         'source_url': 'https://dap.gov.hu/vevo'},
        {'chunk_id': 'SELL', 'document_id': 'seller', 'domain': 'vehicle', 'role': 'seller',
         'text': 'Az eladó bejelenti a gépjármű eladását.', 'title': 'Eladó',
         'source_url': 'https://dap.gov.hu/elado'},
    ]
    (folder / 'fixture.json').write_text(json.dumps({'chunks': chunks}), encoding='utf-8')

    class DenseWithStaleIds:
        def search(self, *args, **kwargs):
            return [('SELL', 0.99), ('BUY', 0.1)]

    result = hybrid_search('gépjármű átírás', 'vehicle', replace(Settings(), data_dir=tmp_path),
                           dense=DenseWithStaleIds(), role='buyer')
    assert [item['chunk_id'] for item in result] == ['BUY']


def test_source_answer_keeps_complete_action_sentences():
    evidence = [{'evidence_id': 'E_B', 'domain': 'vehicle', 'role': 'buyer',
                 'text': 'Az eredetiségvizsgálat igazolja, hogy a gépjármű nem lopott.\n'
                         'Az adásvételi szerződés megkötésétől számított 15 napon belül '
                         'át kell íratnod a gépjárművet.'}]
    result = source_answer('Vásároltam egy autót, mi a teendőm?', evidence)
    assert result.claims
    assert '15 napon belül' in result.claims[0].text
    assert all(claim.supporting_quote in evidence[0]['text'] for claim in result.claims)


def test_context_in_real_ollama_request_even_if_answer_times_out():
    sent = []

    def handler(request):
        payload = json.loads(request.content)
        sent.append(payload)
        raise httpx.ReadTimeout('slow local model', request=request)

    telemetry = Telemetry()
    adapter = OllamaAdapter(replace(Settings(), answer_mode='quick'),
                            transport=httpx.MockTransport(handler), telemetry=telemetry)
    context = answer_context(domains=['vehicle'], role='buyer', stage='after_event',
                             strategy='quick', reference_date='2026-09-20', event_date=None,
                             subtasks={'t1': {'domain': 'vehicle', 'question': 'Vevői teendők'}})
    evidence = [{'evidence_id': 'E_TEST', 'title': 'Hivatalos tájékoztató',
                 'source_url': 'https://dap.gov.hu/example',
                 'text': 'Az adásvételi szerződés után át kell íratni a gépjárművet.'}]
    try:
        with pytest.raises(Exception):
            adapter.answer('Vásároltam egy autót', evidence, [], 'run-prompt', context=context)
    finally:
        adapter.close()
    assert len(sent) == 1
    payload = json.loads(sent[0]['messages'][1]['content'])
    assert payload['context']['life_events'] == ['vehicle']
    assert payload['context']['life_event_labels'] == ['Autóvásárlás vagy -eladás']
    assert payload['context']['role'] == 'buyer'
    assert payload['context']['answer_strategy'] == 'quick'
    assert payload['context']['reference_date'] == '2026-09-20'
    assert payload['evidence'][0]['source_url'] == 'https://dap.gov.hu/example'
    view = final_prompt_view({'user_question': 'Vásároltam egy autót',
                              'answer_fallback': True}, telemetry.prompt_preview('run-prompt'))
    assert view['submitted'] is True
    assert 'Vásároltam egy autót' in view['text']
    assert 'after_event' in view['text']


def test_source_prompt_is_not_mislabeled_as_sent():
    final = {'user_question': 'Lakást vettem', 'domains': ['housing'],
             'answer_context': {'life_events': ['housing'], 'role': 'not_specified',
                                'answer_strategy': 'source', 'reference_date': '2026-09-20'},
             'evidence': [{'title': 'Lakást veszek', 'evidence_id': 'E1',
                           'source_url': 'https://dap.gov.hu/example'}]}
    view = final_prompt_view(final, [])
    assert not view['submitted']
    assert 'NEM KÜLDTÜK EL MODELLNEK' in view['text']
    assert 'Lakást vettem' in view['text']


@pytest.mark.parametrize('health,level', [
    ({'available': True, 'configured_model_found': True, 'models': ['qwen3:4b']}, 'success'),
    ({'available': True, 'configured_model_found': False, 'models': []}, 'warning'),
    ({'available': False, 'models': []}, 'error'),
])
def test_status_is_readable(health, level):
    view = ollama_status(health, 'qwen3:4b')
    assert view['level'] == level
    assert view['title'] and view['detail']
    assert '"available"' not in view['detail']


def test_event_date_default_and_explicit_confirmation():
    pytest.importorskip('langgraph')
    from datetime import date
    from dap_assistant.workflow import initial_state
    day = date(2026, 9, 20)
    default = initial_state('Autót szeretnék venni', reference_date=day)
    assert default['user_context']['event_date'] == '2026-09-20'
    assert default['user_context']['event_date_confirmed'] is False
    selected = initial_state('Autót vásároltam', event_date=day, reference_date=day)
    assert selected['user_context']['event_date_confirmed'] is True
    yesterday = initial_state('Tegnap eladtam az autót', reference_date=day)
    assert yesterday['user_context']['event_date'] == '2026-09-19'
