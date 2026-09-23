"""Conversation regression tests; no paid API, no internet and no Ollama."""
from __future__ import annotations

from dataclasses import replace

import pytest

from dap_assistant.conversation import (is_contextual_followup, is_outside_scope,
                                        is_routing_only, previous_turn_from_messages,
                                        resolve_followup)
from dap_assistant.llm import source_answer
from dap_assistant.presentation.runtime_view import event_from_update, final_prompt_view
from dap_assistant.settings import Settings


def _completed_assistant(domains=None, role='buyer', status='complete'):
    return {'role': 'assistant', 'content': 'Forrásos válasz', 'report': {'final': {
        'domains': domains if domains is not None else ['vehicle'],
        'role': role, 'stage': 'after_event', 'response_status': status,
        'evidence': [{'evidence_id': 'E_TEST', 'text': 'Kormányablakban intézhető.'}],
    }}}


def test_previous_turn_is_taken_from_last_successful_assistant_not_chat_text():
    previous = previous_turn_from_messages([
        {'role': 'user', 'content': 'Álláskeresési járadék'},
        _completed_assistant(['employment'], role=''),
        {'role': 'user', 'content': 'Vásároltam egy autót, milyen teendőim vannak?'},
        _completed_assistant(),
    ])
    assert previous == {'domains': ['vehicle'], 'role': 'buyer', 'stage': 'after_event'}
    assert previous_turn_from_messages([_completed_assistant(status='unsupported')]) is None
    assert previous_turn_from_messages([_completed_assistant(), {'role': 'user', 'content': 'új'}]) is None
    assert previous_turn_from_messages([{'role': 'assistant', 'content': 'hiba'}]) is None
    assert previous_turn_from_messages([_completed_assistant(domains=['unknown'])]) is None


@pytest.mark.parametrize('question', [
    'hol tudom ezeket az ügyeket intézni?',
    'És mennyi a határidő?',
    'Ott hol tudok időpontot foglalni?',
    'Milyen iratok kellenek ehhez?',
    'És milyen támogatásokat igényelhetek?',
    'Mennyi a költsége?',
    'Melyik kormányablakban?',
    'Mi a következő lépés?',
])
def test_followup_detection(question):
    assert is_contextual_followup(question)


def test_resolved_question_uses_only_topic_metadata_not_raw_history():
    previous = {'domains': ['vehicle'], 'role': 'buyer', 'stage': 'after_event'}
    resolved = resolve_followup('hol tudom ezeket az ügyeket intézni?', previous)
    assert 'autó vásárlása' in resolved
    assert 'vevői ügyintézés' in resolved
    assert 'hol tudom ezeket az ügyeket intézni?' in resolved
    assert 'munkaviszony' not in resolved


@pytest.mark.parametrize('question', [
    'Mi az ötös lottó nyerőszámai?',
    'Milyen időjárás lesz holnap?',
    'Ki nyerte tegnap a focimeccset?',
])
def test_out_of_scope_detection(question):
    assert is_outside_scope(question)
    assert not is_contextual_followup(question)


def test_off_topic_not_falsely_matched_in_vehicle_question():
    assert not is_outside_scope('Vásároltam egy használt autót. Hol írathatom át?')


def test_routing_only_requests_do_not_require_ollama_or_a_document_index():
    previous = {'domains': ['vehicle'], 'role': 'buyer', 'stage': 'after_event'}
    assert is_routing_only('Mi az ötös lottó nyerőszámai?', previous, 'employment')
    assert is_routing_only('Mennyi 2 + 2?', previous)
    assert is_routing_only('Hol tudom ezeket az ügyeket intézni?', None)
    assert not is_routing_only('Hol tudom ezeket az ügyeket intézni?', previous)
    assert not is_routing_only('Vásároltam egy használt autót.', None)


def test_where_question_only_returns_office_or_online_source_sentences():
    snippets = [
        {'evidence_id': 'E_PREP', 'text': 'Ellenőrizd a gépjármű műszaki állapotát, és nézd át a szervizkönyvet.'},
        {'evidence_id': 'E_OFFICE', 'text': 'Az átíráshoz keresd fel bármelyik kormányablakot, vagy foglalj időpontot.'},
    ]
    draft = source_answer('Használt autót vettem. Hol tudom ezeket az ügyeket intézni?', snippets)
    assert draft.claims
    assert all('E_OFFICE' in claim.evidence_ids for claim in draft.claims)
    assert all('kormányablak' in claim.text for claim in draft.claims)


def test_where_followup_can_recover_when_qwen_selected_wrong_chunk():
    snippets = [
        {'evidence_id': 'E_PREP', 'text': 'Ellenőrizd a gépjármű műszaki állapotát és szervizkönyvét.'},
        {'evidence_id': 'E_OFFICE', 'text': 'Az átíráshoz keresd fel bármelyik kormányablakot, vagy foglalj időpontot.'},
    ]
    draft = source_answer('Hol tudom ezeket az ügyeket intézni?', snippets, selected_ids=['E_PREP'])
    assert draft.claims and all(claim.evidence_ids == ['E_OFFICE'] for claim in draft.claims)


def test_where_question_does_not_hallucinate_office_if_absent():
    draft = source_answer('Hol tudom ezeket intézni?', [
        {'evidence_id': 'E_PREP', 'text': 'A műszaki állapotot ellenőrizned kell.'},
    ])
    assert not draft.claims


def test_ui_event_contains_resolution_metadata_not_raw_question():
    event = event_from_update((), 'classify_intent', {
        'domains': ['vehicle'], 'role': 'buyer',
        'context_resolution': {'source': 'previous_turn', 'inherited': True},
        'resolved_question': 'SECRET from earlier user question',
    })
    assert event['context_resolution']['inherited'] is True
    assert 'resolved_question' not in event


def test_out_of_scope_prompt_is_not_mislabeled_as_ollama_request():
    preview = final_prompt_view({'classification_status': 'unsupported', 'user_question': 'lottó'}, [])
    assert preview['submitted'] is False
    assert 'NINCS OLLAMA-PROMPT' in preview['text']


@pytest.mark.parametrize('question,expected', [
    ('Vásároltam egy autót, milyen teendőim vannak?', 'vehicle'),
    ('Elvesztettem a munkámat. Milyen teendőim vannak?', 'employment'),
])
def test_explicit_topic_is_not_contextual_just_because_there_is_history(question, expected):
    from dap_assistant.fast_path import classify_explicit
    assert classify_explicit(question).domains == [expected]


def test_real_graph_followup_and_out_of_scope(tmp_path, monkeypatch):
    pytest.importorskip('langgraph')
    import json
    from dap_assistant import workflow

    path = tmp_path / 'processed' / 'vehicle' / 'doc.json'
    path.parent.mkdir(parents=True)
    rows = [
        {'chunk_id': 'test-office', 'document_id': 'fixture-buyer', 'domain': 'vehicle',
         'role': 'buyer', 'text': 'Az átíráshoz keresd fel bármelyik kormányablakot, vagy foglalj időpontot.',
         'title': 'Synthetic vehicle fixture', 'source_url': 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek',
         'retrieved_at': '2026-09-20T00:00:00Z', 'document_version': 'fixture',
         'section_path': ['Ügyintézés'], 'page_number': None},
        {'chunk_id': 'test-prep', 'document_id': 'fixture-buyer', 'domain': 'vehicle',
         'role': 'buyer', 'text': 'A gépjármű műszaki állapotát az adásvétel előtt ellenőrizd.',
         'title': 'Synthetic vehicle fixture', 'source_url': 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek',
         'retrieved_at': '2026-09-20T00:00:00Z', 'document_version': 'fixture',
         'section_path': ['Ellenőrzés'], 'page_number': None},
    ]
    path.write_text(json.dumps({'chunks': rows}), encoding='utf-8')

    class NoOllama:
        def classify(self, *_a, **_kw):
            raise AssertionError('No Ollama classification expected')

        def plan(self, *_a, **_kw):
            raise AssertionError('No Ollama planning expected')

        def answer(self, question, evidence, tools, **_kw):
            return source_answer(question, evidence)

    monkeypatch.setattr(workflow, 'get_llm', lambda *_a, **_kw: NoOllama())
    settings = replace(Settings(), data_dir=tmp_path, llm_provider='dummy',
                       embedding_provider='dummy', answer_mode='source')
    graph = workflow.build_workflow(settings)
    prev = {'domains': ['vehicle'], 'role': 'buyer', 'stage': 'after_event'}
    result = graph.invoke(workflow.initial_state('hol tudom ezeket az ügyeket intézni?',
                                                  previous_turn=prev),
                          config={'configurable': {'thread_id': 'vehicle-followup'},
                                  'recursion_limit': 30})
    assert result['domains'] == ['vehicle']
    assert result['role'] == 'buyer'
    assert result['context_resolution']['source'] == 'previous_turn'
    assert 'kormányablak' in result['final_answer']
    assert 'munkaviszony' not in result['final_answer']

    unrelated = graph.invoke(workflow.initial_state('Mi az ötös lottó nyerőszámai?',
                                                     previous_turn=prev, domain_hint='employment'),
                             config={'configurable': {'thread_id': 'offtopic'},
                                     'recursion_limit': 30})
    assert unrelated['classification_status'] == 'unsupported'
    assert unrelated['domains'] == []
    assert not unrelated['subtasks']
    assert not unrelated['evidence']
    assert not unrelated['tool_results']
    assert unrelated['response_status'] == 'unsupported'
    assert 'lottó' not in unrelated['final_answer'].casefold()
    assert 'DÁP Life Events Assistant' in unrelated['final_answer']

    switched = graph.invoke(workflow.initial_state('Elvesztettem a munkámat. Milyen teendők vannak?',
                                                    previous_turn=prev),
                            config={'configurable': {'thread_id': 'new-topic'},
                                    'recursion_limit': 30})
    assert switched['domains'] == ['employment']
    assert switched['context_resolution']['source'] == 'explicit'
