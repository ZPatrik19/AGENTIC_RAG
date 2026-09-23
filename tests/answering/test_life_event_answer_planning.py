"""Synthetic source-excerpt tests: source attribution, NOT current legal verification."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.fast_path import classify_explicit
from dap_assistant.context_engineering.evidence_selection import assemble_selection_context
from dap_assistant.llm import (OllamaAdapter, _attach_source_quotes, _verified_tool_hints,
                               NaturalResponse, source_answer)
from dap_assistant.presentation.runtime_view import progress_milestones, public_answer_text
from dap_assistant.prompt_engineering.answer_prompt import answer_plan, generation_instruction
from dap_assistant.response.filters import relevant_unit
from dap_assistant.settings import Settings
from dap_assistant.tooling.result_integration import integrate_checklist

QUESTION_SALE = 'Milyen teendőm van, ha eladtam az autómat?'
E1, E2, E3 = ('E_aaaaaaaaaaaaaaaa', 'E_bbbbbbbbbbbbbbbb', 'E_cccccccccccccccc')
DEADLINE = 'Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást.'
UPLOAD = 'Szkenneld be vagy fényképezd le az aláírt adásvételi szerződést.'
PREP = 'A szerződést 4 példányban kell elkészíteni és 2 tanú jelenlétében aláírni.'
SERVICES = 'Keresd elő a szervizkönyvet, mert ezt a vevő a személyes megtekintéskor kérheti.'
SELLER = [
    {'evidence_id': E1, 'chunk_id': 'a'*32, 'domain': 'vehicle', 'role': 'seller',
     'text': DEADLINE + '\nA bejelentést online vagy személyesen megteheted.',
     'source_url': 'https://example.org/test-seller'},
    {'evidence_id': E2, 'chunk_id': 'b'*32, 'domain': 'vehicle', 'role': 'seller',
     'text': UPLOAD + '\n' + PREP + '\n' + SERVICES,
     'source_url': 'https://example.org/test-seller'},
]
WORK_Q = 'Elvesztettem a munkámat. Milyen teendőim vannak?'
REG = 'Az álláskeresőként történő nyilvántartásba vételt a foglalkoztatási osztályon kérheted.'
WORK = [{'evidence_id': E3, 'chunk_id': 'c'*32, 'domain': 'employment',
         'text': REG, 'source_url': 'https://example.org/synthetic-employment'}]


@pytest.mark.parametrize('question,role,stage', [
    (QUESTION_SALE, 'seller', 'after_event'),
    ('Autót adtam el, mit intézzek?', 'seller', 'after_event'),
    ('El szeretném adni az autómat, mire figyeljek?', 'seller', 'planning'),
    (WORK_Q, '', 'after_event'),
    ('Megszűnt a munkaviszonyom, mit intézzek?', '', 'after_event'),
    ('Szeretnék felmondani a munkahelyemen.', '', 'planning'),
])
def test_explicit_life_event_stage(question, role, stage):
    parsed = classify_explicit(question)
    assert parsed is not None and parsed.role == role and parsed.stage == stage


def test_seller_after_event_keeps_upload_and_deadline_not_preparation():
    plan = answer_plan(QUESTION_SALE, 'vehicle', 'seller', '', SELLER)
    assert plan['stage'] == 'after_event'
    assert plan['source_topic_hints'] == ['deadline', 'documents', 'where']
    assert 'ELADTA' in generation_instruction(plan)
    assert relevant_unit(UPLOAD, QUESTION_SALE, 'vehicle', 'seller')
    assert relevant_unit(DEADLINE, QUESTION_SALE, 'vehicle', 'seller')
    assert not relevant_unit(PREP, QUESTION_SALE, 'vehicle', 'seller')
    assert not relevant_unit(SERVICES, QUESTION_SALE, 'vehicle', 'seller')
    # Explicit interest in pre-sale contract preparation must not be censored.
    assert relevant_unit(PREP, 'Eladtam az autómat. Hány példányban kell a szerződés?',
                         'vehicle', 'seller')
    assert relevant_unit(PREP, 'El szeretném adni az autót.', 'vehicle', 'seller')


def test_employment_uses_retrieved_source_topics_not_buyer_seller_or_invented_benefits():
    plan = answer_plan(WORK_Q, 'employment', '', '', WORK)
    assert plan['stage'] == 'after_event'
    assert plan['role'] == ''
    assert plan['source_topic_hints'] == ['where']
    assert 'Munkahely elvesztése' in plan['title']
    assert 'egészségügyi' in generation_instruction(plan).lower()
    assert 'healthcare' not in plan['source_topic_hints']
    assert 'supports' not in plan['source_topic_hints']
    narrow = answer_plan('Mennyi álláskeresési járadékot kaphatok?', 'employment', '', '', WORK)
    assert 'elvesztése után' not in narrow['title']
    assert not narrow['broad_procedural_question']


def test_seller_source_answer_and_model_proposals_exclude_pre_sale_units():
    quoted = source_answer(QUESTION_SALE, SELLER, role='seller')
    assert quoted.claims
    assert all(PREP not in c.text and SERVICES not in c.text for c in quoted.claims)
    proposed = NaturalResponse.model_validate({'claims': [
        {'text': PREP, 'evidence_id': E2, 'category': 'documents'},
        {'text': UPLOAD, 'evidence_id': E2, 'category': 'documents'}]})
    checked = _attach_source_quotes(proposed, SELLER, QUESTION_SALE, role='seller')
    assert all(c.text != PREP for c in checked.claims)
    assert any(c.text == UPLOAD for c in checked.claims)
    assert 'off_topic_for_life_event' in checked.quality_warnings


def test_delivered_checklist_filters_pre_sale_but_preserves_source_exact_upload():
    tool = {'tool': 'get_document_checklist', 'status': 'success',
            'source_evidence_ids': [E2], 'items': [
                {'text': PREP, 'evidence_ids': [E2]},
                {'text': SERVICES, 'evidence_ids': [E2]},
                {'text': UPLOAD, 'evidence_ids': [E2]}]}
    delivered = {'calls': [{'tool_name': 'get_document_checklist', 'status': 'executed',
                            'result_evidence_ids': [E2], 'returned_to_model': True}]}
    added, details = integrate_checklist(
        question=QUESTION_SALE, domain='vehicle', role='seller', evidence=SELLER,
        existing_claims=[], tool_results={}, native_tool_results={
            'native_get_document_checklist_1': tool}, native_tool_trace=delivered)
    assert [c['text'] for c in added] == [UPLOAD]
    assert details['counts']['irrelevant_to_life_event'] == 2
    assert details['counts']['added_verbatim'] == 1
    assert added[0]['origin'] == 'tool_extract' and added[0]['evidence_ids'] == [E2]
    # Not-delivered results never become an assertion, even if the text exists.
    added2, details2 = integrate_checklist(
        question=QUESTION_SALE, domain='vehicle', role='seller', evidence=SELLER,
        existing_claims=[], tool_results={}, native_tool_results={
            'native_get_document_checklist_1': tool}, native_tool_trace={'calls': []})
    assert not added2 and details2['counts']['not_delivered_to_model'] == 1


def test_prompt_tool_hints_only_exact_selected_relevant_source_units():
    tools = [{'tool': 'get_document_checklist', 'status': 'success', 'items': [
        {'text': PREP, 'evidence_ids': [E2]},
        {'text': UPLOAD, 'evidence_ids': [E2]},
        {'text': 'A szerződés nélkül senki nem járhat el.', 'evidence_ids': [E2]}]}]
    hints = _verified_tool_hints(tools, SELLER, QUESTION_SALE, 'seller', 'after_event')
    assert hints == [{'evidence_id': E2, 'source_excerpt': UPLOAD}]
    assert _verified_tool_hints(tools, [SELLER[0]], QUESTION_SALE, 'seller', 'after_event') == []


def test_prompt_for_employment_and_seller_contains_role_stage_and_bounded_hints():
    captured = []
    def handler(request):
        req = json.loads(request.content)
        captured.append(req)
        source = WORK[0] if len(captured) == 1 else SELLER[1]
        claim = REG if len(captured) == 1 else UPLOAD
        result = {'claims': [{'text': claim, 'evidence_id': source['evidence_id'],
                              'category': 'steps' if len(captured) == 1 else 'documents'}]}
        frames = [{'done': False, 'message': {'content': json.dumps(result, ensure_ascii=False)}},
                  {'done': True, 'done_reason': 'stop', 'message': {'content': ''},
                   'eval_count': 100, 'prompt_eval_count': 410}]
        return httpx.Response(200, text='\n'.join(json.dumps(x, ensure_ascii=False) for x in frames) + '\n')
    settings = replace(Settings(), answer_mode='quick', quick_single_pass=True,
                       ollama_quick_num_predict=768, ollama_answer_num_predict=768)
    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(handler))
    try:
        employment = adapter.answer(WORK_Q, WORK, [], context={
            'life_events': ['employment'], 'role': '', 'stage': 'after_event',
            'native_tool_round_trip': True})
        seller_tool = {'tool': 'get_document_checklist', 'status': 'success',
                       'items': [{'text': UPLOAD, 'evidence_ids': [E2]}]}
        seller = adapter.answer(QUESTION_SALE, [SELLER[1]], [seller_tool], context={
            'life_events': ['vehicle'], 'role': 'seller', 'stage': 'after_event',
            'native_tool_round_trip': True})
    finally:
        adapter.close()
    first = json.loads(captured[0]['messages'][1]['content'])
    second = json.loads(captured[1]['messages'][1]['content'])
    assert first['stage'] == 'after_event' and first['role'] == ''
    assert first['response_plan']['source_topic_hints'] == ['where']
    assert 'munkahely' in captured[0]['messages'][0]['content'].lower()
    assert second['stage'] == 'after_event' and second['role'] == 'seller'
    assert second['verified_tool_excerpts'] == [{'evidence_id': E2, 'source_excerpt': UPLOAD}]
    assert 'ELADTA' in captured[1]['messages'][0]['content']
    assert employment.claims and seller.claims


def test_progress_summary_is_real_only_and_debug_data_survives_outside_chat():
    assert progress_milestones([]) == []
    observed = [{'node': 'classify_intent', 'origin': 'main'},
                {'node': 'hybrid_retrieval', 'origin': 'rag_worker:id'},
                {'node': 'prepare_context', 'origin': 'rag_worker:id'},
                {'node': 'execute_tools', 'origin': 'main'}]
    assert len(progress_milestones(observed)) == 4
    assert all('rag_worker:' not in line for line in progress_milestones(observed))
    assert len(progress_milestones(observed + [{'node': 'answer_audit'}])) == 5
    raw = ('**Teendők**\n\n1. [C_7133b1f3ee62789853a7] Tesztállítás. '
           '*(Qwen megfogalmazása)* [Forrás E_afd602e4b8e836db](<https://example.org/x>)')
    public = public_answer_text(raw)
    assert '[C_' not in public and 'E_afd' not in public
    assert 'Qwen megfogalmazása' not in public
    assert '[Hivatalos forrás](<https://example.org/x>)' in public
    assert 'C_7133b1f3ee62789853a7' in raw  # The report retains this exact string.


def test_seller_evidence_selection_spends_budget_on_post_sale_not_pre_sale():
    # Source-exact post-sale items remain; general seller questions should not
    # waste the tight model context on a service book or pre-signing advice.
    packed = assemble_selection_context(SELLER, QUESTION_SALE, domain='vehicle',
                                        role='seller', token_budget=500)
    text = '\n'.join(e['text'] for e in packed.evidence)
    assert DEADLINE in text and UPLOAD in text
    assert PREP not in text and SERVICES not in text
    explicit = assemble_selection_context(SELLER,
        'Eladtam az autómat. Hány példányban kell a szerződés?',
        domain='vehicle', role='seller', token_budget=500)
    assert PREP in '\n'.join(e['text'] for e in explicit.evidence)


def test_employment_source_pack_remains_role_agnostic():
    packed = assemble_selection_context(WORK, WORK_Q, domain='employment',
                                        role='', token_budget=500)
    assert any(REG in e['text'] for e in packed.evidence)
