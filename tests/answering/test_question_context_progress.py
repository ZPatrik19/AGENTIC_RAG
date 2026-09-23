"""Synthetic source and mocked graph/LLM tests, no Ollama, claims or network."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx

from dap_assistant.llm import OllamaAdapter
from dap_assistant.presentation.runtime_view import event_from_update, progress_milestones
from dap_assistant.context_engineering.question_analysis import analyze_question, align_sources
from dap_assistant.prompt_engineering.answer_prompt import answer_plan, generation_instruction
from dap_assistant.settings import Settings


E1 = 'E_1234567890abcdef'
E2 = 'E_abcdef1234567890'
SYNTHETIC_WORK = [
    {'evidence_id': E1, 'domain': 'employment', 'document_id': 'synthetic-guide',
     'source_url': 'https://example.org/synthetic',
     'text': 'A nyilvántartásba vételt a foglalkoztatási osztályon kérheted.'},
    {'evidence_id': E2, 'domain': 'employment', 'document_id': 'synthetic-guide',
     'source_url': 'https://example.org/synthetic',
     'text': 'A kérelmedhez szükséges igazolást az ügyintézés során csatold.'},
]


def test_context_analysis_distinguishes_user_goal_stage_and_channel_without_entitlements():
    broad = analyze_question('Megszűnt a munkaviszonyom, milyen teendőim vannak?',
                             domain='employment', stage='after_event')
    assert broad['role'] == 'not_specified'
    assert broad['stage'] == 'after_event'
    assert broad['goal'] == 'procedure' and broad['scope'] == 'general'
    docs = analyze_question('Milyen iratokat kérjek és hol intézzem személyesen?',
                            domain='employment', stage='unknown')
    assert docs['goal'] == 'documents' and docs['scope'] == 'focused'
    assert set(docs['requested_needs']) >= {'documents', 'where'}
    assert docs['preferred_channel'] == 'in_person'
    assert docs['stage'] == 'unknown'
    benefit = analyze_question('Mennyi álláskeresési járadékot kaphatok?', domain='employment')
    assert benefit['goal'] == 'benefit_amount'
    assert 'eligibility' not in benefit  # No entitlement decision or extracted amount.
    assert 'amount' not in benefit
    assert 'munkaviszonyom' not in repr(broad)  # Do not copy personal text to trace.


def test_alignment_reports_topic_overlap_not_fabricated_requirements():
    question = 'Milyen dokumentumokat és hol kell intéznem?'
    plan = answer_plan(question, 'employment', '', 'unknown', SYNTHETIC_WORK)
    assert plan['question_analysis']['goal'] == 'documents'
    assert plan['source_alignment']['matched_topic_hints'] == ['documents', 'where']
    assert plan['source_alignment']['retrieved_evidence_count'] == 2
    assert 'entailment' not in plan['source_alignment']['method'] or 'not_' in plan['source_alignment']['method']
    no_health = answer_plan('Mire vagyok jogosult az egészségügyi ellátásnál?',
                            'employment', '', 'unknown', SYNTHETIC_WORK)
    assert 'healthcare' not in no_health['source_topic_hints']
    assert no_health['source_alignment']['method'].endswith('not_claim_support_or_document_completeness')
    absent = align_sources({'requested_needs': ['documents']}, [], retrieved_evidence_count=0)
    assert absent['matched_topic_hints'] == [] and absent['unmatched_requested_topic_hints'] == ['documents']
    assert 'iratokat' in generation_instruction(plan).lower() or 'dokumentumokkal' in generation_instruction(plan).lower()


def test_only_completed_graph_updates_appear_as_seven_human_steps():
    sequence = [
        event_from_update((), 'classify_intent', {'question_analysis': {
            'goal': 'procedure', 'stage': 'after_event', 'raw_question': 'SENSITIVE DATA'}}),
        event_from_update((), 'plan_tasks', {'subtasks': {'t1': {'domain': 'employment', 'question': 'X',
                                                                'status': 'running', 'depends_on': []}}}),
        event_from_update(('rag_worker:unwanted-uuid',), 'hybrid_retrieval', {}),
        event_from_update((), 'evidence_gate', {'evidence': SYNTHETIC_WORK}),
        event_from_update((), 'execute_tools', {'native_tool_trace': {'calls': [
            {'status': 'executed', 'returned_to_model': True},
            {'status': 'executed', 'returned_to_model': True},
            {'status': 'failed', 'returned_to_model': False}]}}),
        event_from_update((), 'generate_answer', {}),
        event_from_update((), 'answer_audit', {'answer_validation': {'status': 'partial'}}),
    ]
    summary = progress_milestones(sequence)
    assert len(summary) == 7
    assert 'Élethelyzet azonosítása' in summary[0]
    assert 'Önálló feladatágak kiosztása' in summary[1]
    assert 'Cél: teendők' in summary[0] and 'megtörtént esemény' in summary[0]
    assert '1 keresési feladat' in summary[1]
    assert '1 dokumentum · 2 forrásrészlet' in summary[3]
    assert '2 végrehajtott natív eszköz' in summary[4]
    assert 'részleges eredmény' in summary[-1]
    assert 'rag_worker:' not in '\n'.join(summary)
    assert 'SENSITIVE DATA' not in repr(sequence) and 'SENSITIVE DATA' not in '\n'.join(summary)
    assert len(progress_milestones(sequence[:4] + sequence[5:])) == 6  # no imaginary tools
    assert len(progress_milestones(sequence[:2] + sequence[2:4] * 3)) == 4
    assert len(progress_milestones(sequence[:-1])) == 6  # audit not declared complete in advance


def test_real_compact_answer_prompt_includes_separate_question_and_source_context():
    requests = []

    def handler(request):
        req = json.loads(request.content)
        requests.append(req)
        response = {'claims': [{
            'text': SYNTHETIC_WORK[0]['text'], 'evidence_id': E1, 'category': 'steps'}]}
        frames = [{'done': False, 'message': {'content': json.dumps(response, ensure_ascii=False)}},
                  {'done': True, 'done_reason': 'stop', 'message': {'content': ''},
                   'eval_count': 76, 'prompt_eval_count': 350}]
        return httpx.Response(200, text='\n'.join(json.dumps(frame, ensure_ascii=False)
                                                   for frame in frames) + '\n')

    settings = replace(Settings(), answer_mode='quick', quick_single_pass=True,
                       ollama_quick_num_predict=768, ollama_answer_num_predict=768)
    adapter = OllamaAdapter(settings, transport=httpx.MockTransport(handler))
    try:
        draft = adapter.answer('Megszűnt a munkaviszonyom, milyen teendőim vannak?',
                               SYNTHETIC_WORK, [], context={
                                   'life_events': ['employment'], 'role': '', 'stage': 'after_event',
                                   'native_tool_round_trip': True})
    finally:
        adapter.close()
    assert draft.claims and len(requests) == 1
    payload = json.loads(requests[0]['messages'][1]['content'])
    assert payload['question_analysis']['goal'] == 'procedure'
    assert payload['question_analysis']['stage'] == 'after_event'
    assert payload['response_plan']['source_alignment']['retrieved_evidence_count'] >= 1
    assert payload['response_plan']['source_alignment']['method'].endswith('not_claim_support_or_document_completeness')
    assert requests[0]['options']['num_predict'] == 768


def test_amount_question_requires_retrieved_monetary_unit_not_only_benefit_heading():
    question = 'Mennyi álláskeresési járadékot kaphatok?'
    without_amount = answer_plan(question, 'employment', '', '', [
        {'domain': 'employment', 'text': 'Tájékoztató az álláskeresési járadékról.'}])
    assert without_amount['question_analysis']['goal'] == 'benefit_amount'
    assert 'benefit_amount' in without_amount['source_alignment']['unmatched_requested_topic_hints']
    with_amount = answer_plan(question, 'employment', '', '', [
        {'domain': 'employment', 'text': 'Példa: a járadék összege 2 500 Ft egy tesztkörnyezetben.'}])
    assert 'benefit_amount' in with_amount['source_alignment']['matched_topic_hints']
    # These are lexical cues, not a promise that the synthetic amount is applicable.
    assert with_amount['source_alignment']['method'].endswith('not_claim_support_or_document_completeness')


def test_production_graph_routes_employment_question_context_to_answer(monkeypatch, tmp_path):
    """Exercise real classification and answer nodes with synthetic source text."""
    from dap_assistant.llm import Claim, Draft
    from test_workflow_answer_nodes import _graph

    question = 'Megszűnt a munkaviszonyom. Milyen dokumentumok kellenek személyesen?'
    recorded = {}

    class FakeLLM:
        def answer(self, query, evidence, tools, **kwargs):
            recorded['context'] = kwargs['context']
            return Draft(claims=[Claim(text=SYNTHETIC_WORK[0]['text'],
                                       supporting_quote=SYNTHETIC_WORK[0]['text'],
                                       evidence_ids=[E1], category='where',
                                       origin='model_generated')],
                         model_context_evidence_ids=[E1])

    with monkeypatch.context() as patch:
        module, graph = _graph(patch, FakeLLM(), tmp_path)
        state = module.initial_state(question)
        state.update(graph.nodes['classify_intent'](state))
        assert state['domains'] == ['employment'] and state['role'] == ''
        assert state['question_analysis']['goal'] == 'documents'
        assert state['question_analysis']['stage'] == 'after_event'
        assert state['question_analysis']['preferred_channel'] == 'in_person'
        event = event_from_update((), 'classify_intent', state)
        assert 'question_analysis' in event
        state.update(evidence=SYNTHETIC_WORK, validation={'status': 'passed'})
        state.update(graph.nodes['generate_answer'](state))

    ctx = recorded['context']
    assert ctx['question_analysis']['goal'] == 'documents'
    assert ctx['response_plan']['question_analysis']['preferred_channel'] == 'in_person'
    assert ctx['response_plan']['source_alignment']['matched_topic_hints'] == ['documents']
    assert state['answer_draft']['claims']
