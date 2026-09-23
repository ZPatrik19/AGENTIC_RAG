"""No network / paid inference; exercise streaming and user-facing provenance."""
from __future__ import annotations

from dataclasses import replace
import json
import time

import httpx
import pytest

from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import LLMError, OllamaAdapter, source_answer
from dap_assistant.presentation.runtime_view import active_components, event_from_update, prompt_export, run_export
from dap_assistant.settings import Settings


EVIDENCE = [
    {'evidence_id': 'E_ONE', 'document_id': 'vehicle-seller', 'domain': 'vehicle',
     'title': 'Használt autó eladása', 'source_url': 'https://dap.gov.hu/example',
     'chunk_id': 'chunk-001', 'score': 0.03,
     'text': 'Indítsd el az online bejelentést a Webes Ügysegéddel.\n'
             'Az eladás idejét add meg a bejelentésben.'},
    {'evidence_id': 'E_TWO', 'document_id': 'vehicle-seller', 'domain': 'vehicle',
     'title': 'Szerződés', 'source_url': 'https://dap.gov.hu/example',
     'chunk_id': 'chunk-002', 'score': 0.02,
     'text': 'Az adásvételi szerződést a felek írják alá.'},
]


def test_quick_stream_selects_existing_sources_and_exports_exact_prompt():
    calls = []
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if not payload['stream']:
            # A selection must explicitly acknowledge every unselected need.
            body = {'evidence_by_need': {'steps': ['E_ONE']},
                    'missing_needs': ['insurance', 'deadline', 'documents', 'where']}
            return httpx.Response(200, json={'done': True,
                'message': {'content': json.dumps(body)},
                'prompt_eval_count': 120, 'eval_count': 12, 'eval_duration': 900000000})
        answer = {'claims': [{'evidence_id': 'E_ONE', 'category': 'where',
                   'text': 'A tulajdonosváltást a Webes Ügysegéden jelentheted be.'}],
                  'disclaimer': ''}
        fragments = [{'done': False, 'message': {'content': json.dumps(answer, ensure_ascii=False)}},
                     {'done': True, 'message': {'content': ''}, 'prompt_eval_count': 88,
                      'eval_count': 27, 'eval_duration': 900000000}]
        return httpx.Response(200, content='\n'.join(json.dumps(v, ensure_ascii=False) for v in fragments) + '\n')
    telemetry = Telemetry()
    adapter = OllamaAdapter(replace(Settings(), answer_mode='quick', quick_single_pass=False),
                            transport=httpx.MockTransport(handler), telemetry=telemetry)
    try:
        draft = adapter.answer('Eladtam az autómat. Milyen teendőim vannak?', EVIDENCE, [], 'run-one')
    finally:
        adapter.close()
    assert len(calls) == 2
    assert all(call['think'] is False for call in calls)
    assert calls[0]['stream'] is False and calls[1]['stream'] is True
    assert calls[0]['options']['num_predict'] >= 384
    assert 'available_full_text' in json.loads(calls[0]['messages'][1]['content'])
    assert draft.claims and all(c.supporting_quote in EVIDENCE[0]['text'] for c in draft.claims)
    assert telemetry.live('run-one')['received_bytes'] > 0
    assert [row['phase'] for row in telemetry.snapshot('run-one')['llm_usage']] == ['selection', 'answer']
    assert telemetry.snapshot('run-one')['llm_usage'][-1]['eval_count'] == 27
    preview = telemetry.prompt_preview('run-one')
    assert len(preview) == 2 and all(p['think'] is False for p in preview)
    assert 'Eladtam az autómat' in prompt_export(preview)
    assert 'Eladtam' not in json.dumps(telemetry.snapshot('run-one'))


def test_source_mode_does_not_call_ollama_and_extracts_verbatim_quotes():
    def forbidden(request):
        raise AssertionError('Source-only mode must not call Ollama')

    adapter = OllamaAdapter(replace(Settings(), answer_mode='source'),
                            transport=httpx.MockTransport(forbidden))
    try:
        result = adapter.answer('Eladtam az autót, mit intézzek?', EVIDENCE, [])
    finally:
        adapter.close()
    assert result.claims
    assert all(c.supporting_quote in next(x['text'] for x in EVIDENCE
                                         if x['evidence_id'] == c.evidence_ids[0])
               for c in result.claims)
    assert 'Forrásalapú kivonat' in result.disclaimer


def test_selection_timeout_is_not_replayed_and_workflow_can_mark_partial():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout('no token arrived', request=request)

    adapter = OllamaAdapter(replace(Settings(), answer_mode='quick', quick_single_pass=False),
                            transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(LLMError, match='ReadTimeout'):
            adapter.answer('Eladtam az autót', EVIDENCE, [])
    finally:
        adapter.close()
    assert calls == 1


def test_hard_generation_budget_aborts_continuously_streaming_model():
    calls = []
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if not payload['stream']:
            return httpx.Response(200, json={'message': {'content': json.dumps({
                'evidence_by_need': {'steps': ['E_ONE']},
                'missing_needs': ['insurance', 'deadline', 'documents', 'where']})}})
        class SlowStream(httpx.SyncByteStream):
            def __iter__(self):
                yield b'{"message":{"content":"{"},"done":false}\n'
                time.sleep(0.02)
                yield b'{"message":{"content":"}"},"done":false}\n'
            def close(self):
                pass
        return httpx.Response(200, request=request, stream=SlowStream())
    adapter = OllamaAdapter(replace(Settings(), answer_mode='quick', quick_single_pass=False, ollama_total_timeout_s=0.001),
                            transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(LLMError, match='LLMError'):
            adapter.answer('Eladtam az autót', EVIDENCE, [])
    finally:
        adapter.close()
    assert len(calls) == 2  # selection once, generation once; never replay partial response


def test_event_and_report_from_real_payload_only():
    event = event_from_update((), 'evidence_gate', {
        'domains': ['vehicle'], 'evidence': EVIDENCE,
        'subtasks': {'t1': {'domain': 'vehicle', 'question': 'Eladói teendők',
                            'status': 'complete', 'depends_on': []}},
    })
    assert event['domains'] == ['vehicle']
    assert event['evidence_count'] == 2
    assert len(event['documents']) == 1
    assert event['subtasks'][0]['status'] == 'complete'
    trace = {'spans': [
        {'name': 'main/generate_answer', 'duration_s': 1.2},
        {'name': 'llm_inference', 'duration_s': 1.19},
        {'name': 'rag_subgraph', 'duration_s': .02},
        {'name': 'bm25_retrieval', 'duration_s': .01},
    ], 'llm_usage': []}
    components = active_components(trace)
    assert [c['component'] for c in components] == ['llm_inference', 'bm25_retrieval']
    report = run_export(events=[event], trace=trace,
                        final={'domains': ['vehicle'], 'subtasks': {'t1': {'status': 'complete'}},
                               'answer_validation': {'status': 'passed'}}, elapsed_s=1.3)
    assert report['elapsed_s'] == 1.3
    assert 'user_question' not in report


def test_missing_selected_id_fails_to_source_evidence_without_invention():
    draft = source_answer('Eladtam az autóm', EVIDENCE, ['FAKE_ID'])
    assert draft.claims
    assert all(claim.evidence_ids[0] != 'FAKE_ID' for claim in draft.claims)
