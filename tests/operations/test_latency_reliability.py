"""Real adapter with mocked HTTP, no paid API or synthetic legal gold."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx

from dap_assistant.context_engineering.evidence_selection import (
    assemble_selection_context,
    deterministic_facet_selection,
    is_relevant_evidence,
)
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.llm import OllamaAdapter, source_answer
from dap_assistant.settings import Settings

URL = 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek'


def evidence(item_id, text, domain='vehicle', role='buyer', title='DÁP autóvásárlás'):
    return {'evidence_id': item_id, 'chunk_id': item_id, 'text': text,
            'domain': domain, 'role': role, 'source_url': URL, 'title': title}


BUYER = [
    evidence('E_DEAD', 'Az átírást az adásvételi szerződés megkötésétől számított 15 napon belül kell elintézni.'),
    evidence('E_INS', 'Tulajdonosváltáskor új kötelező gépjármű-felelősségbiztosítást kell kötni.'),
    evidence('E_DOC', 'Az átíráshoz szükséges az adásvételi szerződés és a törzskönyv.'),
    evidence('E_WHERE', 'Az átírás a kormányablakban intézhető.'),
]


def test_quick_is_one_real_qwen_request_with_bounded_context_and_source_coverage():
    calls = []

    def handler(request):
        data = json.loads(request.content)
        calls.append(data)
        response = {'claims': [
            {'evidence_id': 'E_DEAD', 'text': 'Az átírást az adásvételi szerződés megkötésétől számított 15 napon belül kell elintézni.', 'category': 'deadline'},
        ], 'disclaimer': ''}
        return httpx.Response(200, content='\n'.join(json.dumps(frame, ensure_ascii=False) for frame in (
            {'done': False, 'message': {'content': json.dumps(response, ensure_ascii=False)}},
            {'done': True, 'message': {'content': ''}, 'prompt_eval_count': 310,
             'eval_count': 80, 'eval_duration': 1_000_000_000},
        )) + '\n')

    telemetry = Telemetry()
    client = OllamaAdapter(replace(Settings(), answer_mode='quick', ollama_num_ctx=8192,
                                 ollama_quick_num_predict=384, answer_adaptive_output=False),
                           transport=httpx.MockTransport(handler), telemetry=telemetry)
    try:
        draft = client.answer('Vásároltam egy autót, milyen teendőim vannak?', BUYER,
                              [], 'one-call', {'life_events': ['vehicle'], 'role': 'buyer'})
    finally:
        client.close()
    assert len(calls) == 1
    assert calls[0]['stream'] is True and calls[0]['think'] is False
    assert calls[0]['options']['num_ctx'] == 3072
    assert calls[0]['options']['num_predict'] == 384
    assert {'deadline', 'insurance', 'documents', 'where'} <= {c.category for c in draft.claims}
    assert 'szó szerinti' in draft.disclaimer
    assert all(claim.supporting_quote in next(e['text'] for e in BUYER
               if claim.evidence_ids == [e['evidence_id']]) for claim in draft.claims)
    assert len(telemetry.prompt_preview('one-call')) == 1


def test_unrelated_scooter_insurance_not_in_vehicle_prompt_or_fallback():
    scooter = evidence('E_ROLLER', 'Mielőtt rollert vásárol, ellenőrizze a kötelező gépjármű-felelősségbiztosítást.',
                       title='MABISZ – mikromobilitás')
    assert not is_relevant_evidence(scooter, 'Vettem egy autót', 'vehicle', 'buyer')
    items = BUYER + [scooter]
    packed = assemble_selection_context(items, 'Vettem egy autót milyen teendőim vannak?',
                                        domain='vehicle', role='buyer', token_budget=800)
    assert 'E_ROLLER' not in {item['evidence_id'] for item in packed.evidence}
    draft = source_answer('Vettem egy autót milyen teendőim vannak?', items)
    assert all('roller' not in claim.text.casefold() for claim in draft.claims)


def test_detailed_invalid_evidence_id_is_recovered_without_fabrication():
    requests = []

    def handler(request):
        data = json.loads(request.content)
        requests.append(data)
        if not data['stream']:
            return httpx.Response(200, json={'message': {'content': json.dumps({
                'evidence_by_need': {'where': ['E_NONEXISTENT']}, 'missing_needs': []})}})
        generated = {'claims': [{'evidence_id': 'E_WHERE',
                                 'text': 'Az átírást a kormányablakban intézheted.',
                                 'category': 'where'}], 'disclaimer': ''}
        return httpx.Response(200, content=json.dumps({
            'done': True, 'message': {'content': json.dumps(generated, ensure_ascii=False)}}) + '\n')

    adapter = OllamaAdapter(replace(Settings(), answer_mode='detailed'),
                            transport=httpx.MockTransport(handler))
    try:
        draft = adapter.answer('Hol intézhetem az autó átírását?', BUYER, [],
                               context={'life_events': ['vehicle'], 'role': 'buyer'})
    finally:
        adapter.close()
    assert len(requests) == 2
    assert draft.claims
    assert all('E_NONEXISTENT' not in claim.evidence_ids for claim in draft.claims)


def test_deterministic_selector_returns_missing_without_fake_id():
    packed = assemble_selection_context([BUYER[0]], 'Vettem autót, hol intézem?',
                                        domain='vehicle', role='buyer', token_budget=300)
    selected = deterministic_facet_selection(packed, ('deadline', 'where'))
    assert selected['where'] == []
    assert set(selected['deadline']) <= {'E_DEAD'}


def test_employment_invalid_where_id_does_not_crash_detailed_mode():
    employment = [evidence('E_JOB',
        'Az álláskeresőként történő nyilvántartásba vételt a járási hivatal foglalkoztatási osztályán kezdeményezheted.',
        domain='employment', role='general', title='Foglalkoztatási tájékoztató')]
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if not payload['stream']:
            return httpx.Response(200, json={'message': {'content': json.dumps({
                'evidence_by_need': {'where': ['E_WRONG']}, 'missing_needs': []})}})
        content = {'claims': [{'evidence_id': 'E_JOB', 'category': 'where',
                              'text': 'A nyilvántartásba vételt a járási hivatal foglalkoztatási osztályán kezdeményezheted.'}]}
        return httpx.Response(200, content=json.dumps({'message': {'content': json.dumps(content, ensure_ascii=False)},
                                                       'done': True}) + '\n')

    adapter = OllamaAdapter(replace(Settings(), answer_mode='detailed'),
                            transport=httpx.MockTransport(handler))
    try:
        result = adapter.answer('Megszűnt a munkaviszonyom, hol intézzem az álláskeresést?',
                                employment, [], context={'life_events': ['employment']})
    finally:
        adapter.close()
    assert len(calls) == 2
    assert result.claims and all(claim.evidence_ids == ['E_JOB'] for claim in result.claims)


def test_actual_ollama_status_reports_partial_gpu_without_inference():
    from scripts.check_local_ollama import inspect_ollama

    def handler(request):
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [{'name': 'qwen3:4b'}]})
        return httpx.Response(200, json={'models': [
            {'name': 'qwen3:4b', 'size': 4_000_000_000, 'size_vram': 1_000_000_000,
             'context_length': 2048}]})

    report = inspect_ollama(replace(Settings(), ollama_num_ctx=8192), httpx.MockTransport(handler))
    assert report['quick_requested_num_ctx'] == 3072
    assert report['reported_vram_fraction'] == .25
    assert report['model_loaded'] and report['model_installed']
