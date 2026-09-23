"""Regression tests for local Qwen slow responses and redundant work."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.fast_path import classify_explicit, plan_explicit
from dap_assistant.llm import LLMError, OllamaAdapter, select_excerpt
from dap_assistant.rag.retrieval import _chunk_snapshot, _term_counts, bm25_search, diversify_results
from dap_assistant.settings import Settings
from dap_assistant.tooling.tools import requested_tools


def test_fast_classification_handles_original_typo_and_multidomain():
    parsed = classify_explicit('Milyen támogatásokat vehet fel ha megszünt a mukahelyem?')
    assert parsed is not None and parsed.domains == ['employment']
    parsed = classify_explicit('Eladtam az autómat és megszűnt a munkaviszonyom.')
    assert parsed is not None and parsed.domains == ['vehicle', 'employment']
    plan = plan_explicit('Eladtam az autómat és megszűnt a munkaviszonyom.', parsed.domains)
    assert plan is not None and len(plan.tasks) == 2
    assert all(not task.depends_on for task in plan.tasks)
    assert classify_explicit('Mi történik ilyenkor?') is None
    assert classify_explicit('Használt autót szeretnék venni.').role == 'buyer'
    assert classify_explicit('Eladtam az autómat.').role == 'seller'
    assert classify_explicit('Autót vennék és eladnék.').role == ''


def test_answer_context_is_bounded_and_verbatim():
    from dap_assistant.context_engineering.evidence_selection import assemble_selection_context, complete_units
    text = ('Előzmény ' * 180 + '\n' +
            'Álláskeresési járadék: a feltételeket a hatóság vizsgálja.\n' +
            'Melléklet ' * 180)
    # Old excerpt helper is still backwards-compatible; new *production* path
    # only includes complete source units and never slices mid-statement.
    snippet = select_excerpt(text, 'álláskeresési járadék', 300)
    assert len(snippet) <= 300 and snippet in text
    evidence = [{'evidence_id': f'e{i}', 'text': text, 'title': 'Test',
                 'source_url': 'https://dap.gov.hu/'} for i in range(6)]
    assembled = assemble_selection_context(evidence, 'álláskeresési járadék',
                                            domain='employment', token_budget=400)
    assert assembled.evidence
    assert all(doc['text'] in text for doc in assembled.evidence)
    assert all(unit in complete_units(text) for doc in assembled.evidence
               for unit in complete_units(doc['text']))
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if not payload['stream']:
            return httpx.Response(200, json={'message': {'content': json.dumps({
                'evidence_by_need': {'eligibility': ['e0']}, 'missing_needs': []})}})
        result = json.dumps({'claims': [{'text': 'Az álláskeresési járadék feltételeit a hatóság vizsgálja.',
                       'evidence_id': 'e0', 'category': 'eligibility'}], 'disclaimer': ''}, ensure_ascii=False)
        return httpx.Response(200, content=json.dumps({'done': True, 'message': {'content': result}}) + '\n')
    model = OllamaAdapter(replace(Settings(), answer_mode='quick', quick_single_pass=False), transport=httpx.MockTransport(handler))
    try:
        draft = model.answer('álláskeresési járadék jogosultság feltételei', evidence, [], context={
            'life_events': ['employment']})
    finally:
        model.close()
    assert len(requests) == 2
    assert all(req['think'] is False and req['keep_alive'] == '15m' for req in requests)
    assert requests[0]['options']['num_ctx'] == 8192
    assert requests[0]['options']['num_predict'] >= 384
    assert requests[0]['stream'] is False and requests[1]['stream'] is True
    assert len(json.loads(requests[0]['messages'][1]['content'])['available_full_text']) >= 1
    assert draft.claims and draft.claims[0].supporting_quote in text


def test_read_timeout_is_never_retried():
    count = 0

    def timeout(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout('slow local generation', request=request)

    adapter = OllamaAdapter(Settings(), transport=httpx.MockTransport(timeout))
    try:
        with pytest.raises(LLMError, match='ReadTimeout'):
            adapter.classify('autó')
    finally:
        adapter.close()
    assert count == 1


def test_bm25_reuses_tokenization_and_invalidates_content():
    _term_counts.cache_clear()
    first = [{'chunk_id': 'a', 'domain': 'employment', 'text': 'Álláskeresési járadék igénylése'}]
    assert bm25_search('járadék', first, 'employment')
    misses = _term_counts.cache_info().misses
    assert bm25_search('álláskeresési', first, 'employment')
    assert _term_counts.cache_info().misses == misses
    second = [{**first[0], 'text': 'Vállalkozás indítása'}]
    assert not bm25_search('járadék', second, 'employment')
    assert _term_counts.cache_info().misses == misses + 1


def test_chunk_cache_invalidates_when_document_changes(tmp_path):
    import os
    path = tmp_path / 'processed' / 'employment' / 'test.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'chunks': [{'chunk_id': 'a', 'text': 'old'}]}))
    assert _chunk_snapshot(tmp_path)[0]['text'] == 'old'
    path.write_text(json.dumps({'chunks': [{'chunk_id': 'a', 'text': 'new'}]}))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert _chunk_snapshot(tmp_path)[0]['text'] == 'new'


def test_workflow_survives_local_qwen_timeout_with_real_evidence(tmp_path, monkeypatch):
    pytest.importorskip('langgraph')
    from dap_assistant import workflow

    path = tmp_path / 'processed' / 'employment' / 'test.json'
    path.parent.mkdir(parents=True)
    text = ('Álláskeresési járadék igényléséhez hivatalos tájékoztató olvasható. '
            'A jogosultságot az illetékes hatóság állapítja meg.')
    item = {'chunk_id': 'fixture-employment', 'document_id': 'fixture', 'domain': 'employment',
            'role': 'general', 'text': text, 'title': 'Synthetic offline fixture',
            'source_url': 'https://dap.gov.hu/eletesemenyek', 'retrieved_at': '2026-09-20T00:00:00Z',
            'document_version': 'fixture', 'section_path': ['Fixture'], 'page_number': None}
    path.write_text(json.dumps({'chunks': [item]}), encoding='utf-8')

    class FailingGeneration:
        def classify(self, *_args, **_kwargs):
            raise AssertionError('Explicit-domain fast path should skip classification LLM')

        def plan(self, *_args, **_kwargs):
            raise AssertionError('Explicit-domain fast path should skip planning LLM')

        def answer(self, *_args, **_kwargs):
            raise LLMError('Ollama response unavailable or invalid (ReadTimeout)')

    monkeypatch.setattr(workflow, 'get_llm', lambda *args, **kwargs: FailingGeneration())
    settings = replace(Settings(), data_dir=tmp_path, llm_provider='ollama',
                       embedding_provider='dummy', fast_routing=True)
    graph = workflow.build_workflow(settings)
    result = graph.invoke(workflow.initial_state('Milyen támogatások járhatnak, ha megszűnt a munkaviszonyom?'),
                          config={'configurable': {'thread_id': 'timeout-regression'}, 'recursion_limit': 30})
    assert result['answer_fallback'] is True
    assert result['response_status'] == 'partial'
    assert 'Álláskeresési járadék' in result['final_answer']
    assert '[Forrás' in result['final_answer']
    assert not result['tool_results']  # No unrelated deadlines or document checklist.


def test_diversify_without_extra_retrieval():
    rows = [
        {'chunk_id': 'a1', 'document_id': 'A', 'score': .95},
        {'chunk_id': 'a2', 'document_id': 'A', 'score': .94},
        {'chunk_id': 'b1', 'document_id': 'B', 'score': .90},
        {'chunk_id': 'a3', 'document_id': 'A', 'score': .89},
    ]
    chosen = diversify_results(rows, limit=3)
    assert [r['chunk_id'] for r in chosen] == ['a1', 'b1', 'a2']
    assert chosen[0]['score'] == .95


def test_support_query_does_not_trigger_incidental_deadlines_or_checklist():
    assert requested_tools('Milyen támogatásokat vehet fel ha megszünt a mukahelyem?') == (False, False)
    assert requested_tools('Milyen dokumentumok kellenek a járadék igényléséhez?') == (True, False)
    assert requested_tools('Mikor jár le az átírás határideje?') == (True, True)
    assert requested_tools('Eladtam a gépjárművemet. Mit jelentsek be?') == (False, False)
