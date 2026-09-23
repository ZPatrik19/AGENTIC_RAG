"""P6.3: no live Ollama, external services or invented Hungarian legal gold."""
from __future__ import annotations

from dataclasses import replace
import json

import httpx
import pytest

from dap_assistant.response.filters import (
    missing_healthcare_condition, post_purchase_procedure, pre_purchase_background,
)
from dap_assistant.response.quality import missing_facets_hungarian, clean_claims
from dap_assistant.evaluation.comparison import _render_simple_answer, _usage
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.context_engineering.evidence_selection import assemble_selection_context
from dap_assistant.llm import LLMError, NaturalResponse, OllamaAdapter, source_answer, _attach_source_quotes
from dap_assistant.settings import Settings


def make_adapter(handler):
    telemetry = Telemetry()
    adapter = OllamaAdapter(replace(Settings(), answer_mode='quick',
                                    ollama_quick_num_predict=768,
                                    ollama_answer_num_predict=768),
                            transport=httpx.MockTransport(handler), telemetry=telemetry)
    return adapter, telemetry


def stream(frames):
    return httpx.Response(200, content='\n'.join(json.dumps(f, ensure_ascii=False) for f in frames) + '\n')


def test_length_failure_records_real_tokens_before_raising_and_no_json_is_logged():
    secret = '{"private":"SHOULD_NEVER_APPEAR_IN_TRACE"}'
    def handler(request):
        return stream([{'done': False, 'message': {'content': secret}},
            {'done': True, 'message': {'content': ''}, 'done_reason': 'length',
             'eval_count': 768, 'prompt_eval_count': 450}])
    adapter, telemetry = make_adapter(handler)
    try:
        with pytest.raises(LLMError, match='LLMError'):
            adapter._ask('system', 'question', NaturalResponse, run_id='length', phase='answer')
    finally:
        adapter.close()
    trace = telemetry.snapshot('length')
    failure = trace['llm_failures'][0]
    assert failure['failure_kind'] == 'output_token_limit'
    assert failure['done_reason'] == 'length'
    assert failure['eval_count'] == 768
    assert failure['requested_num_predict'] == 768
    assert failure['observed_content_chars'] == len(secret)
    assert failure['usage_source'] == 'ollama_eval_count'
    assert trace['llm_usage'][0]['eval_count'] == 768
    usage = _usage(trace)
    assert usage['llm_calls'] == 0 and usage['llm_responses_with_usage'] == 1
    assert usage['generated_tokens_observed'] == 768
    assert not usage['answer_generation_completed']
    assert 'SHOULD_NEVER_APPEAR_IN_TRACE' not in json.dumps(trace)


def test_schema_validation_diagnostics_without_source_text():
    invalid = json.dumps({'claims': [{'evidence_id': 'E_TEST', 'text': 'Rövid.',
                                     'category': 'steps'}]}, ensure_ascii=False)
    def handler(_):
        return stream([{'done': True, 'done_reason': 'stop', 'message': {'content': invalid},
                        'eval_count': 29, 'prompt_eval_count': 53}])
    adapter, telemetry = make_adapter(handler)
    try:
        with pytest.raises(LLMError):
            adapter._ask('system', 'question', NaturalResponse, run_id='schema', phase='answer')
    finally:
        adapter.close()
    failure = telemetry.snapshot('schema')['llm_failures'][0]
    assert failure['failure_kind'] == 'schema_validation'
    assert failure['validation_error_type'] == 'string_too_short'
    assert failure['eval_count'] == 29 and failure['done_reason'] == 'stop'
    assert 'Rövid.' not in json.dumps(failure)


def test_malformed_json_distinguished_from_schema_and_token_limits():
    def handler(_):
        return stream([{'done': True, 'done_reason': 'stop', 'message': {'content': '{not-json'},
                        'eval_count': 9}])
    adapter, telemetry = make_adapter(handler)
    try:
        with pytest.raises(LLMError):
            adapter._ask('system', 'question', NaturalResponse, run_id='json', phase='answer')
    finally:
        adapter.close()
    failure = telemetry.snapshot('json')['llm_failures'][0]
    assert failure['failure_kind'] == 'json_syntax'
    assert failure['validation_error_type'] == 'json_invalid'
    assert failure['eval_count'] == 9
    assert '{not-json' not in json.dumps(failure)


def test_stream_without_final_frame_keeps_observed_chars_but_token_count_unknown():
    def handler(_):
        return stream([{'done': False, 'message': {'content': 'partial JSON'}}])
    adapter, telemetry = make_adapter(handler)
    try:
        with pytest.raises(LLMError):
            adapter._ask('system', 'question', NaturalResponse, run_id='partial', phase='answer')
    finally:
        adapter.close()
    trace = telemetry.snapshot('partial')
    failure = trace['llm_failures'][0]
    assert failure['failure_kind'] == 'stream_interrupted_or_incomplete'
    assert failure['observed_content_chars'] == len('partial JSON')
    assert failure['eval_count'] is None and failure['usage_source'] == 'unavailable'
    assert _usage(trace)['generated_tokens_observed'] is None
    assert trace['llm_usage'] == []


BUY_QUESTION = 'Vettem egy használt autót. Milyen ügyintézési teendőim vannak?'
URL = 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek'


def item(eid, text, domain='vehicle'):
    return {'evidence_id': eid, 'chunk_id': eid, 'text': text,
            'title': 'Hivatalos tájékoztató', 'source_url': URL,
            'domain': domain, 'role': 'buyer' if domain == 'vehicle' else 'general'}


def test_after_purchase_context_skips_warranty_and_pre_sale_check_without_losing_obligations():
    sources = [
        item('E_PRE', 'Személyes megtekintéskor ellenőrizd a szervizkönyvet és a gépjármű műszaki állapotát.'),
        item('E_WARRANTY', 'A használt gépjárműre is vonatkozik a kellékszavatosság.'),
        item('E_ORIGIN', 'Ha a vásárlás előtt nem történt eredetiségvizsgálat, az átírás előtt el kell végeztetni.'),
        item('E_DEADLINE', 'Az adásvételi szerződés megkötésétől számított 15 napon belül át kell íratnod a gépjárművet.'),
        item('E_INSURANCE', 'Tulajdonosváltáskor új kötelező gépjármű-felelősségbiztosítást kell kötni.'),
        item('E_WHERE', 'Az átírás a kormányablakban intézhető.'),
        item('E_DOCUMENTS', 'Az átíráshoz szükséges az adásvételi szerződés és a törzskönyv.'),
    ]
    assert post_purchase_procedure(BUY_QUESTION, 'vehicle', 'buyer')
    assert pre_purchase_background(sources[0]['text'])
    assert not pre_purchase_background(sources[2]['text'])
    ctx = assemble_selection_context(sources, BUY_QUESTION, domain='vehicle', role='buyer', token_budget=500)
    ids = {part['evidence_id'] for part in ctx.evidence}
    assert 'E_PRE' not in ids and 'E_WARRANTY' not in ids
    assert {'E_ORIGIN', 'E_DEADLINE', 'E_INSURANCE', 'E_WHERE', 'E_DOCUMENTS'} <= ids
    fallback = source_answer(BUY_QUESTION, sources)
    assert all('szervizkönyv' not in c.text.casefold() and 'kellékszavatosság' not in c.text.casefold()
               for c in fallback.claims)
    # Specific pre-sale questions remain eligible for the underlying sources.
    assert not post_purchase_procedure('Vásárlás előtt hogyan vizsgáljam át az autót?', 'vehicle', 'buyer')


def test_healthcare_conditional_claim_requires_condition_in_both_answer_and_quote():
    quote = 'Amíg kapod az álláskeresési járadékot, megmarad a biztosítási jogviszonyod.'
    deictic = 'Ez azt jelenti, hogy nem kell TB járulékot fizetned.'
    assert missing_healthcare_condition('Nem kell TB járulékot fizetned.', deictic, 'healthcare')
    assert not missing_healthcare_condition(quote, quote, 'healthcare')
    assert not missing_healthcare_condition('Ellenőrizd az egészségügyi jogosultságodat.',
                                            'Ellenőrizd az egészségügyi jogosultságodat.', 'healthcare')
    evidence = [item('E_CARE', quote, domain='employment'),
                item('E_UNCONDITIONAL', deictic, domain='employment')]
    draft = source_answer('Elvesztettem a munkámat. Milyen ügyintézési teendőim vannak?', evidence)
    assert not any('nem kell TB' in c.text for c in draft.claims)
    response = NaturalResponse(claims=[{'evidence_id': 'E_CARE', 'category': 'healthcare',
                  'text': 'Amíg kapod az álláskeresési járadékot, megmarad a biztosítási jogviszonyod.'}])
    attached = _attach_source_quotes(response, evidence)
    assert attached.claims and attached.claims[0].supporting_quote == quote
    unsafe = NaturalResponse(claims=[{'evidence_id': 'E_UNCONDITIONAL', 'category': 'healthcare',
                                      'text': 'Nem kell TB járulékot fizetned a munkaviszony megszűnése után.'}])
    assert not _attach_source_quotes(unsafe, evidence).claims


def test_missing_facet_labels_hungarian_are_used_in_baseline_and_hybrid():
    labels = missing_facets_hungarian(['supports', 'documents', 'healthcare'])
    assert 'támogatások és szolgáltatások' in labels
    assert 'szükséges dokumentumok' in labels
    assert 'egészségügyi jogosultság feltételei' in labels
    draft = {'claims': [], 'requested_facet_coverage': {'missing': ['supports', 'healthcare']},
             'disclaimer': ''}
    answer = _render_simple_answer(draft, [])
    assert 'supports' not in answer and 'healthcare' not in answer
    assert 'támogatások' in answer and 'egészségügyi' in answer


def test_audit_guard_rejects_claim_with_unstated_healthcare_condition():
    quote = ('Amíg kapod az álláskeresési járadékot, megmarad a biztosítási jogviszonyod. '
             'Ez azt jelenti, hogy nem kell TB járulékot fizetned.')
    assert missing_healthcare_condition(
        'Amíg kapod az álláskeresési járadékot, nem kell TB járulékot fizetned.',
        'Amíg kapod az álláskeresési járadékot, megmarad a biztosítási jogviszonyod.',
        'healthcare')
    valid, audit = clean_claims([{'category': 'healthcare',
        'text': 'Nem kell TB járulékot fizetned.', 'supporting_quote': quote,
        'evidence_ids': ['E_X']}])
    assert not valid
    assert len(audit['healthcare_condition_issues']) == 1
    valid, audit = clean_claims([{'category': 'healthcare',
        'text': 'Amíg kapod az álláskeresési járadékot, nem kell TB járulékot fizetned.',
        'supporting_quote': quote, 'evidence_ids': ['E_X']}])
    assert len(valid) == 1
    assert audit['healthcare_condition_issues'] == []


def test_successful_json_is_not_confused_with_raw_ollama_usage():
    valid = json.dumps({'claims': [], 'disclaimer': ''})
    def handler(_):
        return stream([{'done': True, 'done_reason': 'stop',
                        'message': {'content': valid}, 'eval_count': 23, 'prompt_eval_count': 120}])
    adapter, telemetry = make_adapter(handler)
    try:
        assert adapter._ask('system', 'question', NaturalResponse,
                            run_id='success', phase='answer').claims == []
    finally:
        adapter.close()
    trace = telemetry.snapshot('success')
    usage = _usage(trace)
    assert usage['llm_calls'] == 1 and usage['llm_responses_with_usage'] == 1
    assert usage['generated_tokens_observed'] == 23
    assert usage['token_usage_complete'] is True
    assert trace['llm_usage'][0]['done_reason'] == 'stop'
    assert trace['llm_failures'] == []


def test_safe_size_limit_has_distinct_failure_category_and_no_raw_output():
    def handler(_):
        return stream([{'done': False, 'message': {'content': 'SENSITIVE_' + 'a' * 65540}}])
    adapter, telemetry = make_adapter(handler)
    try:
        with pytest.raises(LLMError):
            adapter._ask('system', 'question', NaturalResponse, run_id='size', phase='answer')
    finally:
        adapter.close()
    failure = telemetry.snapshot('size')['llm_failures'][0]
    assert failure['failure_kind'] == 'output_size_limit'
    assert failure['eval_count'] is None
    assert failure['observed_content_bytes'] > 65536
    assert 'SENSITIVE_' not in json.dumps(failure)


def test_comparison_exports_diagnostics_to_json_csv_and_readable_report(tmp_path):
    import csv
    from dap_assistant.evaluation.professional import save_run
    result = {'kind': 'rag_profile_comparison_v1', 'run_id': 'p63mock12345',
        'configuration': {}, 'summary': {},
        'rows': [{'mode': 'agentic', 'question_id': 'AUTO_001', 'success': True,
                  'response_status': 'partial', 'metrics': {},
                  'ollama_diagnostics': {
                      'responses': [{'phase': 'answer', 'done_reason': 'length',
                                     'eval_count': 768, 'requested_num_predict': 768}],
                      'failures': [{'phase': 'answer', 'failure_kind': 'output_token_limit',
                                    'done_reason': 'length', 'eval_count': 768,
                                    'validation_error_type': None,
                                    'requested_num_predict': 768,
                                    'observed_content_chars': 2146}],
                  }}]}
    folder = save_run(result, tmp_path)
    exported = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    with (folder / 'rows.csv').open(encoding='utf-8-sig', newline='') as handle:
        csv_row = next(csv.DictReader(handle))
    report = (folder / 'report.md').read_text(encoding='utf-8')
    assert exported['rows'][0]['ollama_diagnostics']['failures'][0]['eval_count'] == 768
    assert csv_row['ollama_failure_kind'] == 'output_token_limit'
    assert csv_row['ollama_eval_count'] == '768'
    assert 'output_token_limit' in report and '768' in report and 'N/A' in report


def test_unverified_healthcare_condition_has_specific_hungarian_caveat():
    draft = {'claims': [], 'text_integrity': {'incomplete_claims': [],
             'model_warnings': ['healthcare_condition_unverified']},
             'requested_facet_coverage': {'missing': ['healthcare']}, 'disclaimer': ''}
    rendered = _render_simple_answer(draft, [])
    assert 'feltétel nélküli állítást kihagytam' in rendered
    assert 'félbeszakadt állításokat' not in rendered
    assert 'healthcare' not in rendered
