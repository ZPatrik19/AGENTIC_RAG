"""P6.4 offline regressions: only source-backed, relevant claims; no invented gold."""
from __future__ import annotations

from dataclasses import replace
import csv

from dap_assistant.response.filters import low_value_procedural_unit
from dap_assistant.response.quality import (clean_claims, is_complete_statement,
                                          source_fallback_notice)
from dap_assistant.evaluation.comparison import (_audit_simple_draft, _model_generated,
                                                   _simple_generate)
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.evaluation.professional import save_run
from dap_assistant.context_engineering.evidence_selection import assemble_selection_context
from dap_assistant.llm import (LLMError, NaturalResponse, _attach_source_quotes,
                               source_answer)
from dap_assistant.settings import Settings

AUTO = 'Vettem egy használt autót Magyarországon. Milyen ügyintézési teendőim vannak?'
WORK = 'Elvesztettem a munkámat. Milyen ügyintézési teendőim vannak?'
TRANSFER = 'Az adásvételi szerződés megkötésétől számított 15 napon belül át kell íratnod a gépjárművet.'
GENERIC_DEADLINE = ('Ha jogszabály eltérően nem rendelkezik, az e rendelet hatálya alá '
                    'tartozó eljárásokban az ügyintézési határidő tíz nap.')
CARE = 'Amíg kapod az álláskeresési járadékot, megmarad a biztosítási jogviszonyod.'
BOOK = ('"TB kiskönyv"/OEP igazolvány a biztosítási jogviszonyról és '
        'az egészségbiztosítási ellátásokról (a munkáltató adja ki), 2026.')


def evidence(eid: str, text: str, domain: str = 'vehicle') -> dict:
    return {'evidence_id': eid, 'chunk_id': eid[2:], 'domain': domain,
            'document_id': 'fixture', 'role': 'buyer' if domain == 'vehicle' else 'general',
            'text': text, 'title': 'Kizárólag szintetikus tesztadat',
            'source_url': 'https://example.invalid/synthetic', 'section_path': []}


def test_fallback_notice_distinguishes_successful_json_without_grounded_claims():
    assert 'válasza elkészült' in source_fallback_notice('model_output_unusable')
    assert 'nem fejezte be' not in source_fallback_notice('model_output_unusable')
    assert 'nem érkezett érvényes' in source_fallback_notice('model_request_failed')


def test_baseline_fallback_explains_unusable_model_without_fake_timeout():
    class EmptyModel:
        def answer(self, question, evidence, *_a, **_kw):
            draft = source_answer(question, evidence)
            draft.quality_warnings.append('model_no_grounded_complete_claims')
            return draft

    settings = replace(Settings(), llm_provider='ollama', answer_mode='quick')
    output = _simple_generate(settings, {
        'category': 'vehicle', 'reference_date': '2026-09-21',
        'question': AUTO,
    }, [evidence('E_TRANSFER', TRANSFER)], Telemetry(), 'p64-unusable', EmptyModel())
    assert output['answer_fallback']
    assert output['answer_fallback_reason'] == 'model_output_unusable'
    assert 'nem fejezte be' not in output['final_answer']
    assert TRANSFER in output['final_answer']


def test_baseline_fallback_for_technical_model_error_uses_distinct_reason():
    class BrokenModel:
        def answer(self, *_a, **_kw):
            raise LLMError('Ollama invalid response')

    output = _simple_generate(replace(Settings(), llm_provider='ollama'), {
        'category': 'vehicle', 'reference_date': '2026-09-21',
        'question': AUTO,
    }, [evidence('E_TRANSFER', TRANSFER)], Telemetry(), 'p64-error', BrokenModel())
    assert output['answer_fallback_reason'] == 'model_request_failed'
    assert 'nem érkezett érvényes' in output['final_answer']


def test_llm_success_is_independent_of_unusable_answer():
    settings = replace(Settings(), llm_provider='ollama', answer_mode='quick')
    telemetry = Telemetry()
    telemetry.llm_success('p64', 'answer')  # JSON schema parsed, but may lack grounded claims
    assert _model_generated(telemetry.snapshot('p64'), settings) is True
    assert _model_generated({'llm_successes': []}, settings) is False
    assert _model_generated(telemetry.snapshot('p64'), replace(settings, answer_mode='source')) is None


def test_broad_auto_question_rejects_generic_processing_deadline_but_not_transfer_deadline():
    assert low_value_procedural_unit(GENERIC_DEADLINE, AUTO, 'vehicle', 'buyer')
    assert not low_value_procedural_unit(TRANSFER, AUTO, 'vehicle', 'buyer')
    assert not low_value_procedural_unit(GENERIC_DEADLINE,
        'Vettem autót. Mi az ügyintézési határidő a rendelet szerint?', 'vehicle', 'buyer')
    source = source_answer(AUTO, [evidence('E_GENERIC', GENERIC_DEADLINE),
                                  evidence('E_TRANSFER', TRANSFER)])
    assert any(TRANSFER == c.text for c in source.claims)
    assert not any('ügyintézési határidő tíz nap' in c.text for c in source.claims)
    clean, report = clean_claims([{'text': GENERIC_DEADLINE,
        'supporting_quote': GENERIC_DEADLINE, 'category': 'deadline',
        'evidence_ids': ['E_GENERIC']}], question=AUTO, domain='vehicle', role='buyer')
    assert not clean and report['low_value_claims']


def test_source_selected_for_model_is_only_source_allowed_for_attached_quote():
    full = evidence('E_TRANSFER', TRANSFER + '\n' + GENERIC_DEADLINE)
    selected = evidence('E_TRANSFER', TRANSFER)
    answer = NaturalResponse(claims=[{'evidence_id': 'E_TRANSFER',
        'text': GENERIC_DEADLINE, 'category': 'deadline'}])
    # Even without a question-specific background filter the unprovided quote
    # must not be retrieved silently from the rest of the same Qdrant chunk.
    assert not _attach_source_quotes(answer, [selected]).claims
    assert _attach_source_quotes(answer, [full]).claims


def test_work_facet_selection_rejects_tb_book_as_entitlement():
    chunks = [evidence('E_BOOK', BOOK, 'employment'),
              evidence('E_CARE', CARE, 'employment'),
              evidence('E_REG', 'Az álláskeresőként történő nyilvántartásba vételt kérelmezni kell.', 'employment')]
    packed = assemble_selection_context(chunks, WORK, domain='employment', token_budget=700)
    assert 'E_CARE' in packed.facet_coverage.get('healthcare', [])
    assert 'E_BOOK' not in packed.facet_coverage.get('healthcare', [])
    assert not is_complete_statement(BOOK, source_unit=True)


def test_work_support_sentence_can_be_labeled_support_by_both_paths():
    quote = 'Álláskeresőként részt vehetsz támogatott nyelvtanfolyamokon, képzéseken.'
    data = [evidence('E_SUPPORT', quote, 'employment')]
    source = source_answer(WORK, data)
    assert any(c.category == 'support' for c in source.claims)
    generated = NaturalResponse(claims=[{'evidence_id': 'E_SUPPORT',
                                         'text': quote, 'category': 'steps'}])
    attached = _attach_source_quotes(generated, data, WORK)
    assert attached.claims and attached.claims[0].category == 'support'


def test_work_navigation_is_not_independent_next_step_and_specific_question_is_unchanged():
    link = 'Az U1 igazolásról bővebben itt tájékozódhat.'
    assert low_value_procedural_unit(link, WORK, 'employment')
    assert not low_value_procedural_unit(link, 'Hol találom az U1 igazolás tájékoztatóját?', 'employment')
    assert not low_value_procedural_unit(link, 'Elvesztettem a munkámat, az U1 igazolásról milyen teendők vannak?', 'employment')
    assert source_answer(WORK, [evidence('E_LINK', link, 'employment')]).claims == []


def test_baseline_audit_rejects_orphan_year_from_original_report():
    assert not is_complete_statement(BOOK)
    audited, _ = _audit_simple_draft({'claims': [{'text': BOOK,
        'supporting_quote': BOOK, 'category': 'healthcare', 'evidence_ids': ['E_BOOK']}]},
        [evidence('E_BOOK', BOOK, 'employment')], question=WORK, domain='employment')
    assert audited['claims'] == []
    assert audited['text_integrity']['incomplete_claims']


def test_employment_selection_preserves_actual_registration_before_navigation():
    data = [evidence('E_U1', 'Az U1 igazolásról bővebben itt tájékozódhat.', 'employment'),
            evidence('E_BOOK', BOOK, 'employment'),
            evidence('E_REGISTER', 'Az álláskeresőként történő nyilvántartásba vételt kérelmezni kell.', 'employment'),
            evidence('E_CARE', CARE, 'employment')]
    packed = assemble_selection_context(data, WORK, domain='employment', token_budget=574)
    ids = [item['evidence_id'] for item in packed.evidence]
    assert 'E_REGISTER' in ids
    assert 'E_CARE' in ids
    assert 'E_U1' not in ids and 'E_BOOK' not in ids
    assert 'E_REGISTER' in packed.facet_coverage.get('steps', [])


def test_report_exports_success_and_fallback_reason_as_distinct_fields(tmp_path):
    result = {'kind': 'rag_profile_comparison_v1', 'run_id': 'p64-test',
              'configuration': {'model': 'qwen3:4b'}, 'summary': {},
              'rows': [{'mode': 'hybrid_rrf', 'question_id': 'AUTO_001',
                        'response_status': 'partial', 'generation_success': True,
                        'model_answer_usable': False,
                        'answer_fallback_reason': 'model_output_unusable',
                        'ollama_diagnostics': {'responses': [{'phase': 'answer',
                            'done_reason': 'stop', 'eval_count': 276}], 'failures': []},
                        'metrics': {}}]}
    folder = save_run(result, tmp_path)
    with (folder / 'rows.csv').open(encoding='utf-8-sig', newline='') as handle:
        row = next(csv.DictReader(handle))
    assert row['generation_success'] == 'True'
    assert row['model_answer_usable'] == 'False'
    assert row['answer_fallback_reason'] == 'model_output_unusable'
    markdown = (folder / 'report.md').read_text(encoding='utf-8')
    assert 'Qwen JSON kész' in markdown
    assert 'model_output_unusable' in markdown
