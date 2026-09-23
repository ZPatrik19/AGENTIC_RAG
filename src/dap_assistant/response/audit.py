"""Audit model claims and assemble a cited answer from verified source units.

No model calls or UI dependencies are performed here.
"""
from __future__ import annotations

import re

from ..orchestration.state import AssistantState
from ..context_engineering.information_needs import answer_facets, is_location_only_question
from .quality import (clean_claims, render_claim_groups, structural_facet_coverage,
                             missing_facets_hungarian, source_fallback_notice,
                             claim_provenance_report)
from ..context_engineering.evidence_selection import requested_facets
from ..prompt_engineering.answer_prompt import answer_plan
from .filters import relevant_unit
from ..tooling.result_integration import integrate_checklist


def audit_answer(state: AssistantState) -> dict:
    by_id = {e['evidence_id']: e for e in state.get('evidence', [])}
    validated = []
    rejected = []
    domain = (state.get('domains') or [''])[0]
    plan = state.get('answer_context', {}).get('response_plan') or answer_plan(
        state.get('user_question', ''), domain, state.get('role', ''),
        state.get('stage', ''), state.get('evidence', []))
    on_topic = []
    excluded = []
    for claim in state['answer_draft'].get('claims', []):
        if (relevant_unit(claim.get('text', ''), state.get('user_question', ''),
                          domain, state.get('role', ''), plan['stage'])
                and relevant_unit(claim.get('supporting_quote', ''),
                                  state.get('user_question', ''), domain,
                                  state.get('role', ''), plan['stage'])):
            on_topic.append(claim)
        else:
            excluded.append(claim.get('text', '')[:120])
    cleaned, integrity = clean_claims(
        on_topic, question=state.get('user_question', ''),
        domain=domain, role=state.get('role', ''))
    integrity['excluded_out_of_situation'] = excluded
    # The pre-generation context plan and post-generation claim audit are
    # deliberately separate; lexical coverage never upgrades to legal truth.
    integrity['context_engineering'] = state.get('context_engineering', {})
    integrity['model_warnings'] = (state['answer_draft'].get('quality_warnings', [])
                                   + (['healthcare_condition_unverified'] if integrity['healthcare_condition_issues'] else []))
    for claim in cleaned:
        ids = claim.get('evidence_ids', [])
        quote = ' '.join(claim.get('supporting_quote', '').split()).casefold()
        linked_text = ' '.join(' '.join(by_id[i]['text'].split()).casefold() for i in ids if i in by_id)
        claim_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', claim.get('text', ''))
        quote_numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', quote)
        # Quoted provenance + numeric consistency are structural guardrails, not legal entailment.
        supported = (ids and all(i in by_id for i in ids) and quote and quote in linked_text
                     and all(number in quote_numbers for number in claim_numbers)
                     and claim.get('text', '').strip())
        if supported:
            validated.append(claim)
        else:
            rejected.append(claim.get('text', '')[:120])
    # Check tool-result UTILIZATION, not only execution. Only full, verbatim
    # named-document source units may become separately labelled tool extracts.
    # This deliberately does NOT certify the source's semantic/legal meaning.
    original_claim_count = len(validated)
    tool_additions, tool_utilization = integrate_checklist(
        question=state.get('user_question', ''),
        domain=domain, role=state.get('role', ''), stage=plan['stage'],
        evidence=state.get('evidence', []), existing_claims=validated,
        tool_results=state.get('tool_results', {}),
        native_tool_results=state.get('native_tool_results', {}),
        native_tool_trace=state.get('native_tool_trace', {}))
    validated.extend(tool_additions)
    # A deterministic calculation can remain useful even if the local Qwen
    # could not produce a usable prose claim. Do not discard a source-checked
    # NAV rate or a purely mathematical loan estimate with the empty draft.
    vehicle_duty = state.get('tool_results', {}).get('vehicle_duty')
    loan_result = state.get('tool_results', {}).get('illustrative_loan')
    benefit_result = state.get('tool_results', {}).get('unemployment_benefit')
    # A tool can return a documented missing-input result without a
    # generated claim. Preserve the explanation and official link.
    calculation_available = bool(vehicle_duty or loan_result or benefit_result)
    if not validated and not calculation_available:
        return {'answer_draft': {**state['answer_draft'], 'claims': []},
                'answer_validation': {'status': 'failed', 'unsupported_claims': rejected,
                                      'text_integrity': integrity,
                                      'tool_utilization': tool_utilization,
                                      'semantic_support': {'status': 'not_evaluated',
                                                           'reason': 'no_approved_human_references'}},
                'response_status': 'partial', 'final_answer': ''}
    coverage = answer_facets(state.get('user_question', ''), validated)
    facet_coverage = structural_facet_coverage(
        requested_facets(state.get('user_question', ''),
                         (state.get('domains') or ['vehicle'])[0],
                         role=state.get('role', '')), validated,
        domain=(state.get('domains') or [''])[0])
    facet_coverage['missing_in_retrieved_context'] = list(
        state.get('context_engineering', {}).get('missing_retrieved_facets', []))
    facet_coverage['generation_diagnostics'] = state.get('answer_context', {}).get(
        'generation_diagnostics', {})
    facet_coverage['lexically_available_but_unanswered'] = [
        facet for facet in facet_coverage['missing']
        if facet in state.get('context_engineering', {}).get('lexically_covered_facets_after', [])]
    if ('costs' in coverage['requested'] and 'costs' not in coverage['covered']
            and ((vehicle_duty and vehicle_duty.get('status') == 'calculated')
                 or (loan_result and loan_result.get('status') == 'calculated'))):
        coverage['covered'].append('costs')
        coverage['missing'].remove('costs')
        coverage['coverage'] = len(coverage['covered']) / len(coverage['requested'])
        coverage['measurement'] = 'structural_coverage_proxy_including_validated_calculation'
    where_question = is_location_only_question(state.get('user_question', ''))
    lines = [
        ('**Hol intézheted a korábban említett ügyet?**' if where_question else plan['title']),
        '',
    ]
    headings = {'steps': 'Teendők', 'deadline': 'Határidők', 'cost': 'A forrásban szereplő díj',
                'where': 'Hol és hogyan intézhető?',
                'documents': 'Dokumentumok és adatok', 'insurance': 'Biztosítás',
                'benefit_amount': 'Álláskeresési járadék összege',
                'support': 'Elérhető támogatások és szolgáltatások',
                'eligibility': 'Jogosultsági feltételek',
                'healthcare': 'Egészségügyi jogosultság'}
    ordered_claims = sorted(validated, key=lambda c: (
        plan['category_order'].index(c.get('category', 'steps'))
        if c.get('category', 'steps') in plan['category_order']
        else len(plan['category_order'])))
    lines.extend(render_claim_groups(
        ordered_claims, by_id, headings,
        show_provenance=any(c.get('origin') in ('model_generated', 'source_supplement', 'tool_extract')
                            for c in validated)))
    if state.get('answer_fallback'):
        lines.insert(0, source_fallback_notice(state.get('answer_fallback_reason', 'model_request_failed')) + '\n')
    if vehicle_duty is not None:
        lines.extend(['', '**Gépjármű-vagyonszerzési illeték (számított tájékoztató érték)**', ''])
        if vehicle_duty.get('status') == 'calculated':
            refs = ' '.join(
                f"[NAV-forrás {eid}](<{by_id[eid]['source_url']}>)"
                for eid in vehicle_duty['source_evidence_ids'] if eid in by_id
            )
            lines.append(
                f"{vehicle_duty['registered_kw']} kW × {vehicle_duty['rate_huf_per_kw']:,} Ft/kW = "
                f"**{vehicle_duty['amount_huf']:,} Ft** (2026-os tábla, a megadott gyártási év alapján). {refs}"
                .replace(',', ' ')
            )
            lines.append('A mentességek és a tényleges hatósági fizetési kötelezettség nem kerültek megállapításra.')
        else:
            lines.append(vehicle_duty.get('reason', 'Nem számítható igazolt összeg.'))
            if vehicle_duty.get('approximate_kw_from_hp'):
                lines.append('A lóerőből becsült kW nem helyettesíti a forgalmiban szereplő hivatalos kW-adatot.')
    if benefit_result is not None:
        lines.extend(['', '**Személyre szabott járadék-kalkuláció**', ''])
        if benefit_result.get('source_evidence_ids'):
            refs = ' '.join(
                f"[Forrás {eid}](<{by_id[eid]['source_url']}>)"
                for eid in benefit_result['source_evidence_ids'] if eid in by_id
            )
            lines.append(benefit_result.get('reason', '') + (' ' + refs if refs else ''))
        else:
            lines.append(benefit_result.get('reason', ''))
        lines.append('[Hivatalos NFSZ járadékkalkulátor](https://nfsz.munka.hu/tart/jaradek_kalkulator)')
    if loan_result and loan_result.get('status') == 'calculated':
        lines.extend(['', '**Tájékoztató hiteltörlesztés – nem banki ajánlat**', '',
                      f"Havi törlesztő: **{loan_result['monthly_payment_huf']:,} Ft**. "
                      f"Összes visszafizetés: {loan_result['total_repayment_huf']:,} Ft. "
                      f"Kamatköltség: {loan_result['total_interest_huf']:,} Ft.".replace(',', ' '),
                      'A számítás kizárólag az általad megadott kamattal és futamidővel készült; '
                      'THM, banki díj és támogatási feltétel nélkül.'])
    if 'costs' in coverage['missing'] and not ((vehicle_duty and vehicle_duty.get('status') == 'calculated')
                                               or (loan_result and loan_result.get('status') == 'calculated')):
        lines.extend(['', '**Költségek**', '',
                      'A kért költségekre nem találtam kellően alátámasztott összeget a '
                      'visszakeresett forrásokban; nem adok meg találgatáson alapuló díjat.'])
    elif 'costs' in coverage['requested']:
        lines.extend(['', 'A felsorolt díjtételek nem jelentenek teljes vételi '
                      'vagy ügyintézési végösszeget.'])
    note = '\n\n' + state['answer_draft'].get('disclaimer', '')
    if facet_coverage['missing']:
        note += ('\n\nNem igazolt témakörök: '
                 + missing_facets_hungarian(facet_coverage['missing'])
                 + '. A lefedettség állítás–forrásidézet alapú szöveges becslés, '
                 'nem teljes körű ügyintézési vagy jogi ellenőrzés.')
    if integrity['incomplete_claims'] or 'model_incomplete_statement' in integrity['model_warnings']:
        note += '\n\nEgyes félbeszakadt állításokat kihagytam; a hiányzó részleteket az eredeti forrásnál ellenőrizd.'
    if integrity['low_value_claims'] or 'model_low_value_statement' in integrity['model_warnings']:
        note += '\n\nA kérdéshez nem közvetlenül kapcsolódó háttérmondatokat kihagytam.'
    if 'healthcare_condition_unverified' in integrity['model_warnings']:
        note += ('\n\nAz egészségügyi jogosultság feltételeit az adott forrásrészlet nem igazolta '
                 'egyértelműen; a feltétel nélküli állítást kihagytam.')
    if rejected or state.get('validation', {}).get('status') != 'passed':
        note += '\n\nEgyes teendők ellenőrzéséhez nem áll rendelkezésre elegendő bizonyíték.'
    incomplete = bool(integrity['incomplete_claims'] or integrity['model_warnings'])
    is_partial = bool(state.get('answer_fallback') or rejected or incomplete or
                      (not original_claim_count and tool_additions) or facet_coverage['missing']
                      or coverage['missing'] or not validated)
    return {'answer_draft': {**state['answer_draft'], 'claims': validated},
            'answer_validation': {'status': 'partial' if is_partial else 'passed',
                                  'unsupported_claims': rejected, 'request_coverage': coverage,
                                  'requested_facet_coverage': facet_coverage,
                                  'text_integrity': integrity,
                                  'response_plan': plan,
                                  'tool_utilization': tool_utilization,
                                  'semantic_support': {'status': 'not_evaluated',
                                                       'reason': 'no_approved_human_references'},
                                  'completion_basis': 'structural_proxy_not_human_verified',
                                  'claim_provenance': claim_provenance_report(
                                      validated, state['answer_draft'].get('model_context_evidence_ids', []))},
            'final_answer': '\n'.join(lines) + note,
            'response_status': 'partial' if is_partial
            or state.get('validation', {}).get('status') != 'passed' else 'complete'}
