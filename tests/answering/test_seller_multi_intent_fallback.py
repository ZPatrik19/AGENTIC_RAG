"""Offline regressions for mixed seller procedure/location/deadline questions."""
from __future__ import annotations

import json
from pathlib import Path

from dap_assistant.response.audit import audit_answer
from dap_assistant.context_engineering.information_needs import is_location_only_question
from dap_assistant.response.fallback import source_answer


QUESTION = 'Eladtam az autómat. Mit kell bejelentenem, hol és meddig?'
LOCATION_QUESTION = 'Hol tudom bejelenteni az autó eladását?'


def evidence(eid: str, text: str) -> dict:
    return {
        'evidence_id': eid,
        'chunk_id': eid,
        'document_id': 'dap-vehicle-seller',
        'domain': 'vehicle',
        'role': 'seller',
        'section_path': ['Tulajdonosváltás bejelentése'],
        'source_url': 'https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-adok-el',
        'text': text,
    }


def test_combined_question_includes_source_deadline_even_if_model_selected_only_where():
    items = [
        evidence('E_DEADLINE', 'Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást.'),
        evidence('E_WHERE', 'A bejelentést online, a Webes Ügysegéden vagy személyesen, bármelyik kormányablakban megteheted.'),
        evidence('E_DOCUMENT', 'Szkenneld be vagy fényképezd le az aláírt adásvételi szerződést.'),
    ]
    draft = source_answer(QUESTION, items, selected_ids=['E_WHERE'], role='seller', stage='after_event')
    assert is_location_only_question(QUESTION) is False
    assert any(c.category == 'deadline' and '15 napon belül' in c.text for c in draft.claims)
    assert any(c.category == 'where' and 'Webes Ügysegéd' in c.text for c in draft.claims)
    state = {'user_question': QUESTION, 'domains': ['vehicle'], 'role': 'seller',
             'stage': 'after_event', 'evidence': items, 'answer_draft': draft.model_dump(),
             'answer_fallback': True, 'answer_fallback_reason': 'model_request_failed'}
    audited = audit_answer(state)
    assert 'Autóeladás után: intézendő ügyek' in audited['final_answer']
    assert 'Hol intézheted a korábban említett ügyet?' not in audited['final_answer']
    assert '15 napon belül' in audited['final_answer']
    assert 'Nem igazolt témakörök: határidők' not in audited['final_answer']


def test_location_only_remains_focused_and_does_not_add_deadlines():
    items = [
        evidence('E_DEADLINE', 'Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást.'),
        evidence('E_WHERE', 'A bejelentést online, a Webes Ügysegéden vagy személyesen, bármelyik kormányablakban megteheted.'),
    ]
    assert is_location_only_question(LOCATION_QUESTION) is True
    draft = source_answer(LOCATION_QUESTION, items, role='seller', stage='after_event')
    assert draft.claims
    assert all(c.category == 'where' for c in draft.claims)


def test_no_source_deadline_does_not_fabricate_one():
    items = [evidence('E_WHERE', 'A bejelentést online, a Webes Ügysegéden vagy személyesen, bármelyik kormányablakban megteheted.')]
    draft = source_answer(QUESTION, items, role='seller', stage='after_event')
    assert not any(c.category == 'deadline' for c in draft.claims)
    audited = audit_answer({'user_question': QUESTION, 'domains': ['vehicle'], 'role': 'seller',
                            'stage': 'after_event', 'evidence': items,
                            'answer_draft': draft.model_dump()})
    assert 'Nem igazolt témakörök:' in audited['final_answer']
    assert 'határidők' in audited['final_answer']


def test_real_local_processed_seller_corpus_still_contains_deadline():
    root = Path(__file__).resolve().parents[2]
    path = root / 'data/processed/vehicle/dap-vehicle-seller.json'
    if not path.exists():
        return  # data fixtures are optional for a source-only update ZIP
    chunks = json.loads(path.read_text(encoding='utf-8'))['chunks']
    items = [{**chunk, 'evidence_id': f'E_{index}'} for index, chunk in enumerate(chunks)]
    draft = source_answer(QUESTION, items, role='seller', stage='after_event')
    assert any('15 napon belül' in c.text for c in draft.claims if c.category == 'deadline')
