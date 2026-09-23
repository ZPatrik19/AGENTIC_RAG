"""Offline LangGraph integration checks for the two supported life events.

These tests require LangGraph but never require Ollama or live web data.
"""
from dataclasses import replace
import json

import pytest

pytest.importorskip('langgraph')

from dap_assistant.settings import Settings
from dap_assistant.workflow import build_workflow, initial_state


def _app_with_local_chunks(tmp_path, domains=('vehicle', 'employment')):
    """An isolated, deterministic corpus with no dependency on downloaded pages."""
    fixtures = {
        'vehicle': ('vehicle-test',
                    'Használt autó adásvételi szerződés, vevői átírás, kötelező '
                    'gépjármű-felelősségbiztosítás és a szükséges dokumentumok.'),
        'employment': ('employment-test',
                       'Munkaviszony megszűnése után foglalkoztatási igazolás, '
                       'álláskeresési járadék és szükséges dokumentumok.'),
    }
    for domain in domains:
        name, text = fixtures[domain]
        path = tmp_path / 'processed' / domain / f'{name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        item = {'chunk_id': name, 'document_id': name, 'domain': domain, 'role': 'general',
                'text': text, 'title': name,
                'source_url': f'https://dap.gov.hu/eletesemenyek/{name}',
                'retrieved_at': '2026-09-20T10:00:00Z', 'document_version': 'test',
                'section_path': [], 'page_number': None}
        path.write_text(json.dumps({'document_id': name, 'chunks': [item]}), encoding='utf-8')
    settings = replace(Settings(), data_dir=tmp_path,
                       llm_provider='dummy', embedding_provider='dummy')
    return build_workflow(settings)


def test_supported_vehicle_and_employment_in_one_workflow(tmp_path):
    app = _app_with_local_chunks(tmp_path)
    result = app.invoke(initial_state(
        'Eladtam az autómat, és megszűnt a munkaviszonyom. Milyen dokumentumok kellenek?'),
        config={'configurable': {'thread_id': 'graph-both-supported'}, 'recursion_limit': 30})
    assert set(result['domains']) == {'vehicle', 'employment'}
    assert result['classification_status'] == 'supported'
    assert {task['domain'] for task in result['subtasks'].values()} == {'vehicle', 'employment'}
    assert set(result['branch_results']) == set(result['subtasks'])
    assert result['final_answer']
    if result.get('evidence'):
        assert result['tool_results']['checklist']['status'] == 'success'
        assert '[Forrás' in result['final_answer']
    else:
        # A tiny synthetic corpus is deliberately NOT a complete legal source.
        assert result['response_status'] == 'partial'


@pytest.mark.parametrize('question,expected_domain,expects_document_checklist', [
    ('Tegnap vettem egy használt autót. Mi a teendőm?', 'vehicle', True),
    ('Eladtam a gépjárművemet. Mit jelentsek be?', 'vehicle', False),
    ('Elvesztettem a munkámat, milyen dokumentumok kellenek?', 'employment', True),
])
def test_supported_single_domain_scenarios(tmp_path, question, expected_domain, expects_document_checklist):
    app = _app_with_local_chunks(tmp_path, (expected_domain,))
    result = app.invoke(initial_state(question),
                        config={'configurable': {'thread_id': f'graph-{expected_domain}-{question[:8]}',
                                                'recursion_limit': 30}})
    assert result['domains'] == [expected_domain]
    assert result['classification_status'] == 'supported'
    assert result['final_answer']
    assert result['branch_results']  # Actual LangGraph worker/subgraph was entered.
    if result.get('evidence'):
        if expects_document_checklist:
            assert result['tool_results']['checklist']['status'] == 'success'
        else:
            # Reporting a vehicle sale is not a request for a document
            # checklist. Do not invent a successful tool call in the graph.
            assert 'checklist' not in result.get('tool_results', {})
    else:
        assert result['response_status'] == 'partial'


@pytest.mark.parametrize('question', [
    'Megvettem a lakást és kaptam kulcsot. Milyen teendőim vannak?',
    'Egyéni vállalkozást szeretnék indítani.',
])
def test_removed_domains_are_out_of_scope_without_retrieval(tmp_path, question):
    app = _app_with_local_chunks(tmp_path)
    result = app.invoke(initial_state(question),
                        config={'configurable': {'thread_id': f'out-of-scope-{question[:8]}',
                                                'recursion_limit': 30}})
    assert result['classification_status'] == 'unsupported'
    assert result['response_status'] == 'unsupported'
    assert result['domains'] == []
    assert result['final_answer']
    assert 'autó' in result['final_answer'].casefold()
    assert not result.get('subtasks')
    assert not result.get('branch_results')
    assert not result.get('evidence')
    assert not result.get('tool_results')
