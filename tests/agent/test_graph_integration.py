"""Genuine offline LangGraph test; skips clearly when the optional package is absent.

This uses a synthetic local corpus and dummy provider, never Ollama/web/GPU.
"""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

pytest.importorskip('langgraph', reason='Install the project dependencies for real graph integration')

from dap_assistant.orchestration.runtime import create_chat_workflow
from dap_assistant.settings import Settings
from dap_assistant.workflow import initial_state


def test_runtime_to_real_graph_two_independent_rag_branches(tmp_path):
    for domain, content in (
        ('vehicle', 'Autóeladás után a tulajdonosváltást be kell jelenteni; a dokumentum az adásvételi szerződés.'),
        ('employment', 'Álláskeresőként regisztrálni lehet; szükséges dokumentum a foglalkoztatási igazolás.'),
    ):
        chunk = {
            'chunk_id': f'{domain}-fixture', 'document_id': f'{domain}-fixture',
            'document_version': 'synthetic-v1', 'domain': domain, 'role': 'general',
            'text': content, 'title': domain, 'section_path': [], 'page_number': None,
            'source_url': f'https://dap.gov.hu/teszt/{domain}',
        }
        location = tmp_path / 'processed' / domain / 'fixture.json'
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(json.dumps({'document_id': chunk['document_id'], 'chunks': [chunk]}), encoding='utf-8')

    config = replace(Settings(), data_dir=tmp_path, llm_provider='dummy',
                     embedding_provider='dummy', max_subtasks=6)
    graph, telemetry = create_chat_workflow(
        provider='dummy', fast=True, answer_mode='quick', data_dir=str(tmp_path),
        embedding_model=config.embedding_model, embedding_provider='dummy',
        read_timeout_s=120, total_timeout_s=240, base_settings=config,
    )
    initial = initial_state('Eladtam az autómat és elvesztettem a munkámat. Milyen dokumentumok kellenek?')
    initial['run_id'] = 'v9-offline-integration'
    result = graph.invoke(initial, config={'configurable': {'thread_id': 'v9-integration'},
                                           'recursion_limit': 50})
    assert set(result['domains']) == {'vehicle', 'employment'}
    assert len(result['subtasks']) == 2
    assert set(result['branch_results']) == set(result['subtasks'])
    assert result['final_answer']
    assert result['response_status'] in ('partial', 'complete')
    spans = telemetry.snapshot('v9-offline-integration')['spans']
    assert any(span['name'] == 'main/classify_intent' for span in spans)
    assert not telemetry.snapshot('v9-offline-integration')['llm_attempts']
