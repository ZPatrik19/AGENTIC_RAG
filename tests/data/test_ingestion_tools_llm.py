from datetime import date
import hashlib
import json

import httpx
import pytest

from dap_assistant.documents.ingestion import ingest_one, load_chunks, parse_html
from dap_assistant.llm import DummyAdapter, LLMError, OllamaAdapter
from dap_assistant.settings import ROOT, Settings
from dap_assistant.documents.sources import Source
from dap_assistant.tooling.tools import DeadlineRule, calculate_deadline, build_document_checklist
from dap_assistant.rag.retrieval import bm25_search


def source():
    return Source(id='dap-test-auto', title='Teszt auto', url='https://dap.gov.hu/example',
                  domain='vehicle', type='html', destination='vehicle')


def test_html_preserves_headings_and_removes_navigation():
    sections = parse_html((ROOT / 'tests' / 'fixtures' / 'vehicle.html').read_bytes())
    assert 'Átírás' in sections[0]['section']
    assert all('Menü nem kerülhet' not in s['text'] for s in sections)
    assert any('Forgalmi engedély' in s['text'] for s in sections)


def test_ingestion_replace_and_stable_chunk_ids(tmp_path):
    content = (ROOT / 'tests' / 'fixtures' / 'vehicle.html').read_bytes()
    meta = {'file': 'raw/vehicle/dap-test-auto.html', 'sha256': hashlib.sha256(content).hexdigest(),
            'retrieved_at': '2026-09-20T00:00:00+00:00'}
    raw = tmp_path / meta['file']
    raw.parent.mkdir(parents=True)
    raw.write_bytes(content)
    file = tmp_path / 'interim' / 'downloads' / 'dap-test-auto.json'
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps(meta))
    first = ingest_one(source(), tmp_path)
    ids = [c['chunk_id'] for c in load_chunks(tmp_path)]
    second = ingest_one(source(), tmp_path)
    assert first == second
    assert ids == [c['chunk_id'] for c in load_chunks(tmp_path)]
    assert len(list((tmp_path / 'processed').glob('*/*.json'))) == 1


def test_bm25_respects_domain(tmp_path):
    chunks = [{'chunk_id': 'a', 'domain': 'vehicle', 'text': 'Gépjármű átírás adásvételi szerződés'},
              {'chunk_id': 'b', 'domain': 'housing', 'text': 'Lakás átírás adásvételi szerződés'}]
    assert bm25_search('gépjármű átírás', chunks, 'vehicle')[0][0] == 'a'
    assert not bm25_search('gépjármű átírás', chunks, 'business')


def test_deadline_fails_closed_and_calculates_only_when_verified():
    rule = DeadlineRule(rule_id='r1', evidence_id='e1', anchor_event='signed_contract',
                        days=3, unit='calendar_day')
    assert calculate_deadline(date(2026, 9, 20), rule, {'e1'})['status'] == 'incomplete'
    verified = rule.model_copy(update={'include_start': False, 'weekend_policy': 'no_extension'})
    assert calculate_deadline(date(2026, 9, 20), verified, {'e1'})['date'] == '2026-09-23'
    assert calculate_deadline(date(2026, 9, 20), verified, set())['status'] == 'incomplete'


def test_checklist_only_from_evidence():
    result = build_document_checklist([{'evidence_id': 'e1', 'text': 'Foglalkoztatási igazolás\nAdásvételi szerződés'}])
    assert len(result['items']) == 2
    assert all(item['evidence_ids'] == ['e1'] for item in result['items'])


def test_dummy_deterministic_multidomain():
    dummy = DummyAdapter()
    question = 'Eladtam az autómat és megszűnt a munkaviszonyom.'
    assert dummy.classify(question) == dummy.classify(question)
    plan = dummy.plan(question, dummy.classify(question).domains)
    assert len(plan.tasks) == 2
    assert all(not task.depends_on for task in plan.tasks)


def test_ollama_schema_think_false_and_error():
    seen = []
    def handler(request):
        data = json.loads(request.content)
        seen.append(data)
        return httpx.Response(200, request=request, json={'message': {'content': json.dumps({'domains': ['vehicle']})}})
    transport = httpx.MockTransport(handler)
    model = OllamaAdapter(Settings(), transport=transport)
    assert model.classify('autó').domains == ['vehicle']
    assert seen[0]['think'] is False and seen[0]['stream'] is False
    assert 'properties' in seen[0]['format']
    def error(request):
        return httpx.Response(503, request=request)
    with pytest.raises(LLMError):
        OllamaAdapter(Settings(), transport=httpx.MockTransport(error)).classify('autó')


def test_pdf_parser_retains_page_number():
    import fitz
    from dap_assistant.documents.ingestion import parse_pdf
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((60, 60), 'Hivatalos dokumentumok ellenorzese.')
    extracted = parse_pdf(pdf.tobytes())
    pdf.close()
    assert extracted[0]['page'] == 1
    assert 'dokumentumok' in extracted[0]['text']


def test_buyer_seller_role_filter():
    docs = [{'chunk_id': 'b', 'domain': 'vehicle', 'role': 'buyer', 'text': 'Gépjármű átírás'},
            {'chunk_id': 's', 'domain': 'vehicle', 'role': 'seller', 'text': 'Gépjármű átírás'}]
    assert [item[0] for item in bm25_search('gépjármű átírás', docs, 'vehicle', role='buyer')] == ['b']
