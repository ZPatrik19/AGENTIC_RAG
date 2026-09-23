
import httpx
import pytest

from dap_assistant.documents.download import DownloadError, download_one
from dap_assistant.documents.sources import Source, load_manifest
from dap_assistant.settings import ROOT


def test_manifest_has_four_domains_and_pdf():
    manifest = load_manifest(ROOT / 'config' / 'document_sources.yaml')
    assert {s.domain for s in manifest.sources} == {'vehicle', 'employment'}
    assert any(s.type == 'pdf' for s in manifest.sources)
    assert len(manifest.sources) == len({s.id for s in manifest.sources})


def test_disallowed_host_rejected():
    with pytest.raises(ValueError):
        Source(id='malicious-doc', title='Malicious', url='https://example.com/attack', domain='vehicle',
               type='html', destination='vehicle')


def sample_source():
    return Source(id='dap-test-auto', title='Teszt auto', url='https://dap.gov.hu/example',
                  domain='vehicle', type='html', destination='vehicle')


def test_download_idempotency_conditional_get_and_hash(tmp_path):
    calls = []
    html = (ROOT / 'tests' / 'fixtures' / 'vehicle.html').read_bytes()

    def handler(request):
        calls.append(request)
        if request.headers.get('If-None-Match') == 'v1':
            return httpx.Response(304, request=request)
        return httpx.Response(200, request=request, content=html,
                              headers={'content-type': 'text/html', 'ETag': 'v1'})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = download_one(client, sample_source(), tmp_path)
        second = download_one(client, sample_source(), tmp_path)
    assert first['status'] == 'updated'
    assert second['status'] == 'unchanged'
    assert calls[1].headers['If-None-Match'] == 'v1'
    assert len(list((tmp_path / 'raw').rglob('*.html'))) == 1


def test_existing_document_survives_error(tmp_path):
    html = (ROOT / 'tests' / 'fixtures' / 'vehicle.html').read_bytes()
    bad = False
    def handler(request):
        if bad:
            return httpx.Response(503, request=request)
        return httpx.Response(200, request=request, content=html, headers={'content-type': 'text/html'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download_one(client, sample_source(), tmp_path)
        bad = True
        with pytest.raises(DownloadError):
            download_one(client, sample_source(), tmp_path, max_attempts=1)
    assert (tmp_path / 'raw' / 'vehicle' / 'dap-test-auto.html').read_bytes() == html


def test_pdf_html_error_response_rejected(tmp_path):
    source = Source(id='nav-pdf-test', title='NAV pdf teszt', url='https://nav.gov.hu/pfile/file?path=test',
                    domain='employment', type='pdf', destination='employment')
    def handler(request):
        return httpx.Response(200, request=request, content=b'<html>Not a PDF document!!!</html>',
                              headers={'content-type': 'text/html'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadError, match='PDF'):
            download_one(client, source, tmp_path)


def test_redirect_to_non_official_host_blocked_before_fetch(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, request=request, headers={'Location': 'https://example.com/private'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadError, match='Unsafe redirect'):
            download_one(client, sample_source(), tmp_path, max_attempts=1)
    assert calls == ['https://dap.gov.hu/example']


def test_duplicate_manifest_ids_rejected(tmp_path):
    source = {'id': 'same-id', 'title': 'Example document', 'url': 'https://dap.gov.hu/ok',
              'domain': 'vehicle', 'type': 'html', 'destination': 'vehicle'}
    path = tmp_path / 'manifest.yaml'
    import yaml
    path.write_text(yaml.safe_dump({'sources': [source, source]}))
    with pytest.raises(ValueError, match='Duplicate'):
        load_manifest(path)
