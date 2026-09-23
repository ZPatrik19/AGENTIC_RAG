"""Offline regressions for the two public answer strategies and official downloads."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dap_assistant.documents.download import DownloadError, download_one
from dap_assistant.settings import _configured_answer_mode
from dap_assistant.documents.sources import Source, load_manifest


def legal_source(url: str = 'https://njt.hu/jogszabaly/2009-304-20-22') -> Source:
    return Source(id='legal-test', title='Official statute', url=url,
                  domain='vehicle', type='html', destination='vehicle')


def test_robots_redirect_to_official_canonical_domain(tmp_path):
    calls: list[str] = []
    html = b'<html><main><h1>Legal document</h1><p>Official document about vehicle ownership.</p></main></html>'

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if url == 'https://njt.hu/robots.txt':
            return httpx.Response(301, headers={'Location': 'https://njt.jog.gov.hu/robots.txt'})
        if url == 'https://njt.jog.gov.hu/robots.txt':
            return httpx.Response(200, text='User-agent: *\nAllow: /jogszabaly/*\n')
        if url == 'https://njt.hu/jogszabaly/2009-304-20-22':
            return httpx.Response(301, headers={'Location': 'https://njt.jog.gov.hu/jogszabaly/2009-304-20-22'})
        if url == 'https://njt.jog.gov.hu/jogszabaly/2009-304-20-22':
            return httpx.Response(200, content=html, headers={'content-type': 'text/html'})
        raise AssertionError(f'Unexpected URL {url}')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = download_one(client, legal_source(), tmp_path, robots_cache={})
    assert result['status'] == 'updated'
    assert calls == [
        'https://njt.hu/robots.txt',
        'https://njt.jog.gov.hu/robots.txt',
        'https://njt.hu/jogszabaly/2009-304-20-22',
        'https://njt.jog.gov.hu/robots.txt',
        'https://njt.jog.gov.hu/jogszabaly/2009-304-20-22',
    ]


def test_robots_redirect_unapproved_host_not_requested(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(301, headers={'Location': 'https://example.com/robots.txt'})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadError, match='Unsafe robots.txt redirect'):
            download_one(client, legal_source(), tmp_path, robots_cache={}, max_attempts=1)
    assert calls == ['https://njt.hu/robots.txt']


def test_robots_policy_denial_blocks_document_fetch(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text='User-agent: *\nDisallow: /jogszabaly/\n')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadError, match='robots.txt disallows'):
            download_one(client, legal_source(), tmp_path, robots_cache={}, max_attempts=1)
    assert calls == ['https://njt.hu/robots.txt']


def test_certificate_error_never_disables_tls_verification(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ConnectError('[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DownloadError, match='TLS verification remains enabled'):
            download_one(client, legal_source(), tmp_path, robots_cache={}, max_attempts=1)
    assert len(requests) == 1
    assert requests[0].url.scheme == 'https'


def test_js_only_portals_are_manifested_but_not_fetched(tmp_path):
    manifest = load_manifest(Path(__file__).resolve().parents[2] / 'config/document_sources.yaml')
    for id_ in ('magyarorszag-jszp', 'magyarorszag-electronic-vehicle-contract'):
        source = next(s for s in manifest.sources if s.id == id_)
        assert source.processing_status == 'disabled'
        with httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail('No network expected'))) as client:
            assert download_one(client, source, tmp_path, robots_cache={})['status'] == 'disabled'


def test_only_two_public_answer_strategies(monkeypatch):
    # The UI and environment expose only the two supported public strategies.
    ui_source = (Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant' / 'ui.py').read_text(encoding='utf-8')
    select_start = ui_source.index('answer_labels = {')
    select_end = ui_source.index('answer_mode = st.selectbox', select_start)
    options = ui_source[select_start:select_end]
    assert "'quick':" in options and "'detailed':" in options
    assert "'source':" not in options
    monkeypatch.setenv('ANSWER_MODE', 'source')
    with pytest.raises(ValueError, match='ANSWER_MODE'):
        _configured_answer_mode()
    monkeypatch.setenv('ANSWER_MODE', 'detailed')
    assert _configured_answer_mode() == 'detailed'


def test_setup_summary_is_readable_and_retains_error_details(capsys):
    from dap_assistant.cli import _print_document_summary

    _print_document_summary('Official source downloads', [
        {'id': 'dap-vehicle-buyer', 'status': 'unchanged'},
        {'id': 'njt-vehicle-insurance', 'status': 'updated'},
        {'id': 'optional-source-error', 'status': 'error',
         'error': 'certificate verification failed\nDetailed traceback must not be printed'},
        {'id': 'magyarorszag-jszp', 'status': 'disabled'},
    ])
    printed = capsys.readouterr().out
    assert '[INFO] Official source downloads:' in printed
    for name in ('unchanged=1', 'updated=1', 'error=1', 'disabled=1'):
        assert name in printed
    assert '[WARN] optional-source-error: certificate verification failed' in printed
    assert 'Detailed traceback' not in printed


def test_setup_invokes_summary_and_keeps_the_existing_index(tmp_path):
    root = Path(__file__).resolve().parents[2]
    setup = (root / 'SETUP.bat').read_text(encoding='utf-8')
    assert 'scripts\\download_documents.py --index --summary' in setup
    assert 'scripts\\launcher_support.py index-check' in setup
    assert 'if not errorlevel 1' in setup  # reuse verified existing index
