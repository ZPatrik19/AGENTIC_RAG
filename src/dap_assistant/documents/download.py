"""Idempotent, allowlisted downloader with atomic writes and HTTP cache validation."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import ssl
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import httpx

from dap_assistant.documents.sources import ALLOWED_HOSTS, Source, load_manifest
from dap_assistant.settings import Settings

LOG = logging.getLogger(__name__)
USER_AGENT = 'DAP-Life-Events-Research-Prototype/0.1 (public documentation; respectful rate limit)'


def _tls_verification_context() -> ssl.SSLContext | bool:
    """Use Windows' trusted root CAs when installed; never disable verification.

    Python's certifi-only bundle may not contain a user's enterprise/OS root.
    This cannot repair a truly invalid server certificate or missing OS roots.
    """
    if sys.platform == 'win32':
        try:
            import truststore
        except ImportError:
            LOG.warning('truststore not installed; using default verified TLS CA bundle')
        else:
            return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return True


class DownloadError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(data)
    os.replace(temp, path)


def _get_allowed(client: httpx.Client, url: str, headers: dict, robots_cache: dict[str, RobotFileParser] | None) -> httpx.Response:
    for _ in range(4):
        parts = urlparse(url)
        if parts.scheme != 'https' or parts.hostname not in ALLOWED_HOSTS or parts.port or parts.username:
            raise DownloadError('Unsafe redirect or source URL')
        if robots_cache is not None and not _robots_allowed(client, url, robots_cache):
            raise DownloadError(f'robots.txt disallows {url}')
        response = client.get(url, headers=headers, follow_redirects=False, timeout=25)
        if response.status_code not in (301, 302, 303, 307, 308):
            return response
        if not response.headers.get('location'):
            raise DownloadError('Redirect has no Location header')
        url = urljoin(url, response.headers['location'])
    raise DownloadError('Too many redirects')


def _validate_response(response: httpx.Response, source: Source) -> None:
    final = urlparse(str(response.url))
    if final.scheme != 'https' or final.hostname not in ALLOWED_HOSTS:
        raise DownloadError('Redirect outside approved HTTPS hosts')
    body = response.content
    content_type = response.headers.get('content-type', '').lower()
    if len(body) < 30 or len(body) > 20_000_000:
        raise DownloadError('Document is empty or exceeds 20 MB')
    if source.type == 'pdf':
        if not body.lstrip().startswith(b'%PDF-'):
            raise DownloadError('Expected PDF magic bytes; server returned another format')
    elif 'html' not in content_type and b'<html' not in body[:2048].lower() and b'<!doctype' not in body[:2048].lower():
        raise DownloadError('Expected HTML content')


def _robots_allowed(client: httpx.Client, url: str, robots_cache: dict[str, RobotFileParser]) -> bool:
    """Fail closed if robots policy cannot be determined; never bypass access controls."""
    origin = urlparse(url)
    host = origin.hostname or ''
    if host not in robots_cache:
        robots_url = f'{origin.scheme}://{host}/robots.txt'
        try:
            # Do NOT use follow_redirects=True: that could fetch an unapproved
            # destination before inspecting its Location header.
            for _ in range(4):
                target = urlparse(robots_url)
                if (target.scheme != 'https' or target.hostname not in ALLOWED_HOSTS
                        or target.port or target.username or target.password):
                    raise DownloadError('Unsafe robots.txt redirect outside approved HTTPS hosts')
                response = client.get(robots_url, timeout=12, follow_redirects=False)
                if response.status_code not in (301, 302, 303, 307, 308):
                    break
                location = response.headers.get('location')
                if not location:
                    raise DownloadError('robots.txt redirect has no Location')
                robots_url = urljoin(robots_url, location)
            else:
                raise DownloadError('Too many robots.txt redirects')
            if response.status_code in (404, 410):
                parser = RobotFileParser()
                parser.parse(['User-agent: *', 'Allow: /'])
            else:
                response.raise_for_status()
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
            robots_cache[host] = parser
        except httpx.HTTPError as exc:
            hint = (' Check the Windows certificate store or your trusted CA chain; '
                    'TLS verification remains enabled.' if 'CERTIFICATE_VERIFY_FAILED' in str(exc) else '')
            raise DownloadError(f'Cannot verify robots.txt for {host}: {exc}.{hint}') from exc
    return robots_cache[host].can_fetch(USER_AGENT, url)


def download_one(
    client: httpx.Client,
    source: Source,
    data_dir: Path,
    *,
    robots_cache: dict[str, RobotFileParser] | None = None,
    max_attempts: int = 3,
) -> dict:
    """Return a structured result, never replace a good existing file with an error page."""
    if source.processing_status == 'disabled':
        return {'id': source.id, 'status': 'disabled'}
    ext = '.pdf' if source.type == 'pdf' else '.html'
    target = data_dir / 'raw' / source.destination / f'{source.id}{ext}'
    meta_file = data_dir / 'interim' / 'downloads' / f'{source.id}.json'
    previous = json.loads(meta_file.read_text(encoding='utf-8')) if meta_file.exists() else {}
    # Do not trust stale metadata if its corresponding raw file disappeared.
    valid_cache = target.exists() and sha256(target.read_bytes()) == previous.get('sha256')
    headers = {}
    if valid_cache and previous.get('etag'):
        headers['If-None-Match'] = previous['etag']
    if valid_cache and previous.get('last_modified'):
        headers['If-Modified-Since'] = previous['last_modified']
    error = None
    for attempt in range(max_attempts):
        try:
            response = _get_allowed(client, source.url, headers, robots_cache)
            if response.status_code == 304 and valid_cache:
                return {'id': source.id, 'status': 'unchanged', 'sha256': previous['sha256']}
            if response.status_code in (408, 429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError('Transient HTTP error', request=response.request, response=response)
            response.raise_for_status()
            _validate_response(response, source)
            digest = sha256(response.content)
            if valid_cache and digest == previous['sha256']:
                return {'id': source.id, 'status': 'unchanged', 'sha256': digest}
            _atomic_write(target, response.content)
            meta = {
                'id': source.id,
                'title': source.title,
                'url': source.url,
                'source_page': source.source_page or source.url,
                'domain': source.domain,
                'type': source.type,
                'file': target.relative_to(data_dir).as_posix(),
                'sha256': digest,
                'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'etag': response.headers.get('etag'),
                'last_modified': response.headers.get('last-modified'),
                'content_type': response.headers.get('content-type'),
            }
            _atomic_write(meta_file, json.dumps(meta, ensure_ascii=False, indent=2).encode())
            return {'id': source.id, 'status': 'updated', 'sha256': digest}
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            error = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in (408, 429, 500, 502, 503, 504):
                break
            if attempt + 1 < max_attempts:
                time.sleep(min(2**attempt, 4))
        except (DownloadError, OSError) as exc:
            error = exc
            break
    raise DownloadError(f'{source.id}: download failed: {error}')


def download_all(settings: Settings, *, check_robots: bool = True) -> list[dict]:
    manifest = load_manifest(settings.manifest)
    results = []
    cache: dict[str, RobotFileParser] = {}
    with httpx.Client(headers={'User-Agent': USER_AGENT}, trust_env=True,
                      verify=_tls_verification_context()) as client:
        for source in manifest.sources:
            try:
                result = download_one(client, source, settings.data_dir, robots_cache=cache if check_robots else None)
            except (DownloadError, OSError) as exc:
                result = {'id': source.id, 'status': 'error', 'error': str(exc)}
            results.append(result)
            LOG.info('document_download %s', json.dumps(result, ensure_ascii=False))
            time.sleep(0.8)
    audit_file = settings.data_dir / 'interim' / 'download_audit.jsonl'
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with audit_file.open('a', encoding='utf-8') as audit:
        for item in results:
            audit.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), **item}, ensure_ascii=False) + '\n')
    return results
