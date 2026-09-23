"""Offline HTML/PDF parse, structural chunking and stable per-document indexing."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
import pymupdf as fitz

from dap_assistant.settings import Settings
from dap_assistant.documents.sources import Source, load_manifest


def parse_html(body: bytes) -> list[dict]:
    """Preserve headings and tabular row boundaries for source-backed fee retrieval.

    Each table row retains its column labels, avoiding accidental pairing of
    e.g. the wrong vehicle-age column with an amount. No OCR or invented PDF.
    """
    soup = BeautifulSoup(body, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'aside']):
        tag.decompose()
    main = soup.find('main') or soup.find('article') or soup.body or soup
    section: list[str] = []
    sections: list[dict] = []
    title_path: list[str] = []

    def flush() -> None:
        nonlocal section
        if section:
            sections.append({'section': list(title_path), 'text': '\n'.join(section), 'page': None})
            section = []

    for el in main.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'table']):
        if el.find_parent(['li', 'table']) is not None:
            continue
        if el.name.startswith('h'):
            flush()
            depth = int(el.name[1])
            title_path = title_path[:depth - 1] + [' '.join(el.get_text(' ', strip=True).split())]
        elif el.name == 'table':
            # Keep each complete legal tariff table as one logical section.
            # Otherwise fixed-size chunking can separate the age headers from rates.
            flush()
            # Repeat section heading within the tariff chunk itself: it provides
            # the rate year if the HTML table has no explicit '2026' header.
            if title_path:
                section.append(' / '.join(title_path))
            rows = el.find_all('tr')
            for row in rows:
                cells = [' '.join(cell.get_text(' ', strip=True).split())
                         for cell in row.find_all(['th', 'td'], recursive=False)]
                if any(cells):
                    section.append(' | '.join(cells))
            flush()
        else:
            content = ' '.join(el.get_text(' ', strip=True).split())
            if content:
                section.append(content)
    flush()
    return sections


def parse_pdf(body: bytes) -> list[dict]:
    result = []
    with fitz.open(stream=body, filetype='pdf') as pdf:
        for n, page in enumerate(pdf, start=1):
            text = '\n'.join(' '.join(line.split()) for line in page.get_text(sort=True).splitlines() if line.strip())
            if text:
                result.append({'section': [f'PDF oldal {n}'], 'text': text, 'page': n})
    if not result:
        raise ValueError('No extractable text in PDF; manual OCR review required')
    return result



# Select specific legal provisions rather than embedding entire statute (hundreds of
# unrelated sections). A missing requested paragraph is an ingestion ERROR, not a
# silent fallback to the full law. The statute is not legal advice or proof of
# historical effectiveness at the user's transaction date.
def select_legal_sections(sections: list[dict], wanted: list[str]) -> list[dict]:
    """Keep complete legal provisions when headings and their paragraphs appear in separate HTML elements."""
    if not wanted:
        return sections
    pattern = re.compile(r"(?m)^\s*(\d{1,3}(?:/[A-Z])?)\s*[.．]?\s*§")
    # Preserve source order, including separate heading/body blocks. A full
    # statute is only assembled in memory; only selected provisions are indexed.
    complete = '\n'.join(sec['text'] for sec in sections if sec.get('text'))
    markers = list(pattern.finditer(complete))
    by_number: dict[str, list[str]] = {}
    for i, match in enumerate(markers):
        number = match.group(1)
        if number not in wanted:
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else len(complete)
        body = complete[match.start():end].strip()
        # Ignore short clickable table-of-contents links and empty § headers.
        if len(body) >= 100:
            by_number.setdefault(number, []).append(body)
    result = []
    for number in wanted:
        matches = by_number.get(number, [])
        if matches:
            body = max(matches, key=len)
            result.append({'section': ['Jogszabály', f'{number}. §'],
                           'text': body, 'page': None})
    if not result:
        raise ValueError('None of the selected legal paragraphs could be extracted; raw statute was preserved')
    return result


def _semantic_units(text: str) -> list[str]:
    """Split documents at structural boundaries while preserving complete list items and table rows."""
    units: list[str] = []
    for raw in text.splitlines():
        line = ' '.join(raw.split()).strip()
        if not line:
            continue
        if len(line) <= 950:
            units.append(line)
            continue
        parts = [part.strip() for part in re.split(r'(?<=[.!?])\s+', line) if part.strip()]
        units.extend(parts or [line])
    return units


def make_chunks(source: Source, metadata: dict, sections: list[dict], max_chars: int = 1500, overlap_units: int = 1) -> list[dict]:
    """Chunk by complete semantic units with heading context, overlap and source provenance."""
    if max_chars < 400 or overlap_units < 0:
        raise ValueError('Invalid chunk configuration')
    chunks: list[dict] = []
    for sec in sections:
        text = sec['text'].strip()
        if not text:
            continue
        units = _semantic_units(text)
        if not units:
            continue
        cursor = 0
        while cursor < len(units):
            packed: list[str] = []
            size = 0
            end = cursor
            while end < len(units):
                unit = units[end]
                projected = size + len(unit) + (1 if packed else 0)
                if packed and projected > max_chars:
                    break
                packed.append(unit)
                size = projected
                end += 1
                if size >= max_chars:
                    break
            content = '\n'.join(packed).strip()
            if content:
                index = len(chunks)
                section_path = list(sec['section'])
                retrieval_text = (
                    f"{source.title}\nTémák: {', '.join(source.topics)}\n"
                    f"Fejezet: {' / '.join(section_path)}\n{content}"
                )
                chunks.append({
                    'chunk_id': hashlib.sha256(f'{source.id}:{metadata["sha256"]}:{index}'.encode()).hexdigest()[:32],
                    'document_id': source.id,
                    'document_version': metadata['sha256'],
                    'source_url': source.url,
                    'title': source.title,
                    'domain': source.domain,
                    'role': source.role,
                    'topics': list(source.topics),
                    'authority': source.authority,
                    'source_priority': source.priority,
                    'retrieved_at': metadata['retrieved_at'],
                    'published_at': None,
                    'effective_from': None,
                    'effective_to': None,
                    'section_path': section_path,
                    'section_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    'page_number': sec['page'],
                    'chunk_kind': 'legal_section' if source.legal_sections else 'administrative_section',
                    'text': content,
                    'retrieval_text': retrieval_text,
                })
            if end >= len(units):
                break
            next_cursor = max(cursor + 1, end - overlap_units)
            cursor = next_cursor
    return chunks


def ingest_one(source: Source, data_dir: Path) -> dict:
    if source.processing_status == 'disabled':
        return {'id': source.id, 'status': 'disabled'}
    metadata_file = data_dir / 'interim' / 'downloads' / f'{source.id}.json'
    if not metadata_file.exists():
        return {'id': source.id, 'status': 'missing_download'}
    meta = json.loads(metadata_file.read_text(encoding='utf-8'))
    stored_path = str(meta["file"]).replace("\\", "/")
    raw_path = data_dir / Path(stored_path)
    raw = raw_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != meta['sha256']:
        raise ValueError(f'Raw file hash mismatch: {source.id}')
    sections = parse_pdf(raw) if source.type == 'pdf' else parse_html(raw)
    if source.legal_sections:
        sections = select_legal_sections(sections, source.legal_sections)
    if not sections or sum(len(sec['text']) for sec in sections) < 80:
        raise ValueError(f'No meaningful extractable content: {source.id}')
    chunks = make_chunks(source, meta, sections)
    output = data_dir / 'processed' / source.domain / f'{source.id}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'document_id': source.id, 'document_version': digest, 'chunks': chunks}
    tmp = output.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, output)
    return {'id': source.id, 'status': 'processed', 'chunks': len(chunks)}


def ingest_all(settings: Settings) -> list[dict]:
    result = []
    for source in load_manifest(settings.manifest).sources:
        try:
            result.append(ingest_one(source, settings.data_dir))
        except (OSError, ValueError, fitz.FileDataError) as exc:
            result.append({'id': source.id, 'status': 'error', 'error': str(exc)})
    return result


def load_chunks(data_dir: Path) -> list[dict]:
    chunks = []
    # Versioned, explicit source allowlist: previously downloaded housing/business
    # JSON remains on disk for the user's safety, but is never searched/indexed.
    manifest_path = data_dir.parent / 'config' / 'document_sources.yaml'
    active_ids = ({s.id for s in load_manifest(manifest_path).sources
                   if s.processing_status != 'disabled'} if manifest_path.exists() else None)
    for path in sorted((data_dir / 'processed').glob('*/*.json')):
        if active_ids is not None and path.stem not in active_ids:
            continue
        try:
            chunks.extend(json.loads(path.read_text(encoding='utf-8'))['chunks'])
        except (OSError, ValueError, KeyError):
            continue
    return chunks
