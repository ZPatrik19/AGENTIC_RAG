"""One-command reproducible download/ingest/index and local status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import logging
import sys

from dap_assistant.documents.download import download_all
from dap_assistant.documents.ingestion import ingest_all
from .settings import Settings
from dap_assistant.documents.sources import load_manifest


def _print_document_summary(label: str, results: list[dict]) -> None:
    """Readable setup output; full per-document audit remains on disk."""
    from collections import Counter

    counts = Counter(item.get('status', 'unknown') for item in results)
    summary = ', '.join(f'{status}={count}' for status, count in sorted(counts.items()))
    print(f'[INFO] {label}: {summary}')
    for item in results:
        if item.get('status') == 'error':
            detail = str(item.get('error', 'unknown error')).splitlines()[0]
            print(f"[WARN] {item.get('id')}: {detail}")


def rebuild(*, download: bool, index: bool, summary: bool = False) -> int:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    if download:
        results = download_all(settings)
        if summary:
            _print_document_summary('Official source downloads', results)
        else:
            print(json.dumps({'download': results}, ensure_ascii=False, indent=2))
        required = {item.id for item in load_manifest(settings.manifest).sources if item.required}
        optional_errors = [r for r in results if r['status'] == 'error' and r['id'] not in required]
        if optional_errors:
            print('[WARN] Optional official sources unavailable: ' + ', '.join(r['id'] for r in optional_errors))
        for r in results:
            if r['status'] == 'error' and r['id'] in required:
                # Offline replay: a verified raw snapshot is still usable.
                if not (settings.data_dir / 'interim' / 'downloads' / f"{r['id']}.json").exists():
                    failures += 1
                else:
                    print('[WARN] Core source network unavailable; using existing verified snapshot: ' + r['id'])
    if index:
        parsed = ingest_all(settings)
        if summary:
            _print_document_summary('Document processing', parsed)
        else:
            print(json.dumps({'ingestion': parsed}, ensure_ascii=False, indent=2))
        required = {item.id for item in load_manifest(settings.manifest).sources if item.required}
        for item in parsed:
            if item['status'] == 'error' and item['id'] in required:
                failures += 1
            elif item['status'] == 'error':
                print('[WARN] Optional document parsing failed: ' + item['id'])
        missing_core = required - {r['id'] for r in parsed if r['status'] == 'processed'}
        failures += len(missing_core)
        if missing_core:
            print('[ERROR] Missing mandatory documents: ' + ', '.join(sorted(missing_core)), file=sys.stderr)
        try:
            from dap_assistant.rag.retrieval import index_all
            indexed = index_all(settings)
            if summary:
                print(f"[OK] Index: {indexed.get('documents', 0)} documents, {indexed.get('chunks', 0)} chunks")
            else:
                print(json.dumps({'index': indexed}, ensure_ascii=False, indent=2))
        except Exception as exc:
            failures += 1
            print(f'Indexing failed: {exc}', file=sys.stderr)
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Read-only official document pipeline')
    parser.add_argument('action', choices=('download', 'ingest', 'index', 'rebuild', 'status'))
    parser.add_argument('--verify-qdrant', action='store_true',
                        help='status only: inspect actual local Qdrant payloads; stop Streamlit first')
    parser.add_argument('--strict', action='store_true',
                        help='status only: exit nonzero unless mandatory sources and Qdrant payloads verify')
    parser.add_argument('--output', type=Path,
                        help='status only: optionally export JSON report (local, gitignored reports/)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if args.action != 'status' and (args.verify_qdrant or args.strict or args.output):
        parser.error('--verify-qdrant, --strict and --output are only supported for status')
    if args.action == 'status':
        settings = Settings()
        from dap_assistant.documents.corpus_status import build_corpus_report, corpus_ready
        report = build_corpus_report(settings, verify_qdrant=args.verify_qdrant)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n', encoding='utf-8')
        print(text)
        if args.strict:
            raise SystemExit(0 if corpus_ready(report) else 1)
        return
    if args.action == 'download':
        code = rebuild(download=True, index=False)
    elif args.action == 'ingest':
        print(json.dumps(ingest_all(Settings()), ensure_ascii=False, indent=2))
        code = 0
    elif args.action == 'index':
        from dap_assistant.rag.retrieval import index_all
        print(json.dumps(index_all(Settings()), ensure_ascii=False, indent=2))
        code = 0
    else:
        code = rebuild(download=True, index=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
