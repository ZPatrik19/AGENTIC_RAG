"""Audit real local extraction and all 20 golden questions; no auto-approved gold.

Without --with-dense: offline BM25 and extraction audit, no model/DB loading.
With --with-dense: real E5 + Qdrant query for all 20 cases (stop Streamlit first).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dap_assistant.evaluation.corpus_quality import (
    embedding_runtime_sanity, extraction_report, golden_diagnostics,
)
from dap_assistant.evaluation.index_health import check_index
from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-dense', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('reports/corpus_golden_audit.json'))
    args = parser.parse_args()
    settings = Settings()
    dense = None
    try:
        if args.with_dense:
            from dap_assistant.rag.retrieval import LocalEmbeddings, LocalQdrant
            dense = LocalQdrant(settings, LocalEmbeddings(settings))
            check_index(settings, dense)
        chunks = load_chunks(settings.data_dir)
        output = {'extraction': extraction_report(settings),
                  'embedding_numeric_smoke': (embedding_runtime_sanity(dense.embeddings, chunks)
                                              if dense else {'status': 'not_run'}),
                  'golden': golden_diagnostics(settings, chunks, dense=dense)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n',
                               encoding='utf-8')
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}')
        return 1
    finally:
        if dense is not None:
            dense.close()
    summary = output['extraction']['summary']
    print(f'[OK] {args.output}: {output["golden"]["question_count"]} kérdés; '
          f'parse/chunk egyezés {summary["reparsed_roundtrip_match"]} dokumentum; '
          f'hiteles gold {output["golden"]["pinned_reviewed_cases"]}/20; '
          f'vektoros keresés: {"igen" if args.with_dense else "nem"}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
