"""Generate the 20-case automatic SILVER/proxy reference from the local corpus.

No human-review claim, no embeddings, API or model calls. Missing facts remain
unevaluable and are printed explicitly. Human-reviewed gold is never changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dap_assistant.evaluation.automatic_reference import ensure_auto_reference
from dap_assistant.evaluation.dataset import AUTO_DATASET, DATASET
from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Settings().data_dir)
    parser.add_argument('--output', type=Path, default=AUTO_DATASET)
    parser.add_argument('--benchmark', type=Path, default=DATASET)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    try:
        payload = ensure_auto_reference(args.data_dir, chunks=load_chunks(args.data_dir),
                                        baseline_path=args.benchmark, destination=args.output,
                                        force=args.force)
    except (ValueError, OSError, KeyError) as exc:
        print(f'[HIBA] {exc}')
        return 1
    ready = payload['reference_ready_cases']
    print(f'[SILVER] Automatikus, NEM emberileg ellenőrzött referencia: {args.output}')
    print(f'[SILVER] {ready}/20 becslésként értékelhető kérdés; a hiányok N/A-k maradnak.')
    for case in payload['cases']:
        audit = case['reference_audit']
        print(f"  {case['question_id']}: {case['reference_status']}; "
              f"chunk={len(case['relevant_chunk_ids'])}; "
              f"hiányzó tény={audit['missing_fact_ids']}; "
              f"hiányzó ág={audit['missing_task_ids']}")
    return 0 if ready == 20 else 2


if __name__ == '__main__':
    raise SystemExit(main())
