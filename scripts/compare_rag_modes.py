"""Compare baseline dense RAG, one-shot hybrid RAG and the production Agentic RAG.

Examples:
  python scripts/compare_rag_modes.py --topic all
  python scripts/compare_rag_modes.py --question-id AUTO_001 --question-id WORK_001
  python scripts/compare_rag_modes.py --topic vehicle --judge
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dap_assistant.evaluation.comparison import run_rag_comparison
from dap_assistant.evaluation.professional import save_run
from dap_assistant.settings import Settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Paired baseline_dense vs hybrid_rrf vs agentic evaluation on the same golden cases.')
    parser.add_argument('--topic', choices=('all', 'vehicle', 'employment'), default='all')
    parser.add_argument('--question-id', action='append', default=None,
                        help='Repeat to evaluate selected AUTO_*/WORK_* cases only.')
    parser.add_argument('--judge', action='store_true',
                        help='Optional local Qwen semantic judge; deterministic metrics remain separate.')
    parser.add_argument('--output', type=Path, default=Path('reports/comparisons'))
    parser.add_argument('--read-timeout', type=float, default=None,
                        help='Ollama inactivity timeout in seconds, shared by all three profiles.')
    parser.add_argument('--total-timeout', type=float, default=None,
                        help='Maximum generation time in seconds, shared by all three profiles.')
    args = parser.parse_args()
    settings = Settings()
    read_timeout = args.read_timeout if args.read_timeout is not None else settings.ollama_read_timeout_s
    total_timeout = args.total_timeout if args.total_timeout is not None else settings.ollama_total_timeout_s
    if not 0 < read_timeout <= total_timeout:
        parser.error('Require 0 < --read-timeout <= --total-timeout.')
    settings = replace(settings, ollama_read_timeout_s=read_timeout,
                       ollama_total_timeout_s=total_timeout)
    try:
        result = run_rag_comparison(
            settings, topic=args.topic, question_ids=args.question_id, local_judge=args.judge,
            progress=lambda done, total: print(f'[COMPARE] {done}/{total}', flush=True),
        )
        folder = save_run(result, args.output)
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}')
        return 1
    print(f'[OK] Baseline / Hybrid / Agentic comparison: {folder}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
