"""Professional functional evaluation CLI.

Examples:
  python scripts/evaluate.py --scope full_workflow --topic all
  python scripts/evaluate.py --scope single_node --target rag/hybrid_retrieval --topic vehicle
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dap_assistant.evaluation.professional import available_targets, run_functional, save_run
from dap_assistant.settings import Settings


def main() -> int:
    targets = available_targets()
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('single_node', 'subflow', 'full_workflow'),
                        default='full_workflow')
    parser.add_argument('--target', default='agentic/full')
    parser.add_argument('--topic', choices=('all', 'vehicle', 'employment'), default='all')
    parser.add_argument('--question-id', action='append', default=None,
                        help='Repeat to select individual AUTO_*/WORK_* cases')
    parser.add_argument('--judge', action='store_true',
                        help='Optional local Qwen judge; only pinned human-reviewed references are scored')
    parser.add_argument('--output', type=Path, default=Path('reports/runs'))
    args = parser.parse_args()
    if args.target not in targets[args.scope]:
        parser.error(f'Invalid target for {args.scope}. Choices: {", ".join(targets[args.scope])}')
    try:
        result = run_functional(
            Settings(), scope=args.scope, target=args.target, topic=args.topic,
            question_ids=args.question_id, local_judge=args.judge,
            progress=lambda done, total: print(f'[EVAL] {done}/{total}', flush=True),
        )
        folder = save_run(result, args.output)
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}')
        return 1
    print(f'[OK] JSON, CSV, Markdown: {folder}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
