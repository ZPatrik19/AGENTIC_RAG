"""Professional 50-200 request load-test CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from dap_assistant.evaluation.professional import available_targets, run_load, save_run
from dap_assistant.settings import Settings


def main() -> int:
    targets = available_targets()
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('single_node', 'subflow', 'full_workflow'),
                        default='full_workflow')
    parser.add_argument('--target', default='agentic/full')
    parser.add_argument('--topic', choices=('all', 'vehicle', 'employment'), default='all')
    parser.add_argument('--count', type=int, default=50,
                        help='Total measured requests, 50..200')
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('reports/runs'))
    args = parser.parse_args()
    if not 50 <= args.count <= 200:
        parser.error('--count must be between 50 and 200')
    if args.target not in targets[args.scope]:
        parser.error(f'Invalid target for {args.scope}. Choices: {", ".join(targets[args.scope])}')
    try:
        result = run_load(
            Settings(), scope=args.scope, target=args.target, topic=args.topic,
            request_count=args.count, concurrency=args.concurrency,
            timeout_s=args.timeout, seed=args.seed, warmup=args.warmup,
            progress=lambda done, total: print(f'[LOAD] {done}/{total}', flush=True),
        )
        folder = save_run(result, args.output)
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}')
        return 1
    print(f'[OK] JSON, CSV, Markdown: {folder}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
