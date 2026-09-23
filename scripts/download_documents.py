"""Run: python scripts/download_documents.py [--index]."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dap_assistant.cli import rebuild  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', action='store_true', help='also parse/embed/index the documents')
    parser.add_argument('--summary', action='store_true', help='short, readable setup report instead of full JSON')
    args = parser.parse_args()
    sys.exit(rebuild(download=True, index=args.index, summary=args.summary))
