"""Pin source content hashes from a reviewed local index; commit output only after manual verification."""
import argparse
import json
from pathlib import Path

from dap_assistant.evaluation.dataset import DATASET, index_versions
from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.settings import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('evaluation/golden_pinned.json'))
    args = parser.parse_args()
    payload = json.loads(DATASET.read_text(encoding='utf-8'))
    versions = index_versions(load_chunks(Settings().data_dir))
    for case in payload['cases']:
        missing = set(case['expected_source_ids']) - versions.keys()
        if missing:
            raise SystemExit(f'Missing expected indexed sources: {sorted(missing)}')
        case['expected_source_versions'] = {doc: versions[doc] for doc in case['expected_source_ids']}
        case['reference_status'] = 'pinned_source_versions_only'
    payload['reference_policy'] = 'Source content hashes pinned to local index; human validation of facts and relevant chunk IDs still required.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[OK] Review actual downloaded source content and commit: {args.output}')


if __name__ == '__main__':
    main()
