"""Run final functional + sequential 50-request full-workflow measurements.

The live UI's embedded database is left alone. Freeze the processed corpus,
verify its fingerprint, and build an isolated real E5/Qdrant benchmark index.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime

from dap_assistant.rag.dense_resources import acquire_dense, release_dense
from dap_assistant.evaluation.automatic_reference import corpus_fingerprint
from dap_assistant.evaluation.professional import run_functional, run_load, save_run
from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.rag.retrieval import index_all
from dap_assistant.settings import Settings


def main() -> None:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    original = Settings()
    root = original.root / 'reports' / 'final_validation'
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'corpus_snapshot'
    if not snapshot.exists():
        shutil.copytree(original.data_dir / 'processed', snapshot / 'processed')
    source_hash = corpus_fingerprint(load_chunks(original.data_dir))
    if corpus_fingerprint(load_chunks(snapshot)) != source_hash:
        raise RuntimeError('Snapshot differs from live processed corpus; choose a fresh output directory')
    settings = replace(original, data_dir=snapshot)
    (snapshot / 'vectorstore' / 'qdrant').mkdir(parents=True, exist_ok=True)
    progress_path = root / 'progress.jsonl'

    def log(stage: str, done: int = 0, total: int = 0, **extra) -> None:
        record = {'time': datetime.now(UTC).isoformat(), 'stage': stage,
                  'done': done, 'total': total, **extra}
        with progress_path.open('a', encoding='utf-8') as out:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=True), flush=True)

    log('initializing', corpus_fingerprint=source_hash)
    dense = acquire_dense(settings)
    if dense is None:
        raise RuntimeError('A real dense index is required for this validation')
    try:
        if not (snapshot / 'vectorstore' / 'index_meta.json').exists():
            log('indexing', **index_all(settings, dense=dense))
        manifest_file = root / 'completed_runs.json'
        manifest = json.loads(manifest_file.read_text(encoding='utf-8')) if manifest_file.exists() else {}
        manifest['corpus_fingerprint'] = source_hash
        manifest['index_note'] = 'Rebuilt E5/Qdrant from identical processed chunks; live index not modified'
        for stage in ('functional', 'load'):
            if stage in manifest:
                log(stage + '_already_saved', path=manifest[stage])
                continue
            log(stage + '_starting')
            def progress(done, total, stage=stage):
                return log(stage, done, total)
            if stage == 'functional':
                result = run_functional(settings, scope='full_workflow', topic='all', progress=progress)
            else:
                result = run_load(settings, scope='full_workflow', topic='all', request_count=50,
                                  concurrency=1, timeout_s=120, seed=42, warmup=1, progress=progress)
            folder = save_run(result, root / stage)
            manifest[stage] = str(folder.relative_to(original.root))
            manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            log(stage + '_saved', path=str(folder))
    finally:
        release_dense(dense)


if __name__ == '__main__':
    main()
