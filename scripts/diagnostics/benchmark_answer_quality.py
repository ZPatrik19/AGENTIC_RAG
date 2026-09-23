"""Replay frozen retrieved evidence using existing comparison audits and metrics.

Capture A before editing production code, then run B with the same fixture.
This isolates generation/context changes; it is not a retrieval comparison.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from dap_assistant.evaluation.comparison import _simple_generate, _simple_metrics, _usage
from dap_assistant.evaluation.dataset import index_versions, load_dataset
from dap_assistant.observability.telemetry import Telemetry
from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.llm import OllamaAdapter
from dap_assistant.rag.rag_graph import build_rag_graph
from dap_assistant.settings import Settings


def compare_saved(folder: Path, label: str, baseline: str = 'A') -> dict:
    """Compare saved runs without performing inference or changing old results."""
    a = json.loads((folder / (baseline + '.json')).read_text(encoding='utf-8'))
    b = json.loads((folder / (label + '.json')).read_text(encoding='utf-8'))
    if a['fixture_sha256'] != b['fixture_sha256']:
        raise ValueError('Cannot compare different evidence fixtures')
    if [r['question_id'] for r in a['rows']] != [r['question_id'] for r in b['rows']]:
        raise ValueError('Cannot compare unfinished or mismatched runs')
    rows = []
    for run in (a, b):
        for row in run['rows']:
            answer_calls = [x for x in row['trace']['llm_usage'] if x['phase'] == 'answer']
            last = answer_calls[-1] if answer_calls else {}
            rows.append({'variant': run['label'], 'question_id': row['question_id'],
                'latency_s': round(row['latency_s'], 2),
                'prompt_tokens': last.get('prompt_eval_count'),
                'generated_tokens': last.get('eval_count'),
                'requested_num_predict': last.get('requested_num_predict'),
                'done_reason': last.get('done_reason'),
                'model_claims': sum(c.get('origin') == 'model_generated'
                                    for c in row['output']['answer_draft']['claims']),
                'fallback': row['output']['answer_fallback'],
                'status': row['output']['response_status'], 'metrics': row['metrics'],
                'context_budget': row['trace']['selection'].get('answer_context_budget'),
                'warnings': row['output']['answer_draft'].get('quality_warnings', [])})
    changed = {key: {'before': value, 'after': b['settings'].get(key)}
               for key, value in a['settings'].items() if b['settings'].get(key) != value}
    report = {'baseline': baseline, 'variant': label, 'fixture_sha256': a['fixture_sha256'],
              'changed_existing_settings': changed,
              'rows': rows, 'limitations': [
                  'Two single-run cases, no statistical or general quality conclusion.',
                  'See changed_existing_settings; adaptive output may change requested num_predict.',
                  'Frozen BM25 evidence from production nodes; no live dense retrieval or native tool cycle.',
                  'Faithfulness, correctness and reference completeness require human review.',
                  'Citation validity and facet coverage are structural proxies, not semantic scores.']}
    (folder / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Válaszgenerálás A/B mérés', '',
        'Azonos rögzített kérdések és evidence; a modellbeállítások az eredeti JSON-riportokban szerepelnek.',
        ('Az adaptív kimeneti keret a meglévő válaszplafonig nőhet. '
         'A mérés a kontextus- és válaszréteget izolálja.'), '',
        '| Változat | Kérdés | Idő (s) | Prompt token | Generált token | Kimeneti limit | Lezárás | Modellállítás | Fallback |',
        '|---|---|---:|---:|---:|---:|---|---:|---|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row[k]) for k in (
            'variant', 'question_id', 'latency_s', 'prompt_tokens', 'generated_tokens',
            'requested_num_predict', 'done_reason', 'model_claims', 'fallback')) + ' |')
    lines.extend(['', ('A forrásazonosító-helyesség mindkét ágon strukturális metrika. '
                  'A correctness, faithfulness és emberileg igazolt completeness nem mérhető '
                  'jóváhagyott referenciák nélkül; értékük N/A.'), '',
                  ('A részleges állapot és a kiszorult bizonyítékok nem rejtett hibák: '
                  'a részletek a comparison.json context_budget/warnings mezőiben szerepelnek. '
                  'A végső válaszok az eredeti futásfájlokban olvashatók.'), '',
                  ('Két kérdés, egy-egy futás; az időt a cache és a gép terhelése is befolyásolja. '
                  'Ebből általános százalékos minőségjavulás nem állítható. '
                  'A korábbi B1/B2/B_final/B_verified kísérletek megmaradtak; '
                  'az auditált végállapotot a fenti változat neve azonosítja.')])
    (folder / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--folder', type=Path, default=Path('reports/answer_quality_ab'))
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    target = args.folder / (args.label + '.json')
    if target.exists():
        parser.error('Choose a new label; existing measurements are never overwritten.')
    settings = Settings()
    fixture = args.folder / 'fixture.json'
    if not fixture.exists():
        # Real production RAG nodes, lexical-only capture to avoid a live UI's
        # Qdrant lock. Both runs replay exactly this evidence, never reretrieve.
        graph = build_rag_graph(settings, dense=None)
        cases = [c for c in load_dataset() if c['question_id'] in ('AUTO_001', 'WORK_001')]
        rows = []
        for case in cases:
            state = graph.invoke({'query': case['question'], 'domain': case['category'],
                                  'role': 'buyer' if case['category'] == 'vehicle' else '',
                                  'task_id': case['question_id'], 'run_id': 'capture'})
            rows.append({'case': case, 'evidence': state['evidence']})
        fixture.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = json.loads(fixture.read_text(encoding='utf-8'))
    chunks = load_chunks(settings.data_dir)
    report = {'label': args.label, 'fixture_sha256': sha256(fixture.read_bytes()).hexdigest(),
              'settings': asdict(settings), 'retrieval': 'frozen_production_nodes_bm25_only',
              'scope': 'answer_stage_no_native_tools', 'rows': []}
    telemetry = Telemetry()
    adapter = OllamaAdapter(settings, telemetry=telemetry)
    try:
        for row in rows:
            case, evidence = row['case'], row['evidence']
            run_id = args.label + '_' + case['question_id']
            started = perf_counter()
            output = _simple_generate(settings, case, evidence, telemetry, run_id, adapter)
            elapsed = perf_counter() - started
            trace = telemetry.snapshot(run_id)
            metrics, diagnostics = _simple_metrics(case, output,
                [e['chunk_id'] for e in evidence], index_versions(chunks), chunks, None)
            report['rows'].append({'question_id': case['question_id'], 'latency_s': elapsed,
                'metrics': metrics, 'diagnostics': diagnostics, 'usage': _usage(trace),
                'trace': trace, 'output': output, 'answer_correctness': None,
                'correctness_reason': 'requires_reviewed_semantic_reference'})
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            print(case['question_id'], round(elapsed, 2), output['response_status'], flush=True)
    finally:
        adapter.close()
    if args.label != 'A' and (args.folder / 'A.json').exists():
        compare_saved(args.folder, args.label)


if __name__ == '__main__':
    main()
