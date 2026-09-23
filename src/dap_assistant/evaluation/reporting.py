"""Persistent benchmark reports and Windows-safe run names.

This module has no retrieval, model or index dependencies. Existing imports from
``evaluation.professional`` remain supported during the migration.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import re


def _run_name_segment(value: object) -> str:
    """Produce a Windows-safe segment for an evaluation scope or graph target."""
    segment = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(value or '').replace('/', '_'))
    return segment.strip('_-')[:90] or 'unknown'


def run_report_stem(result: dict, *, at: datetime | None = None) -> str:
    """Local date/time + evaluation scope + optional measured graph target.

    Example: 2026-09-22_10-35-27_terheleses_teszt_single_node_rag_hybrid_retrieval.
    Use `at` for deterministic tests; saved reports otherwise use local time.
    """
    if at is None:
        raw_timestamp = result.get('timestamp_utc')
        if isinstance(raw_timestamp, str):
            try:
                at = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
            except ValueError:
                at = None
        if at is None:
            at = datetime.now().astimezone()
    if at.tzinfo is not None:
        at = at.astimezone()
    cfg = result.get('configuration') or {}
    measurement = ('terheleses_teszt' if result.get('kind') == 'load_v4'
                   else 'funkcionalis_meres' if result.get('kind') == 'functional_v4'
                   else _run_name_segment(result.get('kind', 'evaluation')))
    scope = _run_name_segment(cfg.get('scope') or result.get('kind', 'evaluation'))
    target = cfg.get('target')
    # A full workflow has no individual node to include in its filename.
    suffix = (f'_{_run_name_segment(target)}'
              if target and target not in ('agentic/full', 'full') else '')
    return f'{at:%Y-%m-%d_%H-%M-%S}_{measurement}_{scope}{suffix}'


def save_run(result: dict, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stem = run_report_stem(result, at=datetime.now().astimezone())
    # Two runs may finish within the same second. Preserve the date/scope/node
    # naming rule without overwriting a previous report.
    folder = output_root / stem
    counter = 2
    while folder.exists():
        folder = output_root / f'{stem}_{counter:02d}'
        counter += 1
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    # CSV keeps one row per case/request and flattens scalar metrics only.
    rows = []
    for item in result.get('rows', []):
        row = {key: value for key, value in item.items()
               if isinstance(value, (str, int, float, bool)) or value is None}
        for key, value in item.get('metrics', {}).items():
            row[f'metric_{key}'] = value
        # In addition to full JSON, include safe Ollama counters in the CSV.
        # Missing authoritative eval_count stays blank, never a fabricated zero.
        diag = item.get('ollama_diagnostics', {})
        if diag:
            responses = [r for r in diag.get('responses', []) if r.get('phase') == 'answer']
            failures = [r for r in diag.get('failures', []) if r.get('phase') == 'answer']
            response = responses[-1] if responses else {}
            failure = failures[-1] if failures else {}
            row.update({
                'ollama_failure_kind': failure.get('failure_kind'),
                'ollama_done_reason': failure.get('done_reason') or response.get('done_reason'),
                'ollama_eval_count': (failure.get('eval_count') if failure else response.get('eval_count')),
                'ollama_validation_error_type': failure.get('validation_error_type'),
                'ollama_requested_num_predict': failure.get('requested_num_predict') or response.get('requested_num_predict'),
                'ollama_observed_content_chars': (failure.get('observed_content_chars')
                                                  if failure else response.get('observed_content_chars')),
            })
        provenance = item.get('claim_provenance', {}) or item.get('details', {}).get('claim_provenance', {})
        counts = provenance.get('counts', {})
        if provenance:
            row.update({
                'claims_model_generated': counts.get('model_generated'),
                'claims_source_supplement': counts.get('source_supplement'),
                'claims_source_only': counts.get('source_only'),
                'claims_tool_extract': counts.get('tool_extract'),
                'tool_checklist_visible_items': item.get('details', {}).get('tool_utilization', {}).get('visible_items'),
                'claim_ids': ';'.join(str(claim.get('claim_id', '')) for claim in provenance.get('claims', [])
                                      if claim.get('claim_id')),
                'claims_quote_in_model_prompt': sum(
                    claim.get('quoted_unit_in_model_prompt') is True
                    for claim in provenance.get('claims', [])),
            })
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with (folder / 'rows.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'report.md').write_text(render_report_markdown(result), encoding='utf-8')
    return folder


def render_report_markdown(result: dict) -> str:
    lines = [
        '# Agentic RAG evaluation report', '',
        f"- Run ID: `{result.get('run_id')}`",
        f"- Timestamp: {result.get('timestamp_utc')}",
        f"- Kind: {result.get('kind')}",
        f"- Model: `{result.get('configuration', {}).get('model')}`",
        f"- Context window: {result.get('configuration', {}).get('context_window')}",
        '', '## Configuration', '', '```json',
        json.dumps(result.get('configuration', {}), ensure_ascii=False, indent=2), '```',
        '', '## Summary', '', '```json',
        json.dumps(result.get('summary', {}), ensure_ascii=False, indent=2, default=str), '```',
        '', '## Methodology note', '',
        'N/A metrics are never converted to zero. Retrieval/context scores on the automatic SILVER dataset '
        'are heuristic proxy estimates, NOT human-verified gold. Local LLM judge scores are fallible estimates.',
    ]
    if result.get('kind') == 'rag_profile_comparison_v1':
        lines.extend(['', '## Ollama answer diagnostics', '',
                      '| Mód | Kérdés | Állapot | Qwen JSON kész | Modellválasz használható | Fallback oka | Ollama-hiba | done_reason | eval_count | Validáció |',
                      '|---|---|---|---|---|---|---|---|---:|---|'])
        for item in result.get('rows', []):
            diag = item.get('ollama_diagnostics', {})
            responses = [r for r in diag.get('responses', []) if r.get('phase') == 'answer']
            failures = [r for r in diag.get('failures', []) if r.get('phase') == 'answer']
            response = responses[-1] if responses else {}
            failure = failures[-1] if failures else {}
            def fmt(value):
                return str(value) if value is not None else 'N/A'
            lines.append('| ' + ' | '.join((
                str(item.get('mode', '')).replace('|', '/'),
                str(item.get('question_id', '')).replace('|', '/'),
                str(item.get('response_status', '')).replace('|', '/'),
                fmt(item.get('generation_success')),
                fmt(item.get('model_answer_usable')),
                str(item.get('answer_fallback_reason', '') or '—').replace('|', '/'),
                fmt(failure.get('failure_kind')),
                fmt(failure.get('done_reason') or response.get('done_reason')),
                fmt(failure.get('eval_count') if failure else response.get('eval_count')),
                fmt(failure.get('validation_error_type')),
            )) + ' |')
        lines.extend(['', '## Állítások eredete', '',
                      '| Mód | Kérdés | Qwen megfogalmazása | Utólagos forráskiegészítés | Csak forráskivonat | Eszközből ellenőrzött kivonat |',
                      '|---|---|---:|---:|---:|---:|'])
        for item in result.get('rows', []):
            provenance = item.get('claim_provenance', {}) or item.get('details', {}).get('claim_provenance', {})
            counts = provenance.get('counts', {})
            lines.append('| ' + ' | '.join((
                str(item.get('mode', '')), str(item.get('question_id', '')),
                str(counts.get('model_generated', 'N/A')),
                str(counts.get('source_supplement', 'N/A')),
                str(counts.get('source_only', 'N/A')),
                str(counts.get('tool_extract', 'N/A')),
            )) + ' |')
        lines.extend(['', 'A megjelenített válaszban szereplő `[C_...]` állításazonosítók ugyanazok, '
                      'mint a JSON `rows[].claim_provenance.claims[].claim_id` és a CSV `claim_ids` értékei. '
                      'Az azonosító a pontos állítástól, az idézett forrásegységtől és a bizonyítékazonosítóktól '
                      'függ; a megjelenítési sorrend vagy a kategóriacímke nem változtatja meg.',
                      'Az egyedi állítások eredete és a Qwen-promptba ténylegesen bekerült '
                      'idézetek jelölése a JSON `rows[].claim_provenance` mezőjében található. '
                      'A forráskiegészítés nem Qwen által megfogalmazott szöveg, a bizonyíték-azonosító '
                      'önmagában nem igazolja, hogy az idézet a promptban szerepelt.',
                      '', 'A Qwen JSON elkészülte nem bizonyítja, hogy a forráshivatkozásokkal '
                       'ellenőrzött válasz felhasználható vagy tartalmilag teljes. A fallback oka '
                       'különbözteti meg a modellkérés hibáját az elkészült, de használhatatlan választól.',
                       'Az `eval_count` csak akkor tényleges outputtokenszám, ha az Ollama '
                       'visszaadta. Hiányzó lezáró keretnél N/A; a karakterek száma nem tokenszám.',
                       'A `complete` kizárólag szerkezeti proxy, nem ember által ellenőrzött '
                       'ügyintézési teljesség. Az állítás és a pontos forrásidézet lexikai lefedettsége '
                       'nem bizonyítja a jogi vagy szemantikai helyességet.'])
    return '\n'.join(lines) + '\n'


# Existing internal imports remain supported.
_markdown_report = render_report_markdown


def list_saved_runs(output_root: Path) -> list[Path]:
    if not output_root.exists():
        return []
    return sorted((path for path in output_root.iterdir() if (path / 'result.json').is_file()), reverse=True)


def load_saved_run(path: Path) -> dict:
    file = path / 'result.json' if path.is_dir() else path
    return json.loads(file.read_text(encoding='utf-8'))
