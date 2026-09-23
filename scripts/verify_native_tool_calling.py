"""Run an ACTUAL Ollama tool-choice/execution/feedback trace on indexed sources.

Usage: python scripts/verify_native_tool_calling.py [--strict]
Does not download documents or open Qdrant; requires local Ollama and processed chunks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from dap_assistant.observability.telemetry import Telemetry  # noqa: E402
from dap_assistant.documents.ingestion import load_chunks  # noqa: E402
from dap_assistant.llm import OllamaAdapter  # noqa: E402
from dap_assistant.settings import Settings  # noqa: E402


def _evidence(settings: Settings) -> list[dict]:
    chunks = [c for c in load_chunks(settings.data_dir) if c.get('domain') == 'vehicle'
              and c.get('document_id') == 'dap-vehicle-buyer' and c.get('text')]
    # Grounded official chunks, with at least one candidate per requested need.
    chosen = []
    for keywords in (('adásvételi szerződés', 'forgalmi engedély', 'dokumentum'),
                     ('15 napon belül', 'eredetiségvizsgálat', 'átírat')):
        matches = [item for item in chunks if any(word in item['text'].casefold() for word in keywords)]
        chosen.extend(matches[:4])
    ids: set[str] = set()
    result = []
    for item in chosen:
        if item['chunk_id'] not in ids:
            ids.add(item['chunk_id'])
            result.append({**item, 'evidence_id': 'E_' + item['chunk_id'][:16]})
    return result[:8]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true',
                        help='Exit nonzero unless both distinct native tools execute and Ollama acknowledges their results.')
    parser.add_argument('--smoke', action='store_true',
                        help='Use two synthetic source excerpts; isolates native Ollama tools from the real document corpus.')
    args = parser.parse_args()
    settings = Settings()
    if settings.llm_provider != 'ollama':
        parser.error('LLM_PROVIDER=ollama is required for an authentic trace.')
    if args.smoke:
        # Synthetic integration fixture only. Never present as current legal advice.
        evidence = [
            {'evidence_id': 'E_aaaaaaaaaaaaaaaa', 'domain': 'vehicle',
             'text': 'Az átíráshoz szükséges az adásvételi szerződés.'},
            {'evidence_id': 'E_bbbbbbbbbbbbbbbb', 'domain': 'vehicle',
             'text': 'A gépjárművet 15 napon belül át kell íratni.'},
        ]
        question = ('Teszt: melyik dokumentum szükséges, és milyen határidő van? '
                    'Hívd meg mindkét deklarált eszközt.')
    else:
        evidence = _evidence(settings)
        if len(evidence) < 2:
            parser.error('Official vehicle buyer chunks missing; download/index documents first.')
        question = ('Használt autót vettem. Milyen dokumentumok szükségesek az átíráshoz, '
                    'és milyen 15 napos határidőket kell betartanom? Kérlek használd '
                    'mindkét rendelkezésre álló forrásellenőrző eszközt.')
    run_id = str(uuid.uuid4())
    telemetry = Telemetry()
    adapter = OllamaAdapter(settings, telemetry=telemetry)
    try:
        _, trace = adapter.call_native_tools(question, evidence, run_id=run_id)
    finally:
        adapter.close()
    # Only names, exact evidence IDs, status, and real Ollama counters are exported.
    record = {'run_id': run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'model': settings.ollama_model, 'native_tool_model': settings.native_tool_model,
              'native_tool_trace': trace, 'evidence_kind': 'synthetic_smoke' if args.smoke else 'official_chunks',
              'ollama_usage': telemetry.snapshot(run_id)['llm_usage']}
    folder = ROOT / 'reports' / 'native_tool_calls'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{run_id}.json'
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'trace': trace, 'report': str(target)}, indent=2, ensure_ascii=False))
    diagnostic = trace.get('selection_diagnostic') or trace.get('ack_diagnostic')
    if diagnostic:
        stage = diagnostic.get('failure_stage')
        if stage == 'generation_length_limit':
            if trace.get('ack_diagnostic') is diagnostic:
                hint = ('A két natív toolhívás már megtörtént, de az eredmények utáni '
                        'visszaigazolás a generálási korlátnál megszakadt; '
                        'ellenőrizd az ack_diagnostic mezőt. Ez nem toolválasztási hiba.')
            else:
                hint = ('A natív toolválasztás elérte a generálási korlátot; '
                        'ellenőrizd a selection_diagnostic thinking_chars és '
                        'tool_calls_observed mezőit.')
        elif stage == 'http_status':
            if diagnostic.get('http_error_category') == 'model_unavailable':
                hint = f'A natív modell nincs telepítve. Futtasd: ollama pull {settings.native_tool_model}'
            else:
                hint = 'Ollama HTTP-hiba: ellenőrizd a http_status és http_error_category mezőt; a szerver nyers hibaszövegét ne oszd meg.'
        elif stage in ('transport', 'stream_read', 'total_timeout'):
            if diagnostic.get('exception_type') in ('ReadTimeout', 'TimeoutError'):
                hint = ('Natív tool-hívás időtúllépés; ez nem bizonyítja, hogy az Ollama nem fut. '
                        'Ellenőrizd a streamed_frames/first_frame_s mezőket, az `ollama ps` kimenetet, '
                        'majd szükség esetén NATIVE_TOOL_READ_TIMEOUT_S=240 és '
                        'NATIVE_TOOL_TOTAL_TIMEOUT_S=360 beállítással próbáld újra.')
            else:
                hint = 'Ollama-kapcsolat vagy streaming válasz megszakadt; ellenőrizd a diagnosztikát.'
        else:
            hint = 'Ollama válaszformátum-hiba: ellenőrizd a failure_stage és exception_type mezőt.'
        print(f'[NATIVE-TOOL DIAGNOSTIC] {hint}', file=sys.stderr)
    if trace.get('ack_stream', {}).get('truncated'):
        print('[NATIVE-TOOL DIAGNOSTIC] A tooleredmények Ollamához visszakerültek, '
              'de a visszaigazoló válasz csonkolódott; ez nem teljes válaszgenerálás.',
              file=sys.stderr)
    called = {call['tool_name'] for call in trace['calls']
              if call['status'] == 'executed' and call['returned_to_model']}
    if args.strict and (called != {'get_document_checklist', 'get_deadline_mentions'}
                        or not trace['result_returned_to_model']):
        print('Two distinct native tools were NOT verified in this real run; inspect the trace.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
