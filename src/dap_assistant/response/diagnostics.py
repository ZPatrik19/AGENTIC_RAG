"""Report observed Ollama token metadata, distinguishing requested context from measured usage."""
from __future__ import annotations

from math import isfinite
import json


PHASES = {
    'answer': 'Végső válasz',
    'native_tool_selection': 'Natív eszközválasztás',
    'native_tool_result_ack': 'Eszközeredmény nyugtázása',
    'selection': 'Bizonyítékkiválasztás',
    'reasoning': 'Modellkérés',
}


def _number(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _seconds(ns: object) -> float | None:
    if isinstance(ns, (float, int)) and not isinstance(ns, bool) and isfinite(ns) and ns >= 0:
        return round(ns / 1_000_000_000, 2)
    return None


def _think_label(request: dict | None, phase: str, native_model: str | None) -> str:
    if phase in ('native_tool_selection', 'native_tool_result_ack'):
        if isinstance(native_model, str) and native_model:
            return ('Nincs think paraméter (instruct)' if 'instruct' in native_model.casefold()
                    else 'Kikapcsolva kérve (think=false)')
        return 'Nem ismert (natív toolhívás)'
    if not isinstance(request, dict):
        return 'Nem áll rendelkezésre a kérés'
    # Presence and value matter: omitted does NOT imply that thinking was disabled.
    if 'think' not in request or request['think'] is None:
        # Telemetry prompt_preview uses None for an omitted think parameter.
        return 'Nincs think paraméter (modellfüggő)'
    if request['think'] is False:
        return 'Kikapcsolva kérve (think=false)'
    if request['think'] is True:
        return 'Bekapcsolva kérve (think=true)'
    return 'Think opció: ' + str(request['think'])[:32]


def _request_evidence_ids(request: dict | None) -> list[str] | None:
    """Inspect submitted user JSON; only keep evidence IDs, never its text."""
    if not isinstance(request, dict):
        return None
    for message in request.get('messages') or []:
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        text = message.get('content')
        if not isinstance(text, str) or len(text) > 256_000:
            continue
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get('evidence'), list):
            return list(dict.fromkeys(str(item['evidence_id']) for item in payload['evidence']
                        if isinstance(item, dict) and isinstance(item.get('evidence_id'), str)
                        and item['evidence_id'].startswith('E_')))[:64]
    return None


def _failure_label(failure: dict) -> str:
    return {
        'input_context_budget': 'Bemeneti kontextuskeret túllépése',
        'output_token_limit': 'Kimeneti tokenkorlát',
        'total_timeout': 'Teljes időkorlát',
        'http_timeout': 'Olvasási időtúllépés',
        'stream_interrupted_or_incomplete': 'Megszakadt adatfolyam',
        'json_syntax': 'Hibás JSON',
        'schema_validation': 'Sémaellenőrzési hiba',
        'native_tool_ack_truncated_after_receipt': 'Levágott eszköznyugtázás',
    }.get(failure.get('failure_kind'), failure.get('failure_kind') or failure.get('type') or 'Nem ismert hiba')


def token_diagnostics(trace: dict, prompts: list[dict], final: dict) -> dict:
    """Match measured chat usage to call order; label previews and unknown counts accurately."""
    usage = [x for x in trace.get('llm_usage', []) if isinstance(x, dict)]
    failures = [x for x in trace.get('llm_failures', []) if isinstance(x, dict)]
    preview = [x for x in prompts if isinstance(x, dict)]
    prompt_by_phase: dict[str, list[dict]] = {}
    for item in preview:
        prompt_by_phase.setdefault(str(item.get('phase')), []).append(item)
    calls_by_phase: dict[str, int] = {}
    native = final.get('native_tool_trace') or trace.get('native_tool_trace') or {}
    native_model = native.get('model')
    streams = {
        'native_tool_selection': native.get('selection_stream') or {},
        'native_tool_result_ack': native.get('ack_stream') or {},
    }
    entries: list[dict] = []
    failures_used: set[int] = set()
    for item in usage:
        phase = str(item.get('phase') or 'unspecified')
        index = calls_by_phase.get(phase, 0)
        calls_by_phase[phase] = index + 1
        candidates = prompt_by_phase.get(phase) or []
        request = candidates[min(index, len(candidates) - 1)] if candidates else None
        # Native tool prompts are intentionally NOT recorded; the tool metadata
        # includes only model, requested budgets and observed thinking characters.
        requested_ctx = _number(item.get('requested_num_ctx'))
        requested_predict = _number(item.get('requested_num_predict'))
        prompt_count = _number(item.get('prompt_eval_count'))
        generated = _number(item.get('eval_count'))
        done = item.get('done_reason') if isinstance(item.get('done_reason'), str) else None
        matched_failure = None
        for failure_index, failure in enumerate(failures):
            if failure_index in failures_used or failure.get('phase') != phase:
                continue
            f_predict = _number(failure.get('requested_num_predict'))
            f_done = failure.get('done_reason')
            if (f_predict is not None and f_predict != requested_predict) or (f_done is not None and f_done != done):
                continue
            if failure.get('failure_kind') == 'output_token_limit' and done != 'length':
                continue
            matched_failure = failure
            failures_used.add(failure_index)
            break
        stream = streams.get(phase, {})
        thinking_chars = _number(item.get('observed_thinking_chars'))
        if thinking_chars is None:
            thinking_chars = _number(stream.get('thinking_chars'))
        request_evidence_ids = _request_evidence_ids(request) if phase == 'answer' else None
        ratio = (prompt_count / requested_ctx if prompt_count is not None and requested_ctx else None)
        output_ratio = (generated / requested_predict if generated is not None and requested_predict else None)
        if matched_failure:
            status = _failure_label(matched_failure)
        elif done == 'length':
            status = ('Eszköznyugtázás levágva' if phase == 'native_tool_result_ack'
                      else 'Kimeneti tokenkorlát')
        elif done == 'stop':
            status = 'Természetes leállás (nem minőségi értékelés)'
        else:
            status = 'Nincs lezárási metaadat'
        entries.append({
            'phase': phase, 'label': PHASES.get(phase, phase),
            'model': (item.get('model') or native_model if phase.startswith('native_tool_')
                      else item.get('model')) or (request or {}).get('model'),
            'requested_num_ctx': requested_ctx,
            'requested_num_predict': requested_predict,
            'prompt_eval_count': prompt_count,
            'eval_count': generated,
            'done_reason': done,
            'think': _think_label(request, phase, native_model),
            'observed_thinking_chars': thinking_chars,
            'observed_thinking_tokens': None,  # Ollama does not separately return this count.
            'request_evidence_count': (len(request_evidence_ids) if request_evidence_ids is not None else None),
            'request_evidence_ids': request_evidence_ids,
            'requested_window_ratio': round(ratio, 4) if ratio is not None else None,
            'output_budget_ratio': round(output_ratio, 4) if output_ratio is not None else None,
            'elapsed_s': _seconds(item.get('total_duration')),
            'status': status,
            'usage_source': item.get('usage_source'),
            'complete_ollama_metadata': bool(item.get('done') and done),
        })
    # A timeout can have partial streamed bytes but no final usage frame; do not
    # silently drop that attempt or fabricate its token counts.
    for i, failure in enumerate(failures):
        if i in failures_used:
            continue
        phase = str(failure.get('phase') or 'unspecified')
        requested_ctx = _number(failure.get('requested_num_ctx'))
        requested_predict = _number(failure.get('requested_num_predict'))
        entries.append({
            'phase': phase, 'label': PHASES.get(phase, phase), 'model': None,
            'requested_num_ctx': requested_ctx,
            'requested_num_predict': requested_predict,
            'prompt_eval_count': _number(failure.get('prompt_eval_count')),
            'eval_count': _number(failure.get('eval_count')),
            'done_reason': failure.get('done_reason'),
            'think': _think_label(None, phase, native_model),
            'observed_thinking_chars': _number(failure.get('observed_thinking_chars')),
            'observed_thinking_tokens': None,
            'request_evidence_count': None, 'request_evidence_ids': None,
            'requested_window_ratio': None, 'output_budget_ratio': None,
            'elapsed_s': (round(failure['elapsed_s'], 2)
                          if isinstance(failure.get('elapsed_s'), (int, float))
                          and isfinite(failure['elapsed_s']) and failure['elapsed_s'] >= 0 else None),
            'status': _failure_label(failure),
            'usage_source': failure.get('usage_source'),
            'complete_ollama_metadata': False,
        })
    answers = [x for x in entries if x['phase'] == 'answer']
    actual_answer_responses = [x for x in usage if x.get('phase') == 'answer']
    generated = final.get('answer_context', {}).get('generation_diagnostics') or {}
    if not answers:
        outcome = 'Nincs válaszgeneráló Ollama-hívás (vagy nem érkezett róla mérés).'
    elif any(x['done_reason'] == 'length' for x in answers):
        if (answers[-1]['done_reason'] == 'stop' and not final.get('answer_fallback')
                and generated.get('model_claim_count', 0) > 0):
            outcome = 'Az első válasz tokenkorlátos volt; a helyreállító kérés természetesen lezárult.'
        else:
            outcome = 'Válaszgenerálás közben kimeneti tokenkorlátot ért el a modell.'
    elif any(x['status'] in ('Teljes időkorlát', 'Olvasási időtúllépés', 'Megszakadt adatfolyam')
             for x in answers):
        outcome = 'A modellkérés megszakadt vagy időtúllépést ért el; a tokenszám hiányozhat.'
    elif answers[-1]['done_reason'] == 'stop':
        if final.get('answer_fallback') or generated.get('status') == 'completed_but_unusable':
            outcome = 'A generálás természetesen lezárult, de nem született használható végső modellválasz.'
        elif generated.get('uncovered_facets_proxy'):
            outcome = 'A generálás természetesen lezárult; egyes kért témák lefedettsége bizonytalan.'
        else:
            outcome = 'A generálás természetesen lezárult; a tartalmi teljesség külön ellenőrzendő.'
    else:
        outcome = 'A válasz lezárási oka nem igazolható a rendelkezésre álló metaadatokból.'
    # If the request finished without a final Ollama frame, usage stays unknown.
    if not actual_answer_responses and answers:
        outcome += ' Nincs hiteles Ollama-végkeret az érintett kéréshez.'
    return {'calls': entries, 'answer_outcome': outcome,
            'request_budget': trace.get('selection', {}).get('answer_context_budget') or
                              trace.get('selection', {}).get('answer_request_budget') or {},
            'context_engineering': final.get('context_engineering') or
                                   final.get('answer_context', {}).get('context_engineering') or {},
            'generation_diagnostics': generated,
            'notes': (
                'num_ctx/num_predict: a kérésben megadott keret, nem a szerver által igazolt effektív ablak. '
                'prompt_eval_count/eval_count: kizárólag a szerver válaszából származó tokenszám. '
                'A prompt/num_ctx arány csak a kért ablak kihasználtságának jelzése; nem bizonyítja, '
                'hogy minden eredeti forrásrészt feldolgozott a modell. A forrásrészletek száma '
                'az összeállított prompt JSON-jából származik, de nem bizonyítja a modell általi '
                'megértést vagy az Ollama esetleges belső csonkolásának hiányát. '
                'A thinking karakterek nem thinking tokenek; külön thinking tokenszám nem ismert.'),
    }
