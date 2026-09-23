"""Pure presentation/observability helpers shared by Streamlit and offline tests."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone

LABELS = {
    'vehicle': 'Autóvásárlás vagy -eladás',
    'employment': 'Munkahely elvesztése',
}


def available_life_events(chunks: list[dict]) -> list[str]:
    """Present each indexed life event once, in a stable human-friendly order."""
    available = {chunk.get('domain') for chunk in chunks}
    return [LABELS[domain] for domain in LABELS if domain in available]

ROLE_LABELS = {'buyer': 'Vevő', 'seller': 'Eladó', '': 'Nincs pontosítva'}
STRATEGY_LABELS = {
    'quick': 'Gyors Qwen · információigényes forrásválasztás + természetes válaszgenerálás',
    'source': 'Forrásalapú · LLM nélkül',
    'detailed': 'Részletes Qwen · strukturált válasz',
    'timeout_source_fallback': 'Biztonságos forrásalapú tartalék válasz',
    'model_error_source_fallback': 'Forrásalapú válasz sikertelen modellkérés után',
    'source_fallback_unusable_model': 'Forrásalapú válasz: a modell állításai nem voltak felhasználhatók',
    'dummy': 'Tesztmód · determinisztikus',
    'verified_fee_extract': 'Forrásból kiemelt díj · külön LLM-hívás nélkül',
}

STAGES = {
    'classify_intent': 'Élethelyzet felismerése',
    'clarify_query': 'Pontosítás szükséges',
    'plan_tasks': 'Feladatok és függőségek megtervezése',
    'process_query': 'Keresőkérdés előkészítése',
    'hybrid_retrieval': 'Hivatalos dokumentumok keresése',
    'rerank_results': 'Találatok rangsorolása',
    'evaluate_evidence': 'Források ellenőrzése',
    'prepare_context': 'Forrásrészletek előkészítése',
    'rag_worker': 'Dokumentumkeresési ág lezárása',
    'evidence_gate': 'Bizonyítékok és feladatok összevezetése',
    'engineer_context': 'Kontextus és részkérdés-lefedettség ellenőrzése',
    'execute_tools': 'Szükséges eszközök futtatása',
    'generate_answer': 'Válasz összeállítása helyi modellel',
    'answer_audit': 'Hivatkozások ellenőrzése',
    'safe_response': 'Biztonságos részleges válasz',
}


# Exact node names from Telemetry.measure in workflow.py and rag_graph.py.
# These are not inferred from timestamps of Streamlit update messages.
_PROGRESS_STEPS = (
    ('classify_intent', 'Élethelyzet azonosítása', 'main/classify_intent'),
    ('plan_tasks', 'Önálló feladatágak kiosztása', 'main/plan_tasks'),
    ('process_query', 'Keresőkérdés előkészítése', 'rag/process_query'),
    ('hybrid_retrieval', 'BM25 és vektoros keresés', 'rag/hybrid_retrieval'),
    ('rerank_results', 'Találatok újrarangsorolása', 'rag/rerank_results'),
    ('evaluate_evidence', 'RAG-bizonyítékok előzetes vizsgálata', 'rag/evaluate_evidence'),
    ('prepare_context', 'RAG-forrásrészletek összeállítása', 'rag/prepare_context'),
    ('evidence_gate', 'RAG-ágak eredményeinek egyesítése', 'main/evidence_gate'),
    ('engineer_context', 'Források kiválasztása a válaszhoz', 'main/engineer_context'),
    ('execute_tools', 'Szükséges számítások és eszközök', 'main/execute_tools'),
    ('generate_answer', 'Hivatkozott válasz generálása', 'main/generate_answer'),
    ('answer_audit', 'Válasz és forráskapcsolatok vizsgálata', 'main/answer_audit'),
)


def duration_label(seconds: float) -> str:
    """Human-readable measured seconds without displaying a small span as 0.0 s."""
    if seconds < 0.001:
        return '< 0,001 s'
    digits = 3 if seconds < 0.01 else 2 if seconds < 0.1 else 1
    return f'{seconds:.{digits}f}'.replace('.', ',') + ' s'


def execution_steps(events: list[dict], trace: dict | None = None) -> list[dict]:
    """Show completed nodes and observed durations; concurrent work must not be summed as wall time."""
    trace = trace or {}
    by_node: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for event in events:
        node = event.get('node')
        counts[node] = counts.get(node, 0) + 1
        by_node[node] = event
    measured: dict[str, list[float]] = {}
    for span in trace.get('spans', []):
        if not isinstance(span, dict):
            continue
        name, duration = span.get('name'), span.get('duration_s')
        if (isinstance(name, str) and isinstance(duration, (int, float))
                and not isinstance(duration, bool) and math.isfinite(duration)
                and duration >= 0):
            measured.setdefault(name, []).append(float(duration))
    output = []
    for node, label, span_name in _PROGRESS_STEPS:
        if node not in by_node:
            continue
        event = by_node[node]
        parts: list[str] = []
        if node == 'classify_intent':
            context = event.get('question_analysis') or {}
            goal_labels = {
                'procedure': 'teendők', 'benefit_amount': 'összeg',
                'eligibility': 'jogosultság', 'cost': 'költség',
                'deadline': 'határidő', 'documents': 'dokumentumok',
                'where': 'ügyintézés helye', 'supports': 'támogatások',
            }
            stage_labels = {'after_event': 'megtörtént esemény',
                            'planning': 'tervezés'}
            if context.get('goal') in goal_labels:
                parts.append('Cél: ' + goal_labels[context['goal']])
            if context.get('stage') in stage_labels:
                parts.append(stage_labels[context['stage']])
        elif node == 'plan_tasks':
            count = event.get('task_count')
            if isinstance(count, int) and count > 0:
                parts.append(f'{count} keresési feladat')
        elif node == 'evidence_gate':
            docs, count = event.get('documents', []), event.get('evidence_count')
            if docs and isinstance(count, int):
                parts.append(f'{len(docs)} dokumentum · {count} forrásrészlet')
        elif node == 'engineer_context':
            count = event.get('context_candidate_chunks')
            missing = event.get('missing_retrieved_facets') or []
            if isinstance(count, int):
                parts.append(f'{count} kontextusjelölt')
            if missing:
                parts.append(f'{len(missing)} résztéma forrás nélkül')
        elif node == 'execute_tools':
            native = event.get('native_executed_tool_count', 0)
            if native:
                parts.append(f'{native} végrehajtott natív eszköz')
            elif event.get('tools'):
                parts.append(f"{len(event['tools'])} helyi eszközeredmény")
        elif node == 'answer_audit' and event.get('validation') == 'partial':
            parts.append('részleges eredmény')
        spans = measured.get(span_name, [])
        seconds = sum(spans) if spans else None
        if seconds is not None:
            parts.insert(0, duration_label(seconds))
        if counts[node] > 1:
            parts.append(f'{counts[node]} lezárt futás')
        output.append({'node': node, 'label': label, 'details': ' · '.join(parts),
                       'seconds': seconds, 'measured_runs': len(spans),
                       'event_runs': counts[node], 'validation': event.get('validation')})
    return output


def progress_milestones(events: list[dict], trace: dict | None = None) -> list[str]:
    """Up to eight real completed stages; attach times only if measured."""
    return [f"✓ **{row['label']}**" + (f" · {row['details']}" if row['details'] else '')
            for row in execution_steps(events, trace)]



def grouped_progress_milestones(events: list[dict], trace: dict | None = None) -> list[str]:
    """Backward-compatible public entry point for Streamlit's progress view."""
    from .ui_progress import grouped_progress_milestones as render_progress

    return render_progress(events, trace)


def public_answer_text(raw: str) -> str:
    """Hide diagnostic labels in chat only; keep exact provenance in exports."""
    visible = re.sub(r'(?m)(?<=\s)\[C_[0-9a-f]{12,}\][ \t]*', '', raw)
    # A claim may start immediately after a heading or a list marker.
    visible = re.sub(r'(?m)^([ \t]*(?:\d+\.|-)[ \t]*)\[C_[0-9a-f]{12,}\][ \t]*',
                     r'\1', visible)
    visible = re.sub(r'\s*\*\((?:Qwen megfogalmazása|utólagos forráskiegészítés|'
                     r'eszközből ellenőrzött, szó szerinti forráskivonat|eszközből származó, szó szerinti forrásrészlet)\)\*', '', visible)
    visible = re.sub(r'\[(?:NAV-forrás|Forrás) E_[0-9a-f]{12,}\]\((<https://[^\s<>]+>)\)',
                     r'[Hivatalos forrás](\1)', visible)
    return visible


def event_from_update(namespace: tuple, node: str, delta: dict) -> dict:
    """Derive UI information exclusively from real LangGraph update payloads."""
    event = {
        'origin': ' / '.join(namespace) if namespace else 'main',
        'node': node, 'label': STAGES.get(node, node),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    if 'domains' in delta:
        event['domains'] = list(delta['domains'])
    if 'classification_status' in delta:
        event['classification_status'] = delta['classification_status']
    if 'question_analysis' in delta:
        # Store safe, bounded tags only; never echo the question into trace.
        allowed = ('domain', 'role', 'stage', 'goal', 'scope', 'requested_needs', 'preferred_channel', 'method')
        event['question_analysis'] = {key: delta['question_analysis'][key]
                                      for key in allowed if key in delta['question_analysis']}
    if 'context_resolution' in delta:
        # Metadata only: do not copy the user's question into the default trace export.
        event['context_resolution'] = dict(delta['context_resolution'])
    if 'role' in delta:
        event['role'] = delta['role']
    if 'stage' in delta:
        event['stage'] = delta['stage']
    if 'subtasks' in delta:
        event['task_count'] = len(delta['subtasks'])
        event['subtasks'] = [{
            'task_id': key,
            'domain': value.get('domain', ''),
            'question': value.get('question', ''),
            'depends_on': value.get('depends_on', []),
            'status': value.get('status', ''),
        } for key, value in delta['subtasks'].items()]
    if 'evidence' in delta:
        unique_documents = {}
        for item in delta['evidence']:
            document_id = item.get('document_id') or item.get('source_url', '')
            unique_documents.setdefault(document_id, (
                document_id, item.get('title', ''), item.get('source_url', ''),
            ))
        event['documents'] = list(unique_documents.values())
        event['evidence_count'] = len(delta['evidence'])
    if 'context_engineering' in delta:
        plan = delta['context_engineering']
        event['context_candidate_chunks'] = plan.get('context_candidate_chunks')
        event['missing_retrieved_facets'] = list(plan.get('missing_retrieved_facets', []))
    if isinstance(delta.get('context_evidence'), list):
        event['context_selected_count'] = len(delta['context_evidence'])
    if 'native_tool_trace' in delta:
        calls = delta['native_tool_trace'].get('calls', [])
        event['native_executed_tool_count'] = sum(
            isinstance(call, dict) and call.get('status') == 'executed'
            and call.get('returned_to_model') is True for call in calls)
        event['native_attempted_tool_count'] = len(calls)
    if 'tool_results' in delta:
        event['tools'] = list(delta['tool_results'])
        # The progress display needs only local result status, not user data,
        # calculation inputs, amounts or full tool output.
        event['local_tool_statuses'] = {
            key: value.get('status', '') if isinstance(value, dict) else ''
            for key, value in delta['tool_results'].items()
        }
    if 'answer_validation' in delta:
        event['validation'] = delta['answer_validation'].get('status')
    if 'retrieval_status' in delta:
        event['retrieval_status'] = delta['retrieval_status']
    if 'answer_strategy' in delta:
        event['answer_strategy'] = delta['answer_strategy']
    return event


def active_components(trace: dict) -> list[dict]:
    """Exclusive leaf/node costs, not double-counted parent durations."""
    spans = trace.get('spans', [])
    totals: dict[str, float] = {}
    for span in spans:
        name = span['name']
        # A nested parent span is deliberately excluded from exclusive ranking.
        if name in ('main/rag_worker', 'rag_subgraph', 'main/generate_answer',
                    'rag/hybrid_retrieval'):
            continue
        totals[name] = totals.get(name, 0.0) + span['duration_s']
    return [{'component': key, 'seconds': round(value, 4)}
            for key, value in sorted(totals.items(), key=lambda item: -item[1])]


def run_export(*, events: list[dict], trace: dict, final: dict, elapsed_s: float) -> dict:
    """No raw prompt or user question in default export; explicit prompt download is separate."""
    return {
        'schema_version': 1,
        'elapsed_s': round(elapsed_s, 3),
        'domains': final.get('domains', []),
        'task_statuses': {k: v.get('status') for k, v in final.get('subtasks', {}).items()},
        'response_status': final.get('response_status'),
        'answer_strategy': final.get('answer_strategy'),
        'answer_validation': final.get('answer_validation', {}),
        'events': events, 'telemetry': trace,
    }


def prompt_export(preview: list[dict]) -> str:
    """Explicitly downloaded only; may contain user-supplied personal information."""
    return json.dumps({'warning': 'May contain user text; handle as personal data.',
                       'model_requests': preview}, ensure_ascii=False, indent=2)


def final_prompt_view(final: dict, prompts: list[dict]) -> dict:
    """Show the *real* final request, or explicitly label a local non-LLM brief.

    An Ollama timeout must not hide the exact request that was already sent.
    For source/dummy modes no model prompt is invented or presented as sent.
    """
    sent = next((p for p in reversed(prompts) if p.get('phase') == 'answer'), None)
    if sent:
        lines = [f"MODEL: {sent.get('model', '?')}",
                 f"THINKING: {sent.get('think')}",
                 'OUTPUT FORMAT: JSON schema (see technical prompt view)']
        for message in sent.get('messages', []):
            lines.extend(('', message.get('role', 'unknown').upper() + ':', message.get('content', '')))
        return {'submitted': True, 'text': '\n'.join(lines),
                'label': 'Az Ollamának ténylegesen elküldött utolsó válaszprompt'}

    # A selection request is a genuine Ollama call even when it timed out before
    # the separate generation request. Never label it as an unsent local brief.
    selected = next((p for p in reversed(prompts) if p.get('phase') == 'selection'), None)
    if selected:
        lines = [f"MODEL: {selected.get('model', '?')}",
                 f"THINKING: {selected.get('think')}",
                 'PHASE: evidence selection (generation was NOT submitted)',
                 'OUTPUT FORMAT: JSON schema (see technical prompt view)']
        for message in selected.get('messages', []):
            lines.extend(('', message.get('role', 'unknown').upper() + ':', message.get('content', '')))
        return {'submitted': True, 'text': '\n'.join(lines),
                'label': 'Az Ollamának ténylegesen elküldött bizonyítékkiválasztási prompt; válaszgenerálás nem indult'}

    if final.get('classification_status') in ('unsupported', 'needs_clarification'):
        return {
            'submitted': False,
            'text': 'NINCS OLLAMA-PROMPT — a kérdés nem tartozik a támogatott élethelyzetekhez '
                    'vagy a kontextus pontosítása szükséges. Nem történt modellhívás vagy dokumentumkeresés.',
            'label': 'A feltételes domain routing lezárta a kérést modellhívás nélkül',
        }

    context = final.get('answer_context', {})
    evidence = final.get('evidence', [])
    lines = [
        'HELYI ÖSSZEÁLLÍTÁSI UTASÍTÁS — NEM KÜLDTÜK EL MODELLNEK',
        'Feladat: készíts kizárólag a forrásokkal alátámasztható magyar ügyintézési kivonatot.',
        f"Élethelyzetek: {', '.join(context.get('life_events', final.get('domains', [])))}",
        f"Szerepkör: {context.get('role', final.get('role', 'not_specified'))}",
        f"Stratégia: {context.get('answer_strategy', final.get('answer_strategy', 'source'))}",
        f"Referencia-dátum: {context.get('reference_date', final.get('reference_date', ''))}",
        f"Esemény dátuma: {context.get('event_date') or 'nincs megerősítve'}",
        f"Esemény dátuma megerősítve: {context.get('event_date_confirmed', False)}",
        f"Kérdés: {final.get('user_question', '')}",
        'Források:',
    ]
    lines.extend(f"- {e.get('title', '')} ({e.get('evidence_id', '')}): {e.get('source_url', '')}"
                 for e in evidence)
    return {'submitted': False, 'text': '\n'.join(lines),
            'label': 'Forrásalapú válasz összeállítási utasítása (nem LLM-hívás)'}


def ollama_status(health: dict, model: str) -> dict:
    """Human-readable operational status; avoid exposing raw API JSON in the UI."""
    if not health.get('available'):
        return {'level': 'error', 'title': 'Ollama nem érhető el',
                'detail': 'Indítsd el az Ollamát, majd ismételd meg az ellenőrzést.'}
    if not health.get('configured_model_found'):
        return {'level': 'warning', 'title': 'Ollama elérhető, a modell hiányzik',
                'detail': f'A szükséges modell: {model}. Telepítés: ollama pull {model}'}
    return {'level': 'success', 'title': 'Ollama elérhető · modell telepítve',
            'detail': f'Kiválasztott helyi modell: {model}. Nem kell újratelepíteni.'}
