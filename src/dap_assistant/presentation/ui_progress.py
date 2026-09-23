"""Detailed user-facing progress; renders only observed LangGraph updates/spans.

Keep presentation here, separate from the workflow, telemetry and export model.
No model calls, filesystem reads or Streamlit imports.
"""
from __future__ import annotations

import math
import re

from .runtime_view import LABELS, ROLE_LABELS, duration_label, execution_steps


# These labels are for the human-facing Streamlit progress panel. They do not
# rename actual LangGraph nodes, spans, benchmark keys or exported events.
_PROGRESS_LABELS = {
    'classify_intent': 'Élethelyzet és kérési cél felismerése',
    'plan_tasks': 'Részfeladatok és függőségek megtervezése',
    'process_query': 'Keresőkifejezések előkészítése',
    'hybrid_retrieval': 'Dokumentumtalálatok visszakeresése',
    'rerank_results': 'Találatok újrarangsorolása',
    'evaluate_evidence': 'Forráslefedettség előzetes ellenőrzése',
    'prepare_context': 'RAG-találatok átadása a fő folyamatnak',
    'evidence_gate': 'Keresési ágak eredményeinek összesítése',
    'engineer_context': 'Válaszhoz szükséges források előkészítése',
    'execute_tools': 'Kalkulátorok és segédeszközök kezelése',
    'generate_answer': 'Válasz előállítása a forrásokból',
    'answer_audit': 'Állítások és hivatkozások ellenőrzése',
}

_PROGRESS_GROUPS = (
    ('A kérés értelmezése', ('classify_intent',)),
    ('Részfeladatok megtervezése', ('plan_tasks',)),
    ('Dokumentumok felkutatása',
     ('process_query', 'hybrid_retrieval', 'rerank_results',
      'evaluate_evidence', 'prepare_context', 'evidence_gate')),
    ('Bizonyítékok előkészítése', ('engineer_context',)),
    ('Kalkulátorok és eszközök', ('execute_tools',)),
    ('Válaszalkotás és ellenőrzés', ('generate_answer', 'answer_audit')),
)

_TOOL_LABELS = {
    'checklist': 'Iratlista',
    'vehicle_duty': 'Gépjármű-illeték',
    'unemployment_benefit': 'Álláskeresési járadék',
    'illustrative_loan': 'Szemléltető törlesztő',
}


def _progress_detail(node: str, event: dict) -> list[str]:
    """Short, safe UI hints from recorded metadata, never from user text."""
    if node == 'classify_intent':
        result = []
        domains = event.get('domains') or []
        known_domains = [LABELS[d] for d in domains if d in LABELS]
        if known_domains:
            result.append(', '.join(known_domains))
        role = event.get('role') or (event.get('question_analysis') or {}).get('role')
        if role in ('seller', 'buyer'):
            result.append(ROLE_LABELS[role])
        goal = (event.get('question_analysis') or {}).get('goal')
        goals = {
            'procedure': 'ügyintézési teendők', 'benefit_amount': 'ellátás összege',
            'eligibility': 'jogosultsági feltételek', 'cost': 'költségek',
            'deadline': 'határidők', 'documents': 'szükséges iratok',
            'where': 'ügyintézés helye', 'supports': 'támogatások',
        }
        if goal in goals:
            result.append('Cél: ' + goals[goal])
        stage = (event.get('question_analysis') or {}).get('stage') or event.get('stage')
        if stage in ('after_event', 'planning'):
            result.append('megtörténtként értelmezve' if stage == 'after_event'
                          else 'tervezett esemény')
        return result
    if node == 'plan_tasks':
        count = event.get('task_count')
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            return []
        result = [f'{count} tervezett feladatág']
        tasks = event.get('subtasks') or []
        if len(tasks) == count:
            independent = sum(not task.get('depends_on') for task in tasks if isinstance(task, dict))
            if independent > 1:
                result.append(f'{independent} függőség nélküli ág')
            elif independent == 1 and count > 1:
                result.append('függőségekkel ütemezve')
        return result
    if node == 'evidence_gate':
        documents = event.get('documents') or []
        evidence_count = event.get('evidence_count')
        if isinstance(evidence_count, int) and not isinstance(evidence_count, bool):
            return [f'{len(documents)} dokumentum · {evidence_count} összegyűjtött részlet']
        return []
    if node == 'engineer_context':
        result = []
        selected = event.get('context_selected_count')
        candidates = event.get('context_candidate_chunks')
        if isinstance(selected, int) and not isinstance(selected, bool):
            result.append(f'{selected} kontextushoz kiválasztott részlet')
        elif isinstance(candidates, int) and not isinstance(candidates, bool):
            result.append(f'{candidates} kontextusjelölt')
        missing = event.get('missing_retrieved_facets') or []
        if missing:
            result.append(f'{len(missing)} témához nincs lexikai forrásegyezés')
        return result
    if node == 'execute_tools':
        statuses = event.get('local_tool_statuses') or {}
        labels = []
        for key in statuses:
            # Explicit allowlist; dynamic tool keys and arguments never go to Markdown.
            label = (_TOOL_LABELS.get(key, 'Határidő-ellenőrzés' if
                   re.fullmatch(r'deadline_\d+', key) else '')
                   if isinstance(key, str) else '')
            if label and label not in labels:
                labels.append(label)
        result = [', '.join(labels)] if labels else []
        done = sum(status in ('success', 'calculated', 'ok') for status in statuses.values())
        incomplete = sum(status in ('incomplete', 'needs_user_input', 'unavailable', 'error')
                         for status in statuses.values())
        if done:
            result.append(f'{done} sikeres helyi eszközeredmény')
        if incomplete:
            result.append(f'{incomplete} további adatot igénylő / nem számítható eredmény')
        native = event.get('native_executed_tool_count', 0)
        if native:
            result.append(f'{native} sikeres natív eszközhívás')
        attempted = event.get('native_attempted_tool_count', 0)
        if attempted > native:
            result.append(f'{attempted - native} nem sikeres / nem átadott natív hívás')
        if not statuses and not attempted:
            result.append('nem volt szükség külön eszközre')
        return result
    if node == 'generate_answer':
        strategy = event.get('answer_strategy')
        if strategy in ('source', 'dummy'):
            return ['helyi, modell nélküli válasz']
        if isinstance(strategy, str) and 'fallback' in strategy:
            return ['forrásalapú tartalékválasz']
        if strategy in ('quick', 'detailed'):
            return ['helyi LLM']
        return []
    if node == 'answer_audit':
        status = event.get('validation')
        if status == 'partial':
            return ['részleges eredmény']
        if status == 'failed':
            return ['nem sikerült ellenőrizhető választ összeállítani']
    return []


def _measured_spans(trace: dict) -> dict[str, list[float]]:
    """Accept only completed, finite, nonnegative measured intervals."""
    measured: dict[str, list[float]] = {}
    for span in trace.get('spans', []):
        if not isinstance(span, dict):
            continue
        name, duration = span.get('name'), span.get('duration_s')
        if (isinstance(name, str) and isinstance(duration, (int, float))
                and not isinstance(duration, bool) and math.isfinite(duration)
                and duration >= 0):
            measured.setdefault(name, []).append(float(duration))
    return measured


def _rag_span_rows(events: list[dict], trace: dict, existing: set[str]) -> list[dict]:
    """Display observed RAG node spans, including inner nodes absent from graph updates."""
    if not any(event.get('node') in ('rag_worker', 'evidence_gate') for event in events):
        return []
    measured = _measured_spans(trace)
    rows = []
    for node in ('process_query', 'hybrid_retrieval', 'rerank_results',
                 'evaluate_evidence', 'prepare_context'):
        spans = measured.get('rag/' + node, [])
        if node in existing or not spans:
            continue
        details = [duration_label(sum(spans))]
        if len(spans) > 1:
            details.append(f'{len(spans)} mért végrehajtás')
        rows.append({'node': node, 'details': ' · '.join(details),
                     'seconds': sum(spans), 'measured_runs': len(spans)})
    return rows


def _retrieval_breakdown(trace: dict) -> str:
    """Render measured retrieval components; absent dense search stays absent."""
    lookup = (
        ('bm25_retrieval', 'BM25'),
        ('embedding_query', 'Embedding'),
        ('dense_retrieval', 'Vektoros keresés'),
        ('retrieval_fusion', 'RRF összevezetés'),
    )
    measured = _measured_spans(trace)
    return ' · '.join(f'{label}: {duration_label(sum(measured[name]))}'
                     for name, label in lookup if measured.get(name))


def grouped_progress_milestones(events: list[dict], trace: dict | None = None) -> list[str]:
    """Six compact phases based only on real updates and completed RAG spans.

    The 6 phases are sections, not six mandatory operations. No fabricated
    timings, false tool successes or inferred legal evidence quality.
    """
    trace = trace or {}
    rows = {row['node']: row for row in execution_steps(events, trace)}
    for row in _rag_span_rows(events, trace, set(rows)):
        rows[row['node']] = row
    latest_events = {event.get('node'): event for event in events}
    groups: list[str] = []
    for title, nodes in _PROGRESS_GROUPS:
        completed = [rows[node] for node in nodes if node in rows]
        if not completed:
            continue
        lines = []
        for row in completed:
            node = row['node']
            event = latest_events.get(node, {})
            # Timing was taken from the exact instrumented node; counts from
            # actual update payloads, never from an aggregate parent span.
            seconds = row.get('seconds')
            parts = [duration_label(seconds)] if seconds is not None else []
            parts.extend(_progress_detail(node, event))
            runs = max(row.get('event_runs') or 0, row.get('measured_runs') or 0)
            if runs > 1:
                parts.append(f'{runs} végrehajtás (összesített node-idő)')
            suffix = ' · '.join(parts)
            line = '✓ **' + _PROGRESS_LABELS[node] + '**' + (' · ' + suffix if suffix else '')
            if node == 'hybrid_retrieval':
                breakdown = _retrieval_breakdown(trace)
                if breakdown:
                    line += '\n\n  ↳ ' + breakdown
            lines.append(line)
        groups.append('**' + title + '**\n\n' + '\n\n'.join(lines))
    return groups
