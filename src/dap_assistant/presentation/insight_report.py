"""Render the chatbot insight report without owning the chat lifecycle."""
from __future__ import annotations

from contextlib import contextmanager
import json
import plotly.graph_objects as go
import streamlit as st

from .runtime_view import LABELS, active_components, duration_label, execution_steps, run_export


@contextmanager
def detail_section(title: str, *, expanded: bool = False, inside_insight: bool = False):
    """Never nest st.expander inside the per-answer Betekintés expander."""
    if inside_insight:
        with st.container(border=True):
            st.markdown('**' + title + '**')
            yield
    else:
        with st.expander(title, expanded=expanded):
            yield


def render_evidence(evidence: list[dict], prefix: str) -> None:
    if not evidence:
        st.info('Ehhez a futáshoz nem áll rendelkezésre visszakeresett forrás.')
        return
    for i, item in enumerate(evidence):
        with st.container(border=True):
            st.markdown(f"**{item.get('title', 'Hivatalos dokumentum')}** · `{item.get('domain', '')}`")
            st.caption(f"Szövegrészlet: `{item.get('chunk_id', '')}` · RRF: {item.get('score', 0):.5f} (rangsorjel, nem valószínűség)")
            st.caption(
                f"BM25 #{item.get('bm25_rank') or '—'} · "
                f"Dense #{item.get('dense_rank') or '—'} · "
                f"RRF összetevők: {item.get('rrf_components', {})} · "
                f"Információigények: {', '.join(item.get('facet_matches', [])) or '—'}"
            )
            if item.get('source_url', '').startswith('https://'):
                st.link_button('Eredeti hivatalos forrás', item['source_url'], key=f'{prefix}-source-{i}')
            st.code(item.get('text', ''), language=None)


def render_report(data: dict, prefix: str, *, inside_insight: bool = False) -> None:
    final, events, trace, elapsed_s = (
        data['final'], data['events'], data['trace'], data['elapsed_s'],
    )
    overview = st.columns(4)
    overview[0].metric('Teljes idő', f'{elapsed_s:.1f} s')
    overview[1].metric('Részfeladatok', len(final.get('subtasks', {})))
    overview[2].metric('RAG keresések', sum(
        item.get('search_attempt', 0) for item in final.get('branch_results', {}).values()))
    overview[3].metric('Eszközfutások', len(final.get('tool_results', {})) + sum(
        c.get('status') == 'executed' for c in final.get('native_tool_trace', {}).get('calls', [])))
    st.caption('Válaszstratégia: ' + final.get('answer_strategy', 'nincs') +
               ' · Audit: ' + final.get('answer_validation', {}).get('status', 'nem futott'))
    resolution = final.get('context_resolution', {})
    if resolution.get('inherited'):
        st.info('A kérdést az előző sikeres ügyintézési válaszhoz kapcsoltam. '
                'Az előző élethelyzet és szerepkör alapján újra kerestem a dokumentumokban.')
    elif resolution.get('source') == 'explicit':
        st.caption('Témafelismerés: a mostani kérdésben egyértelműen megnevezett élethelyzet.')
    plan = final.get('answer_context', {}).get('response_plan', {})
    question_analysis = plan.get('question_analysis') or final.get('question_analysis', {})
    source_alignment = plan.get('source_alignment', {})
    if question_analysis:
        with detail_section('Kérdéskontextus és források illeszkedése', inside_insight=inside_insight):
            goals = {
                'procedure': 'Általános ügyintézési teendők',
                'benefit_amount': 'Támogatás / ellátás összege',
                'eligibility': 'Jogosultsági feltételek',
                'cost': 'Díjak és költségek', 'deadline': 'Határidők',
                'documents': 'Szükséges dokumentumok', 'where': 'Ügyintézés helye',
                'supports': 'Elérhető támogatások', 'information': 'Tájékoztatás',
            }
            stages = {'after_event': 'Már megtörtént esemény',
                      'planning': 'Tervezett esemény', 'unknown': 'Nem egyértelmű'}
            st.caption('A kérdésből becsült cél: ' + goals.get(
                question_analysis.get('goal', ''), 'Nem egyértelmű') +
                ' · Időbeli helyzet: ' + stages.get(
                question_analysis.get('stage', ''), 'Nem egyértelmű'))
            if source_alignment.get('source_topic_hints'):
                st.write('Visszakeresett szövegben felismert témák: ' +
                         ', '.join(source_alignment['source_topic_hints']))
            if source_alignment.get('unmatched_requested_topic_hints'):
                st.info('A kiválasztott részletekben nem találtam egyértelmű '
                        'témakörjelzést: ' + ', '.join(
                            source_alignment['unmatched_requested_topic_hints']))
            st.caption('Kulcsszóalapú kontextusbecslés; nem bizonyítja az állítások '
                       'jogi vagy szemantikai helyességét, és nem ellenőrzi a teljes dokumentumállományt.')
    components = active_components(trace)
    if components:
        chart = go.Figure(go.Bar(x=[r['seconds'] for r in components],
                                 y=[r['component'] for r in components], orientation='h'))
        chart.update_layout(height=max(280, len(components) * 30), margin=dict(l=10, r=10, t=15, b=15),
                            xaxis_title='Mért idő (másodperc)', yaxis=dict(autorange='reversed'))
        st.plotly_chart(chart, use_container_width=True, key=f'{prefix}-chart')
        st.caption('A grafikonból a befoglaló node-idők kimaradnak; az átfedő ágak időtartama nem adható össze.')
    usage = trace.get('llm_usage', [])
    if usage:
        last = usage[-1]
        cols = st.columns(3)
        cols[0].metric('Prompt token', last.get('prompt_eval_count') or '—')
        cols[1].metric('Generált token', last.get('eval_count') or '—')
        generation_ns = last.get('eval_duration') or 0
        speed = (last.get('eval_count') or 0) / (generation_ns / 1e9) if generation_ns else None
        cols[2].metric('Token/s', f'{speed:.1f}' if speed else '—')
    with detail_section('Fejlesztői retrieval és evidence diagnosztika', inside_insight=inside_insight):
        branch = final.get('branch_results', {})
        for task_id, result in branch.items():
            st.markdown(f"**{task_id} · találatok és lefedettség**")
            st.caption(f"Keresési körök: {result.get('search_attempt', 0)} · "
                       f"Információigények: {', '.join(result.get('requested_facets', [])) or '—'}")
            rows = []
            before = result.get('pre_rerank_chunk_ids', [])
            after = result.get('ranked_chunk_ids', [])
            model_selection = trace.get('selection', {}).get('chosen_by_need')
            model_selected_ids = ({eid for ids in model_selection.values() for eid in ids}
                                  if isinstance(model_selection, dict) else None)
            for item in result.get('candidate_debug', []):
                cid = item.get('chunk_id')
                rows.append({
                    'Chunk': cid, 'Dokumentum': item.get('document_id'),
                    'BM25 hely': item.get('bm25_rank'),
                    'Dense hely': item.get('dense_rank'),
                    'BM25 pont': item.get('bm25_score'),
                    'Dense pont': item.get('dense_score'),
                    'RRF (nem valószínűség)': item.get('score'),
                    'RRF összetevők': str(item.get('rrf_components', {})),
                    'Keresésenkénti eredeti helyek': str(item.get('retrieval_traces', [])),
                    'Rerank pont (nem RRF)': item.get('rerank_score'),
                    'Fusion hely': before.index(cid) + 1 if cid in before else None,
                    'Rerank hely': after.index(cid) + 1 if cid in after else None,
                    'Résztémák': ', '.join(item.get('facet_matches') or []),
                    'RAG által elfogadva': cid in {i.get('chunk_id') for i in result.get('evidence', [])},
                    'Válaszhoz kiválasztva': (
                        ('E_' + cid[:16]) in model_selected_ids
                        if model_selected_ids is not None and cid else None),
                })
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            st.json({'Információigényenkénti találatok (nem gold relevancia)':
                         result.get('selected_facet_evidence', {}),
                     'Kimaradt információigények': result.get('missing_information', []),
                     'Elutasított chunkazonosítók': result.get('rejected_chunk_ids', [])})
        selection = trace.get('selection', {})
        if selection:
            st.markdown('**Bizonyítékkiválasztás · tokenkeret**')
            st.json(selection)
            st.caption('A becsült tokenkeret nem azonos a Qwen tényleges tokenizerével. '
                       'A pontos bemeneti és kimeneti tokenszámot az Ollama-kérések mért metaadatai adják.')
        diagnostics = final.get('answer_context', {}).get('generation_diagnostics') or {}
        if diagnostics:
            st.markdown('**Válaszgenerálás vége és lefedettsége**')
            st.json(diagnostics)
            st.caption('A done_reason=length a kimeneti tokenkorlátot jelzi. A done_reason=stop '
                       'mellett hiányzó résztéma tartalmi hiány lehet; a lefedettség csak '
                       'kulcsszóalapú diagnosztika, nem szemantikai bizonyíték.')
        usage = trace.get('llm_usage', [])
        if usage:
            st.markdown('**Ollama-hívásonkénti tokenek**')
            st.dataframe([{'Fázis': x.get('phase'), 'Prompt token': x.get('prompt_eval_count'),
                           'Kimeneti token': x.get('eval_count'),
                           'Kiértékelés (s)': round((x.get('eval_duration') or 0)/1e9, 3)}
                          for x in usage], hide_index=True, use_container_width=True)
    st.markdown('**Valóban lefutott LangGraph-lépések és részfeladatok – mért idők**')
    steps = execution_steps(events, trace)
    if steps:
        st.dataframe([{
            'Lezárt lépés': row['label'],
            'Mért node-idő': (duration_label(row['seconds'])
                              if row['seconds'] is not None else 'nem mérhető'),
            'Részletek': row['details'].split(' · ', 1)[-1]
                       if row['seconds'] is not None and ' · ' in row['details'] else
                       ('' if row['seconds'] is not None else row['details']),
            'Lefutások': row['event_runs'],
        } for row in steps], hide_index=True, use_container_width=True)
    else:
        st.info('Ehhez a futáshoz nincs lezárt LangGraph-lépés.')
    st.caption('A mért idők node-onkéntiek, nem a teljes feldolgozási idő részeinek összegei: '
               'a keresési ágak párhuzamosan és ismételten is futhatnak. '
               'Ahol nincs telemetria, ott nem jelenítünk meg becsült időt.')
    subtasks = final.get('subtasks') or {}
    branches = final.get('branch_results') or {}
    if subtasks:
        with detail_section(f'Keresési részfeladatok ({len(subtasks)})', expanded=True, inside_insight=inside_insight):
            task_rows = []
            for index, (task_id, task) in enumerate(subtasks.items(), 1):
                branch = branches.get(task_id) or {}
                seconds = branch.get('elapsed_s')
                task_rows.append({
                    'Feladat': index,
                    'Élethelyzet': LABELS.get(task.get('domain'), task.get('domain') or '—'),
                    'Állapot': task.get('status') or '—',
                    'Keresési körök': branch.get('search_attempt', '—'),
                    'Forrásrészletek': len(branch['evidence']) if 'evidence' in branch else '—',
                    'Mért futásidő': (duration_label(seconds) if isinstance(seconds, (int, float))
                                      and seconds >= 0 else 'nem mérhető'),
                })
            st.dataframe(task_rows, hide_index=True, use_container_width=True)
            st.caption('A részfeladat ideje az adott RAG-ág saját mérése. '
                       'Párhuzamos ágak futásideje átfedhet.')
    if final.get('administrative_steps'):
        with detail_section('Ellenőrizhető ügyintézési lépések és forrásaik', inside_insight=inside_insight):
            st.dataframe([{'Kategória': step['category'], 'Teendő / forrásidézet': step['description'],
                           'Feltételes': step['conditional'],
                           'Csatorna': ', '.join(step['official_channels']),
                           'Forrás ID': ', '.join(step['source_evidence_ids'])}
                          for step in final['administrative_steps']], hide_index=True, use_container_width=True)
    with detail_section('Végrehajtási napló', inside_insight=inside_insight):
        for event in events:
            st.write(f"{event['origin']} / {event['label']}")
        st.json({k: value.get('status') for k, value in final.get('subtasks', {}).items()})
        utilization = final.get('answer_validation', {}).get('tool_utilization') or {}
        if utilization:
            st.markdown('**Eszközeredmény → végső válasz (ellenőrzött felhasználás)**')
            st.json(utilization)
            st.caption('A forrásidézet technikai egyezése nem bizonyít szemantikai vagy jogi helyességet.')
        native = final.get('native_tool_trace') or {}
        if native:
            st.markdown('**Natív LLM tool calling (valódi Ollama-választás)**')
            st.json(native)  # Sanitized names, IDs and statuses, not raw source or prompt.
        if final.get('tool_results'):
            for key, tool_result in final['tool_results'].items():
                st.markdown(f"**{key} · {tool_result.get('status', '?')}**")
                st.json(tool_result)
    export = run_export(events=events, trace=trace, final=final, elapsed_s=elapsed_s)
    st.download_button('Futási riport exportálása (JSON)',
                       data=json.dumps(export, ensure_ascii=False, indent=2, default=str),
                       file_name=f'dap-run-{prefix}.json', mime='application/json',
                       key=f'{prefix}-report')
