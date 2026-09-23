"""Product-oriented Streamlit UI using actual LangGraph updates and Ollama telemetry.

The graph runs once per chat submission in a background thread. Only the main
Streamlit thread updates widgets; poll loops never restart an inference on rerun.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from queue import Empty, Queue
import time
import uuid
from zoneinfo import ZoneInfo

import streamlit as st

from dap_assistant.presentation.insight_report import detail_section, render_evidence, render_report
from dap_assistant.conversation import is_routing_only, previous_turn_from_messages

from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.presentation.ui_flow import render_flow
from dap_assistant.llm import ollama_health
from dap_assistant.presentation.runtime_view import (LABELS, ROLE_LABELS,
                                   available_life_events, event_from_update,
                                   final_prompt_view, ollama_status, prompt_export,
                                   grouped_progress_milestones,
                                   public_answer_text)
from dap_assistant.settings import Settings
from dap_assistant.presentation.ui_presentation import (
    example_questions_for_domain, official_source_overview, submitted_question,
)
from dap_assistant.response.diagnostics import token_diagnostics
from dap_assistant.workflow import initial_state
from dap_assistant.orchestration.runtime import open_chat_index, create_chat_workflow

@st.cache_resource(show_spinner='Embedding index megnyitása…')
def dense_resource(data_dir: str, embedding_model: str, provider: str):
    """One Qdrant local client per path, across all UI variants (avoids file locks)."""
    return open_chat_index(data_dir, embedding_model, provider)


@st.cache_resource(show_spinner=False)
def workflow_resource(provider: str, fast: bool, answer_mode: str, data_dir: str,
                      embedding_model: str, embedding_provider: str, read_timeout_s: float,
                      total_timeout_s: float, skip_dense: bool = False):
    # Streamlit caches the acquired index and the compiled workflow independently.
    dense = None if skip_dense else dense_resource(data_dir, embedding_model, embedding_provider)
    return create_chat_workflow(
        provider=provider, fast=fast, answer_mode=answer_mode,
        data_dir=data_dir, embedding_model=embedding_model,
        embedding_provider=embedding_provider, read_timeout_s=read_timeout_s,
        total_timeout_s=total_timeout_s, dense=dense,
    )



def render_prompt(prompts: list[dict], prefix: str, *, inside_insight: bool = False) -> None:
    if not prompts:
        st.info('Ebben a módban nem volt LLM-prompt. A dokumentumkeresés ettől függetlenül lefuthat.')
        return
    st.info('Az alábbiak a ténylegesen az Ollamának küldött kérések. Nem a modell belső gondolatmenete.')
    st.warning('A prompt tartalmazhatja a kérdésed és a dokumentumok szövegét. Csak tudatosan exportáld.')
    for i, prompt in enumerate(prompts):
        with detail_section(f"{i + 1}. {prompt['phase']} · {prompt['model']}",
                            expanded=i == len(prompts) - 1, inside_insight=inside_insight):
            st.caption(f"Gondolkodási mód: {prompt['think']} · paraméterek: {prompt['options']}")
            for message in prompt['messages']:
                st.markdown('**' + {'system': 'Rendszerutasítás', 'user': 'Felhasználói bemenet', 'assistant': 'Asszisztensválasz'}.get(message['role'], message['role']) + '**')
                st.code(message['content'], language='text')
            st.caption('A JSON-kimenet elvárt sémája')
            st.json(prompt['schema'])
    st.download_button('Az aktuális prompt(ok) mentése (JSON)', prompt_export(prompts),
                       file_name=f'dap-prompt-{prefix}.json', mime='application/json',
                       key=f'{prefix}-prompt')


def render_final_prompt(final: dict, prompts: list[dict], prefix: str) -> None:
    view = final_prompt_view(final, prompts)
    st.subheader('Végső prompt')
    st.caption(view['label'])
    if not view['submitted']:
        st.info('Ez a forrásalapú válasz ellenőrizhető összeállítási utasítása; nem történt ilyen Ollama-hívás.')
    elif final.get('answer_fallback'):
        st.warning('Ezt a promptot elküldtük a Qwen modellnek, de nem érkezett időben teljes, érvényes válasz.')
    st.code(view['text'], language='text')
    st.download_button('Végső prompt mentése (TXT)', view['text'],
                       file_name=f'dap-final-prompt-{prefix}.txt', mime='text/plain',
                       key=f'{prefix}-final-prompt')
    st.caption('A prompt személyes adatokat is tartalmazhat. Csak szándékosan töltsd le.')



def render_token_diagnostics(report: dict, prompts: list[dict]) -> None:
    """Show measured Ollama answer/tool tokens without exposing prompt content."""
    diagnostics = token_diagnostics(report.get('trace') or {}, prompts, report.get('final') or {})
    st.write(diagnostics['answer_outcome'])
    st.caption(diagnostics['notes'])
    if diagnostics.get('request_budget'):
        st.caption('Teljes kéréskeret: becslés, külön kimeneti és biztonsági tartalékkal.')
        st.json(diagnostics['request_budget'])
    rows = diagnostics['calls']
    if not rows:
        st.info('Nem áll rendelkezésre Ollama-kérésmérés ehhez a válaszhoz. '
                'Forrásból összeállított válasznál ez nem hiba.')
        return
    answer_rows = [row for row in rows if row['phase'] == 'answer']
    last = answer_rows[-1] if answer_rows else None
    if last:
        def measured(value, suffix=''):
            return f'{value}{suffix}' if value is not None else 'N/A · nincs mérés'
        def percent(value):
            return f'{value * 100:.1f}%' if value is not None else 'N/A · nincs mérés'
        cols = st.columns(4)
        cols[0].metric('Kért kontextus (num_ctx)', measured(last['requested_num_ctx']))
        cols[1].metric('Feldolgozott prompt token', measured(last['prompt_eval_count']))
        cols[2].metric('Kért kimeneti keret', measured(last['requested_num_predict']))
        cols[3].metric('Generált token', measured(last['eval_count']))
        st.caption('Utolsó válaszgeneráló kérés · prompt/kért kontextus: '
                   + percent(last['requested_window_ratio']) + ' · kimenet/keret: '
                   + percent(last['output_budget_ratio']) + '. Ez nem az effektív kontextus mérése.')
        if last['request_evidence_count'] is not None:
            st.caption('A modellkérés összeállított JSON-jában '
                       f"{last['request_evidence_count']} forrásrészlet szerepelt. "
                       'Ez az elküldött prompt tartalmáról szól, nem a modell megértéséről.')
    st.markdown('**Kérések és mért eredményeik**')
    st.dataframe([{
        'Fázis': row['label'], 'Modell': row['model'] or '—',
        'num_ctx (kért)': row['requested_num_ctx'],
        'num_predict (kért)': row['requested_num_predict'],
        'prompt_eval_count': row['prompt_eval_count'],
        'eval_count': row['eval_count'],
        'Promptba tett forrásrészlet': row['request_evidence_count'],
        'done_reason': row['done_reason'] or 'nem ismert',
        'Thinking beállítás': row['think'],
        'Thinking karakter (nem token)': row['observed_thinking_chars'],
        'Idő (s)': row['elapsed_s'], 'Értelmezés': row['status'],
    } for row in rows], hide_index=True, use_container_width=True)
    if any(row['requested_window_ratio'] is not None and row['requested_window_ratio'] >= .9
           for row in rows):
        st.warning('Legalább egy kérés promptja megközelítette a kért num_ctx értéket. '
                   'Érdemes a forráskiválasztást és az Ollama tényleges bemenetét ellenőrizni; '
                   'ez önmagában nem bizonyít kontextuslevágást.')
    if any(row['observed_thinking_chars'] for row in rows):
        st.info('Legalább egy Ollama-válasz thinking mezőt tartalmazott. A karakterekből nem '
                'számítható hiteles thinking-tokenszám; a generálási keret felhasználását '
                'csak az Ollama összesített eval_count értéke jelzi.')
    if diagnostics['context_engineering']:
        st.markdown('**Kontextus-előkészítés (becsült keret, nem tényleges tokenhasználat)**')
        safe = diagnostics['context_engineering']
        st.json({key: safe.get(key) for key in (
            'requested_num_ctx', 'excerpt_budget_estimate', 'context_candidate_chunks',
            'duplicate_chunk_ids_removed', 'missing_retrieved_facets', 'budget_kind')
            if key in safe})
    if diagnostics['generation_diagnostics']:
        st.markdown('**Válaszlezárás és hiányzó témák – automatikus jelzés**')
        st.json({key: diagnostics['generation_diagnostics'].get(key) for key in (
            'status', 'failure_kinds', 'uncovered_facets_proxy', 'token_limit_vs_content_gap',
            'model_claim_count') if key in diagnostics['generation_diagnostics']})
    st.caption('A válasz állításainak helyességét és teljességét ez a technikai mérés nem értékeli.')


def render_insight(report: dict, prompts: list[dict], prefix: str) -> None:
    """Separate source evidence, runtime flow, model telemetry and developer diagnostics."""
    final = report['final']
    with st.expander('Betekintés · források, folyamat és technikai adatok', expanded=False):
        source_tab, evidence_tab, flow_tab, tokens_tab, developer_tab = st.tabs([
            'Hivatalos források', 'Forrásrészletek', 'Folyamat',
            'Tokenek és válaszlevágás', 'Fejlesztői adatok',
        ])
        with source_tab:
            st.markdown('#### Hivatalos dokumentumok')
            st.caption('A futásban visszakeresett dokumentumok. Nem mindegyikből került '
                       'állítás a válaszba; a konkrét teendőknél külön hivatkozások szerepelnek. '
                       'A dokumentum letöltési ideje nem igazolja a szabály aktuális hatályosságát.')
            sources = official_source_overview(final.get('evidence') or [])
            if not sources:
                st.info('Ehhez a válaszhoz nincs visszakeresett dokumentum.')
            for source in sources:
                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    left.write(source['title'])
                    left.caption(f"{source['chunk_count']} visszakeresett forrásrészlet")
                    if source['source_url']:
                        right.link_button('Megnyitás', source['source_url'],
                                          key=f"{prefix}-overview-source-{source['document_id']}")
        with evidence_tab:
            st.markdown('#### Visszakeresett forrásrészletek')
            st.caption('Az eredeti szövegek és a keresési rangsorok külön láthatók. '
                       'Az RRF rangsorpontszám, nem relevanciavalószínűség.')
            render_evidence(final.get('evidence') or [], f'{prefix}-raw')
        with flow_tab:
            render_flow(report, prefix)
        with tokens_tab:
            st.markdown('#### Modellkérések és tokenmérések')
            render_token_diagnostics(report, prompts)
        with developer_tab:
            st.caption('Hibakereséshez: a promptok és forráskivonatok személyes '
                       'adatot is tartalmazhatnak. Exportálás előtt ellenőrizd őket.')
            detail_tab, final_prompt_tab, ollama_tab = st.tabs([
                'Teljes futási napló', 'Végső prompt', 'Ollama-kérések',
            ])
            with detail_tab:
                render_report(report, f'{prefix}-developer', inside_insight=True)
            with final_prompt_tab:
                render_final_prompt(final, prompts, f'{prefix}-developer')
            with ollama_tab:
                render_prompt(prompts, f'{prefix}-developer', inside_insight=True)


def stream_workflow(app, initial: dict, config: dict, outbox: Queue) -> dict:
    """Background work produces data only: never call st.* outside main thread."""
    try:
        for namespace, delta in app.stream(initial, config=config,
                                           stream_mode='updates', subgraphs=True):
            for node, update in delta.items():
                if isinstance(update, dict):
                    outbox.put(event_from_update(namespace, node, update))
        return app.get_state(config).values
    finally:
        outbox.put(None)



st.title('DÁP Élethelyzet-asszisztens')
st.caption('Nem hivatalos, kizárólag tájékoztató prototípus · helyi modell · hivatalos források')
base = Settings()
if 'messages' not in st.session_state:
    st.session_state.messages = []

today_budapest = datetime.now(ZoneInfo('Europe/Budapest')).date()
with st.sidebar:
    st.header('Beszélgetés')
    if st.button('Új beszélgetés', use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop('example_question', None)
        st.rerun()
    active_case = previous_turn_from_messages(st.session_state.messages)
    if active_case:
        topics = ', '.join(LABELS.get(d, d) for d in active_case['domains'])
        role_label = ROLE_LABELS.get(active_case['role'], active_case['role']) if active_case['role'] else ''
        st.caption('Aktív élethelyzet: ' + topics + (f' · {role_label}' if role_label else ''))
    domain_display = st.selectbox('Téma (opcionális)', ('Automatikus', 'Autó', 'Munkahely'),
                                  help='Automatikus módban a kérdés alapján ismerem fel az élethelyzetet.')
    domain_hint = {'Automatikus': '', 'Autó': 'vehicle', 'Munkahely': 'employment'}[domain_display]
    with st.expander('Kérdésötletek', expanded=True):
        st.caption('Válassz egy kérdést: kattintás után ugyanúgy elküldöm, mintha beírtad volna.')
        for group_index, (label, examples) in enumerate(example_questions_for_domain(domain_hint)):
            st.markdown(f'**{label}**')
            for question_index, example in enumerate(examples):
                if st.button(example, key=f'example-{group_index}-{question_index}',
                             use_container_width=True):
                    st.session_state.example_question = example
        st.caption('Saját kérdésnél írd le, mi történt, és mire vagy kíváncsi. '
                   'Pontos határidőhöz add meg az esemény dátumát, ha ismert. '
                   'Személyes azonosítót ne írj be.')
    event_date_confirmed = st.checkbox(
        'Megadom az esemény tényleges dátumát', value=False,
        help='Csak akkor jelöld be, ha az esemény valóban már megtörtént, '
             'és ismered a dátumát. A mai napot nem tekintjük automatikusan eseménydátumnak.',
    )
    event_date = (st.date_input('Esemény dátuma', value=today_budapest, key='event_date')
                  if event_date_confirmed else None)
    with st.expander('Haladó beállítások és rendszerállapot', expanded=False):
        mode = st.selectbox('LLM-üzemmód', ('ollama', 'dummy'),
                            index=0 if base.llm_provider == 'ollama' else 1,
                            format_func=lambda name: 'Helyi Ollama' if name == 'ollama' else 'Tesztüzemmód (dummy)',
                            help='Dummy: determinisztikus teszt, nem LLM-válasz.')
        answer_labels = {
            'quick': 'Gyors Qwen',
            'detailed': 'Részletes Qwen (lassabb)',
        }
        answer_mode = st.selectbox('Válaszstratégia', list(answer_labels),
                                   index=list(answer_labels).index(base.answer_mode)
                                   if base.answer_mode in answer_labels else 0,
                                   format_func=lambda item: answer_labels[item])
        fast = st.checkbox('Gyors élethelyzet-felismerés', value=base.fast_routing)
        st.caption(
            f'LLM időkorlátok (.env): read={base.ollama_read_timeout_s:g} s · '
            f'total={base.ollama_total_timeout_s:g} s. '
            'A Chatbot oldal ezeket nem írja felül.'
        )
        st.caption('A gyors mód kért kontextuskerete: '
                   f'{min(base.ollama_num_ctx, base.ollama_quick_num_ctx)} token. '
                   'A tényleges kimeneti tokenhasználat az Ollama-válaszban ellenőrizhető.')
        settings = replace(base, llm_provider=mode, answer_mode=answer_mode,
                           fast_routing=fast)
        chunks = load_chunks(settings.data_dir)
        st.caption(f"{settings.ollama_model} · {len(chunks)} szövegrészlet · "
                   f"{len({c['document_id'] for c in chunks})} dokumentum")
        st.caption('Támogatott élethelyzetek: ' + ', '.join(available_life_events(chunks)))
        if st.button('Ollama állapot ellenőrzése', use_container_width=True):
            st.session_state.ollama_status = ollama_status(
                ollama_health(settings), settings.ollama_model)
        if 'ollama_status' in st.session_state:
            health_view = st.session_state.ollama_status
            getattr(st, health_view['level'])(health_view['title'])
            st.caption(health_view['detail'])
        previous = [m for m in st.session_state.messages if m.get('report')]
        if previous:
            times = [m['report']['elapsed_s'] for m in previous]
            st.caption(f'Lezárt válaszok: {len(times)} · '
                       f'átlagos feldolgozás: {sum(times)/len(times):.1f} s')


# Only warm the real Ollama workflow. In dummy/test mode, the first render
# must NOT load SentenceTransformer or open the embedded Qdrant index.
# Settings reads providers at construction time, including pytest monkeypatches.
if chunks and 'workflow_prepared' not in st.session_state and mode == 'ollama':
    with st.spinner('Egyszeri előkészítés: embeddingmodell és ügyintézési workflow betöltése…'):
        try:
            workflow_resource(mode, fast, answer_mode, str(settings.data_dir),
                              settings.embedding_model, settings.embedding_provider,
                              float(settings.ollama_read_timeout_s),
                              float(settings.ollama_total_timeout_s))
            st.session_state.workflow_prepared = True
        except Exception as exc:
            st.warning(f'A munkafolyamat előkészítése sikertelen: {exc}. A következő kérdésnél újrapróbálható.')

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('report'):
            render_insight(message['report'], message.get('prompts', []), f'hist-{index}')

typed_question = st.chat_input('Miben segíthetek az ügyintézésben?')
question = submitted_question(typed_question, st.session_state.pop('example_question', None))
if question:
    # Read the latest *successful* assistant result, not a concatenation of raw
    # chat strings. No previous personal details enter the next LLM prompt.
    previous_turn = previous_turn_from_messages(st.session_state.messages)
    routing_only = is_routing_only(question, previous_turn, domain_hint)
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        if not chunks and not routing_only:
            st.error('Nincs feldolgozott dokumentum. Futtasd: python scripts/download_documents.py --index')
        elif (not routing_only and mode == 'ollama' and answer_mode != 'source'
              and not ollama_health(settings).get('configured_model_found')):
            st.error('Ollama / qwen3:4b nem elérhető. Indítsd az Ollamát, vagy válassz forrásalapú módot.')
        else:
            run_id = str(uuid.uuid4())
            thread_id = str(uuid.uuid4())
            try:
                app, telemetry = workflow_resource(mode, fast, answer_mode,
                                                   str(settings.data_dir), settings.embedding_model,
                                                   settings.embedding_provider,
                                                   float(settings.ollama_read_timeout_s),
                                                   float(settings.ollama_total_timeout_s),
                                                   skip_dense=routing_only)
                initial = initial_state(
                    question, '', domain_hint,
                    event_date if event_date_confirmed else None,
                    reference_date=today_budapest,
                    previous_turn=previous_turn,
                )
                initial['run_id'] = run_id
                config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': 30}
                queue: Queue = Queue()
                events: list[dict] = []
                stage = 'Indítás'
                started = time.perf_counter()
                with st.status('Kérdés feldolgozása…', expanded=True) as status:
                    lines = st.empty()
                    progress = st.empty()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(stream_workflow, app, initial, config, queue)
                        while not future.done() or not queue.empty():
                            try:
                                event = queue.get(timeout=0.3)
                            except Empty:
                                event = None
                            if event is not None:
                                events.append(event)
                                stage = event['label']
                                lines.markdown('\n\n'.join(grouped_progress_milestones(events, telemetry.snapshot(run_id))) or
                                               'Élethelyzet feldolgozása…')
                            live = telemetry.live(run_id)
                            elapsed = time.perf_counter() - started
                            if live.get('phase') == 'answer' and not any(
                                    e.get('node') == 'generate_answer' for e in events):
                                # Ollama's phase timer, not the whole workflow time.
                                model_elapsed = live.get('elapsed_s', 0.0)
                                progress.caption(
                                    f'Helyi modell válasza készül · modellhívás: {model_elapsed:.1f} s '
                                    f'· teljes feldolgozás: {elapsed:.1f} s')
                                status.update(label='Válasz készítése a kiválasztott forrásokból…')
                            elif live.get('phase') in ('classify', 'plan', 'selection'):
                                progress.caption(f'Feldolgozás · {elapsed:.1f} s · {stage}')
                            else:
                                progress.caption(f'Feldolgozás · {elapsed:.1f} s · {stage}')
                        final = future.result()
                        # Completed spans are available after the final graph update;
                        # refresh the same status with measured per-step timings.
                        completed_trace = telemetry.snapshot(run_id)
                        lines.markdown('\n\n'.join(grouped_progress_milestones(events, completed_trace)) or
                                       'Nem történt lezárt LangGraph-lépés.')
                        finished_s = time.perf_counter() - started
                        progress.caption(f'Teljes feldolgozás: {finished_s:.1f} s · '
                                         'a fenti lépések ideje külön mérés')
                elapsed = time.perf_counter() - started
                trace = telemetry.snapshot(run_id)
                prompts = telemetry.prompt_preview(run_id)
                if final.get('response_status') == 'unsupported':
                    status.update(label='Témán kívüli kérdés · keresés és modellhívás nélkül',
                                  state='complete', expanded=False)
                elif final.get('response_status') == 'needs_clarification':
                    status.update(label='Pontosítás szükséges · keresés és modellhívás nélkül',
                                  state='complete', expanded=False)
                elif final.get('answer_fallback'):
                    status.update(label='Részleges válasz · részletek a Betekintésben',
                                  state='complete', expanded=False)
                else:
                    status.update(label='Válasz elkészült', state='complete', expanded=False)
                answer = public_answer_text(final.get('final_answer') or
                    'Nem áll rendelkezésre megfelelő bizonyíték a válaszhoz.')
                if final.get('response_status') == 'unsupported':
                    st.info('A kérdés nem kapcsolódik a két támogatott ügyintézési élethelyzethez.')
                elif final.get('response_status') == 'needs_clarification':
                    st.info('Nem található egyértelmű ügyintézési kontextus; kérlek, nevezd meg az élethelyzetet.')
                elif final.get('answer_fallback'):
                    st.warning('Nem készült teljes, ellenőrizhető modellválasz; '
                               'a forrásalapú tájékoztatás részleges.')
                elif final.get('answer_strategy') == 'source':
                    st.info('Forrásalapú üzemmód: nem történt válaszgeneráló LLM-hívás.')
                elif final.get('answer_strategy') == 'verified_fee_extract':
                    st.info('Konkrét díjkérdés: az összeget szó szerint, a visszakeresett '
                            'dokumentumból emeltem ki. Nem történt külön válaszgeneráló LLM-hívás.')
                st.markdown(answer)
                report = {'final': final, 'events': events, 'trace': trace, 'elapsed_s': elapsed}
                render_insight(report, prompts, run_id)
                st.session_state.messages.append({'role': 'assistant', 'content': answer,
                                                  'report': report, 'prompts': prompts})
            except Exception as exc:
                st.error(f'A futás nem fejeződött be: {type(exc).__name__}: {exc}')
                st.caption('Nem történt hatósági ügyintézés. Ellenőrizd az indexet és a helyi modellt.')
                st.session_state.messages.append({'role': 'assistant', 'content': 'A futás technikai hiba miatt megszakadt.'})
