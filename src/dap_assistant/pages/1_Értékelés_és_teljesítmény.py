"""Professzionális, magyar nyelvű Agentic RAG Evaluation Dashboard."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import re

import streamlit as st

from dap_assistant.evaluation.dataset import active_dataset, load_dataset
from dap_assistant.evaluation.automatic_reference import ensure_auto_reference
from dap_assistant.documents.ingestion import load_chunks
from dap_assistant.evaluation.professional import (
    available_targets, list_saved_runs, load_saved_run, run_functional, run_load, save_run,
    run_report_stem, render_report_markdown,
)
from dap_assistant.evaluation.presentation.ui_labels import metric_label, component_label, error_message
from dap_assistant.evaluation.presentation.ui_metric_visibility import (
    visible_functional_result, visible_functional_csv,
    visible_load_result, visible_load_csv,
)
from dap_assistant.evaluation.presentation.ui_metric_layout import (
    execution_success, metric_sections, node_latency_rows, selected_rag_nodes,
)
from dap_assistant.settings import Settings

try:
    import plotly.express as px
except ImportError:
    px = None

st.set_page_config(page_title='Értékelés és teljesítmény', layout='wide')
# Keep cards readable on ultrawide screens without introducing new UI packages.
st.markdown('''
<style>
[data-testid="stMainBlockContainer"] {max-width: 1360px; padding-top: 1.5rem;}
@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] {padding-left: 1rem; padding-right: 1rem;}
}
</style>
''', unsafe_allow_html=True)
st.title('Értékelés és teljesítmény')
st.caption(
    'Professzionális Agentic RAG mérés: node/részfolyamat/teljes workflow · retrieval · kontextus · '
    'hivatkozások · agentic döntések · terheléses teljesítmény.'
)

settings = Settings()
reports_root = settings.root / 'reports' / 'runs'
reports_root.mkdir(parents=True, exist_ok=True)
# The local corpus is parsed once per Streamlit rerun. No Ollama/embedding calls.
indexed = load_chunks(settings.data_dir)
try:
    ensure_auto_reference(settings.data_dir, chunks=indexed)
    reference_error = None
except (ValueError, OSError, KeyError) as exc:
    reference_error = str(exc)
cases = load_dataset(active_dataset())
targets = available_targets()


METRIC_HELP = {
    'retrieval_recall_at_5': 'A verzióhoz kötött, automatikusan előállított relevanciareferenciák közül mennyi került a Top 5 találatba (nem kézzel hitelesített).',
    'retrieval_precision_at_5': 'A Top 5 visszaadott találat mekkora része releváns.',
    'retrieval_mrr': 'Az első releváns találat ranghelyének reciprok átlaga.',
    'context_recall': 'Az LLM tényleges kontextusában megtalálható elvárt bizonyítékok aránya.',
    'context_precision': 'Ranghelyérzékeny nDCG-alapú relevanciaminőség a végső kontextusban.',
    'context_coverage': 'Bizonyítékkal lefedett elvárt részfeladatok aránya.',
    'citation_accuracy': 'Technikai ellenőrzés: a hivatkozott forrás és az idézet létezik-e; NEM igazolja az állítás szemantikai helyességét.',
    'citation_coverage': 'Hivatkozást igénylő állítások közül mennyi rendelkezik forrással.',
    'abstention_accuracy': 'Helyesen válaszol-e, illetve helyesen tartózkodik/pontosít-e bizonyítékhiány esetén.',
    'tool_selection_accuracy': 'Az elfogadható eszközkészlethez képest helyes tool-választás.',
    'subtask_coverage': 'Az összetett kérdés elvárt részfeladataiból mennyit ismert fel az agent.',
    'tool_call_efficiency': 'Tool-lefedettség hibás, redundáns és ismételt hívások büntetésével.',
    'workflow_success_rate': 'Technikailag hiba nélkül eljutott-e a workflow a végállapotba.',
}

LABELS = {
    **{key: metric_label(key) for key in METRIC_HELP},
    'retrieval_precision_at_5': 'Precision@5',
    'context_recall': 'Context Recall',
    'context_precision': 'Context Precision',
    'context_coverage': 'Context Coverage',
    'citation_accuracy': 'Hivatkozások technikai érvényessége',
    'citation_coverage': 'Citation Coverage',
    'abstention_accuracy': 'Abstention Accuracy',
    'tool_selection_accuracy': 'Tool Selection Accuracy',
    'subtask_coverage': 'Subtask Coverage',
    'tool_call_efficiency': 'Tool Call Efficiency',
    'workflow_success_rate': 'Workflow Success Rate',
}


def fmt_metric(item: dict | None) -> str:
    if not item or item.get('value') is None:
        return 'N/A'
    return f"{item['value']:.1%}"


def metric_card(container, key: str, item: dict | None, *, isolated_rag: bool = False) -> None:
    value = fmt_metric(item)
    evaluated = (item or {}).get('evaluated', 0)
    total = (item or {}).get('total', 0)
    help_text = METRIC_HELP.get(key, '')
    if isolated_rag and key in ('context_recall', 'context_precision', 'context_coverage'):
        help_text = ('A mért RAG node/részfolyamat által kiadott bizonyítéklista és a '
                     'referencia összevetése; nem az LLM tényleges promptjának mérése.')
    with container.container(border=True):
        st.metric(LABELS.get(key, metric_label(key)), value, help=help_text)
        st.caption(f'Értékelt esetek: {evaluated}/{total}')


def render_functional(result: dict) -> None:
    st.subheader('Funkcionális értékelés eredménye')
    cfg = result['configuration']
    st.caption(
        f"Run ID: {result['run_id']} · Modell: {cfg['model']} · Kontextus: {cfg['context_window']} · "
        f"Szint: {cfg['scope']} / {cfg['target']} · Esetek: {result['summary']['evaluated_cases']}"
    )
    metrics = result['summary']['metrics']

    scope = cfg.get('scope', 'full_workflow')
    target = cfg.get('target', 'agentic/full')
    nodes = selected_rag_nodes(scope, target)
    if nodes:
        node_title = component_label('rag/' + nodes[0]) if len(nodes) == 1 else f'RAG részfolyamat · {target}'
        st.markdown(f'### {node_title}')
        if cfg.get('node_metric_schema_version') != 2:
            st.warning('Ez a mentett node-riport a korábbi metrikaszámítással készült. '
                       'Indíts új mérést: a régi, fixture-ből örökölt rangsorok nem számolódnak át.')
        st.caption('Csak a kiválasztott gráfrész kimenete értékelhető. '
                   'A fixture-ként lefutó előző node-ok mutatói nem tartoznak az izolált node-hoz.')
        per_node = node_latency_rows(result.get('rows', []), nodes)
        success = execution_success(result.get('rows', []))
        tech_cols = st.columns(2, gap='medium')
        with tech_cols[0].container(border=True):
            st.metric('Tesztfutások technikai sikere',
                      f'{success:.1%}' if success is not None else 'N/A',
                      help='A mért cél és a bemeneti állapot előkészítésének sikere; nem válaszminőség.')
        with tech_cols[1].container(border=True):
            if len(per_node) == 1:
                elapsed = per_node[0]['mean_s']
                st.metric('Mért node átlagos futásideje',
                          f'{elapsed:.3f} s' if elapsed is not None else 'N/A',
                          help='Csak a rag/<node> telemetry span; a fixture-futtatás ideje nélkül.')
            else:
                st.metric('Mért RAG node-ok száma', len(nodes),
                          help='A részfolyamatban közvetlenül mért node-ok száma.')
        if len(per_node) > 1:
            st.dataframe([{
                'Node': component_label(item['node']),
                'Hívások': item['calls'],
                'Átlagos saját futásidő (s)': round(item['mean_s'], 3)
                if item['mean_s'] is not None else None,
            } for item in per_node], hide_index=True, use_container_width=True)
    sections = metric_sections(scope, target)
    if not sections:
        st.info('Ehhez a node-hoz nincs önálló, referenciaalapú minőségi mutató. '
                'Az itt mért technikai futásadat nem helyettesíti a retrieval- vagy válaszminőséget.')
    for section in sections:
        st.markdown(f'### {section.title}')
        st.caption(section.description)
        cols = st.columns(len(section.keys), gap='medium')
        for col, key in zip(cols, section.keys):
            metric_card(col, key, metrics.get(key), isolated_rag=bool(nodes))

    st.markdown('### Kérdésenkénti eredmények')
    summary_rows = []
    for row in result['rows']:
        summary_rows.append({
            'ID': row['question_id'], 'Téma': row['topic'], 'Nehézség': row['difficulty'],
            'Sikeres': row['success'], 'Válaszidő (s)': round(row['latency_s'], 3),
            'Válasz állapota': row['response_status'], 'Hiba': row.get('error') or '',
        })
    st.dataframe(summary_rows, hide_index=True, use_container_width=True)

    for row in result['rows']:
        with st.expander(f"{row['question_id']} · {row['question']}"):
            left, right = st.columns(2)
            with left:
                st.markdown('**Expected Facts**')
                st.json(row.get('expected_facts', []), expanded=False)
                st.markdown('**Expected Subtasks**')
                st.json(row.get('expected_subtasks', []), expanded=False)
                st.markdown('**Retrieved Sources**')
                st.json(row.get('retrieved_sources', []), expanded=False)
                st.markdown('**Final Context (chunk IDs)**')
                st.json(row.get('final_context', []), expanded=False)
            with right:
                st.markdown('**Generated Answer**')
                st.write(row.get('generated_answer') or 'N/A ezen az értékelési szinten.')
                st.markdown('**Hivatkozások technikai validálása (nem tényellenőrzés)**')
                st.write(row.get('citation_validation') if row.get('citation_validation') is not None else 'N/A')
                st.markdown('**Evaluation Metrics**')
                st.json(row.get('metrics', {}), expanded=False)
            st.markdown('**Node Execution Trace**')
            st.json(row.get('node_execution_trace', {}), expanded=False)


# These are displayed only if the measurement is present in the selected run.
# A retrieval-only graph does not generate LLM tokens and must not show LLM stats.
LOAD_METRICS = (
    ('mean_s', 'Átlagos latency', 's', 'A teljes mérési kérés átlagos válaszideje másodpercben.'),
    ('p50_s', 'P50', 's', 'A kérések legalább fele ennél nem lassabb (medián).'),
    ('p95_s', 'P95', 's', 'A mért kérések 95%-a legfeljebb ennyi ideig tartott.'),
    ('p99_s', 'P99', 's', 'A mért kérések 99%-a legfeljebb ennyi ideig tartott.'),
    ('throughput_qps', 'Throughput', 'qps', 'A mért idő alatt másodpercenként befejezett kérések száma.'),
    ('error_rate', 'Error Rate', 'percent', 'A kivétellel vagy hibával végződött mért kérések aránya.'),
    ('timeout_rate', 'Timeout Rate', 'percent', 'Időtúllépett vagy a beállított SLA-időt meghaladó kérések aránya.'),
    ('ttft_mean_s', 'TTFT átlag', 's', 'Az első generált tokenig eltelt idő; csak akkor látható, ha a modellről tényleges token-időzítés érkezett.'),
    ('generation_tokens_per_second_mean', 'Generation Speed', 'tokens', 'Az Ollama tényleges outputtoken- és generálási időadataiból számolt token/s.'),
    ('context_window_utilization_mean', 'Context Utilization', 'percent', 'Az LLM-hívás során jelentett prompttokenek és a rendelkezésre álló kontextus aránya.'),
)


def _load_metric_value(value: float, kind: str) -> str:
    if kind == 'percent':
        return f'{value:.1%}'
    if kind == 'qps':
        return f'{value:.2f} kérés/s'
    if kind == 'tokens':
        return f'{value:.1f} tok/s'
    return f'{value:.3f} s'


def _balanced_groups(items: list, max_columns: int) -> list[list]:
    """Use balanced rows without placeholders or single wide cards after a full row."""
    groups = []
    index = 0
    while index < len(items):
        remaining = len(items) - index
        if remaining == max_columns + 1:
            size = (max_columns + 1) // 2
        else:
            size = min(max_columns, remaining)
        groups.append(items[index:index + size])
        index += size
    return groups


def _load_card(container, label: str, value: str, help_text: str) -> None:
    with container.container(border=True):
        st.metric(label, value, help=help_text)


def render_load(result: dict) -> None:
    st.subheader('Terheléses mérés eredménye')
    result = visible_load_result(result)
    s = result['summary']
    cfg = result['configuration']
    st.caption(
        f"Run ID: {result['run_id']} · {cfg['request_count']} kérés · párhuzamosság: {cfg['concurrency']} · "
        f"seed: {cfg['seed']} · cél: {cfg['scope']} / {cfg['target']}"
    )
    # Do not create blank metric columns or label an unmeasured LLM metric N/A.
    measured = [(label, _load_metric_value(s[key], kind), help_text)
                for key, label, kind, help_text in LOAD_METRICS if s.get(key) is not None]
    for group in _balanced_groups(measured, 3):
        for column, (label, value, help_text) in zip(st.columns(len(group), gap='medium'), group):
            _load_card(column, label, value, help_text)

    resources = s.get('resource_summary') or {}
    resource_metrics = [
        ('CPU átlag / csúcs', _pair(resources, 'cpu_percent_mean', 'cpu_percent_peak', '%'),
         'A Python-folyamat CPU-használata; többmagos gépen 100% fölé is mehet.'),
        ('RAM csúcs (Python)', _mib(resources.get('process_rss_peak_bytes')),
         'A Python-folyamat legnagyobb mért rezidens memóriája, nem a teljes rendszeré.'),
        ('GPU átlag / csúcs', _pair(resources, 'gpu_percent_mean', 'gpu_percent_peak', '%'),
         'GPU-kihasználtság, ha a mérőeszköz visszaadta.'),
        ('VRAM csúcs', f"{resources['vram_used_mib_peak']:.0f} MiB"
         if resources.get('vram_used_mib_peak') is not None else 'N/A',
         'A GPU-memória legnagyobb mért foglaltsága.'),
    ]
    resource_metrics = [item for item in resource_metrics if item[1] != 'N/A']
    if resource_metrics:
        st.markdown('### Erőforrás-használat')
        for group in _balanced_groups(resource_metrics, 2):
            for column, (label, value, help_text) in zip(st.columns(len(group), gap='medium'), group):
                _load_card(column, label, value, help_text)

    if px is not None and result['rows']:
        rows = result['rows']
        st.markdown('### Teljesítménygrafikonok')
        st.plotly_chart(px.line(rows, x='request_index', y='latency_s', title='Válaszidő lekérdezésenként',
                                labels={'request_index': 'Kérés', 'latency_s': 'Latency (s)'}),
                        use_container_width=True)
        st.plotly_chart(px.histogram(rows, x='latency_s', nbins=25, title='Válaszidő-eloszlás',
                                     labels={'latency_s': 'Latency (s)'}), use_container_width=True)
        ttft_rows = [{'Kérés': row['request_index'], 'TTFT (s)': row['llm_performance']['ttft_s']}
                     for row in rows if (row.get('llm_performance') or {}).get('ttft_s') is not None]
        if ttft_rows:
            st.plotly_chart(px.line(ttft_rows, x='Kérés', y='TTFT (s)', title='Time to First Token'),
                            use_container_width=True)
        speed_rows = [{'Kérés': row['request_index'], 'Token/s': row['llm_performance']['generation_tokens_per_second']}
                      for row in rows if (row.get('llm_performance') or {}).get('generation_tokens_per_second') is not None]
        if speed_rows:
            st.plotly_chart(px.line(speed_rows, x='Kérés', y='Token/s', title='Generálási sebesség'),
                            use_container_width=True)
        context_rows = [
            {'Kérés': row['request_index'], 'Kihasználtság':
             row['llm_performance']['context_window_utilization']['ratio']}
            for row in rows if ((row.get('llm_performance') or {}).get('context_window_utilization') or {}).get('ratio') is not None
        ]
        if context_rows:
            st.plotly_chart(px.line(context_rows, x='Kérés', y='Kihasználtság', title='Kontextusablak-kihasználtság'),
                            use_container_width=True)
        samples = result.get('resources', {}).get('samples', [])
        if samples:
            resource_rows = []
            base = samples[0]['timestamp_s']
            for sample in samples:
                resource_rows.append({
                    'Idő (s)': sample['timestamp_s'] - base,
                    'CPU %': sample.get('cpu_percent'), 'GPU %': sample.get('gpu_percent'),
                    'RAM %': sample.get('system_ram_percent'), 'VRAM MiB': sample.get('vram_used_mib'),
                })
            active_series = [key for key in ('CPU %', 'GPU %', 'RAM %')
                             if any(item.get(key) is not None for item in resource_rows)]
            if active_series:
                st.plotly_chart(px.line(resource_rows, x='Idő (s)', y=active_series,
                                        title='CPU / GPU / RAM terhelés'), use_container_width=True)
            if any(row['VRAM MiB'] is not None for row in resource_rows):
                st.plotly_chart(px.line(resource_rows, x='Idő (s)', y='VRAM MiB', title='VRAM használat'),
                                use_container_width=True)
        component_rows = [
            {'Komponens': component_label(name), 'Átlag (s)': data['mean_s'],
             'P95 (s)': data['p95_s'], 'Hívások': data['invocations']}
            for name, data in s.get('component_stats', {}).items()
        ]
        if component_rows:
            st.plotly_chart(px.bar(component_rows, x='Komponens', y='Átlag (s)',
                                   title='Node-onkénti végrehajtási idő'), use_container_width=True)
            st.dataframe(component_rows, hide_index=True, use_container_width=True)

    st.caption('A node-idők inkluzív és párhuzamos szakaszokat tartalmazhatnak; nem adhatók össze a teljes falióra szerinti latencyként.')


def _pair(data: dict, left: str, right: str, suffix: str) -> str:
    a, b = data.get(left), data.get(right)
    if a is None or b is None:
        return 'N/A'
    return f'{a:.1f}{suffix} / {b:.1f}{suffix}'


def _mib(value) -> str:
    return f'{value / 2**20:.1f} MiB' if value is not None else 'N/A'


config_tab, results_tab, history_tab = st.tabs(['Konfiguráció és futtatás', 'Aktuális eredmény', 'Mentett riportok'])

with config_tab:
    test_type = st.radio('Teszt típusa', ['Funkcionális értékelés', 'Terheléses teszt'], horizontal=True,
                         help='Funkcionális: referenciaalapú metrikák. Terheléses: latency, throughput és erőforrás-használat.')
    topic_label = st.selectbox('Témakör', ['Mindkettő', 'Gépjárművásárlás', 'Munkaviszony megszűnése'],
                               help='A mért kérések melyik 20 kérdéses adatkészlet-részből kerüljenek kiválasztásra.')
    topic = {'Mindkettő': 'all', 'Gépjárművásárlás': 'vehicle', 'Munkaviszony megszűnése': 'employment'}[topic_label]

    scope_label = st.selectbox('Értékelési szint', ['Teljes Agentic workflow', 'Egyetlen RAG node', 'RAG részfolyamat'],
                               help='Teljes válaszfolyamat vagy egy kiválasztott, külön futtatott RAG-komponens.')
    scope = {'Teljes Agentic workflow': 'full_workflow', 'Egyetlen RAG node': 'single_node', 'RAG részfolyamat': 'subflow'}[scope_label]
    target_options = list(targets[scope])
    target_labels = {
        'agentic/full': 'Teljes Agentic workflow',
        'rag/process_query': 'Keresőkérdés előkészítése · process_query',
        'rag/hybrid_retrieval': 'Hibrid keresés · hybrid_retrieval',
        'rag/rerank_results': 'Találatok újrarangsorolása · rerank_results',
        'rag/evaluate_evidence': 'Bizonyítékok ellenőrzése · evaluate_evidence',
        'rag/prepare_context': 'Kontextus összeállítása · prepare_context',
        'rag/query_to_retrieval': 'Kérdésfeldolgozás → keresés',
        'rag/query_to_rerank': 'Kérdésfeldolgozás → keresés → rangsorolás',
        'rag/retrieval_to_rerank': 'Keresés → rangsorolás',
        'rag/full_subgraph': 'Teljes RAG algráf',
    }
    if not target_options:
        st.error('Ehhez az értékelési szinthez nincs reprodukálható gráfcél.')
        st.stop()
    target = st.selectbox('Tesztelt node / részfolyamat', target_options,
                          index=0, key=f'evaluation_target_{scope}',
                          format_func=lambda name: target_labels.get(name, name),
                          help='Csak a tényleges gráfban létező, önállóan reprodukálható célok jelennek meg.')
    st.caption(f'Kiválasztott gráfcél: `{target}`')

    st.markdown('**Futtatási környezet és konfiguráció**')
    st.code(f'Modell: {settings.ollama_model}\nBeállított context window: {settings.ollama_num_ctx}\n'
            f'Gyors Qwen-kérés kontextuskerete: {min(settings.ollama_num_ctx, settings.ollama_quick_num_ctx)}\n'
            f'Answer mode: {settings.answer_mode}', language='text')

    if test_type == 'Funkcionális értékelés':
        filtered = [case for case in cases if topic == 'all' or case['category'] == topic]
        ids = st.multiselect('Tesztkérdések', [case['question_id'] for case in filtered],
                             default=[case['question_id'] for case in filtered],
                             format_func=lambda qid: f"{qid} · {next(c['question'] for c in filtered if c['question_id']==qid)}")
        if not ids:
            st.info('Válassz ki legalább egy tesztkérdést a benchmark indításához.')
        if reference_error:
            st.error('Az értékelési adatok automatikus előkészítése nem sikerült: ' + reference_error)
        if scope == 'full_workflow':
            st.caption('A benchmark válaszgenerálásának időkerete; a chatbot saját időkorlátját nem változtatja meg.')
            timeout_cols = st.columns(2)
            eval_read_timeout = timeout_cols[0].number_input(
                'Benchmark Ollama olvasási timeout (s)', min_value=10.0, max_value=600.0,
                value=max(120.0, float(settings.ollama_read_timeout_s)), step=10.0)
            eval_total_timeout = timeout_cols[1].number_input(
                'Benchmark Ollama teljes timeout (s)', min_value=10.0, max_value=900.0,
                value=max(180.0, float(settings.ollama_total_timeout_s)), step=10.0)
            if eval_total_timeout < eval_read_timeout:
                st.warning('A teljes timeout ne legyen kisebb az olvasási timeoutnál.')
        else:
            eval_read_timeout = settings.ollama_read_timeout_s
            eval_total_timeout = settings.ollama_total_timeout_s
        if st.button('Funkcionális értékelés indítása', type='primary',
                     disabled=bool(not ids or (scope == 'full_workflow' and eval_total_timeout < eval_read_timeout))):
            bar = st.progress(0.0, text='Értékelés előkészítése…')
            def progress(done, total):
                bar.progress(done / total, text=f'Értékelés: {done}/{total}')
            try:
                with st.spinner('A kiválasztott valódi node-ok/workflow futtatása…'):
                    eval_settings = replace(settings, ollama_read_timeout_s=eval_read_timeout,
                                            ollama_total_timeout_s=eval_total_timeout)
                    result = run_functional(eval_settings, scope=scope, target=target, topic=topic,
                                            question_ids=ids, local_judge=False, progress=progress)
                    result = visible_functional_result(result)
                    folder = save_run(result, reports_root)
                    st.session_state['professional_eval_result'] = result
                    st.session_state['professional_eval_folder'] = str(folder)
                st.success(f'Kész. Mentett futás: {folder.name}')
            except Exception as exc:
                st.error(error_message(exc))
                st.code(f'{type(exc).__name__}: {exc}')
            finally:
                bar.empty()
    else:
        count = st.slider('Lekérdezések száma', 50, 200, 50, step=10,
                          help='Ennyi mért kérés kerül a statisztikába; a warm-up ettől külön fut.')
        concurrency = st.select_slider('Párhuzamosság', options=[1, 2, 4], value=1,
                                       help='Ennyi kérés futhat egyidejűleg. A nagyobb párhuzamosság terhelheti az Ollamát és az indexet.')
        timeout = st.number_input('Timeout / kérés (s)', min_value=5.0, max_value=300.0, value=60.0, step=5.0,
                                  help='Kérésenkénti időkeret és SLA-küszöb másodpercben; a futó workflow teljes megszakítását nem garantálja.')
        seed = st.number_input('Véletlen seed', min_value=0, max_value=999999, value=42, step=1,
                               help='A tesztkérdések véletlen kiválasztásának kezdőértéke. Ugyanazzal a seeddel és adatkészlettel ugyanaz a kérdéssorozat áll elő.')
        warmup = st.number_input('Warm-up kérések', min_value=0, max_value=5, value=1, step=1,
                                 help='Ennyi bemelegítő kérés fut a mérés előtt, hogy a kezdeti modell-/indexbetöltés kevésbé torzítsa a latencyt. Nem része az 50–200 mért kérésnek.')
        st.caption('A warm-up kérések nem számítanak bele a normál latency percentilisekbe.')
        if st.button('Terheléses teszt indítása', type='primary'):
            bar = st.progress(0.0, text='Terheléses mérés előkészítése…')
            def progress(done, total):
                bar.progress(done / total, text=f'Feldolgozva: {done}/{total}')
            try:
                with st.spinner('Valós helyi mérés folyamatban…'):
                    result = run_load(settings, scope=scope, target=target, topic=topic,
                                      request_count=int(count), concurrency=int(concurrency),
                                      timeout_s=float(timeout), seed=int(seed), warmup=int(warmup), progress=progress)
                    result = visible_load_result(result)
                    folder = save_run(result, reports_root)
                    st.session_state['professional_eval_result'] = result
                    st.session_state['professional_eval_folder'] = str(folder)
                st.success(f'Kész. Mentett futás: {folder.name}')
            except Exception as exc:
                st.error(error_message(exc))
                st.code(f'{type(exc).__name__}: {exc}')
            finally:
                bar.empty()

with results_tab:
    result = st.session_state.get('professional_eval_result')
    if result is None:
        st.info('Még nincs aktuális futás. Indíts mérést a Konfiguráció és futtatás fülön.')
    else:
        result = visible_functional_result(result)
        result = visible_load_result(result)
        if result['kind'] == 'functional_v4':
            render_functional(result)
        else:
            render_load(result)
        folder = st.session_state.get('professional_eval_folder')
        dated_folder = bool(folder and re.fullmatch(
            r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[a-zA-Z0-9_-]+',
            Path(folder).name))
        export_stem = Path(folder).name if dated_folder else run_report_stem(result)
        st.download_button('JSON export', json.dumps(result, ensure_ascii=False, indent=2, default=str),
                           file_name=f'{export_stem}.json', mime='application/json')
        if folder and (Path(folder) / 'rows.csv').is_file():
            csv_content = (Path(folder) / 'rows.csv').read_bytes()
            if result['kind'] == 'functional_v4':
                csv_content = visible_functional_csv(csv_content)
            elif result['kind'] == 'load_v4':
                csv_content = visible_load_csv(csv_content)
            st.download_button('CSV export', csv_content,
                               file_name=f'{export_stem}.csv', mime='text/csv')
        if folder and (Path(folder) / 'report.md').is_file():
            markdown_content = (render_report_markdown(result).encode('utf-8')
                                if result['kind'] == 'load_v4'
                                else (Path(folder) / 'report.md').read_bytes())
            st.download_button('Markdown riport', markdown_content,
                               file_name=f'{export_stem}.md', mime='text/markdown')

with history_tab:
    runs = list_saved_runs(reports_root)
    if not runs:
        st.info('Nincs mentett futás.')
    else:
        chosen = st.selectbox('Korábbi futás', runs, format_func=lambda p: p.name)
        if st.button('Mentett futás betöltése'):
            st.session_state['professional_eval_result'] = visible_load_result(
                visible_functional_result(load_saved_run(chosen)))
            st.session_state['professional_eval_folder'] = str(chosen)
            st.success('A futás betöltve. Nyisd meg az Aktuális eredmény fület.')
