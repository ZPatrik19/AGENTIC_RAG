"""Flow-only Streamlit presentation. No model calls or state mutations."""
from __future__ import annotations

import streamlit as st

from dap_assistant.presentation.runtime_view import duration_label, execution_steps
from dap_assistant.presentation.ui_presentation import insight_counts
from dap_assistant.presentation.workflow_visualization import execution_dot, rag_subgraph_dot


def render_flow(report: dict, prefix: str) -> None:
    """Give measured steps, executed main graph and RAG branches separate views."""
    final = report['final']
    events = report.get('events') or []
    counts = insight_counts(final, report.get('elapsed_s'))
    steps_tab, graph_tab, branches_tab = st.tabs([
        'Lépések és idők', 'LangGraph-gráf', 'RAG-részfeladatok',
    ])

    with steps_tab:
        st.markdown('#### Futási lépések és időmérések')
        if counts['partial']:
            st.warning('A válasz részleges. A lefutott lépések önmagukban nem '
                       'bizonyítják a válasz teljességét.')
        cols = st.columns(4)
        cols[0].metric('Teljes idő', f"{counts['elapsed_s']:.1f} s"
                       if counts['elapsed_s'] is not None else 'nincs mérés')
        cols[1].metric('Dokumentum', counts['documents'])
        cols[2].metric('Forrásrészlet', counts['chunks'])
        cols[3].metric('Eszközeredmény', counts['tool_results'])
        steps = execution_steps(events, report.get('trace') or {})
        if steps:
            st.dataframe([{
                'Lépés': row['label'],
                'Mért idő': (duration_label(row['seconds'])
                              if row['seconds'] is not None else 'nincs mérés'),
                'Részletek': (row['details'].split(' · ', 1)[-1]
                              if row['seconds'] is not None and ' · ' in row['details']
                              else (row['details'] if row['seconds'] is None else '')),
            } for row in steps], hide_index=True, use_container_width=True)
        else:
            st.info('Ehhez a válaszhoz nem áll rendelkezésre lezárt gráfesemény.')
        st.caption('A lépések saját mért idői átfedhetnek: nem adhatók össze '
                   'a teljes feldolgozási idővé.')

    with graph_tab:
        st.markdown('#### Technikai LangGraph-gráf (node-ok, függőségek és eszközök)')
        st.caption('Ez az adott kérdés futásának ábrája. A teljes, statikus rendszerterv '
                   'a bal oldali „Teljes architektúra” oldalon érhető el.')
        diagram = execution_dot(final, events)
        st.graphviz_chart(diagram, use_container_width=True)
        st.caption('A fő node-ok rögzített gráfeseményekből, a részfeladatok az állapotból '
                   'származnak. RAG worker csak ágeredménnyel látható. '
                   'A szaggatott él tervezett függőség, nem időrendi trace.')
        st.download_button('Futási gráf mentése (DOT)', diagram,
                           file_name=f'dap-workflow-{prefix}.dot',
                           mime='text/vnd.graphviz', key=f'{prefix}-flow-graph')

    with branches_tab:
        st.markdown('#### RAG algráf és önálló részfeladatok')
        branches = final.get('branch_results') or {}
        if not branches:
            st.info('Ehhez a futáshoz nincs rögzített RAG-ágeredmény.')
            return
        tasks = final.get('subtasks') or {}
        choices = [key for key in tasks if key in branches]
        choices.extend(key for key in branches if key not in tasks)
        selected = st.selectbox('Részfeladat', choices,
                                key=f'{prefix}-rag-branch',
                                format_func=lambda key: (
                                    f"{key} · {(tasks.get(key) or {}).get('facet') or 'keresés'}"))
        task, branch = tasks.get(selected) or {}, branches[selected]
        with st.container(border=True):
            if task.get('question'):
                st.write('**Keresési fókusz:** ' + task['question'])
            st.caption(f"Állapot: {task.get('status', 'ismeretlen')} · "
                       f"Keresési körök: {branch.get('search_attempt', '—')} · "
                       f"Forrásrészletek: {len(branch.get('evidence') or [])} · "
                       f"Mért futásidő: {duration_label(branch['elapsed_s']) if isinstance(branch.get('elapsed_s'), (int, float)) and branch['elapsed_s'] >= 0 else 'nincs mérés'}")
            st.graphviz_chart(rag_subgraph_dot(selected, branch), use_container_width=True)
            st.caption('Az ötnode-os algráf szerkezetét mutatja. Az ágeredmény önmagában '
                       'nem igazolja minden belső node egyedi lefutását.')
