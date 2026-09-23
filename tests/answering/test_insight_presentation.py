"""P6.12: source-backed overview, careful telemetry, and privacy-first disclosure."""
from pathlib import Path
import ast

from dap_assistant.presentation.ui_presentation import official_source_overview, insight_counts


UI_PATH = Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant' / 'ui.py'


def test_source_overview_groups_actual_document_ids_and_preserves_https():
    docs = official_source_overview([
        {'document_id': 'dap-seller', 'title': 'DÁP eladói útmutató',
         'source_url': 'https://dap.gov.hu/elado', 'chunk_id': 'c1'},
        {'document_id': 'dap-seller', 'title': 'DÁP eladói útmutató',
         'source_url': 'https://dap.gov.hu/elado', 'chunk_id': 'c2'},
        {'document_id': 'nfsz-reg', 'title': 'NFSZ nyilvántartás',
         'source_url': 'https://nfsz.munka.hu/nyilvantartas', 'chunk_id': 'c3'},
    ])
    assert [x['document_id'] for x in docs] == ['dap-seller', 'nfsz-reg']
    assert docs[0]['chunk_count'] == 2
    assert docs[0]['source_url'] == 'https://dap.gov.hu/elado'
    assert docs[1]['chunk_count'] == 1


def test_source_overview_never_links_untrusted_or_missing_urls():
    docs = official_source_overview([
        {'document_id': 'bad', 'title': 'Forrás', 'source_url': 'javascript:alert(1)'},
        {'document_id': 'bad', 'title': 'Forrás'},
        {'document_id': 'other', 'source_url': 'http://not-secure.local'},
    ])
    assert [(d['chunk_count'], d['source_url']) for d in docs] == [(2, ''), (1, '')]


def test_insight_counts_do_not_treat_attempted_native_calls_as_success():
    final = {
        'response_status': 'partial', 'answer_fallback': True,
        'evidence': [{'document_id': 'd1'}, {'document_id': 'd1'}, {'document_id': 'd2'}],
        'tool_results': {'t1': {'status': 'success'}},
        'native_tool_trace': {'calls': [
            {'status': 'executed', 'returned_to_model': True},
            {'status': 'executed', 'returned_to_model': False},
            {'status': 'refused', 'returned_to_model': False},
        ]},
    }
    assert insight_counts(final, 5.4) == {
        'documents': 2, 'chunks': 3, 'tool_results': 2,
        'elapsed_s': 5.4, 'partial': True,
    }
    assert insight_counts(final, float('nan'))['elapsed_s'] is None
    assert insight_counts(final, None)['elapsed_s'] is None


def test_developer_prompt_and_ollama_tabs_are_only_inside_insight():
    source = UI_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source)
    insight = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'render_insight')
    main_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'tabs']
    insight_source = ast.get_source_segment(source, insight)
    assert "with st.expander('Betekintés" in insight_source
    assert "'Fejlesztői adatok'" in insight_source
    assert 'render_final_prompt(final, prompts,' in insight_source
    assert 'render_prompt(prompts,' in insight_source
    assert 'with st.expander(' not in insight_source.split('with st.expander(', 1)[-1]
    assert len(main_calls) == 2  # Betekintés -> main tabs -> developer tabs; flow tabs are delegated
    assert 'live_sidebar(' not in source
    assert 'event_date if event_date_confirmed else None' in source
    assert 'st.date_input(' in source


def test_chat_progress_stays_with_chat_and_finished_status_collapses():
    source = UI_PATH.read_text(encoding='utf-8')
    assert "st.status('Kérdés feldolgozása…', expanded=True)" in source
    assert "status.update(label='Válasz elkészült', state='complete', expanded=False)" in source
    assert "render_insight(report, prompts, run_id)" in source
    assert "render_insight(message['report'], message.get('prompts', []), f'hist-{index}')" in source
