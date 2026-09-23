from pathlib import Path

from dap_assistant.evaluation.presentation.ui_labels import component_label, error_message, metric_label

ROOT = Path(__file__).resolve().parents[2]


def test_local_qdrant_error_is_explained_in_hungarian():
    error = RuntimeError('Storage folder X is already accessed by another instance of Qdrant client')
    response = error_message(error)
    assert 'Qdrant' in response and 'Állítsd le' in response


def test_known_metrics_and_components_have_hungarian_labels():
    assert metric_label('domain_accuracy') == 'Élethelyzet-felismerés pontossága'
    assert component_label('llm_inference') == 'LLM-következtetés'


def test_streamlit_navigation_and_evaluation_are_hungarian():
    launcher = (ROOT / 'scripts' / 'start_streamlit.py').read_text(encoding='utf-8')
    assert "'Chatbot.py'" in launcher
    assert (ROOT / 'src' / 'dap_assistant' / 'Chatbot.py').is_file()
    pages = list((ROOT / 'src' / 'dap_assistant' / 'pages').glob('*.py'))
    assert len(pages) == 2
    assert {'1_Értékelés_és_teljesítmény.py', '2_Teljes_architektura.py'} == {p.name for p in pages}
    content = next(p for p in pages if 'Értékelés' in p.name).read_text(encoding='utf-8')
    for text in ('Funkcionális értékelés', 'Terheléses mérés', 'Mentett riportok',
                 'Kérdésenkénti eredmények', 'Futtatási környezet és konfiguráció'):
        assert text in content


def test_per_question_columns_are_localized_without_changing_export_keys():
    from dap_assistant.evaluation.presentation.ui_labels import localized_row

    original = {'question_id': 'V01', 'success': True, 'latency_s': 1.5,
                'metrics': {'domain_accuracy': 1.0}}
    table_row = localized_row(original)
    assert table_row['Kérdésazonosító'] == 'V01'
    assert table_row['Élethelyzet-felismerés pontossága'] == 1.0
    assert 'question_id' in original


def test_streamlit_entrypoint_reexecutes_ui_on_rerun():
    content = (ROOT / 'src' / 'dap_assistant' / 'Chatbot.py').read_text(encoding='utf-8')
    assert 'runpy.run_path(' in content
    assert not any(line.strip() == 'from dap_assistant.ui import *' for line in content.splitlines())
