"""Only executes on installations that include Streamlit; never downloads a model."""
from pathlib import Path
import pytest

streamlit = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_streamlit_initial_render_without_documents(monkeypatch, tmp_path):
    # The app does not invoke the graph/model on initial render.
    monkeypatch.setenv('LLM_PROVIDER', 'dummy')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'dummy')
    # A real corpus may already exist. Dummy initial render must not load
    # sentence-transformers / open Qdrant even when Settings was imported earlier.
    from dap_assistant.rag import dense_resources
    opened_indexes = []
    monkeypatch.setattr(dense_resources, 'acquire_dense', lambda settings: opened_indexes.append(settings))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant' / 'ui.py'))
    at.run(timeout=30)
    assert not at.exception
    assert next(box.value for box in at.selectbox if box.label == 'LLM-üzemmód') == 'dummy'
    assert opened_indexes == []


def test_chat_ui_does_not_override_env_timeouts():
    ui_source = (Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant' / 'ui.py').read_text(encoding='utf-8')
    assert "st.slider('LLM időkeret" not in ui_source
    assert 'A Chatbot oldal ezeket nem írja felül.' in ui_source
    assert 'ollama_total_timeout_s=float(timeout_s)' not in ui_source
