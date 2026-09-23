"""Architecture UI is static, versioned and available in Docker."""
from __future__ import annotations

import ast
from pathlib import Path

from dap_assistant.presentation.architecture_assets import architecture_assets

ROOT = Path(__file__).resolve().parents[2]


def test_assets_resolve_from_package_not_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths = architecture_assets()
    for name in ('svg', 'png', 'dot', 'mermaid'):
        assert paths[name].is_file(), name
    assert paths['dot'].read_text(encoding='utf-8').startswith('digraph')
    assert 'execute_tools' in paths['dot'].read_text(encoding='utf-8')
    assert 'process_query' in paths['dot'].read_text(encoding='utf-8')


def test_assets_path_can_be_used_with_an_alternate_project_root(tmp_path):
    paths = architecture_assets(tmp_path)
    assert paths['svg'] == tmp_path / 'docs' / 'architecture' / 'full_workflow.svg'
    assert not paths['svg'].exists()


def test_sidebar_navigation_and_architecture_are_read_only():
    launcher = (ROOT / 'src/dap_assistant/Chatbot.py').read_text(encoding='utf-8')
    page = (ROOT / 'src/dap_assistant/pages/2_Teljes_architektura.py').read_text(encoding='utf-8')
    ast.parse(page)
    assert launcher.count('st.Page(') == 3
    assert "title='Teljes architektúra'" in launcher
    assert 'st.image(' in page and 'st.download_button(' in page
    for expensive in ('build_workflow(', 'load_chunks(', 'acquire_dense(', 'ollama_health('):
        assert expensive not in page


def test_flow_has_separate_graph_and_subtask_sections():
    ui = (ROOT / 'src/dap_assistant/ui.py').read_text(encoding='utf-8')
    flow = (ROOT / 'src/dap_assistant/presentation/ui_flow.py').read_text(encoding='utf-8')
    assert "'Hivatalos források', 'Forrásrészletek', 'Folyamat'" in ui
    assert "'Lépések és idők', 'LangGraph-gráf', 'RAG-részfeladatok'" in flow
    assert 'render_flow(report, prefix)' in ui
    assert 'execution_dot(final, events)' in flow
    assert 'rag_subgraph_dot(selected, branch)' in flow
    assert 'st.expander(' not in flow


def test_docker_includes_versioned_architecture_docs():
    docker = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    ignore = (ROOT / '.dockerignore').read_text(encoding='utf-8').splitlines()
    assert 'COPY --chown=appuser:appuser docs/architecture/ ./docs/architecture/' in docker
    assert 'docs/' not in ignore

class _FakeStreamlit:
    """Minimal read-only page harness, not a substitute for browser rendering."""

    def __init__(self):
        self.images = []
        self.downloads = []
        self.errors = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def tabs(self, names):
        return [self for _ in names]

    def image(self, source, **_):
        self.images.append(source)

    def download_button(self, label, **_):
        self.downloads.append(label)

    def error(self, text):
        self.errors.append(text)

    def __getattr__(self, name):
        if name in {'set_page_config', 'title', 'caption', 'markdown', 'info', 'warning'}:
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def test_architecture_page_renders_static_assets_without_streamlit_runtime(monkeypatch):
    import runpy
    import sys

    fake = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    runpy.run_path(str(ROOT / 'src/dap_assistant/pages/2_Teljes_architektura.py'))
    assert fake.images == [str(architecture_assets()['svg'])]
    assert len(fake.downloads) == 4
    assert not fake.errors


def test_architecture_page_shows_actionable_missing_assets_message(monkeypatch, tmp_path):
    import runpy
    import sys
    from dap_assistant.presentation import architecture_assets as assets_module

    fake = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    original = assets_module.architecture_assets
    monkeypatch.setattr(assets_module, 'architecture_assets',
                        lambda: original(tmp_path))
    runpy.run_path(str(ROOT / 'src/dap_assistant/pages/2_Teljes_architektura.py'))
    assert not fake.images
    assert 'docs/architecture/full_workflow.svg' in fake.errors[0]
