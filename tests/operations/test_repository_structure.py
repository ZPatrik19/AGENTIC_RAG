"""Layout contracts with no Ollama, Docker daemon, network, or paid API."""
from pathlib import Path
import ast
import pytest

from dap_assistant import llm
from dap_assistant.response import models, fallback
from dap_assistant.documents import index_metadata
from dap_assistant.evaluation import index_health
from dap_assistant.response.audit import audit_answer
from dap_assistant.orchestration.state import AssistantState, WorkerState


ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'src' / 'dap_assistant'
PACKAGE = PKG


def test_ui_helpers_and_tool_implementations_are_grouped_once():
    for name in ('ui_flow', 'ui_progress', 'ui_presentation',
                 'workflow_visualization', 'architecture_assets'):
        assert (PKG / 'presentation' / f'{name}.py').is_file()
        assert not (PKG / f'{name}.py').exists()
    for name in ('tools', 'calculators', 'native_tool_calling'):
        assert (PKG / 'tooling' / f'{name}.py').is_file()
        assert not (PKG / f'{name}.py').exists()


def test_docs_have_one_current_page_per_subject_and_no_history():
    expected = {'ARCHITEKTURA_ES_FEJLESZTES_HU.md', 'UZEMELTETES_HU.md',
                'ERTEKELES_ES_BENCHMARK_HU.md'}
    assert {p.name for p in (ROOT / 'docs').glob('*.md')} == expected
    assert not (ROOT / 'docs' / 'history').exists()
    assert (ROOT / 'docs' / 'architecture' / 'full_workflow.svg').is_file()


def test_evaluation_keeps_baseline_silver_and_regeneration_inputs():
    actual = {p.name for p in (ROOT / 'evaluation').glob('*.json')}
    # The automatic SILVER proxy is generated locally and gitignored.
    assert actual in ({'golden_v4.json'}, {'golden_v4.json', 'golden_auto_v4.json'})
    assert not (ROOT / 'evaluation' / 'archive').exists()


def test_launchers_do_not_override_env_response_budgets():
    windows = ''.join((ROOT / name).read_text(encoding='utf-8')
                      for name in ('SETUP.bat', 'RUN.bat'))
    assert 'set "OLLAMA_ANSWER_NUM_PREDICT=' not in windows
    assert 'set "OLLAMA_QUICK_NUM_PREDICT=' not in windows
    linux = (ROOT / 'run.sh').read_text(encoding='utf-8')
    assert 'scripts/start_streamlit.py' in linux
    assert 'scripts/launcher_support.py ollama-check' in linux


def test_scripts_keep_main_entrypoints_and_group_maintenance():
    for name in ('download_documents', 'prepare_golden_review', 'evaluate',
                 'benchmark', 'start_streamlit', 'launcher_support'):
        assert (ROOT / 'scripts' / f'{name}.py').is_file()
    assert (ROOT / 'scripts/maintenance/pin_golden.py').is_file()
    assert (ROOT / 'scripts/diagnostics/run_final_validation.py').is_file()


@pytest.mark.parametrize(('category', 'module'), [
    ('documents', 'sources'),
    ('documents', 'download'),
    ('documents', 'ingestion'),
    ('documents', 'corpus_status'),
    ('documents', 'index_metadata'),
    ('rag', 'retrieval'),
    ('rag', 'rag_graph'),
    ('rag', 'graph_contract'),
    ('rag', 'dense_resources'),
    ('response', 'models'),
    ('response', 'fallback'),
])
def test_single_canonical_module_location(category: str, module: str) -> None:
    assert (PACKAGE / category / f'{module}.py').is_file()
    assert not (PACKAGE / f'{module}.py').exists()


def test_document_layer_does_not_import_evaluation_layer() -> None:
    for file in (PACKAGE / 'documents').glob('*.py'):
        tree = ast.parse(file.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('dap_assistant.evaluation'), file
            if isinstance(node, ast.Import):
                assert all(not name.name.startswith('dap_assistant.evaluation')
                           for name in node.names), file


@pytest.mark.parametrize('name', [
    'Classification', 'PlannedTask', 'Plan', 'Claim', 'Draft',
    'FastSelection', 'FacetSelection', 'NaturalClaim', 'NaturalResponse',
])
def test_llm_schema_api_reexports_identical_class(name: str) -> None:
    assert getattr(llm, name) is getattr(models, name)


def test_source_only_fallback_reexports_identical_functions() -> None:
    assert llm.source_answer is fallback.source_answer
    assert llm.select_excerpt is fallback.select_excerpt


def test_shared_index_metadata_is_not_duplicated() -> None:
    assert index_health.metadata_problems is index_metadata.metadata_problems
    assert index_health.corpus_fingerprint is index_metadata.corpus_fingerprint


def test_canonical_architecture_is_documented() -> None:
    doc = (ROOT / 'docs' / 'ARCHITEKTURA_ES_FEJLESZTES_HU.md').read_text(encoding='utf-8')
    assert 'documents/' in doc and 'rag/' in doc and 'prompt_engineering/' in doc and 'context_engineering/' in doc
    assert 'fallback.py' in doc
    assert (ROOT / 'docs/architecture/full_workflow.svg').is_file()


def test_audit_and_state_are_one_way_dependencies():
    audit_source = (PKG / 'response' / 'audit.py').read_text(encoding='utf-8')
    assert 'from ..workflow import' not in audit_source
    assert 'import streamlit' not in audit_source
    assert 'import langgraph' not in audit_source
    assert 'def audit_answer(' in audit_source
    assert AssistantState.__name__ == 'AssistantState'
    assert WorkerState.__name__ == 'WorkerState'
    assert callable(audit_answer)


def test_node_logic_is_separated_from_public_graph_assembly():
    workflow_source = (PKG / 'workflow.py').read_text(encoding='utf-8')
    node_source = (PKG / 'orchestration' / 'workflow_nodes.py').read_text(encoding='utf-8')
    workflow_tree = ast.parse(workflow_source)
    node_tree = ast.parse(node_source)
    assert len([x for x in ast.walk(node_tree) if isinstance(x, ast.FunctionDef) and x.name == 'answer_audit']) == 1
    assert 'return audit_answer(state)' in node_source
    assert 'graph.add_node(name, nodes.measured(name, fn))' in workflow_source
    assert "('answer_audit', nodes.answer_audit)" in workflow_source
    build = next(x for x in workflow_tree.body if isinstance(x, ast.FunctionDef) and x.name == 'build_workflow')
    assert build.end_lineno - build.lineno + 1 < 60


def test_canonical_evaluation_implementation_and_presentation_home():
    from dap_assistant.evaluation import professional, load_testing
    assert professional.run_load is load_testing.run_load
    assert professional._llm_perf is load_testing._llm_perf
    assert (PKG / 'presentation' / 'insight_report.py').is_file()
    assert (PKG / 'ui.py').read_text(encoding='utf-8').count('def render_report(') == 0
