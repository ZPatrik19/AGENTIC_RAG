"""Validate the GitHub Actions job graph without network access or Docker."""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/ci.yml'


def _load_workflow() -> dict:
    # BaseLoader preserves the GitHub "on" key (PyYAML's YAML 1.1 bool resolver does not).
    workflow = yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    assert isinstance(workflow, dict)
    return workflow


def test_required_jobs_form_a_valid_dependency_graph() -> None:
    jobs = _load_workflow()['jobs']
    expected = {
        'preflight', 'repository-quality', 'code-quality', 'offline-tests',
        'graph-integration', 'dependency-security', 'package-build',
        'docker-smoke', 'final-gate',
    }
    assert set(jobs) == expected
    dependencies = {
        job: ([value['needs']] if isinstance(value.get('needs'), str) else value.get('needs', []))
        for job, value in jobs.items()
    }
    assert all(set(required) <= expected - {job} for job, required in dependencies.items())
    assert set(dependencies['graph-integration']) == {
        'code-quality', 'offline-tests', 'repository-quality'
    }
    assert set(dependencies['docker-smoke']) == {
        'graph-integration', 'package-build', 'dependency-security'
    }
    assert set(dependencies['final-gate']) == expected - {'final-gate'}

    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(job: str) -> None:
        assert job not in visiting, f'Cycle involving {job}'
        if job in visited:
            return
        visiting.add(job)
        for requirement in dependencies[job]:
            visit(requirement)
        visiting.remove(job)
        visited.add(job)

    for job in expected:
        visit(job)
    assert visited == expected


def test_optional_docker_does_not_mask_required_failures() -> None:
    workflow = _load_workflow()
    jobs = workflow['jobs']
    assert workflow['on']['workflow_dispatch']['inputs']['run_docker']['default'] == 'false'
    assert jobs['docker-smoke']['if'] == (
        "github.event_name == 'workflow_dispatch' && inputs.run_docker == true"
    )
    gate = jobs['final-gate']
    assert gate['if'] == 'always()'
    script = gate['steps'][0]['run']
    assert 'check_required offline-tests "$TESTS"' in script
    assert 'check_required graph-integration "$GRAPH"' in script
    assert 'check_required docker-smoke "$DOCKER"' in script
    assert '[[ "$DOCKER" != skipped ]]' in script


def test_real_graph_target_paths_and_streamlit_health_are_current() -> None:
    workflow = _load_workflow()
    assert workflow['jobs']['offline-tests']['strategy']['matrix']['python-version'] == [
        '3.12', '3.13', '3.14'
    ]
    graph_script = next(
        step['run'] for step in workflow['jobs']['graph-integration']['steps']
        if step.get('name') == 'Run the main graph and RAG subgraph'
    )
    for name in (
        'test_graph_integration.py',
        'test_mock_graph_integration.py',
        'test_native_tool_calling_graph.py',
    ):
        assert name in graph_script
        assert (ROOT / 'tests/agent' / name).is_file()
    assert 'test_v9_graph_integration.py' not in graph_script
    assert 'test_v10_mock_graph_integration.py' not in graph_script
    docker_script = next(
        step['run'] for step in workflow['jobs']['docker-smoke']['steps']
        if step.get('name') == 'Start Streamlit with dummy providers'
    )
    assert '127.0.0.1:8501/_stcore/health' in docker_script
    assert 'EMBEDDING_PROVIDER=dummy' in docker_script
