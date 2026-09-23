"""Regression contracts for the P0/P1 cleanup, without live external services."""
from __future__ import annotations

import re
from pathlib import Path
from typing import get_args, get_type_hints

import yaml

from dap_assistant.orchestration.state import TaskRecord

ROOT = Path(__file__).resolve().parents[2]


def test_task_status_contract_matches_workflow_values():
    statuses = set(get_args(get_type_hints(TaskRecord)['status']))
    assert statuses == {'pending', 'running', 'complete', 'partial', 'failed'}
    assert 'completed' not in statuses


def test_all_shared_compose_defaults_match_env_template():
    template = dict(line.split('=', 1) for line in (ROOT / '.env.example').read_text(
        encoding='utf-8').splitlines() if '=' in line and not line.lstrip().startswith('#'))
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text(encoding='utf-8'))
    env = compose['services']['assistant']['environment']
    assert set(template) <= set(env)
    for key, expected in template.items():
        if key == 'OLLAMA_BASE_URL':
            assert env[key] == '${OLLAMA_DOCKER_URL:-http://host.docker.internal:11434}'
        elif key == 'DATA_DIR':
            assert env[key] == expected == 'data'
        else:
            match = re.fullmatch(r'\$\{' + re.escape(key) + r':-(.*)\}', env[key])
            assert match is not None and match.group(1) == expected, key


def test_legacy_evaluation_and_reference_paths_are_absent():
    evaluation = ROOT / 'src/dap_assistant/evaluation'
    for obsolete in ('runner.py', 'assisted_reference.py', 'telemetry.py'):
        assert not (evaluation / obsolete).exists()
    assert not (ROOT / 'scripts/prepare_assisted_reference.py').exists()
    # Optional SILVER proxies must not be required on a clean Git checkout.
    json_names = {p.name for p in (ROOT / 'evaluation').glob('*.json')}
    assert json_names in ({'golden_v4.json'}, {'golden_v4.json', 'golden_auto_v4.json'})
    assert 'evaluation/golden_auto_v4.json' in (ROOT / '.gitignore').read_text(encoding='utf-8')
    assert 'ENV_CONFIG_VERSION' not in (ROOT / '.env.example').read_text(encoding='utf-8')
    assert 'ANSWER_EVIDENCE_LIMIT' not in (ROOT / '.env.example').read_text(encoding='utf-8')
    assert 'ANSWER_EXCERPT_CHARS' not in (ROOT / '.env.example').read_text(encoding='utf-8')
    assert not list((ROOT / 'tests').rglob('test_v[0-9]*.py'))
