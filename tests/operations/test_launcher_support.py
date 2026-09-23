"""Windows launcher preflight logic; executable batch execution needs Windows."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[2]
spec = spec_from_file_location('launcher_support', ROOT / 'scripts' / 'launcher_support.py')
launcher = module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)

sync_spec = spec_from_file_location('sync_env_script', ROOT / 'scripts' / 'config' / 'sync_env.py')
env_sync = module_from_spec(sync_spec)
sys.modules[sync_spec.name] = env_sync
sync_spec.loader.exec_module(env_sync)


class FakeResponse:
    def __init__(self, models: list[str]):
        self.payload = json.dumps({'models': [{'name': name} for name in models]}).encode()

    def __enter__(self):
        return io.BytesIO(self.payload)

    def __exit__(self, *_):
        return False


def test_detects_existing_ollama_model_without_install(monkeypatch):
    monkeypatch.setenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    monkeypatch.setenv('OLLAMA_MODEL', 'qwen3:4b')
    monkeypatch.setattr(launcher, 'urlopen', lambda *a, **k: FakeResponse(['qwen3:4b']))
    code, message = launcher.ollama_status()
    assert code == 0
    assert 'no reinstall or pull' in message
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: None)
    assert launcher.pull_model() == 0  # exits before executing subprocess.run


def test_detects_missing_model(monkeypatch):
    monkeypatch.setenv('OLLAMA_MODEL', 'qwen3:4b')
    monkeypatch.setattr(launcher, 'urlopen', lambda *a, **k: FakeResponse(['other:latest']))
    assert launcher.ollama_status()[0] == launcher.EXIT_MODEL_MISSING


def test_offline_ollama_distinguishes_installed_and_missing(monkeypatch):
    def offline(*_args, **_kwargs):
        raise URLError('connection refused')
    monkeypatch.setattr(launcher, 'urlopen', offline)
    monkeypatch.setattr(launcher, 'ollama_executable', lambda: 'C:/Ollama/ollama.exe')
    assert launcher.ollama_status()[0] == launcher.EXIT_SERVER_OFFLINE
    monkeypatch.setattr(launcher, 'ollama_executable', lambda: None)
    assert launcher.ollama_status()[0] == launcher.EXIT_OLLAMA_MISSING


def test_start_does_not_spawn_a_second_server(monkeypatch):
    monkeypatch.setattr(launcher, 'urlopen', lambda *a, **k: FakeResponse(['qwen3:4b']))
    with patch.object(launcher.subprocess, 'Popen', side_effect=AssertionError('duplicate server')):
        assert launcher.start_ollama() == 0


def test_existing_env_is_preserved_and_dummy_can_be_selected(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.delenv('LLM_PROVIDER', raising=False)
    (tmp_path / '.env').write_text('OLLAMA_MODEL=qwen3:4b\nLLM_PROVIDER=ollama\nCUSTOM_VALUE=abc\n')
    assert launcher.config_value('LLM_PROVIDER', '') == 'ollama'
    assert launcher.set_provider('dummy') == 0
    assert launcher.config_value('LLM_PROVIDER', '') == 'dummy'
    assert 'CUSTOM_VALUE=abc' in (tmp_path / '.env').read_text()


def test_index_check_detects_incomplete_and_complete_lexical_index(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'dummy')
    monkeypatch.setenv('DATA_DIR', 'data')
    conf = tmp_path / 'config'
    conf.mkdir()
    (conf / 'document_sources.yaml').write_text('sources:\n  - id: first\n  - id: second\n')
    assert launcher.index_status() == 1
    processed = tmp_path / 'data' / 'processed' / 'vehicle'
    processed.mkdir(parents=True)
    (processed / 'first.json').write_text(json.dumps({'document_id': 'first', 'chunks': [{'text': 'x'}]}))
    assert launcher.index_status() == 1
    (processed / 'second.json').write_text(json.dumps({'document_id': 'second', 'chunks': [{'text': 'y'}]}))
    # A valid index must include signed model/source metadata, not just files.
    assert launcher.index_status() == 1
    vector = tmp_path / 'data' / 'vectorstore'
    vector.mkdir()
    (vector / 'index_meta.json').write_text(json.dumps({
        'embedding_model': 'intfloat/multilingual-e5-small',
        'embedding_text_template': 'title / section_path + text (v2)',
        'active_source_ids': ['first', 'second'],
        'document_versions': {'first': 'v1', 'second': 'v2'},
        'chunk_count': 2,
    }))
    assert launcher.index_status() == 0
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'sentence_transformers')
    assert launcher.index_status() == 1



def test_env_sync_migrates_old_generated_defaults_without_overwriting_custom_values(monkeypatch, tmp_path):
    (tmp_path / '.env.example').write_text(
        'OLLAMA_BASE_URL=http://localhost:11434\n'
        'OLLAMA_READ_TIMEOUT_S=180\nOLLAMA_TOTAL_TIMEOUT_S=480\n'
        'NATIVE_TOOL_MODEL=qwen3:4b\nAGENTIC_ANSWER_RECOVERY_MODEL=qwen3:4b\n',
        encoding='utf-8',
    )
    (tmp_path / '.env').write_text(
        'ENV_CONFIG_VERSION=1\n'
        'OLLAMA_BASE_URL=[http://localhost:11434](http://localhost:11434)\n'
        'OLLAMA_READ_TIMEOUT_S=120\nOLLAMA_TOTAL_TIMEOUT_S=240\n'
        'AGENTIC_ANSWER_RECOVERY_MODEL=qwen3:4b-instruct\nCUSTOM_VALUE=keep-me\n',
        encoding='utf-8',
    )
    assert env_sync.sync_env(tmp_path) == 0
    result = (tmp_path / '.env').read_text(encoding='utf-8')
    assert 'ENV_CONFIG_VERSION' not in result
    assert 'OLLAMA_BASE_URL=http://localhost:11434' in result
    assert 'OLLAMA_READ_TIMEOUT_S=180' in result
    assert 'OLLAMA_TOTAL_TIMEOUT_S=480' in result
    assert 'NATIVE_TOOL_MODEL=qwen3:4b' in result
    assert 'AGENTIC_ANSWER_RECOVERY_MODEL=qwen3:4b' in result
    assert 'CUSTOM_VALUE=keep-me' in result
    assert (tmp_path / '.env.pre-setup.bak').exists()


def test_env_sync_preserves_explicit_nonlegacy_timeout(monkeypatch, tmp_path):
    (tmp_path / '.env.example').write_text(
        'OLLAMA_TOTAL_TIMEOUT_S=480\n', encoding='utf-8')
    (tmp_path / '.env').write_text(
        'OLLAMA_TOTAL_TIMEOUT_S=360\n', encoding='utf-8')
    assert env_sync.sync_env(tmp_path) == 0
    result = (tmp_path / '.env').read_text(encoding='utf-8')
    assert 'OLLAMA_TOTAL_TIMEOUT_S=360' in result
    assert 'ENV_CONFIG_VERSION' not in result


def test_env_sync_removes_old_marker_once_and_preserves_subsequent_timeout(tmp_path):
    (tmp_path / '.env.example').write_text('OLLAMA_TOTAL_TIMEOUT_S=480\n', encoding='utf-8')
    (tmp_path / '.env').write_text(
        'ENV_CONFIG_VERSION=1\nOLLAMA_TOTAL_TIMEOUT_S=240\n', encoding='utf-8')
    assert env_sync.sync_env(tmp_path) == 0
    assert (tmp_path / '.env').read_text(encoding='utf-8').strip() == 'OLLAMA_TOTAL_TIMEOUT_S=480'

    (tmp_path / '.env').write_text('OLLAMA_TOTAL_TIMEOUT_S=240\n', encoding='utf-8')
    assert env_sync.sync_env(tmp_path) == 0
    assert (tmp_path / '.env').read_text(encoding='utf-8') == 'OLLAMA_TOTAL_TIMEOUT_S=240\n'


def test_env_sync_removes_only_known_obsolete_knobs(tmp_path):
    (tmp_path / '.env.example').write_text('LLM_PROVIDER=ollama\n', encoding='utf-8')
    (tmp_path / '.env').write_text(
        'ANSWER_EVIDENCE_LIMIT=3\nANSWER_EXCERPT_CHARS=520\nCUSTOM_VALUE=keep\n',
        encoding='utf-8',
    )
    assert env_sync.sync_env(tmp_path) == 0
    result = (tmp_path / '.env').read_text(encoding='utf-8')
    assert 'ANSWER_EVIDENCE_LIMIT' not in result
    assert 'ANSWER_EXCERPT_CHARS' not in result
    assert 'CUSTOM_VALUE=keep' in result
    assert 'LLM_PROVIDER=ollama' in result

def test_launchers_are_stepwise_and_setup_starts_run():
    setup = (ROOT / 'SETUP.bat').read_text(encoding='ascii').lower()
    run = (ROOT / 'RUN.bat').read_text(encoding='ascii').lower()
    assert '[step 7/7]' in setup
    assert '[step 5/5]' in run
    assert 'scripts\\download_documents.py --index' in setup
    assert 'call "%~dp0run.bat"' in setup
    assert 'ollama-check' in setup and 'ollama-check' in run
    assert 'scripts\\config\\sync_env.py' in setup
    assert 'save llm_provider=dummy to .env' not in setup
    assert 'scripts\\start_streamlit.py' in run
    assert 'cmd /k' not in run
    assert 'cmd /k' not in run


def test_streamlit_launcher_opens_browser_and_propagates_server_exit(monkeypatch, tmp_path):
    spec = spec_from_file_location('start_streamlit', ROOT / 'scripts' / 'start_streamlit.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    calls = []
    def fake_popen(command, **kwargs):
        calls.append(('opener', command, kwargs))
        return object()
    def fake_call(command, **kwargs):
        calls.append(('server', command, kwargs))
        return 17
    monkeypatch.setattr(module.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(module.subprocess, 'call', fake_call)
    assert module.main(8501) == 17
    assert calls[0][1][-1] == 'http://localhost:8501'
    assert calls[1][1][1:4] == ['-m', 'streamlit', 'run']
    assert calls[1][1][-2:] == ['--server.fileWatcherType', 'none']
    assert ['--browser.gatherUsageStats', 'false'] == calls[1][1][-4:-2]
    assert (tmp_path / 'logs' / 'browser-launch.log').exists()
