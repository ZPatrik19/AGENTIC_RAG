"""Dependency-free preflight for the Windows SETUP/RUN launchers.

This module intentionally uses the stdlib: it must work before pip installation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
EXIT_MODEL_MISSING = 2
EXIT_SERVER_OFFLINE = 3
EXIT_OLLAMA_MISSING = 4


def config_value(key: str, default: str) -> str:
    """Read environment first, then an unquoted value from the local .env file."""
    if key in os.environ:
        return os.environ[key]
    env_file = ROOT / '.env'
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding='utf-8-sig').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            name, value = line.split('=', 1)
            if name.strip() == key:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                return value
    return default


def ollama_executable() -> str | None:
    custom = config_value('OLLAMA_EXE', '').strip()
    candidates = [custom] if custom else []
    found = shutil.which('ollama')
    if found:
        candidates.append(found)
    local = os.environ.get('LOCALAPPDATA', '')
    program_files = os.environ.get('ProgramFiles', '')
    for base in (local, program_files):
        if base:
            candidates.extend((str(Path(base) / 'Programs' / 'Ollama' / 'ollama.exe'),
                               str(Path(base) / 'Ollama' / 'ollama.exe')))
    for path in candidates:
        resolved = shutil.which(path) or (path if Path(path).is_file() else None)
        if resolved:
            return resolved
    return None


def ollama_status() -> tuple[int, str]:
    base = config_value('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
    model = config_value('OLLAMA_MODEL', 'qwen3:4b')
    parsed = urlparse(base)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        return EXIT_SERVER_OFFLINE, f'[ERROR] Invalid OLLAMA_BASE_URL: {base}'
    try:
        request = Request(base + '/api/tags', headers={'Accept': 'application/json'})
        with urlopen(request, timeout=3) as response:
            models = json.load(response).get('models', [])
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
        if ollama_executable() is None:
            return EXIT_OLLAMA_MISSING, '[WARN] Ollama API is offline and Ollama executable was not found.'
        return EXIT_SERVER_OFFLINE, f'[WARN] Ollama is installed, but its API at {base} is not reachable ({type(exc).__name__}).'
    installed = {entry.get('name', entry.get('model', '')) for entry in models if isinstance(entry, dict)}
    if model in installed:
        message = f'[OK] Ollama API is online. Model {model} is installed; no reinstall or pull needed.'
        # The separate native-tool/recovery model is optional for basic answers.
        # Warn rather than downloading it or silently modifying .env.
        if config_value('NATIVE_TOOL_CALLING_ENABLED', 'true').lower() in ('true', '1', 'yes'):
            tool_model = config_value('NATIVE_TOOL_MODEL', 'qwen3:4b').strip()
            if tool_model and tool_model not in installed:
                message += f' [WARN] Native tool model {tool_model} is missing; configure/install it separately.'
        if config_value('AGENTIC_ANSWER_RECOVERY_ENABLED', 'true').lower() in ('true', '1', 'yes'):
            recovery_model = config_value('AGENTIC_ANSWER_RECOVERY_MODEL', 'qwen3:4b').strip()
            if recovery_model and recovery_model not in installed and recovery_model != config_value('NATIVE_TOOL_MODEL', 'qwen3:4b').strip():
                message += f' [WARN] Answer recovery model {recovery_model} is missing.'
        return 0, message
    return EXIT_MODEL_MISSING, f'[WARN] Ollama API is online, but model {model} is not installed.'


def start_ollama() -> int:
    status, message = ollama_status()
    if status in (0, EXIT_MODEL_MISSING):
        print('[OK] Ollama API is already running; no second server will be started.')
        return 0
    executable = ollama_executable()
    if not executable:
        print('[ERROR] Ollama executable was not found.', file=sys.stderr)
        return EXIT_OLLAMA_MISSING
    base = urlparse(config_value('OLLAMA_BASE_URL', 'http://localhost:11434'))
    if base.hostname not in ('localhost', '127.0.0.1', '::1'):
        print('[ERROR] The configured Ollama URL is not local; refusing to start another local server.', file=sys.stderr)
        return EXIT_SERVER_OFFLINE
    print(f'[INFO] Starting existing Ollama installation: {executable}', flush=True)
    flags = 0
    if sys.platform == 'win32':
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen([executable, 'serve'], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags)
    except OSError as exc:
        print(f'[ERROR] Could not start Ollama: {exc}', file=sys.stderr)
        return EXIT_SERVER_OFFLINE
    for _ in range(20):
        time.sleep(0.5)
        current, _ = ollama_status()
        if current in (0, EXIT_MODEL_MISSING):
            print('[OK] Ollama API is online.', flush=True)
            return 0
    print('[ERROR] Ollama was launched, but its API did not become reachable. Check the Ollama logs.', file=sys.stderr)
    return EXIT_SERVER_OFFLINE


def pull_model() -> int:
    code, _ = ollama_status()
    if code == 0:
        print('[OK] Requested model already exists. Download skipped.')
        return 0
    if code != EXIT_MODEL_MISSING:
        print('[ERROR] Ollama must be online before pulling a model.', file=sys.stderr)
        return code
    executable = ollama_executable()
    if not executable:
        print('[ERROR] Ollama CLI is not available to download the model.', file=sys.stderr)
        return EXIT_OLLAMA_MISSING
    model = config_value('OLLAMA_MODEL', 'qwen3:4b')
    print(f'[INFO] Downloading missing model: {model} (this may require several GB).', flush=True)
    try:
        result = subprocess.run([executable, 'pull', model], check=False)
    except OSError as exc:
        print(f'[ERROR] Model download failed: {exc}', file=sys.stderr)
        return 1
    if result.returncode:
        print('[ERROR] Model download failed. You can retry: ollama pull ' + model, file=sys.stderr)
        return result.returncode
    code, message = ollama_status()
    print(message)
    return code


def set_provider(provider: str) -> int:
    if provider not in ('ollama', 'dummy'):
        return 1
    path = ROOT / '.env'
    existing = path.read_text(encoding='utf-8') if path.exists() else ''
    lines = existing.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith('LLM_PROVIDER='):
            lines[i] = f'LLM_PROVIDER={provider}'
            replaced = True
            break
    if not replaced:
        lines.append(f'LLM_PROVIDER={provider}')
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    print(f'[OK] Saved LLM_PROVIDER={provider} to .env')
    return 0


def index_status() -> int:
    data = Path(config_value('DATA_DIR', 'data'))
    if not data.is_absolute():
        data = ROOT / data
    manifest = ROOT / 'config' / 'document_sources.yaml'
    # Deliberately parse ONLY document IDs and required flags with stdlib: this
    # preflight runs before PyYAML/pip. Full validation is in sources.py.
    ids: list[str] = []
    required: set[str] = set()
    current = None
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if stripped.startswith('- id:'):
                current = stripped.split(':', 1)[1].strip().strip('"\'')
                ids.append(current)
            elif stripped == 'required: true' and current:
                required.add(current)
    docs = set()
    chunks = 0
    for document_id in ids:
        path = next(iter(sorted((data / 'processed').glob(f'*/{document_id}.json'))), None)
        if path is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('document_id') == document_id and payload.get('chunks'):
                docs.add(document_id)
                chunks += len(payload['chunks'])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    embedding = config_value('EMBEDDING_PROVIDER', 'sentence_transformers')
    vector_dir = data / 'vectorstore' / 'qdrant'
    vectors = vector_dir.is_dir() and any(p.is_file() for p in vector_dir.rglob('*'))
    index_meta = data / 'vectorstore' / 'index_meta.json'
    metadata_valid = False
    try:
        meta = json.loads(index_meta.read_text(encoding='utf-8'))
        metadata_valid = (meta.get('embedding_model') == config_value('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')
                          and meta.get('embedding_text_template') == 'title / section_path + text (v2)'
                          and meta.get('active_source_ids') == sorted(ids)
                          and set(meta.get('document_versions', {})) == docs
                          and meta.get('chunk_count') == chunks)
    except (OSError, ValueError, TypeError):
        pass
    if required <= docs and docs and metadata_valid and (embedding == 'dummy' or vectors):
        print(f'[OK] Ready: {len(docs)} / {len(ids)} official sources; '
              f'{chunks} chunks, required={len(required)}, embedding={embedding}.')
        if len(docs) < len(ids):
            print(f'[WARN] {len(ids) - len(docs)} optional sources unavailable; rerun the downloader to retry.')
        return 0
    print(f'[WARN] Index preparation needed: {len(docs)}/{len(ids)} documents; '
          f'{chunks} chunks; required missing={sorted(required - docs)}; '
          f'metadata {"valid" if metadata_valid else "outdated/missing"}.')
    return 1



def runtime_status() -> int:
    """Display effective non-secret configuration, never change it."""
    keys = (
        ('LLM_PROVIDER', 'ollama'),
        ('OLLAMA_MODEL', 'qwen3:4b'),
        ('ANSWER_MODE', 'detailed'),
        ('OLLAMA_NUM_CTX', '8192'),
        ('OLLAMA_QUICK_NUM_CTX', '3072'),
        ('OLLAMA_ANSWER_NUM_PREDICT', '900'),
        ('OLLAMA_QUICK_NUM_PREDICT', '384'),
        ('OLLAMA_READ_TIMEOUT_S', '180'),
        ('OLLAMA_TOTAL_TIMEOUT_S', '480'),
        ('NATIVE_TOOL_MODEL', 'qwen3:4b'),
        ('AGENTIC_ANSWER_RECOVERY_MODEL', 'qwen3:4b'),
        ('EMBEDDING_DEVICE', 'cpu'),
        ('MAX_SUBTASKS', '6'),
    )
    for name, default in keys:
        print(f'[INFO] {name}={config_value(name, default)}')
    print('[INFO] Ollama chooses CPU/GPU placement automatically. Run: ollama ps')
    return 0


def port_status(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.8)
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            print(f'[ERROR] Port {port} is already occupied. Stop the previous server before running RUN.bat.', file=sys.stderr)
            return 1
    print(f'[OK] Port {port} is available.')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Local launcher health checks (no third-party packages)')
    parser.add_argument('action', choices=('ollama-check', 'ollama-start', 'ollama-pull',
                                           'ollama-path', 'mode', 'dummy-check', 'set-provider', 'index-check', 'port-check', 'runtime-status'))
    parser.add_argument('value', nargs='?')
    args = parser.parse_args()
    if args.action == 'runtime-status':
        return runtime_status()
    if args.action == 'mode':
        print(config_value('LLM_PROVIDER', 'ollama'))
        return 0
    if args.action == 'dummy-check':
        return 0 if config_value('LLM_PROVIDER', 'ollama').lower() == 'dummy' else 1
    if args.action == 'set-provider':
        return set_provider(args.value or '')
    if args.action == 'ollama-path':
        executable = ollama_executable()
        if executable:
            print(executable)
            return 0
        return EXIT_OLLAMA_MISSING
    if args.action == 'ollama-check':
        status, message = ollama_status()
        print(message, flush=True)
        return status
    if args.action == 'ollama-start':
        return start_ollama()
    if args.action == 'ollama-pull':
        return pull_model()
    if args.action == 'index-check':
        return index_status()
    if args.action == 'port-check':
        return port_status(int(args.value or '8501'))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
