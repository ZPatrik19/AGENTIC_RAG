#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MIN_VERSION='import sys; assert (3, 12) <= sys.version_info[:2] < (3, 15)'
if [[ -e .venv && ! -x .venv/bin/python ]]; then
  echo '[ERROR] .venv exists but its Python executable is missing. Remove only .venv and rerun setup.' >&2
  exit 1
fi
if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c "$MIN_VERSION" >/dev/null 2>&1; then
  echo '[WARN] Existing .venv is incompatible. Python 3.12-3.14 is required.'
  echo '[INFO] Only .venv will be deleted. .env, data and the Ollama model are preserved.'
  if [[ ! -t 0 ]]; then
    echo '[ERROR] Rerun interactively, or remove only .venv manually.' >&2
    exit 1
  fi
  read -r -p 'Recreate .venv now? [y/N]: ' answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo '[ERROR] Setup cancelled.' >&2; exit 1; }
  rm -rf -- .venv
fi
if [[ ! -x .venv/bin/python ]]; then
  for candidate in python3.12 python3.13 python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$MIN_VERSION" >/dev/null 2>&1; then
      echo "[INFO] Creating .venv with $($candidate --version)..."
      "$candidate" -m venv .venv
      break
    fi
  done
fi
[[ -x .venv/bin/python ]] || { echo '[ERROR] Install Python 3.12-3.14 with venv support.' >&2; exit 1; }
.venv/bin/python -c "$MIN_VERSION"
.venv/bin/python -m pip install -e '.[rag,dev]'
.venv/bin/python scripts/config/sync_env.py
.venv/bin/python -c "from dap_assistant.settings import Settings; Settings(); print('[OK] .env configuration validated.')"
mkdir -p data/raw data/interim data/processed data/vectorstore logs reports
.venv/bin/python scripts/launcher_support.py runtime-status
if [[ "$(.venv/bin/python scripts/launcher_support.py mode)" == ollama ]]; then
  .venv/bin/python scripts/launcher_support.py ollama-check || \
    echo '[WARN] Ollama/model unavailable. Start it and install the model named in .env before running ./run.sh.'
fi
.venv/bin/python scripts/launcher_support.py index-check || \
  echo '[WARN] To build the local index: .venv/bin/python scripts/download_documents.py --index'
printf '[OK] Setup complete. Start with: ./run.sh\n'
