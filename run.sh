#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=.venv/bin/python
[[ -x "$PYTHON" ]] || { echo '[ERROR] Run ./setup.sh first.' >&2; exit 1; }
"$PYTHON" -c 'import sys; assert (3,12) <= sys.version_info[:2] < (3,15), "Python 3.12-3.14 required; rerun ./setup.sh"'
"$PYTHON" -c 'import streamlit, langgraph' || { echo '[ERROR] Missing dependencies: rerun ./setup.sh' >&2; exit 1; }
"$PYTHON" -c 'from dap_assistant.settings import Settings; Settings(); print("[OK] .env configuration validated.")'
"$PYTHON" scripts/launcher_support.py runtime-status
if [[ "$("$PYTHON" scripts/launcher_support.py mode)" == ollama ]]; then
  if ! "$PYTHON" scripts/launcher_support.py ollama-check; then
    echo '[ERROR] Ollama or the configured model is unavailable.' >&2
    echo '[INFO] Start Ollama and install OLLAMA_MODEL from .env, or explicitly use LLM_PROVIDER=dummy ./run.sh.' >&2
    exit 1
  fi
fi
"$PYTHON" scripts/launcher_support.py index-check || echo '[WARN] The search index needs preparation; run scripts/download_documents.py --index.'
"$PYTHON" scripts/launcher_support.py port-check 8501
# The shared launcher waits for Streamlit health before opening the browser.
exec "$PYTHON" scripts/start_streamlit.py
