# syntax=docker/dockerfile:1
# The LLM runs in Ollama on the host; only Streamlit and local RAG run here.
FROM python:3.12-slim-bookworm AS dependencies

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
RUN python -m venv "$VIRTUAL_ENV"

# Install dependencies separately from application sources so code changes
# do not trigger another costly sentence-transformers / PyTorch download.
COPY pyproject.toml ./pyproject.toml
RUN python -c "import pathlib, tomllib; p = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); deps = p['project']['dependencies'] + p['project']['optional-dependencies']['rag']; pathlib.Path('/tmp/runtime-requirements.txt').write_text('\\n'.join(deps) + '\\n', encoding='utf-8')" \
    && pip install -r /tmp/runtime-requirements.txt

FROM python:3.12-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/huggingface \
    XDG_CACHE_HOME=/app/models/cache \
    HOME=/home/appuser

# libgomp1 supports CPU inference; graphviz supplies 'dot' for diagrams.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates graphviz libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data/vectorstore/qdrant /app/models /app/reports \
    && chown -R appuser:appuser /app /home/appuser

COPY --from=dependencies /opt/venv /opt/venv
WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser evaluation/ ./evaluation/
COPY --chown=appuser:appuser docs/architecture/ ./docs/architecture/

USER appuser
EXPOSE 8501

# Only proves Streamlit is answering; it does not assert corpus or Ollama readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).close()" || exit 1

CMD ["python", "-m", "streamlit", "run", "src/dap_assistant/Chatbot.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]
