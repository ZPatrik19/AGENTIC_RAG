"""Offline container-configuration regression checks (no Docker daemon needed)."""
from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import tomllib
import os

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_uses_local_embedded_qdrant_without_server(compose: dict) -> None:
    assert set(compose["services"]) == {"assistant"}
    volumes = compose["services"]["assistant"]["volumes"]
    assert any(v["target"] == "/app/data/vectorstore/qdrant" and v["type"] == "volume" for v in volumes)
    assert any(v["target"] == "/app/data" and v["type"] == "bind" for v in volumes)


def test_compose_uses_host_ollama_and_cpu_embeddings(compose: dict) -> None:
    env = compose["services"]["assistant"]["environment"]
    assert "host.docker.internal:11434" in env["OLLAMA_BASE_URL"]
    assert "OLLAMA_DOCKER_URL" in env["OLLAMA_BASE_URL"]
    assert "EMBEDDING_DEVICE" in env and "cpu" in env["EMBEDDING_DEVICE"]
    assert "MAX_SUBTASKS" in env and "6" in env["MAX_SUBTASKS"]


def test_compose_keeps_reports_and_benchmark_on_host(compose: dict) -> None:
    volumes = compose["services"]["assistant"]["volumes"]
    targets = {item["target"]: item for item in volumes}
    assert targets["/app/reports"]["source"] == "./reports"
    assert targets["/app/evaluation"]["source"] == "./evaluation"
    assert targets["/app/config"]["read_only"] is True


def test_dockerfile_nonroot_and_app_entrypoint() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in dockerfile
    assert "--server.address=0.0.0.0" in dockerfile
    assert "src/dap_assistant/Chatbot.py" in dockerfile
    assert "/_stcore/health" in dockerfile
    assert "COPY --chown=appuser:appuser src/ ./src/" in dockerfile


def test_requirements_extraction_uses_project_metadata(tmp_path: Path) -> None:
    original = Path(__file__).resolve().parents[2] / "Dockerfile"
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    # The Docker RUN command uses Unix shell quoting; validate it on Unix hosts only.
    if os.name == "nt":
        pytest.skip("Docker RUN shell validation is Unix-only")
    if not pyproject.exists():
        pytest.skip("Run in the project root with its pyproject.toml")
    (tmp_path / "pyproject.toml").write_bytes(pyproject.read_bytes())
    line = next(line for line in original.read_text(encoding="utf-8").splitlines()
                if line.startswith("RUN python -c "))
    command = line.removeprefix("RUN ").removesuffix(" \\")
    output = tmp_path / "requirements.txt"
    command = command.replace("/tmp/runtime-requirements.txt", str(output))
    completed = subprocess.run(shlex.split(command), cwd=tmp_path, check=True, capture_output=True, text=True)
    assert completed.returncode == 0
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    expected = metadata["project"]["dependencies"] + metadata["project"]["optional-dependencies"]["rag"]
    requirements = output.read_text(encoding="utf-8").splitlines()
    assert requirements == expected


def test_dockerignore_excludes_sensitive_and_large_inputs() -> None:
    excluded = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for entry in (".env", ".venv/", "data/", "reports/", "tests/", ".github/"):
        assert entry in excluded
