"""Guard the Python 3.12-3.14 compatibility contract."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

import pytest
from typing import NotRequired, get_origin, get_type_hints

from dap_assistant.conversation import PreviousTurn


ROOT = Path(__file__).resolve().parents[2]


def test_conversation_typeddict_has_optional_focus() -> None:
    # With postponed annotations, TypedDict's __optional_keys__ can be wrong;
    # this module evaluates annotations at class definition time instead.
    assert "focus" in PreviousTurn.__optional_keys__
    assert "domains" in PreviousTurn.__required_keys__
    hints = get_type_hints(PreviousTurn, include_extras=True)
    assert get_origin(hints["focus"]) is NotRequired


def test_supported_python_baseline_is_consistent() -> None:
    package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert package["project"]["requires-python"] == ">=3.12,<3.15"
    assert package["tool"]["ruff"]["target-version"] == "py312"
    assert any(line.startswith("FROM python:3.12-slim")
               for line in (ROOT / "Dockerfile").read_text().splitlines())
    ci_path = ROOT / ".github/workflows/ci.yml"
    if not ci_path.is_file():
        pytest.skip("A feltöltött projektből a .github könyvtár hiányzik; CI nem ellenőrizhető.")
    ci = ci_path.read_text(encoding="utf-8")
    assert "python-version: ['3.12', '3.13', '3.14']" in ci


def test_launchers_reject_old_virtualenvs() -> None:
    setup = (ROOT / "SETUP.bat").read_text(encoding="ascii")
    run = (ROOT / "RUN.bat").read_text(encoding="ascii")
    assert "for %%V in (3.12 3.13 3.14)" in setup
    assert "Recreate the incompatible .venv now?" in setup
    assert "(3, 12) <= sys.version_info[:2] < (3, 15)" in setup
    assert "(3, 12) <= sys.version_info[:2] < (3, 15)" in run
    assert "python3.12 python3.13 python3.14" in (ROOT / "setup.sh").read_text()
    assert "(3,12) <= sys.version_info[:2] < (3,15)" in (ROOT / "run.sh").read_text()


def test_no_unsupported_typing_backports_in_source() -> None:
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing_extensions":
                assert not ({alias.name for alias in node.names} & {"NotRequired", "Required"})
