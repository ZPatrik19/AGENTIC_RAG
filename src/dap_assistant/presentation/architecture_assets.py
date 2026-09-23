"""Locate checked-in architecture diagrams independently of the working directory."""
from __future__ import annotations

from pathlib import Path


FILE_NAMES = {
    'svg': 'full_workflow.svg',
    'png': 'full_workflow.png',
    'dot': 'full_workflow.dot',
    'mermaid': 'full_workflow.mmd',
}


def architecture_assets(project_root: Path | None = None) -> dict[str, Path]:
    """Return known, repository-controlled architecture assets only."""
    root = project_root if project_root is not None else Path(__file__).resolve().parents[3]
    directory = root / 'docs' / 'architecture'
    return {kind: directory / filename for kind, filename in FILE_NAMES.items()}
