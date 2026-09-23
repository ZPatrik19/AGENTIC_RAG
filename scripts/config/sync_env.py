"""Synchronize ``.env`` with the template without changing custom values."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_ENV_MIGRATIONS = {
    "OLLAMA_READ_TIMEOUT_S": {"35": "180", "120": "180"},
    "OLLAMA_TOTAL_TIMEOUT_S": {"45": "480", "240": "480"},
    "NATIVE_TOOL_MODEL": {"qwen3:4b-instruct": "qwen3:4b"},
    "AGENTIC_ANSWER_RECOVERY_MODEL": {"qwen3:4b-instruct": "qwen3:4b"},
}

# These knobs no longer affect runtime behavior. Preserve unknown user keys;
# remove only keys that are confirmed to be obsolete in this application.
_OBSOLETE_KEYS = frozenset({'ENV_CONFIG_VERSION', 'ANSWER_EVIDENCE_LIMIT', 'ANSWER_EXCERPT_CHARS'})


def _clean_env_value(name: str, value: str) -> str:
    """Normalize known copy/paste artifacts without rewriting custom values."""
    if name == "OLLAMA_BASE_URL":
        match = re.fullmatch(r"\[(https?://[^\]]+)\]\((https?://[^)]+)\)", value.strip())
        if match and match.group(1) == match.group(2):
            return match.group(1)
    return value


def _parse_template(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() not in _OBSOLETE_KEYS:
            values[name.strip()] = value.strip()
    return values


def sync_env(root: Path = ROOT) -> int:
    """Create or safely migrate ``.env`` while preserving explicit user choices."""
    template = root / ".env.example"
    target = root / ".env"
    if not template.is_file():
        print("[ERROR] .env.example is missing.", file=sys.stderr)
        return 1

    if not target.exists():
        shutil.copyfile(template, target)
        print("[OK] Created .env from .env.example.")
        return 0

    raw = target.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    current: dict[str, str] = {}
    line_index: dict[str, int] = {}
    cleaned_lines: list[str] = []
    format_changed = False

    for raw_line in lines:
        original_line = raw_line
        stripped = raw_line.strip()
        if stripped.startswith("*#"):
            body = stripped[2:]
            if body.endswith("*"):
                body = body[:-1]
            raw_line = "#" + body
            stripped = raw_line.strip()
            format_changed = raw_line != original_line or format_changed

        cleaned_lines.append(raw_line)
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        current[name] = value.strip()
        line_index[name] = len(cleaned_lines) - 1

    # Only an *explicitly versioned* old generated file can be migrated safely.
    # A 240-second timeout in an unversioned file may be an intentional choice.
    raw_version = current.get("ENV_CONFIG_VERSION")
    migrate_old_defaults = raw_version is not None and raw_version.isdigit() and int(raw_version) < 2
    changed: list[str] = ["comment-format"] if format_changed else []

    # The migration marker is no longer a public environment setting. Once
    # removed, subsequent SETUP runs must not rewrite the user's timeout.
    for obsolete in sorted(_OBSOLETE_KEYS & current.keys()):
        cleaned_lines[line_index[obsolete]] = ""
        del current[obsolete]
        changed.append(f"removed {obsolete}")

    for name, value in list(current.items()):
        normalized = _clean_env_value(name, value)
        if migrate_old_defaults:
            normalized = _ENV_MIGRATIONS.get(name, {}).get(normalized, normalized)
        if normalized != value:
            cleaned_lines[line_index[name]] = f"{name}={normalized}"
            current[name] = normalized
            changed.append(name)

    template_values = _parse_template(template)
    missing = [name for name in template_values if name not in current]
    if missing:
        cleaned_lines.extend(["", "# Added by SETUP from the current .env.example"])
        for name in missing:
            cleaned_lines.append(f"{name}={template_values[name]}")
            current[name] = template_values[name]
            changed.append(name)

    if changed:
        backup = root / ".env.pre-setup.bak"
        backup.write_text(raw, encoding="utf-8")
        target.write_text("\n".join(cleaned_lines).rstrip() + "\n", encoding="utf-8")
        print("[OK] Updated .env safely; previous file saved as .env.pre-setup.bak.")
        print("[INFO] Updated/added keys: " + ", ".join(sorted(set(changed))))
    else:
        print("[OK] Existing .env already matches the current configuration schema.")
    return 0


def main() -> int:
    return sync_env()


if __name__ == "__main__":
    raise SystemExit(main())
