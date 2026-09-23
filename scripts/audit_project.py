"""Check repository structure, import-time dependencies and exact AST duplicates."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

AREAS = ("src", "scripts", "tests")


_PURE_PACKAGES = frozenset({
    'context_engineering', 'documents', 'inference', 'observability', 'orchestration',
    'prompt_engineering', 'rag', 'response', 'tooling',
})


def inspect_import_boundaries(root: Path) -> dict:
    """Inspect top-level imports only; lazy imports and runtime cycles are out of scope.

    This structural check does not prove that deferred imports are intentional or safe.
    """
    package = root / 'src' / 'dap_assistant'
    if not package.is_dir():
        return {'import_cycles': [], 'presentation_boundary_violations': []}
    modules = {}
    for file in package.rglob('*.py'):
        name = '.'.join(file.relative_to(package).with_suffix('').parts)
        if name.endswith('.__init__'):
            name = name.removesuffix('.__init__')
        modules[name] = file
    graph = {name: set() for name in modules}
    violations = []
    for name, file in sorted(modules.items()):
        try:
            tree = ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
        except (OSError, UnicodeError, SyntaxError):
            continue  # The existing audit() reports unparsable files separately.
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = name.split('.')[:-1] if file.name != '__init__.py' else name.split('.')
                    prefix = parent[:len(parent) - node.level + 1]
                    target = '.'.join(prefix + ([node.module] if node.module else []))
                else:
                    target = node.module or ''
                targets = [target]
                # "from . import sibling" imports a concrete sibling module.
                if not node.module:
                    targets.extend(target + '.' + alias.name for alias in node.names)
            for target in targets:
                if name.split('.')[0] in _PURE_PACKAGES and (
                    target == 'streamlit' or target.startswith('streamlit.')
                ):
                    violations.append(f'{name}:{node.lineno}: {target}')
                local = target.removeprefix('dap_assistant.')
                if local in modules and local != name:
                    graph[name].add(local)
    visiting, visited, cycles = [], set(), set()

    def walk(name: str) -> None:
        if name in visiting:
            chain = visiting[visiting.index(name):]
            rotations = [tuple(chain[index:] + chain[:index]) for index in range(len(chain))]
            cycles.add(min(rotations))
            return
        if name in visited:
            return
        visiting.append(name)
        for dependency in sorted(graph[name]):
            walk(dependency)
        visiting.pop()
        visited.add(name)

    for name in sorted(graph):
        walk(name)
    return {
        'import_cycles': [list(cycle) for cycle in sorted(cycles)],
        'presentation_boundary_violations': sorted(violations),
    }


def audit(root: Path, *, large_module_lines: int = 500) -> dict:
    """Collect parse errors, large modules and exact function-body matches without imports.

    Matching ASTs are review candidates, not proof of redundant or dead code.
    """
    if large_module_lines <= 0:
        raise ValueError("large_module_lines must be positive")
    root = root.resolve()
    files = [path for area in AREAS for path in (root / area).rglob("*.py")
             if path.is_file() and "__pycache__" not in path.parts]
    duplicates: dict[str, list[str]] = defaultdict(list)
    large = []
    invalid = []
    tests = 0
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            tree = ast.parse("\n".join(lines), filename=relative)
        except (OSError, SyntaxError, UnicodeError) as exc:
            invalid.append({"path": relative, "error": type(exc).__name__})
            continue
        if len(lines) >= large_module_lines:
            large.append({"path": relative, "lines": len(lines)})
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("test_") and relative.startswith("tests/"):
                tests += 1
            if len(node.body) < 3:
                continue
            # Name-independent exact structural equality is only a review hint:
            # same semantics may have different syntax and vice versa.
            digest = sha256(ast.dump(node.args, include_attributes=False).encode()
                            + ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False).encode()).hexdigest()
            duplicates[digest].append(f"{relative}:{node.lineno}:{node.name}")
    return {
        **inspect_import_boundaries(root),
        "python_files": len(files),
        "python_files_by_area": dict(Counter(path.relative_to(root).parts[0] for path in files)),
        "test_functions": tests,
        "large_modules": sorted(large, key=lambda item: -item["lines"]),
        "exact_function_candidates": [members for members in duplicates.values()
                                      if len(members) > 1],
        "unparseable_files": invalid,
        "limitations": "AST duplicates are not dead-code proof; pytest parametrization adds cases.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--large-module-lines", type=int, default=500)
    args = parser.parse_args()
    result = audit(args.root, large_module_lines=args.large_module_lines)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if any(result[key] for key in (
        "unparseable_files", "import_cycles", "presentation_boundary_violations"
    )) else 0


if __name__ == "__main__":
    raise SystemExit(main())
