"""Architecture dependency contracts.

The scanner deliberately includes function-local imports: moving an import into a
function delays the cycle, but it does not repair the dependency direction.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "factorzen"


def _factorzen_import_targets(path: Path) -> set[str]:
    # Some legacy modules still carry a UTF-8 BOM; Python imports them normally,
    # so the architecture scanner must use the same BOM-tolerant decoding.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.ImportFrom):
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("factorzen."):
                    targets.add(alias.name.split(".", 2)[1])
        if module and module.startswith("factorzen."):
            targets.add(module.split(".", 2)[1])
    return targets


def test_lower_layers_do_not_depend_on_orchestration_layers() -> None:
    """Dependency flow stays acyclic even when local imports are considered."""
    forbidden = {
        "agents": {"pipelines"},
        "core": {"daily"},
        "daily": {"discovery"},
        "discovery": {"agents", "pipelines"},
        "llm": {"agents"},
    }
    violations: list[str] = []
    for source_layer, forbidden_targets in forbidden.items():
        for path in sorted((PACKAGE_ROOT / source_layer).rglob("*.py")):
            bad = _factorzen_import_targets(path) & forbidden_targets
            for target in sorted(bad):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)} -> {target}")

    assert violations == [], "forbidden FactorZen dependencies:\n" + "\n".join(violations)


def test_top_level_package_dependency_graph_is_acyclic() -> None:
    """No package cycle may be hidden behind a function-local import."""
    graph: dict[str, set[str]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.relative_to(PACKAGE_ROOT).parts[0]
        graph.setdefault(source, set()).update(_factorzen_import_targets(path) - {source})

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            start = visiting.index(node)
            return [*visiting[start:], node]
        if node in visited:
            return None
        visiting.append(node)
        for target in sorted(graph.get(node, set())):
            cycle = visit(target)
            if cycle:
                return cycle
        visiting.pop()
        visited.add(node)
        return None

    found = next((cycle for node in sorted(graph) if (cycle := visit(node))), None)
    assert found is None, "top-level FactorZen dependency cycle: " + " -> ".join(found or [])


def _is_artifact_path_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(
        ("workspace/", "data/")
    )


def _looks_like_path_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and any(
        marker in node.id.lower() for marker in ("root", "dir", "path")
    )


def test_runtime_artifact_path_defaults_derive_from_settings() -> None:
    """Runtime defaults must follow the configured roots; prose strings are irrelevant."""
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path == PACKAGE_ROOT / "config" / "settings.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            candidates: list[ast.AST | None] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                candidates.extend(node.args.defaults)
                candidates.extend(node.args.kw_defaults)
            elif isinstance(node, ast.AnnAssign):
                if _looks_like_path_name(node.target):
                    candidates.append(node.value)
            elif isinstance(node, ast.Assign):
                if any(_looks_like_path_name(target) for target in node.targets):
                    candidates.append(node.value)
            elif isinstance(node, ast.Call):
                candidates.extend(kw.value for kw in node.keywords if kw.arg == "default")
                if isinstance(node.func, ast.Name) and node.func.id == "Path" and node.args:
                    candidates.append(node.args[0])
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) >= 2
                ):
                    candidates.append(node.args[1])
            if any(_is_artifact_path_literal(candidate) for candidate in candidates):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")

    assert violations == [], "hard-coded runtime artifact paths:\n" + "\n".join(violations)


def test_cli_entrypoint_does_not_assemble_the_argparse_tree() -> None:
    """Parser declarations have their own seam instead of growing ``cli/main.py``."""
    path = PACKAGE_ROOT / "cli" / "main.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    parser_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"add_parser", "add_argument", "add_subparsers"}
    ]
    assert parser_calls == []
