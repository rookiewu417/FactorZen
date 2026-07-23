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


def test_architecture_dependencies_suite():
    """Dependency flow stays acyclic even when local imports are considered.；No package cycle may be hidden behind a function-local import.；Runtime defaults must follow the configured roots; prose strings are irrelevant.；Parser declarations have their own seam instead of growing ``cli/main.py``."""
    # -- 原 test_lower_layers_do_not_depend_on_orchestration_layers --
    def _section_0_test_lower_layers_do_not_depend_on_orchestration_layers():
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

    _section_0_test_lower_layers_do_not_depend_on_orchestration_layers()

    # -- 原 test_top_level_package_dependency_graph_is_acyclic --
    def _section_1_test_top_level_package_dependency_graph_is_acyclic():
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

    _section_1_test_top_level_package_dependency_graph_is_acyclic()

    # -- 原 test_runtime_artifact_path_defaults_derive_from_settings --
    def _section_2_test_runtime_artifact_path_defaults_derive_from_settings():
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

    _section_2_test_runtime_artifact_path_defaults_derive_from_settings()

    # -- 原 test_cli_entrypoint_does_not_assemble_the_argparse_tree --
    def _section_3_test_cli_entrypoint_does_not_assemble_the_argparse_tree():
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

    _section_3_test_cli_entrypoint_does_not_assemble_the_argparse_tree()


def _is_artifact_path_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(
        ("workspace/", "data/")
    )


def _looks_like_path_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and any(
        marker in node.id.lower() for marker in ("root", "dir", "path")
    )


# ── *_DIR 产物目录零引用守卫 ────────────────────────────────────────────────

# 白名单：常量名 → 保留理由（不许空理由）。任务 E 后 FACTOR_STORE_DIR 已接线，不进白名单。
_DIR_ZERO_REF_ALLOWLIST: dict[str, str] = {
    "WORKSPACE_OPS_DIR": (
        "运维杂项目录；tools/ 脚本消费（repair_raw_partition/ingest_minute），"
        "src/factorzen 包内零引用为设计（运维与产品路径分离）"
    ),
    "EXECUTION_DIR": (
        "纸面执行会话根；CLI live 经 --session-dir 显式传入任意路径，"
        "SessionStore 不读常量（会话可落任意目录）"
    ),
    "OPS_DIR": (
        "仅作 OPS_SITE_DIR/OPS_STATE_DIR 的父路径中间量；"
        "生产读站点/状态用后两者，本常量无直接消费点"
    ),
    "CONFIG_DIR": (
        "workspace/configs 约定路径；CLI/流水线经 --config 显式路径注入，"
        "未默认绑定该常量"
    ),
    "DATA_DIR": (
        "data 根；包内经 DATA_RAW/DATA_CACHE 等子常量间接使用；"
        "tools/download_tushare_lake 直接引用，src 层保持子路径粒度"
    ),
    "OUTPUT_DIR": (
        "历史 artifacts 根；日频评估已废除 factors/results 双写，仅 OUTPUT_INTRADAY_* "
        "等派生路径在 settings 内保留，包外无直接引用"
    ),
    "COMMON_DIR": "历史源码路径别名（core/），现用包 import 而非路径常量，保留供文档/脚本",
    "REPORTING_DIR": "历史 reports 路径别名，现用 factorzen.reports 包导入，保留供文档/脚本",
    "NOTEBOOKS_DIR": "研究 notebook 目录约定，产品代码不读写 notebook，仅路径声明",
    "TESTS_DIR": "测试根路径约定，运行时由 pytest 发现 tests/，产品代码不引用",
}


def _settings_dir_constant_names() -> list[str]:
    """解析 config/settings.py 顶层 ``*_DIR`` 赋值名。"""
    settings = PACKAGE_ROOT / "config" / "settings.py"
    tree = ast.parse(settings.read_text(encoding="utf-8-sig"), filename=str(settings))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.endswith("_DIR"):
                    names.append(t.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id.endswith("_DIR")
        ):
            names.append(node.target.id)
    return names


def test_settings_dir_constants_have_consumers_or_whitelist():
    """每个 settings ``*_DIR`` 在 src/factorzen（除 settings 自身）须有引用，否则白名单。"""
    import re

    names = _settings_dir_constant_names()
    assert names, "settings.py 应声明至少一个 *_DIR"

    zero: list[str] = []
    for name in names:
        found = False
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            if path == PACKAGE_ROOT / "config" / "settings.py":
                continue
            text = path.read_text(encoding="utf-8-sig")
            if re.search(rf"\b{re.escape(name)}\b", text):
                found = True
                break
        if not found:
            zero.append(name)

    allow = _DIR_ZERO_REF_ALLOWLIST
    # 白名单项必须有非空理由
    empty_reasons = [k for k, v in allow.items() if not (v and str(v).strip())]
    assert not empty_reasons, f"白名单理由为空: {empty_reasons}"

    unexpected_zero = [n for n in zero if n not in allow]
    stale_allow = [n for n in allow if n not in zero]
    # 已有引用却仍在白名单 → 提示清理（不 fail 也可，但更严：fail 逼清理）
    msgs: list[str] = []
    if unexpected_zero:
        msgs.append(
            "零引用 *_DIR 未进白名单（接线或加入白名单并注明理由）: "
            + ", ".join(unexpected_zero)
        )
    if stale_allow:
        msgs.append(
            "白名单项已有引用，请从白名单移除: " + ", ".join(stale_allow)
        )
    assert not msgs, "\n".join(msgs)


