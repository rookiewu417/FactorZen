"""registry library provider：load_library_factors 注入 expression 型（Batch 2）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from factorzen.discovery.library_provider import load_library_factors


def _write_lib(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _default_name(expr: str) -> str:
    return f"mined_{hashlib.sha1(expr.encode()).hexdigest()[:8]}"


@pytest.fixture
def reg_mod():
    """daily registry 模块；teardown reset 全局单例——LibFactor 注入若滞留会
    污染同进程后续测试文件（全量跑实锤过 test_daily_factors 次序失败）。"""
    import factorzen.daily.factors.registry as reg

    yield reg
    reg._registry.reset()


# ── 1. 基本注入 ──────────────────────────────────────────────────────────────


def test_load_library_factors_registers_expression_records(tmp_path, reg_mod):
    named = "lib_prov_named_alpha"
    expr_named = "rank(close)"
    expr_anon = "neg(rank(close))"
    anon_name = _default_name(expr_anon)
    _write_lib(
        tmp_path,
        "ashare",
        [
            {
                "expression": expr_named,
                "market": "ashare",
                "kind": "expression",
                "name": named,
                "status": "active",
            },
            {
                "expression": expr_anon,
                "market": "ashare",
                "kind": "expression",
                "status": "probation",
                # name 缺省 → default_name_for_expression
            },
        ],
    )
    n = load_library_factors(market="ashare", root=str(tmp_path))
    assert n == 2

    cls_named = reg_mod.get_factor(named)
    inst = cls_named()
    assert inst.name == named
    assert inst.expression == expr_named
    assert inst.lookback_days >= 60
    assert "[active]" in inst.description

    cls_anon = reg_mod.get_factor(anon_name)
    inst_anon = cls_anon()
    assert inst_anon.expression == expr_anon
    assert "[probation]" in inst_anon.description


# ── 2. 冲突让位 ──────────────────────────────────────────────────────────────


def test_load_library_factors_yields_to_existing(tmp_path, reg_mod, caplog):
    from factorzen.daily.factors.base import DailyFactor

    conflict = "lib_prov_conflict_builtin"
    # 先注册同名假因子（模拟 workspace/builtin 占用）
    fake = type(
        "FakeConflict",
        (DailyFactor,),
        {
            "name": conflict,
            "frequency": "daily",
            "description": "fake occupant",
            "lookback_days": 20,
            "compute": lambda self, ctx: None,
        },
    )
    assert reg_mod._registry.register(fake, override=True) is True

    _write_lib(
        tmp_path,
        "ashare",
        [
            {
                "expression": "rank(vol)",
                "market": "ashare",
                "kind": "expression",
                "name": conflict,
                "status": "active",
            }
        ],
    )
    with caplog.at_level("WARNING"):
        n = load_library_factors(market="ashare", root=str(tmp_path))
    assert n == 0
    assert any("让位" in r.message or conflict in r.message for r in caplog.records)
    # 仍是假因子，非 LibFactor
    assert reg_mod.get_factor(conflict) is fake


# ── 3. python 型跳过 ─────────────────────────────────────────────────────────


def test_load_library_factors_skips_python_kind(tmp_path, reg_mod):
    py_name = "lib_prov_python_skip_xyz"
    _write_lib(
        tmp_path,
        "ashare",
        [
            {
                "expression": f"py::{py_name}",
                "market": "ashare",
                "kind": "python",
                "name": py_name,
                "impl": py_name,
                "status": "active",
            },
            {
                "expression": "rank(amount)",
                "market": "ashare",
                "kind": "expression",
                "name": "lib_prov_expr_ok",
                "status": "active",
            },
        ],
    )
    n = load_library_factors(market="ashare", root=str(tmp_path))
    assert n == 1
    reg_mod.get_factor("lib_prov_expr_ok")
    with pytest.raises(KeyError):
        reg_mod.get_factor(py_name)


# ── 4. 幂等 ──────────────────────────────────────────────────────────────────


def test_load_library_factors_idempotent(tmp_path, reg_mod, caplog):
    name = "lib_prov_idempotent_once"
    _write_lib(
        tmp_path,
        "ashare",
        [
            {
                "expression": "rank(high)",
                "market": "ashare",
                "kind": "expression",
                "name": name,
                "status": "active",
            }
        ],
    )
    n1 = load_library_factors(market="ashare", root=str(tmp_path))
    assert n1 == 1
    names_after_1 = reg_mod.list_factors()
    assert names_after_1.count(name) == 1

    with caplog.at_level("WARNING"):
        n2 = load_library_factors(market="ashare", root=str(tmp_path))
    assert n2 == 0
    names_after_2 = reg_mod.list_factors()
    assert names_after_2.count(name) == 1
    # 二次 load 不因自身已注入再刷「让位」warning
    assert not any("让位" in r.message and name in r.message for r in caplog.records)


# ── 5. 损坏库文件 ────────────────────────────────────────────────────────────


def test_load_library_factors_tolerates_corrupt_jsonl(tmp_path, reg_mod):
    path = tmp_path / "ashare.jsonl"
    path.write_text(
        '{"expression":"rank(low)","market":"ashare","kind":"expression","name":"lib_prov_ok_corrupt","status":"active"}\n'
        "NOT_JSON_LINE\n"
        '{"expression":"neg(rank(low))","market":"ashare","kind":"expression","name":"lib_prov_ok2_corrupt","status":"correlated"}\n',
        encoding="utf-8",
    )
    n = load_library_factors(market="ashare", root=str(tmp_path))
    assert n == 2
    reg_mod.get_factor("lib_prov_ok_corrupt")
    reg_mod.get_factor("lib_prov_ok2_corrupt")


# ── 6. CLI 冒烟 ──────────────────────────────────────────────────────────────


def test_cmd_factor_list_includes_library_factor(tmp_path, reg_mod, monkeypatch, capsys):
    import argparse

    from factorzen.cli import main as cli

    name = "lib_prov_cli_list_visible"
    _write_lib(
        tmp_path,
        "ashare",
        [
            {
                "expression": "rank(open)",
                "market": "ashare",
                "kind": "expression",
                "name": name,
                "status": "active",
            }
        ],
    )
    # 把默认库根指到 tmp（load_library_factors 无 root 参数时用 DEFAULT_ROOT）
    monkeypatch.setattr(
        "factorzen.discovery.factor_library.DEFAULT_ROOT",
        str(tmp_path),
    )
    # 同时 patch daily registry 里 load 用的默认（若已 import DEFAULT_ROOT 为值则走函数内再 import）
    args = argparse.Namespace(freq="daily")
    rc = cli._cmd_factor_list(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert name in out
