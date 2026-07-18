"""因子库 python 型（kind）+ 库池物化分派单测。TDD、mock 离线。"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────


def _panel(n_days: int = 4, n_stocks: int = 3, *, offset: float = 0.0) -> pl.DataFrame:
    rows = []
    for d in range(n_days):
        dt = date(2024, 1, 2) + timedelta(days=d)
        for i in range(n_stocks):
            rows.append({
                "trade_date": dt,
                "ts_code": f"{i:06d}.SH",
                "factor_value": float(i + 1) + offset + d * 0.01,
            })
    return pl.DataFrame(rows)


def _daily_grid(n_days: int = 4, n_stocks: int = 3) -> pl.DataFrame:
    """与库池 expression 路径同口径的网格帧（仅 trade_date/ts_code + 占位价列）。"""
    rows = []
    for d in range(n_days):
        dt = date(2024, 1, 2) + timedelta(days=d)
        for i in range(n_stocks):
            rows.append({
                "trade_date": dt,
                "ts_code": f"{i:06d}.SH",
                "close": 10.0 + i,
                "close_adj": 10.0 + i,
            })
    return pl.DataFrame(rows)


def _write_lib(root: Path, market: str, records: list[dict]) -> None:
    import json

    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


# ── 1. from_dict 旧行兼容 + round-trip ───────────────────────────────────────


def test_factor_record_legacy_from_dict_defaults_kind_expression():
    from factorzen.discovery.factor_library import FactorRecord

    r = FactorRecord.from_dict({"expression": "rank(close)", "market": "ashare"})
    assert r.kind == "expression"
    assert r.name is None
    assert r.impl is None

    d = r.to_dict()
    assert d["kind"] == "expression"
    assert "name" in d and "impl" in d
    r2 = FactorRecord.from_dict(d)
    assert r2.kind == "expression"
    assert r2.expression == "rank(close)"


# ── 2. identity helpers ──────────────────────────────────────────────────────


def test_python_identity_helpers_deterministic():
    from factorzen.discovery.factor_library import (
        default_name_for_expression,
        is_python_identity,
        python_identity,
    )

    assert python_identity("hf_resiliency") == "py::hf_resiliency"
    assert is_python_identity("py::hf_resiliency") is True
    assert is_python_identity("rank(close)") is False
    assert is_python_identity(None) is False
    assert is_python_identity("") is False
    assert is_python_identity("py:") is False

    expr = "rank(close)"
    expected = f"mined_{hashlib.sha1(expr.encode()).hexdigest()[:8]}"
    assert default_name_for_expression(expr) == expected
    # 确定性：多次一致
    assert default_name_for_expression(expr) == default_name_for_expression(expr)


# ── 3. _save_library name 回填 ───────────────────────────────────────────────


def test_save_library_backfills_name_idempotent(tmp_path):
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _normalize,
        _save_library,
        default_name_for_expression,
        load_library,
        python_identity,
    )

    expr_rec = FactorRecord(expression="rank(close)", market="ashare")
    py_rec = FactorRecord(
        expression=python_identity("foo"), market="ashare", kind="python",
    )
    _save_library("ashare", [expr_rec, py_rec], root=str(tmp_path))

    lib = load_library("ashare", root=str(tmp_path))
    by_expr = {r.expression: r for r in lib}
    assert by_expr["rank(close)"].name == default_name_for_expression(
        _normalize("rank(close)")
    )
    assert by_expr[python_identity("foo")].name == "foo"
    assert by_expr[python_identity("foo")].kind == "python"

    # 幂等：再 save 名字不变
    names_before = {r.expression: r.name for r in lib}
    _save_library("ashare", lib, root=str(tmp_path))
    lib2 = load_library("ashare", root=str(tmp_path))
    names_after = {r.expression: r.name for r in lib2}
    assert names_before == names_after


# ── 4. _record_from_candidate ────────────────────────────────────────────────


def test_record_from_candidate_python_identity_and_explicit_keys():
    from factorzen.discovery.factor_library import (
        _record_from_candidate,
        python_identity,
    )

    rec = _record_from_candidate(
        {"expression": python_identity("bar"), "ic_train": 0.05},
        norm_expr=python_identity("bar"),
        market="ashare",
        eval_window=("20200101", "20240101"),
        universe="csi300",
        horizon=1,
        run_id="r1",
        session_dir="s1",
        git_sha="abc",
        now="2026-07-18",
        prev=None,
    )
    assert rec.kind == "python"
    assert rec.name == "bar"
    assert rec.impl == "bar"
    assert rec.expression == python_identity("bar")

    # 显式键优先于推断
    rec2 = _record_from_candidate(
        {
            "expression": python_identity("bar"),
            "kind": "python",
            "name": "custom_name",
            "impl": "custom_impl",
        },
        norm_expr=python_identity("bar"),
        market="ashare",
        eval_window=(None, None),
        universe=None,
        horizon=None,
        run_id=None,
        session_dir=None,
        git_sha=None,
        now="2026-07-18",
        prev=None,
    )
    assert rec2.kind == "python"
    assert rec2.name == "custom_name"
    assert rec2.impl == "custom_impl"


# ── 5. materialize_python_panel ──────────────────────────────────────────────


def test_materialize_python_panel_offline(monkeypatch):
    from datetime import datetime

    from factorzen.daily.factors.base import DailyFactor
    from factorzen.discovery import python_factor as pyf
    from factorzen.discovery.python_factor import materialize_python_panel

    class FakeFactor(DailyFactor):
        name = "fake_py_factor"
        lookback_days = 3
        required_data = ["daily"]
        description = "test"

        def compute(self, ctx):
            # 从 expanded_start 到 end 造面板：扩窗行 + 窗口内行
            start_d = datetime.strptime(ctx.start, "%Y%m%d").date()
            end_d = datetime.strptime(ctx.end, "%Y%m%d").date()
            exp_d = datetime.strptime(ctx.expanded_start, "%Y%m%d").date()
            rows = []
            d = exp_d
            while d <= end_d:
                if d.weekday() < 5:
                    for i in range(2):
                        rows.append({
                            "trade_date": d,
                            "ts_code": f"{i:06d}.SH",
                            "factor_value": 1.0 if d >= start_d else -99.0,
                        })
                d += timedelta(days=1)
            return pl.DataFrame(rows)

    # 先正常 import 再 patch，避免 string-target 首次导入陷阱
    import factorzen.daily.factors.registry as reg_mod

    monkeypatch.setattr(reg_mod, "get_factor", lambda name: FakeFactor)
    # 离线：不碰 membership / tushare
    monkeypatch.setattr(
        pyf, "_load_universe_codes",
        lambda start, end, universe: ["000000.SH", "000001.SH"],
    )
    # 日历扩窗：mock expanded_start 路径（prev_trade_date 会碰日历缓存）
    from factorzen.daily.data import context as ctx_mod

    real_expanded = ctx_mod.FactorDataContext.expanded_start

    def _fake_expanded(self):
        # start 往前 lookback_days 个自然日（测试不依赖交易日历）
        d = datetime.strptime(self.start, "%Y%m%d").date() - timedelta(
            days=self.lookback_days + 2
        )
        return d.strftime("%Y%m%d")

    monkeypatch.setattr(
        ctx_mod.FactorDataContext, "expanded_start", property(_fake_expanded),
    )

    start, end = "20240110", "20240115"
    out = materialize_python_panel(
        "fake_py_factor", start, end, "csi300", market="ashare",
    )
    assert set(out.columns) == {"trade_date", "ts_code", "factor_value"}
    # 过滤后只剩 [start, end] 且无扩窗哨兵值
    assert out["factor_value"].min() >= 0.0
    tds = out["trade_date"].to_list()
    start_d = datetime.strptime(start, "%Y%m%d").date()
    end_d = datetime.strptime(end, "%Y%m%d").date()
    for td in tds:
        if hasattr(td, "year"):
            assert start_d <= td <= end_d
        else:
            s = str(td).replace("-", "")[:8]
            assert start <= s <= end

    # 非 ashare
    with pytest.raises(ValueError, match=r"A股|ashare"):
        materialize_python_panel("x", start, end, "csi300", market="crypto")

    # 未注册
    def _boom(name):
        raise KeyError(name)

    monkeypatch.setattr(reg_mod, "get_factor", _boom)
    with pytest.raises(ValueError, match=r"未注册|not registered|未知"):
        materialize_python_panel("no_such", start, end, "csi300", market="ashare")

    # restore property ref unused
    _ = real_expanded


# ── 6. build_library_pool 分派 ───────────────────────────────────────────────


def test_build_library_pool_dispatches_python_and_expression(tmp_path):
    from factorzen.discovery.factor_library import (
        build_library_pool,
        python_identity,
    )

    daily = _daily_grid()
    py_key = python_identity("fake_py")
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "kind": "expression", "ic_train": 0.05},
        {"expression": py_key, "market": "ashare", "status": "active",
         "kind": "python", "name": "fake_py", "impl": "fake_py",
         "ic_train": 0.04},
    ])

    # 假面板比网格多一天 + 多一股，验证 inner-join 限制到网格
    extra = _panel(n_days=5, n_stocks=4, offset=10.0)

    def _mat(name: str) -> pl.DataFrame:
        assert name == "fake_py"
        return extra

    pool = build_library_pool(
        "ashare", daily, root=str(tmp_path), compact=False,
        python_materializer=_mat,
    )
    assert set(pool.keys()) == {"rank(close)", py_key}
    py_panel = pool[py_key]
    grid_keys = set(zip(
        daily["trade_date"].to_list(), daily["ts_code"].to_list(), strict=True,
    ))
    panel_keys = set(zip(
        py_panel["trade_date"].to_list(), py_panel["ts_code"].to_list(), strict=True,
    ))
    assert panel_keys.issubset(grid_keys)
    assert py_panel.height == daily.height  # 全网格有值


def test_build_library_pool_skips_python_without_universe_or_materializer(
    tmp_path, caplog,
):
    import logging

    from factorzen.discovery.factor_library import (
        build_library_pool,
        python_identity,
    )

    daily = _daily_grid()
    py_key = python_identity("orphan")
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "kind": "expression", "ic_train": 0.05},
        {"expression": py_key, "market": "ashare", "status": "active",
         "kind": "python", "name": "orphan", "impl": "orphan",
         "ic_train": 0.04},
    ])
    with caplog.at_level(logging.WARNING):
        pool = build_library_pool(
            "ashare", daily, root=str(tmp_path), compact=False,
        )
    assert set(pool.keys()) == {"rank(close)"}
    assert py_key not in pool
    assert any("python" in m.lower() or "universe" in m.lower()
               for m in caplog.messages)


def test_build_library_pool_compact_python_dispatch(tmp_path):
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
        python_identity,
    )

    daily = _daily_grid()
    py_key = python_identity("fake_py")
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "kind": "expression", "ic_train": 0.05},
        {"expression": py_key, "market": "ashare", "status": "active",
         "kind": "python", "name": "fake_py", "impl": "fake_py",
         "ic_train": 0.04},
    ])
    fake = _panel()

    pool = build_library_pool(
        "ashare", daily, root=str(tmp_path), compact=True,
        python_materializer=lambda name: fake,
    )
    assert isinstance(pool, CompactLibraryPool)
    assert set(pool.keys()) == {"rank(close)", py_key}
    py_panel = pool[py_key]
    assert set(py_panel.columns) >= {"trade_date", "ts_code", "factor_value"}
    assert py_panel.height > 0


def test_pool_cache_python_key_guards_stale_hit(tmp_path):
    """脏缓存防线：先无 universe 建缓存（python 被跳过）→ 后带 universe 装载必须失效。

    纯 expression 库键恒 None（universe 无关，不无谓失效）。
    """
    from factorzen.discovery.factor_library import (
        load_pool_cache,
        python_identity,
        python_pool_cache_key,
        write_pool_cache,
    )

    lib_root = tmp_path / "lib"
    lib_root.mkdir()
    # 纯 expression 库：键恒 None
    _write_lib(lib_root, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active"},
    ])
    assert python_pool_cache_key(
        "ashare", root=str(lib_root), statuses=("active",), universe="csi300",
    ) is None
    # 加入 python 记录后：无 universe → "<missing>"，有 → universe，注入 → "<injected>"
    _write_lib(lib_root, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active"},
        {"expression": python_identity("foo"), "market": "ashare",
         "status": "active", "kind": "python", "name": "foo"},
    ])
    key_missing = python_pool_cache_key(
        "ashare", root=str(lib_root), statuses=("active",), universe=None,
    )
    assert key_missing == "<missing>"
    assert python_pool_cache_key(
        "ashare", root=str(lib_root), statuses=("active",), universe="csi300",
    ) == "csi300"
    assert python_pool_cache_key(
        "ashare", root=str(lib_root), statuses=("active",), universe=None,
        injected=True,
    ) == "<injected>"

    # 空池缓存 + key="<missing>"：同键命中返回 {}，异键（补了 universe）失效
    from factorzen.discovery.factor_library import library_file_hash

    cache_dir = tmp_path / "cache"
    meta = {
        "market": "ashare",
        "statuses": ["active"],
        "eval_start": None,
        "library_hash": library_file_hash("ashare", str(lib_root)),
        "prepped_height": 12,
        "prepped_date_min": "2024-01-02",
        "prepped_date_max": "2024-01-05",
        "python_pool_key": key_missing,
    }
    write_pool_cache({}, cache_dir, meta=meta)
    common = dict(
        market="ashare", root=str(lib_root), statuses=("active",),
        eval_start=None, expect_height=12,
        expect_date_min="2024-01-02", expect_date_max="2024-01-05",
    )
    assert load_pool_cache(cache_dir, **common, python_key="<missing>") == {}
    assert load_pool_cache(cache_dir, **common, python_key="csi300") is None


def test_build_library_pool_skips_python_panel_with_duplicate_keys(
    tmp_path, caplog,
):
    """重复 (trade_date, ts_code) 是作者 bug：legacy 面板失真、compact 列错位，须响亮跳过。"""
    import logging

    from factorzen.discovery.factor_library import (
        build_library_pool,
        python_identity,
    )

    daily = _daily_grid()
    py_key = python_identity("dupe_py")
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "kind": "expression", "ic_train": 0.05},
        {"expression": py_key, "market": "ashare", "status": "active",
         "kind": "python", "name": "dupe_py", "impl": "dupe_py",
         "ic_train": 0.04},
    ])
    good = _panel()
    duped = pl.concat([good, good.head(2)])  # 前两行键重复

    for compact in (False, True):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            pool = build_library_pool(
                "ashare", daily, root=str(tmp_path), compact=compact,
                python_materializer=lambda name: duped,
            )
        assert set(pool.keys()) == {"rank(close)"}, f"compact={compact}"
        assert any("重复" in m for m in caplog.messages), f"compact={compact}"
