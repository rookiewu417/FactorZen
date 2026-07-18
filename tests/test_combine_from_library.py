"""combine from-library：因子库选品 → 物化 → 四方法 OOS。

mock 形态照抄 test_combine_from_session：string-target monkeypatch prepare_mining_daily。
注意 monkeypatch 首次导入陷阱：先 import 模块再 patch 其属性。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.pipelines import factor_combine


def _daily(n_stocks=40, n_days=200, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c, px = f"{i:06d}.SZ", 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "close_adj": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _write_lib(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _expr_rec(
    expression: str,
    *,
    name: str | None = None,
    status: str = "active",
    ic_train: float = 0.05,
    **extra,
) -> dict:
    d = {
        "expression": expression,
        "market": "ashare",
        "status": status,
        "kind": "expression",
        "ic_train": ic_train,
        "name": name,
    }
    d.update(extra)
    return d


def test_combine_from_library_end_to_end(tmp_path, monkeypatch):
    """3 条 expression active → 跑通；factors_used 是 name；manifest/返回字段齐全。"""
    # 先 import 再 patch，避免 string-target 首次导入陷阱
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    _write_lib(lib, "ashare", [
        _expr_rec("rank(close)", name="f_close", ic_train=0.08),
        _expr_rec("ts_mean(vol,5)", name="f_vol", ic_train=0.06),
        _expr_rec("neg(rank(ts_std(close,10)))", name="f_vol_neg", ic_train=0.04),
    ])
    res = factor_combine.combine_from_library(
        market="ashare",
        library_root=str(lib),
        start="20230103",
        end="20231231",
        universe=None,
        horizon=5,
        train_days=60,
        test_days=15,
        decorr_threshold=1.0,
        out_dir=str(tmp_path / "out"),
    )
    comp = res["comparison"]
    methods = set(comp["method"].to_list())
    assert {"equal_weight", "ic_weighted", "max_ir"} <= methods
    assert comp.height >= 3
    # 可读 name，不是 factor_{i} / 表达式原文
    assert set(res["factors_used"]) == {"f_close", "f_vol", "f_vol_neg"}
    assert res["factors_status"] == {
        "f_close": "active", "f_vol": "active", "f_vol_neg": "active",
    }
    assert res["skipped_materialize"] == []
    assert res["dropped_correlated"] == []
    assert res["market"] == "ashare"
    assert res["statuses"] == ["active"]
    assert res["library_hash"] is not None
    assert "run_dir" in res
    assert res.get("truncated_from") is None


def test_combine_from_library_statuses_filter(tmp_path, monkeypatch):
    """probation 默认不入选；statuses 含 probation 则入选。"""
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    _write_lib(lib, "ashare", [
        _expr_rec("rank(close)", name="a_active", status="active", ic_train=0.08),
        _expr_rec("rank(vol)", name="a_prob", status="probation", ic_train=0.07),
        _expr_rec("rank(high)", name="b_active", status="active", ic_train=0.05),
    ])
    # 默认 statuses=active → 只有 2 条 active；应能跑
    res = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        start="20230103", end="20231231",
        train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o1"),
    )
    assert "a_prob" not in res["factors_used"]
    assert set(res["factors_used"]) == {"a_active", "b_active"}

    # 含 probation → 3 条入选
    res2 = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        statuses=("active", "probation"),
        start="20230103", end="20231231",
        train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o2"),
    )
    assert "a_prob" in res2["factors_used"]
    assert len(res2["factors_used"]) == 3


def test_combine_from_library_python_with_expression(tmp_path, monkeypatch):
    """python 面板注入 + expression 同台进组合；universe=None + python → ValueError。"""
    import factorzen.discovery.factor_library as fl
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    daily = _daily()
    # 与网格同口径的假 python 面板
    py_panel = daily.select([
        "trade_date", "ts_code",
        (pl.col("close_adj") * 0.01).alias("factor_value"),
    ])

    def _fake_mat(r, df, *, market, universe, python_materializer, start, end):
        # 只服务我们的 fake_py；其它走原路径（本测无）
        if (r.name or "") == "fake_py" or fl.is_python_identity(r.expression):
            return (
                df.select(["trade_date", "ts_code"])
                .join(py_panel, on=["trade_date", "ts_code"], how="inner")
            )
        return None

    monkeypatch.setattr(fl, "_materialize_python_on_grid", _fake_mat)

    lib = tmp_path / "lib"
    py_key = fl.python_identity("fake_py")
    _write_lib(lib, "ashare", [
        _expr_rec("rank(close)", name="e1", ic_train=0.08),
        {
            "expression": py_key, "market": "ashare", "status": "active",
            "kind": "python", "name": "fake_py", "impl": "fake_py",
            "ic_train": 0.06,
        },
        _expr_rec("ts_mean(vol,5)", name="e2", ic_train=0.04),
    ])

    res = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        start="20230103", end="20231231", universe="csi300",
        train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o"),
    )
    assert "fake_py" in res["factors_used"]
    assert "e1" in res["factors_used"] and "e2" in res["factors_used"]

    with pytest.raises(ValueError, match=r"universe|python"):
        factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            start="20230103", end="20231231", universe=None,
            train_days=60, test_days=15, out_dir=str(tmp_path / "o2"),
        )


def test_combine_from_library_skip_bad_expression(tmp_path, monkeypatch):
    """坏表达式跳过并记入 skipped_materialize；剩余 ≥2 仍跑。"""
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    bad = "this_is_not_a_valid_expr_zzz()"
    _write_lib(lib, "ashare", [
        _expr_rec("rank(close)", name="ok1", ic_train=0.08),
        _expr_rec(bad, name="bad_one", ic_train=0.07),
        _expr_rec("ts_mean(vol,5)", name="ok2", ic_train=0.05),
    ])
    res = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        start="20230103", end="20231231",
        train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o"),
    )
    assert bad in res["skipped_materialize"]
    assert "bad_one" not in res["factors_used"]
    assert set(res["factors_used"]) == {"ok1", "ok2"}


def test_combine_from_library_needs_two_and_top_n(tmp_path, monkeypatch):
    """<2 记录 → ValueError；top_n 截断记 truncated_from。"""
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

    lib = tmp_path / "lib"
    _write_lib(lib, "ashare", [
        _expr_rec("rank(close)", name="only_one", ic_train=0.08),
    ])
    with pytest.raises(ValueError, match="不足 2 个"):
        factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            start="20230103", end="20231231",
            out_dir=str(tmp_path / "o"),
        )

    _write_lib(lib, "ashare", [
        _expr_rec("rank(close)", name="n1", ic_train=0.09),
        _expr_rec("rank(vol)", name="n2", ic_train=0.07),
        _expr_rec("rank(high)", name="n3", ic_train=0.05),
        _expr_rec("ts_mean(vol,5)", name="n4", ic_train=0.03),
    ])
    res = factor_combine.combine_from_library(
        market="ashare", library_root=str(lib),
        start="20230103", end="20231231",
        top_n=2, train_days=60, test_days=15, decorr_threshold=1.0,
        out_dir=str(tmp_path / "o2"),
    )
    assert res["truncated_from"] == 4
    assert len(res["factors_used"]) == 2
    # |ic_train| 降序：n1, n2
    assert res["factors_used"] == ["n1", "n2"]


def test_combine_from_library_cli_parser_smoke():
    """CLI 参数解析冒烟：不炸、statuses 逗号解析正确。"""
    from factorzen.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "combine", "from-library",
        "--start", "20230103",
        "--end", "20231231",
        "--market", "ashare",
        "--statuses", "active,probation",
        "--top-n", "10",
        "--universe", "csi300",
        "--library-root", "/tmp/lib",
    ])
    assert args.combine_command == "from-library"
    assert args.market == "ashare"
    assert args.statuses == ("active", "probation")
    assert args.top_n == 10
    assert args.universe == "csi300"
    assert args.library_root == "/tmp/lib"
    assert args.start == "20230103"
    assert callable(args.func)

    # 非法 status
    with pytest.raises(SystemExit):
        parser.parse_args([
            "combine", "from-library",
            "--start", "20230103", "--end", "20231231",
            "--statuses", "active,bogus",
        ])
