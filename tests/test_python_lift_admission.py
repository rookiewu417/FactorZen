"""python 因子 lift 准入：物化分派 / CLI / rebuild / forward。离线 mock，不碰 tushare。"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta

import polars as pl

# ── helpers ──────────────────────────────────────────────────────────────────


def _grid(n_days: int = 4, n_stocks: int = 3) -> pl.DataFrame:
    rows = []
    for d in range(n_days):
        dt = date(2024, 1, 2) + timedelta(days=d)
        for i in range(n_stocks):
            rows.append({
                "trade_date": dt,
                "ts_code": f"{i:06d}.SH",
                "close": 10.0 + i,
                "close_adj": 10.0 + i,
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "vol": 1e5,
                "amount": 1e7,
            })
    return pl.DataFrame(rows)


def _panel_extra(
    n_days: int = 6, n_stocks: int = 5, *, offset: float = 0.0,
) -> pl.DataFrame:
    """比 grid 更大的假面板（含 grid 外 (date,code)，用于验证 inner-join 裁剪）。"""
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


def _lift_row(expr, *, lift=0.005, lift_se=0.001, lift_second_half=0.004, **extra):
    d = {
        "expression": expr,
        "lift": lift,
        "lift_se": lift_se,
        "lift_first_half": 0.01,
        "lift_second_half": lift_second_half,
        "baseline": 0.04,
        "passed": True,
    }
    d.update(extra)
    return d


def _meta(**kw):
    base = {
        "session_dir": "sess/abc",
        "run_id": "run42",
        "universe": "csi300",
        "eval_start": "20200101",
        "eval_end": "20260101",
        "horizon": 5,
        "git_sha": "deadbeef",
        "now": "2026-07-14",
    }
    base.update(kw)
    return base


# ── 1. _materializer_from_prepped py:: 分派 ───────────────────────────────────


def test_materializer_from_prepped_python_dispatch(monkeypatch, caplog):
    from factorzen.discovery.factor_library import python_identity
    from factorzen.discovery.lift_test import _materializer_from_prepped

    prepped = _grid(n_days=4, n_stocks=3)
    fake = _panel_extra(n_days=6, n_stocks=5, offset=1.0)
    calls: list[dict] = []

    def fake_mat(name, start, end, universe, *, market="ashare"):
        calls.append({
            "name": name, "start": start, "end": end,
            "universe": universe, "market": market,
        })
        return fake

    monkeypatch.setattr(
        "factorzen.discovery.python_factor.materialize_python_panel", fake_mat,
    )

    mat = _materializer_from_prepped(
        prepped, {"close": "close"},
        python_universe="csi300",
        python_market="ashare",
    )
    out = mat(python_identity("fake_py"))
    assert out is not None
    assert set(out.columns) >= {"trade_date", "ts_code", "factor_value"}
    # 限制到 prepped 网格：最多 4d × 3 stocks
    assert out.height <= prepped.height
    keys = out.select(["trade_date", "ts_code"]).unique()
    grid_keys = prepped.select(["trade_date", "ts_code"]).unique()
    joined = keys.join(grid_keys, on=["trade_date", "ts_code"], how="anti")
    assert joined.height == 0
    assert calls and calls[0]["name"] == "fake_py"
    assert calls[0]["universe"] == "csi300"

    # universe 缺失 → None + warning 一次
    with caplog.at_level(logging.WARNING):
        mat_no = _materializer_from_prepped(prepped, {"close": "close"})
        assert mat_no(python_identity("fake_py")) is None
        assert mat_no(python_identity("other")) is None  # 第二次不重复刷屏
    warn_msgs = [r.message for r in caplog.records if "python_universe" in r.message]
    assert len(warn_msgs) == 1

    # 表达式候选不受影响（parse 路径；坏 expr → None 不崩）
    assert mat("not_a_valid_expr!!!") is None


# ── 2. run_lift_tests 混合候选 ───────────────────────────────────────────────


def test_run_lift_tests_mixed_expression_and_python(monkeypatch):
    from factorzen.discovery.factor_library import python_identity
    from factorzen.discovery.lift_test import run_lift_tests

    daily = _grid(n_days=5, n_stocks=3)
    py_key = python_identity("hf_mock")
    expr_key = "rank(close)"

    def mat(expr: str):
        return pl.DataFrame({
            "trade_date": daily["trade_date"],
            "ts_code": daily["ts_code"],
            "factor_value": [float(i % 3) for i in range(daily.height)],
        })

    active = {
        "lib_a": pl.DataFrame({
            "trade_date": daily["trade_date"],
            "ts_code": daily["ts_code"],
            "factor_value": [0.1 + 0.01 * i for i in range(daily.height)],
        }),
    }
    ret_df = pl.DataFrame({
        "trade_date": daily["trade_date"],
        "ts_code": daily["ts_code"],
        "ret": [0.01 * ((i % 3) - 1) for i in range(daily.height)],
    })

    rows = run_lift_tests(
        [
            {"expression": expr_key, "residual_ic_train": 0.006, "ic_train": 0.02},
            {
                "expression": py_key, "kind": "python", "name": "hf_mock",
                "impl": "hf_mock", "residual_ic_train": 0.005, "ic_train": 0.01,
            },
        ],
        market="ashare",
        daily=daily,
        top_m=None,
        active_factor_dfs=active,
        ret_df=ret_df,
        materialize_candidate=mat,
        lift_workers=1,
    )
    assert len(rows) == 2
    exprs = {r["expression"] for r in rows}
    assert exprs == {expr_key, py_key}
    py_row = next(r for r in rows if r["expression"] == py_key)
    assert py_row["expression"] == py_key  # 哨兵保持
    # W3：候选身份原样回传（有则拷入；expression 型无 kind/name/impl 则不加键）
    assert py_row.get("kind") == "python"
    assert py_row.get("name") == "hf_mock"
    assert py_row.get("impl") == "hf_mock"
    expr_row = next(r for r in rows if r["expression"] == expr_key)
    assert "kind" not in expr_row and "name" not in expr_row and "impl" not in expr_row


# ── 3. upsert_lift_admissions python + market 守卫 ───────────────────────────


def test_upsert_lift_admissions_python_kind_and_crypto_guard(tmp_path):
    from factorzen.discovery.factor_library import (
        load_library,
        python_identity,
        upsert_lift_admissions,
    )

    py_key = python_identity("hf_mock")
    out = upsert_lift_admissions(
        [
            _lift_row(
                py_key, kind="python", name="hf_mock", impl="hf_mock",
                ic_train=0.02, holdout_ic=0.01,
            ),
        ],
        market="ashare",
        root=str(tmp_path),
        meta=_meta(),
        allow_active=False,  # cap → probation
    )
    assert out["added_probation"] == 1
    assert out["errors"] == []
    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    rec = lib[py_key]
    assert rec.kind == "python"
    assert rec.name == "hf_mock"
    assert rec.status == "probation"
    assert rec.admission_track == "lift"

    # market=crypto 的 py:: 行 → errors
    out2 = upsert_lift_admissions(
        [_lift_row(py_key, kind="python", name="hf_mock")],
        market="crypto",
        root=str(tmp_path / "crypto_root"),
        meta=_meta(universe=None),
        allow_active=False,
    )
    assert out2["added_active"] == 0
    assert out2["added_probation"] == 0
    assert len(out2["errors"]) == 1
    assert "ashare" in out2["errors"][0]["error"]
    assert load_library("crypto", root=str(tmp_path / "crypto_root")) == []


# ── 4. rebuild fresh 携带 python ─────────────────────────────────────────────


def test_rebuild_preserves_python_lift_and_single(tmp_path):
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        python_identity,
        rebuild,
    )

    py_lift = python_identity("py_lift_fac")
    py_single = python_identity("py_single_fac")
    recs = [
        FactorRecord(
            expression=py_lift, market="ashare", status="probation",
            admission_track="lift", kind="python", name="py_lift_fac",
            impl="py_lift_fac", ic_train=0.01, lift=0.005, lift_se=0.001,
            lift_second_half=0.004, added_at="2026-07-01", updated_at="2026-07-01",
            universe="csi300",
        ),
        FactorRecord(
            expression=py_single, market="ashare", status="active",
            admission_track="single", kind="python", name="py_single_fac",
            impl="py_single_fac", ic_train=0.03, holdout_ic=0.02,
            added_at="2026-07-01", updated_at="2026-07-01",
            universe="csi300",
        ),
    ]
    _save_library("ashare", recs, root=str(tmp_path))

    def evaluate(exprs):
        return []  # sources 为空路径

    def lift_runner(cands, *, active_factor_dfs=None, **kw):
        expr = cands[0]["expression"]
        # 维持 lift 轨
        return [_lift_row(expr, lift=0.006, lift_se=0.001, lift_second_half=0.003)]

    rebuild(
        "ashare",
        sources=[],
        eval_window=("20200101", "20260101"),
        universe="csi300",
        horizon=5,
        evaluate=evaluate,
        git_sha="abc",
        now="2026-07-14",
        root=str(tmp_path),
        fresh=True,
        lift_runner=lift_runner,
        active_factor_dfs={},
    )
    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert py_lift in lib
    assert py_single in lib
    assert lib[py_lift].kind == "python"
    assert lib[py_single].kind == "python"
    assert lib[py_single].status == "active"  # 原样写回
    # lift 轨经复审维持 active（mock lift 过门）
    assert lib[py_lift].status in ("active", "probation")

    man_path = tmp_path / "rebuild_ashare_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert man["preserved_python"] == 2


# ── 5. forward _materialize_panel py:: 分派 ──────────────────────────────────


def test_forward_materialize_panel_python(monkeypatch):
    from factorzen.discovery.factor_library import python_identity
    from factorzen.discovery.forward_track import _materialize_panel

    prepped = _grid(n_days=3, n_stocks=2)
    fake = _panel_extra(n_days=5, n_stocks=4)

    def fake_mat(name, start, end, universe, *, market="ashare"):
        assert name == "fwd_py"
        assert universe == "csi300"
        return fake

    monkeypatch.setattr(
        "factorzen.discovery.python_factor.materialize_python_panel", fake_mat,
    )
    out = _materialize_panel(
        python_identity("fwd_py"),
        prepped,
        None,
        python_universe="csi300",
        python_market="ashare",
    )
    assert out is not None
    assert out.height <= prepped.height
    anti = (
        out.select(["trade_date", "ts_code"]).unique()
        .join(
            prepped.select(["trade_date", "ts_code"]).unique(),
            on=["trade_date", "ts_code"],
            how="anti",
        )
    )
    assert anti.height == 0

    # universe 缺失 → None
    assert _materialize_panel(
        python_identity("fwd_py"), prepped, None,
    ) is None


# ── 6. CLI 参数校验 ──────────────────────────────────────────────────────────


def _ns(**kw):
    base = dict(
        session=None,
        factor=None,
        market="ashare",
        start="20200101",
        end="20201231",
        universe=None,
        top_m=20,
        threshold=None,
        seed=0,
        library_root=None,
        apply=False,
        dry_run=False,
        se_mult=1.0,
        allow_active=False,
        admission_start=None,
        admission_end=None,
        horizon=None,
        lift_workers=1,
        top_n=50,
        symbols=None,
        intraday_leaves=False,
        intraday_freq="5min",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_lift_test_factor_requires_universe(monkeypatch, capsys):
    from factorzen.cli.main import _cmd_factor_library_lift_test

    # 避免后续装配；校验应在更早 return 2
    rc = _cmd_factor_library_lift_test(
        _ns(factor=["momentum_20d"], universe=None, market="ashare"),
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "--universe" in err


def test_cli_lift_test_factor_rejects_crypto(monkeypatch, capsys):
    from factorzen.cli.main import _cmd_factor_library_lift_test

    rc = _cmd_factor_library_lift_test(
        _ns(factor=["momentum_20d"], universe="csi300", market="crypto"),
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "ashare" in err


def test_cli_lift_test_factor_unregistered(monkeypatch, capsys):
    from factorzen.cli.main import _cmd_factor_library_lift_test

    def boom(name):
        raise KeyError(name)

    monkeypatch.setattr(
        "factorzen.daily.factors.registry.get_factor", boom,
    )
    rc = _cmd_factor_library_lift_test(
        _ns(factor=["no_such_factor_xyz"], universe="csi300", market="ashare"),
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "未注册" in err or "no_such_factor_xyz" in err
