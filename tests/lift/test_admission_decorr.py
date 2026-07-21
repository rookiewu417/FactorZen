"""合并自: test_python_lift_admission.py, test_lift_decorrelation_gate.py
目标: test_admission_decorr.py

--- 来源 test_python_lift_admission.py ---
python 因子 lift 准入：物化分派 / CLI / rebuild / forward。离线 mock，不碰 tushare。

--- 来源 test_lift_decorrelation_gate.py ---
W1：lift 准入轨接相关性门。

背景：`upsert_lift_admissions` 体内从无 `_decorrelate` 调用（全仓唯一调用点是
`upsert`，`rebuild` 是委托它），所以整条 lift 准入轨从未去重。实测 150 条候选按
threshold=0.005 准入 12 条，两两 |rho| 最大 0.965、>=0.7 有 8/66 对，聚成的独立
信号仅 4-5 个——「准入数 != 信息量」。

机理：残差 lift 是对**冻结基线**算的，同批候选共用同一基线快照，所以两条互为
重复的候选拿到相同的正 lift，双双准入。语义正确的解法是「每准入一条就并入基线
再测下一条」（贪心 re-lift），0.7 相关门是它的廉价代理。

本文件的第一优先级是 P0 锚：`_decorrelate` 会**无条件覆写 status**，
而 lift 轨默认 `allow_active=False` 把 decision=active 压成 probation。
直接接线会把全部 capped probation 冲成 active，运营护栏整体失效。
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# ==== 来自 test_python_lift_admission.py ====
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


def test_python_materializer_dispatch_suite(caplog):
    """test_materializer_from_prepped_python_dispatch；test_forward_materialize_panel_python"""
    # -- 原 test_materializer_from_prepped_python_dispatch --
    def _section_0_test_materializer_from_prepped_python_dispatch(mp, caplog):
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

        mp.setattr(
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

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_materializer_from_prepped_python_dispatch(mp, caplog)

    # -- 原 test_forward_materialize_panel_python --
    def _section_1_test_forward_materialize_panel_python(mp):
        from factorzen.discovery.factor_library import python_identity
        from factorzen.discovery.forward_track import _materialize_panel

        prepped = _grid(n_days=3, n_stocks=2)
        fake = _panel_extra(n_days=5, n_stocks=4)

        def fake_mat(name, start, end, universe, *, market="ashare"):
            assert name == "fwd_py"
            assert universe == "csi300"
            return fake

        mp.setattr(
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

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_forward_materialize_panel_python(mp)


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


def test_python_upsert_rebuild_suite(tmp_path):
    """test_upsert_lift_admissions_python_kind_and_crypto_guard；test_rebuild_preserves_python_lift_and_single"""
    # -- 原 test_upsert_lift_admissions_python_kind_and_crypto_guard --
    def _section_0_test_upsert_lift_admissions_python_kind_and_crypto_guard(tmp_path):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_upsert_lift_admissions_python_kind_and_crypto_guard(_tp0)

    # -- 原 test_rebuild_preserves_python_lift_and_single --
    def _section_1_test_rebuild_preserves_python_lift_and_single(tmp_path):
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

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_rebuild_preserves_python_lift_and_single(_tp1)


# ── 4. rebuild fresh 携带 python ─────────────────────────────────────────────


# ── 5. forward _materialize_panel py:: 分派 ──────────────────────────────────


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


def test_cli_lift_factor_validate_suite(capsys):
    """test_cli_lift_test_factor_requires_universe；test_cli_lift_test_factor_rejects_crypto；test_cli_lift_test_factor_unregistered"""
    # -- 原 test_cli_lift_test_factor_requires_universe --
    def _section_0_test_cli_lift_test_factor_requires_universe(mp, capsys):
        from factorzen.cli.main import _cmd_factor_library_lift_test

        # 避免后续装配；校验应在更早 return 2
        rc = _cmd_factor_library_lift_test(
            _ns(factor=["momentum_20d"], universe=None, market="ashare"),
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "--universe" in err

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cli_lift_test_factor_requires_universe(mp, capsys)

    # -- 原 test_cli_lift_test_factor_rejects_crypto --
    def _section_1_test_cli_lift_test_factor_rejects_crypto(mp, capsys):
        from factorzen.cli.main import _cmd_factor_library_lift_test

        rc = _cmd_factor_library_lift_test(
            _ns(factor=["momentum_20d"], universe="csi300", market="crypto"),
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "ashare" in err

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cli_lift_test_factor_rejects_crypto(mp, capsys)

    # -- 原 test_cli_lift_test_factor_unregistered --
    def _section_2_test_cli_lift_test_factor_unregistered(mp, capsys):
        from factorzen.cli.main import _cmd_factor_library_lift_test

        def boom(name):
            raise KeyError(name)

        mp.setattr(
            "factorzen.daily.factors.registry.get_factor", boom,
        )
        rc = _cmd_factor_library_lift_test(
            _ns(factor=["no_such_factor_xyz"], universe="csi300", market="ashare"),
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "未注册" in err or "no_such_factor_xyz" in err

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cli_lift_test_factor_unregistered(mp, capsys)


# ==== 来自 test_lift_decorrelation_gate.py ====
_DATES = [f"2024-01-{d:02d}" for d in range(1, 21)]
_CODES = [f"{i:06d}.SZ" for i in range(40)]


def _panel(values: np.ndarray) -> pl.DataFrame:
    """values: (n_date, n_stock) → 长表面板。"""
    return pl.DataFrame({
        "trade_date": np.repeat(_DATES, len(_CODES)),
        "ts_code": np.tile(_CODES, len(_DATES)),
        "factor_value": values.reshape(-1).astype(float),
    })


def _make_panels(seed: int = 0) -> dict[str, pl.DataFrame]:
    """构造一组已知相关结构的面板：

    - ``dup_a`` / ``dup_b``：完全相同（rho = 1.0）
    - ``indep``：独立
    - ``lib_x``：库内既有 active 因子，与上述都不相关
    """
    rng = np.random.default_rng(seed)
    shape = (len(_DATES), len(_CODES))
    base = rng.standard_normal(shape)
    return {
        "dup_a": _panel(base),
        "dup_b": _panel(base.copy()),
        "indep": _panel(rng.standard_normal(shape)),
        "lib_x": _panel(rng.standard_normal(shape)),
    }


def _materializer(panels: dict[str, pl.DataFrame]):
    def _mat(expr: str) -> pl.DataFrame | None:
        return panels.get(expr)
    return _mat


def _row(expr: str, lift: float, *, se: float = 0.0005) -> dict:
    """一行 lift 结果。lift 远超阈值 → decision=active。

    注意真实 lift 结果行**没有 ir_train 字段**（实证：replay_150 parquet 仅 12 列），
    这正是贪心排序键必须改用 lift 的原因。此处也刻意不给 ir_train。
    """
    return {
        "expression": expr,
        "lift": lift,
        "lift_se": se,
        "lift_first_half": lift,
        "lift_second_half": lift,
        "n_blocks": 20,
        "lift_metric": "residual_ic_v1",
    }


def _seed_library(tmp_path, panels) -> str:
    """库里先放一条 single 轨 active 因子（lib_x）。"""
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    rec = FactorRecord(
        expression="lib_x", market="ashare", status="active",
        admission_track="single", ir_train=0.5,
    )
    _save_library("ashare", [rec], root=str(tmp_path))
    return str(tmp_path)


# ── P0 锚：cap 不得被去相关冲掉 ──────────────────────────────────────────────

def test_decorrelate_p0_skip_suite(tmp_path):
    """【W1 第一验收锚】allow_active=False 下走完去相关，capped probation 仍是 probation。；物化器缺失 → 跳过去相关 + 显式告警，status 一律不动。；门无事可做时（单条准入 + 库内无 active）只落标志、不告警。"""
    # -- 原 test_capped_probation_survives_decorrelation --
    def _section_0_test_capped_probation_survives_decorrelation(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

        panels = _make_panels()
        root = _seed_library(tmp_path, panels)
        out = upsert_lift_admissions(
            [_row("indep", 0.02)],
            market="ashare", root=root,
            materialize=_materializer(panels),
            allow_active=False,
        )
        assert out.get("capped_active") == 1, "前提不成立：这一行本应触发 cap"
        rec = {r.expression: r for r in load_library("ashare", root=root)}["indep"]
        assert rec.status == "probation", f"cap 被去相关冲成 {rec.status}"
        assert rec.admission_decision == "active", "统计裁决原文丢失"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_capped_probation_survives_decorrelation(_tp0)

    # -- 原 test_missing_materializer_skips_gate_without_touching_status --
    def _section_1_test_missing_materializer_skips_gate_without_touching_status(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

        panels = _make_panels()
        root = _seed_library(tmp_path, panels)
        with pytest.warns(UserWarning, match="去相关"):
            out = upsert_lift_admissions(
                [_row("indep", 0.02)],
                market="ashare", root=root,
                allow_active=False,   # 不传 materialize
            )
        assert out.get("decorrelation_skipped") is True
        rec = {r.expression: r for r in load_library("ashare", root=root)}["indep"]
        assert rec.status == "probation"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_missing_materializer_skips_gate_without_touching_status(_tp1)

    # -- 原 test_noop_skip_sets_flag_without_warning_noise --
    def _section_2_test_noop_skip_sets_flag_without_warning_noise(tmp_path):
        from factorzen.discovery.factor_library import upsert_lift_admissions

        with warnings.catch_warnings():
            warnings.simplefilter("error")   # 任何 warning 都会变成异常
            out = upsert_lift_admissions(
                [_row("indep", 0.02)],
                market="ashare", root=str(tmp_path),   # 空库
                allow_active=False,
            )
        assert out.get("decorrelation_skipped") is True

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_noop_skip_sets_flag_without_warning_noise(_tp2)


# ── 去重本体 ────────────────────────────────────────────────────────────────

def test_decorrelate_gate_body_suite(tmp_path):
    """rho=1.0 的一对同批进入 → 一条留、一条标 correlated + max_corr_in_lib 回填。；【陷阱 3 反例锚】贪心顺序由 lift 决定，不是表达式字母序。；不相关的候选不该被误杀（门的特异性）。；与库内既有 active 因子高相关的候选 → correlated（跨批去重，不只批内）。；D2：库内既有 probation 不进比较池（试用因子不该挡新候选）。；状态机单调性：已 forward-confirmed 的 active 不得被同批高 lift 新候选挤成 correlated。；计数口径：被改判 correlated 的不应继续计进 added_active/added_probation。"""
    # -- 原 test_duplicate_pair_one_marked_correlated --
    def _section_0_test_duplicate_pair_one_marked_correlated(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

        panels = _make_panels()
        root = _seed_library(tmp_path, panels)
        upsert_lift_admissions(
            [_row("dup_a", 0.02), _row("dup_b", 0.01)],
            market="ashare", root=root,
            materialize=_materializer(panels),
            allow_active=False,
        )
        by = {r.expression: r for r in load_library("ashare", root=root)}
        statuses = {by["dup_a"].status, by["dup_b"].status}
        assert statuses == {"probation", "correlated"}, f"得到 {statuses}"
        loser = by["dup_a"] if by["dup_a"].status == "correlated" else by["dup_b"]
        assert loser.max_corr_in_lib is not None
        assert loser.max_corr_in_lib > 0.99, f"rho 应≈1.0，得 {loser.max_corr_in_lib}"
        assert loser.correlated_with in ("dup_a", "dup_b")
        assert loser.admission_decision == "active", "被判 correlated 不应抹掉裁决原文"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_duplicate_pair_one_marked_correlated(_tp0)

    # -- 原 test_greedy_order_follows_lift_not_alphabet --
    def _section_1_test_greedy_order_follows_lift_not_alphabet(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

        panels = _make_panels()
        root = _seed_library(tmp_path, panels)
        upsert_lift_admissions(
            [_row("dup_a", 0.010), _row("dup_b", 0.050)],   # b 的 lift 高 5 倍，但字母序靠后
            market="ashare", root=root,
            materialize=_materializer(panels),
            allow_active=False,
        )
        by = {r.expression: r for r in load_library("ashare", root=root)}
        assert by["dup_b"].status == "probation", "lift 高者应占住位"
        assert by["dup_a"].status == "correlated", "lift 低者应被标 correlated"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_greedy_order_follows_lift_not_alphabet(_tp1)

    # -- 原 test_independent_candidates_all_admitted --
    def _section_2_test_independent_candidates_all_admitted(tmp_path):
        from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

        panels = _make_panels()
        root = _seed_library(tmp_path, panels)
        upsert_lift_admissions(
            [_row("indep", 0.02), _row("dup_a", 0.03)],
            market="ashare", root=root,
            materialize=_materializer(panels),
            allow_active=False,
        )
        by = {r.expression: r for r in load_library("ashare", root=root)}
        assert by["indep"].status == "probation"
        assert by["dup_a"].status == "probation"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_independent_candidates_all_admitted(_tp2)

    # -- 原 test_library_active_blocks_correlated_candidate --
    def _section_3_test_library_active_blocks_correlated_candidate(tmp_path):
        from factorzen.discovery.factor_library import (
            FactorRecord,
            _save_library,
            load_library,
            upsert_lift_admissions,
        )

        panels = _make_panels()
        # 库内 active 因子就是 dup_a；候选 dup_b 与它 rho=1.0
        _save_library("ashare", [FactorRecord(
            expression="dup_a", market="ashare", status="active",
            admission_track="single", ir_train=0.5,
        )], root=str(tmp_path))
        upsert_lift_admissions(
            [_row("dup_b", 0.02)],
            market="ashare", root=str(tmp_path),
            materialize=_materializer(panels),
            allow_active=False,
        )
        by = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
        assert by["dup_b"].status == "correlated"
        assert by["dup_b"].correlated_with == "dup_a"

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_library_active_blocks_correlated_candidate(_tp3)

    # -- 原 test_library_probation_does_not_block --
    def _section_4_test_library_probation_does_not_block(tmp_path):
        from factorzen.discovery.factor_library import (
            FactorRecord,
            _save_library,
            load_library,
            upsert_lift_admissions,
        )

        panels = _make_panels()
        _save_library("ashare", [FactorRecord(
            expression="dup_a", market="ashare", status="probation",
            admission_track="lift", ir_train=0.5,
        )], root=str(tmp_path))
        upsert_lift_admissions(
            [_row("dup_b", 0.02)],
            market="ashare", root=str(tmp_path),
            materialize=_materializer(panels),
            allow_active=False,
        )
        by = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
        assert by["dup_b"].status == "probation", "库内 probation 不应挡住新候选"

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_library_probation_does_not_block(_tp4)

    # -- 原 test_forward_confirmed_active_not_downgraded_by_new_candidate --
    def _section_5_test_forward_confirmed_active_not_downgraded_by_new_candidate(tmp_path):
        from factorzen.discovery.factor_library import (
            FactorRecord,
            _save_library,
            load_library,
            upsert_lift_admissions,
        )

        panels = _make_panels()
        _save_library("ashare", [FactorRecord(
            expression="dup_a", market="ashare", status="active",
            admission_track="lift", forward_confirmed_at="2026-06-01",
        )], root=str(tmp_path))

        upsert_lift_admissions(
            # dup_b 的 lift 远高于 dup_a，且两者 rho=1.0
            [_row("dup_a", 0.010), _row("dup_b", 0.900)],
            market="ashare", root=str(tmp_path),
            materialize=_materializer(panels), allow_active=False,
        )
        by = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
        assert by["dup_a"].status == "active", "已确认状态被撤销了"
        assert by["dup_a"].forward_confirmed_at == "2026-06-01"
        assert by["dup_b"].status == "correlated", "新候选应被已确认者挡住"

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    _section_5_test_forward_confirmed_active_not_downgraded_by_new_candidate(_tp5)

    # -- 原 test_correlated_records_removed_from_admission_counts --
    def _section_6_test_correlated_records_removed_from_admission_counts(tmp_path):
        from factorzen.discovery.factor_library import upsert_lift_admissions

        panels = _make_panels()
        root = _seed_library(tmp_path, panels)
        out = upsert_lift_admissions(
            [_row("dup_a", 0.02), _row("dup_b", 0.01), _row("indep", 0.03)],
            market="ashare", root=root,
            materialize=_materializer(panels), allow_active=False,
        )
        assert out.get("correlated") == 1
        # 3 条准入、1 条被判重复 → 只剩 2 条真进库
        assert out["added_active"] + out["added_probation"] == 2, out

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_correlated_records_removed_from_admission_counts(_tp6)


# ── 双路径一致性（登记簿要求）────────────────────────────────────────────────

def test_same_verdict_as_single_track(tmp_path):
    """两条轨对同一对因子的去相关裁决必须一致——复用 `_decorrelate` 单点的意义。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert,
        upsert_lift_admissions,
    )

    panels = _make_panels()
    lib_seed = [FactorRecord(
        expression="dup_a", market="ashare", status="active",
        admission_track="single", ir_train=0.5,
    )]

    # 单因子轨
    single_root = tmp_path / "single"
    single_root.mkdir()
    _save_library("ashare", list(lib_seed), root=str(single_root))
    upsert(
        "ashare",
        [{"expression": "dup_b", "ic_train": 0.05, "ir_train": 0.3,
          "holdout_ic": 0.03, "dsr_pvalue": 0.01, "n_holdout_days": 300}],
        eval_window=("2024-01-01", "2024-01-20"), universe=None, horizon=1,
        run_id=None, session_dir=None, git_sha=None, now="2026-07-18",
        materialize=_materializer(panels), root=str(single_root),
    )
    single_status = {
        r.expression: r for r in load_library("ashare", root=str(single_root))
    }["dup_b"].status

    # lift 轨
    lift_root = tmp_path / "lift"
    lift_root.mkdir()
    _save_library("ashare", list(lib_seed), root=str(lift_root))
    upsert_lift_admissions(
        [_row("dup_b", 0.02)],
        market="ashare", root=str(lift_root),
        materialize=_materializer(panels), allow_active=False,
    )
    lift_status = {
        r.expression: r for r in load_library("ashare", root=str(lift_root))
    }["dup_b"].status

    assert single_status == "correlated"
    assert lift_status == "correlated", (
        f"双路径漂移：单因子轨判 {single_status}、lift 轨判 {lift_status}"
    )


# ── 接线锚：能力做完 ≠ 调用方传了（从最外层出发，不用 inspect.signature）──────

def test_materialize_wiring_suite(tmp_path):
    """`fz factor-library lift-test --apply` 必须把物化器传进去。；team session 末 lift 钩子同样必须传物化器（第二个调用方，不能只修一侧）。"""
    # -- 原 test_cli_lift_apply_wires_materializer --
    def _section_0_test_cli_lift_apply_wires_materializer(tmp_path, mp):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser
        from tests.lift.test_cli_lift_apply import _patch_lift_deps, _write_gray_session

        run_dir = _write_gray_session(tmp_path)
        upsert_calls: list = []
        _patch_lift_deps(mp, upsert_calls=upsert_calls)

        args = build_parser().parse_args([
            "factor-library", "lift-test", "--session", str(run_dir),
            "--market", "ashare", "--start", "20200101", "--end", "20201231",
            "--set", "library_root=" + str(tmp_path / "lib"), "--apply",
        ])
        assert cli_main._cmd_factor_library_lift_test(args) == 0
        assert len(upsert_calls) == 1
        got = upsert_calls[0]
        mat = got.get("materialize") or got.get("compact_materialize")
        assert mat is not None, "CLI 未把物化器传给 upsert_lift_admissions，相关性门静默失效"
        assert callable(mat)

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cli_lift_apply_wires_materializer(_tp0, mp)

    # -- 原 test_team_lift_hook_wires_materializer --
    def _section_1_test_team_lift_hook_wires_materializer(mp):
        import factorzen.discovery.factor_library as fl

        calls: list = []

        def fake_upsert(rows, **kw):
            calls.append(kw)
            return {"added_active": 0, "added_probation": 0, "rejected": 0}

        mp.setattr(fl, "upsert_lift_admissions", fake_upsert)

        import inspect

        import factorzen.agents.team_orchestrator as to

        src = inspect.getsource(to)
        idx = src.find("adm = upsert_lift_admissions(")
        assert idx > 0, "调用点不存在了——本测试需跟着重写"
        call_src = src[idx:idx + src[idx:].find("\n        )")]
        assert "materialize=" in call_src, (
            "team lift 钩子未传物化器，相关性门在 session 自动路径上静默失效"
        )

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_team_lift_hook_wires_materializer(mp)


# ── 零回归锚：单因子轨默认行为逐位不变 ───────────────────────────────────────

def test_decorrelate_unit_defaults_suite():
    """`_decorrelate` 新增参数必须带默认值且默认行为不变（A股单因子轨零回归底线）。；默认 `preserve_status=False` 时 `compact_of=None` 仍全置 active（现语义不变）。；`preserve_status=True` 只允许下调到 correlated，绝不上调。"""
    # -- 原 test_decorrelate_default_behavior_unchanged --
    def _section_0_test_decorrelate_default_behavior_unchanged():
        from factorzen.discovery.factor_library import FactorRecord, _decorrelate

        panels = _make_panels()

        def compact_of(expr):
            p = panels.get(expr)
            if p is None:
                return None
            return p["factor_value"].to_numpy().reshape(len(_DATES), len(_CODES))

        affected = [
            FactorRecord(expression="dup_a", market="ashare", ir_train=0.2),
            FactorRecord(expression="dup_b", market="ashare", ir_train=0.9),
            FactorRecord(expression="indep", market="ashare", ir_train=0.5),
        ]
        n = _decorrelate(affected, [], compact_of, 0.7)
        by = {r.expression: r for r in affected}
        assert n == 1
        # 默认排序键仍是 |ir_train| 降序 → dup_b(0.9) 先占位，dup_a(0.2) 被标 correlated
        assert by["dup_b"].status == "active"
        assert by["dup_a"].status == "correlated"
        assert by["indep"].status == "active"

    _section_0_test_decorrelate_default_behavior_unchanged()

    # -- 原 test_decorrelate_none_compact_still_forces_active_by_default --
    def _section_1_test_decorrelate_none_compact_still_forces_active_by_default():
        from factorzen.discovery.factor_library import FactorRecord, _decorrelate

        affected = [FactorRecord(expression="e1", market="ashare", status="probation")]
        assert _decorrelate(affected, [], None, 0.7) == 0
        assert affected[0].status == "active"

    _section_1_test_decorrelate_none_compact_still_forces_active_by_default()

    # -- 原 test_decorrelate_preserve_status_blocks_upgrade --
    def _section_2_test_decorrelate_preserve_status_blocks_upgrade():
        from factorzen.discovery.factor_library import FactorRecord, _decorrelate

        affected = [FactorRecord(expression="e1", market="ashare", status="probation")]
        assert _decorrelate(affected, [], None, 0.7, preserve_status=True) == 0
        assert affected[0].status == "probation"

    _section_2_test_decorrelate_preserve_status_blocks_upgrade()


