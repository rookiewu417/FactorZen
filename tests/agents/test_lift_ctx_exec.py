"""
test_exec_convention.py：可实现成交口径：`compute_fwd_returns` 的 `exec_lag` / `exec_price_col`。
test_lift_ctx_wiring.py：F3：LiftEvalContext 三消费方接通（team hook / CLI lift-test / rebuild 复审）。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns
from tests._cli_lift_mocks import patch_cli_lift_pre_gates


# ==== 来自 test_exec_convention.py ====
def _px(closes: list[float], opens: list[float] | None = None) -> pl.DataFrame:
    """单只票的价格序列，日期递增。"""
    n = len(closes)
    d = {
        "ts_code": ["X"] * n,
        "trade_date": [f"2024-01-{i + 1:02d}" for i in range(n)],
        "close": closes,
    }
    if opens is not None:
        d["open"] = opens
    return pl.DataFrame(d)


def test_exec_convention_compute_suite():
    """默认（exec_lag=0）必须逐位等于 close[t+h]/close[t] − 1。；exec_lag=1 ⇒ 用 price[t+2]/price[t+1] − 1。；exec_price_col='open' ⇒ 完全走 open 列，close 不参与。；h=2 且 exec_lag=1 ⇒ price[t+3]/price[t+1] − 1。；shift 必须按 ts_code 分组——跨股票串价会污染边界日。"""
    # -- 原 test_default_unchanged_close_to_close --
    def _section_0_test_default_unchanged_close_to_close():
        df = compute_fwd_returns(_px([100.0, 110.0, 121.0]), horizons=[1])
        got = df["fwd_ret_1d"].to_list()
        assert got[0] == pytest.approx(0.10, abs=1e-12)
        assert got[1] == pytest.approx(0.10, abs=1e-12)
        assert got[2] is None  # 末日无前向价

    _section_0_test_default_unchanged_close_to_close()

    # -- 原 test_exec_lag_shifts_entry_and_exit --
    def _section_1_test_exec_lag_shifts_entry_and_exit():
        px = _px([100.0, 200.0, 210.0, 420.0])
        base = compute_fwd_returns(px, horizons=[1])["fwd_ret_1d"].to_list()
        lag1 = compute_fwd_returns(px, horizons=[1], exec_lag=1)["fwd_ret_1d"].to_list()

        assert base[0] == pytest.approx(1.00, abs=1e-12)   # 200/100 − 1
        assert lag1[0] == pytest.approx(0.05, abs=1e-12)   # 210/200 − 1
        assert lag1[1] == pytest.approx(1.00, abs=1e-12)   # 420/210 − 1
        assert lag1[2] is None and lag1[3] is None         # 尾部越界

    _section_1_test_exec_lag_shifts_entry_and_exit()

    # -- 原 test_exec_price_col_uses_open --
    def _section_2_test_exec_price_col_uses_open():
        px = _px([999.0, 999.0, 999.0, 999.0], opens=[10.0, 20.0, 25.0, 50.0])
        got = compute_fwd_returns(
            px, horizons=[1], exec_lag=1, exec_price_col="open")["fwd_ret_1d"].to_list()
        assert got[0] == pytest.approx(0.25, abs=1e-12)   # open[2]/open[1] = 25/20
        assert got[1] == pytest.approx(1.00, abs=1e-12)   # open[3]/open[2] = 50/25

    _section_2_test_exec_price_col_uses_open()

    # -- 原 test_horizon_and_lag_compose --
    def _section_3_test_horizon_and_lag_compose():
        px = _px([1.0, 2.0, 4.0, 6.0, 12.0])
        got = compute_fwd_returns(px, horizons=[2], exec_lag=1)["fwd_ret_2d"].to_list()
        assert got[0] == pytest.approx(2.00, abs=1e-12)   # 6/2 − 1
        assert got[1] == pytest.approx(2.00, abs=1e-12)   # 12/4 − 1

    _section_3_test_horizon_and_lag_compose()

    # -- 原 test_per_code_isolation --
    def _section_4_test_per_code_isolation():
        df = pl.DataFrame({
            "ts_code": ["A", "A", "B", "B"],
            "trade_date": ["2024-01-01", "2024-01-02"] * 2,
            "close": [10.0, 20.0, 100.0, 300.0],
        })
        got = compute_fwd_returns(df, horizons=[1]).sort(["ts_code", "trade_date"])
        v = got["fwd_ret_1d"].to_list()
        assert v[0] == pytest.approx(1.0)   # A: 20/10
        assert v[1] is None                 # A 末日，**不得**借用 B 的价格
        assert v[2] == pytest.approx(2.0)   # B: 300/100
        assert v[3] is None

    _section_4_test_per_code_isolation()


def test_ret_col_fallback_respects_lag():
    """无价格列时走单日收益复利，exec_lag 须跳过前 lag 步。

    ret = [0.5, 0.1, 0.2, ...]：h=1 且 lag=1 ⇒ 取 ret[t+2] 而非 ret[t+1]。
    """
    df = pl.DataFrame({
        "ts_code": ["X"] * 4,
        "trade_date": [f"2024-01-{i + 1:02d}" for i in range(4)],
        "ret_1d": [0.5, 0.1, 0.2, 0.3],
    })
    got = compute_fwd_returns(df, horizons=[1], exec_lag=1)["fwd_ret_1d"].to_list()
    assert got[0] == pytest.approx(0.2, abs=1e-12)   # ret[2]
    assert got[1] == pytest.approx(0.3, abs=1e-12)   # ret[3]


def test_invalid_args_raise():
    """负 exec_lag、不存在的 exec_price_col 必须显式报错，不得静默。"""
    px = _px([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="exec_lag"):
        compute_fwd_returns(px, horizons=[1], exec_lag=-1)
    with pytest.raises(ValueError, match="exec_price_col"):
        compute_fwd_returns(px, horizons=[1], exec_price_col="nope")


def test_exec_wiring_signature_suite():
    """`_build_ret_panel` 必须把两个参数透传下去，而不是吞掉。；`DataBundle.build` 必须透传，否则护栏仍在评不可实现的收益。；`make_lift_context` 必须把口径写进 ctx，供 lift 裁决读取。；`_session_end_auto_lift` 必须接住口径——它是 lift 裁决的入口。；`fz mine team` 必须暴露两个旗标，且默认值 = 历史行为。"""
    # -- 原 test_lift_ret_panel_threads_exec_args --
    def _section_0_test_lift_ret_panel_threads_exec_args():
        from factorzen.discovery.lift_test import _build_ret_panel

        df = pl.DataFrame({
            "ts_code": ["X"] * 4,
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "close": [100.0, 100.0, 100.0, 100.0],   # close 恒定 ⇒ close 口径必为 0
            "open_adj": [10.0, 20.0, 25.0, 50.0],
        })
        base = _build_ret_panel(df, horizon=1)
        assert all(v == pytest.approx(0.0) for v in base["ret"].to_list())

        got = _build_ret_panel(df, horizon=1, exec_lag=1, exec_price_col="open_adj")
        vals = got["ret"].to_list()
        assert vals[0] == pytest.approx(0.25, abs=1e-12)   # 25/20
        assert vals[1] == pytest.approx(1.00, abs=1e-12)   # 50/25

    _section_0_test_lift_ret_panel_threads_exec_args()

    # -- 原 test_databundle_threads_exec_args --
    def _section_1_test_databundle_threads_exec_args():
        from factorzen.discovery.scoring import DataBundle

        df = pl.DataFrame({
            "ts_code": ["X"] * 6,
            "trade_date": [f"2024-01-{i + 1:02d}" for i in range(6)],
            "close": [100.0] * 6,
            "close_adj": [100.0] * 6,
            "open_adj": [10.0, 20.0, 25.0, 50.0, 60.0, 70.0],
        })
        base = DataBundle.build(df)
        assert all(v == pytest.approx(0.0)
                   for v in base.fwd_returns["fwd_ret_1d"].drop_nulls().to_list())

        got = DataBundle.build(df, exec_lag=1, exec_price_col="open_adj")
        v = got.fwd_returns["fwd_ret_1d"].to_list()
        assert v[0] == pytest.approx(0.25, abs=1e-12)   # open[2]/open[1] = 25/20
        assert v[1] == pytest.approx(1.00, abs=1e-12)   # open[3]/open[2] = 50/25

    _section_1_test_databundle_threads_exec_args()

    # -- 原 test_lift_context_carries_exec_args --
    def _section_2_test_lift_context_carries_exec_args():
        from factorzen.discovery.lift_test import make_lift_context

        df = pl.DataFrame({
            "ts_code": ["X", "X"],
            "trade_date": ["2024-01-01", "2024-01-02"],
            "close": [1.0, 2.0],
        })
        d = make_lift_context("ashare", df, prepped=df)
        assert d.exec_lag == 0 and d.exec_price_col is None      # 默认不变

        c = make_lift_context("ashare", df, prepped=df,
                              exec_lag=1, exec_price_col="open_adj")
        assert c.exec_lag == 1 and c.exec_price_col == "open_adj"

    _section_2_test_lift_context_carries_exec_args()

    # -- 原 test_session_end_auto_lift_accepts_exec_args --
    def _section_3_test_session_end_auto_lift_accepts_exec_args():
        import inspect

        from factorzen.agents.team_orchestrator import _session_end_auto_lift

        sig = inspect.signature(_session_end_auto_lift)
        assert "exec_lag" in sig.parameters
        assert "exec_price_col" in sig.parameters
        assert sig.parameters["exec_lag"].default == 0
        assert sig.parameters["exec_price_col"].default is None

    _section_3_test_session_end_auto_lift_accepts_exec_args()

    # -- 原 test_cli_parser_exposes_exec_flags --
    def _section_4_test_cli_parser_exposes_exec_flags():
        from factorzen.cli.parser import build_parser

        class _Stub:
            def __getattr__(self, _n):
                return lambda *a, **k: 0

        p = build_parser(_Stub())
        for cmd in ("team", "agent", "search"):
            base = ["mine", cmd, "--start", "20200101", "--end", "20201231"]
            ns = p.parse_args(base)
            # CLI 默认可实现口径
            assert ns.exec_lag == 1, cmd
            assert ns.exec_price_col == "open_adj", cmd
            ns2 = p.parse_args([
                *base, "--exec-lag", "0", "--exec-price-col", "close",
            ])
            # 显式旧口径逃生口
            assert ns2.exec_lag == 0, cmd
            assert ns2.exec_price_col == "close", cmd
            ns3 = p.parse_args([
                *base, "--exec-lag", "1", "--exec-price-col", "open_adj",
            ])
            assert ns3.exec_lag == 1, cmd
            assert ns3.exec_price_col == "open_adj", cmd

    _section_4_test_cli_parser_exposes_exec_flags()


# ── 接线层：参数必须真的一路传到底 ────────────────────────────────
# 记忆库有一条老坑「能力层↔接线层漂移」：功能实现完 + 单测绿，但 pipeline
# 没透传，用户用不了。且 `inspect.signature` 断言零判别力——必须从外层出发
# 用**可观测的数值差异**验证。


# ==== 来自 test_lift_ctx_wiring.py ====
# ── helpers ──────────────────────────────────────────────────────────────────


def _as_ymd(v) -> str | None:
    """统一成 YYYYMMDD 字符串，便于与 holdout 边界比较。"""
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y%m%d")
    s = str(v).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:4] + s[5:7] + s[8:10]
    return s.replace("-", "")[:8]


def _mock_daily(n_days: int = 120, n_stocks: int = 8, seed: int = 1) -> pl.DataFrame:
    import datetime as dt

    import numpy as np

    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _panel(n_dates: int, n_stocks: int = 10, start=date(2022, 1, 3)):
    days, d = [], start
    while len(days) < n_dates:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        for dd in days:
            rows.append({
                "trade_date": dd, "ts_code": c,
                "factor_value": float(hash((c, dd)) % 100) / 100.0,
            })
    return pl.DataFrame(rows)


def _holdout_and_mat(n_days: int = 120):
    daily = _mock_daily(n_days=n_days)
    dates = daily["trade_date"].unique().sort()
    cut = dates[int(len(dates) * 0.8)]
    holdout = daily.filter(pl.col("trade_date") >= cut)

    def mat(expr):
        return _panel(80, start=cut)

    return daily, holdout, mat


def _state_with_lift_queue(exprs: list[str]):
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_QUEUE

    st = AgentState(seed=1)
    for e in exprs:
        st.attempts.append(AttemptRecord(
            iteration=0, hypothesis="h", expression=e,
            compile_ok=True, ic_train=0.02, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=1.0, n_train=100,
            residual_ic_train=0.01,
            reject_category=REJECT_CATEGORY_LIFT_QUEUE,
            reject_reason="x(lift队列)",
        ))
    st.n_gray_zone = len(exprs)
    return st


class _FakeCtx:
    leaf_map = None


def _write_gray_session(tmp_path: Path, *, holdout_start: str | None = None) -> Path:
    run_dir = tmp_path / "session1"
    run_dir.mkdir(parents=True, exist_ok=True)
    man: dict = {
        "attempts": [
            {
                "expression": "rank(close)",
                "reject_category": "gray_zone",
                "residual_ic_train": 0.02,  # ≥ DEFAULT_GRAY_IC_FLOOR（避开 sub-floor 防呆）
                "n_residual_holdout_days": 100,
            },
        ],
        "candidates": [],
        "params": {"start": "20200101", "end": "20201231", "holdout_ratio": 0.2},
    }
    if holdout_start is not None:
        man["holdout_start"] = holdout_start
    (run_dir / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False), encoding="utf-8",
    )
    return run_dir


def _fake_daily_full() -> pl.DataFrame:
    """可过 _preprocess_daily 的最小帧（含 vol）。"""
    return pl.DataFrame({
        "trade_date": [date(2020, 1, 2), date(2020, 1, 3)],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "close": [10.0, 10.5],
        "open": [9.9, 10.4],
        "high": [10.2, 10.6],
        "low": [9.8, 10.3],
        "vol": [1e6, 1.1e6],
        "amount": [1e7, 1.1e7],
        "close_adj": [10.0, 10.5],
    })


# ── 1. team hook 窗口落盘 ────────────────────────────────────────────────────


def test_lift_ctx_wiring_suite(monkeypatch, tmp_path, capsys):
    """组测收到的 ctx.admission_start == holdout 首日；meta 含 admission_start/end/horizon。；显式 materialize_candidate 注入时不构造默认 / prepped materializer。；库内 lift 记录 horizon=10 → 默认 runner 调 run_lift_tests 时 ctx.horizon==10。；装配返回 None → 现有空帧报错路径不变（exit 1）。；manifest 含 holdout_start → run_lift_tests 收到推导出的 admission_start。"""
    # -- 原 test_team_hook_admission_window_in_ctx_and_meta --
    def _section_0_test_team_hook_admission_window_in_ctx_and_meta(mp):
        from factorzen.agents.team_orchestrator import _session_end_auto_lift
        from factorzen.discovery.lift_test import DEFAULT_HORIZON

        state = _state_with_lift_queue(["ts_mean(close, 5)"])
        daily, holdout, mat = _holdout_and_mat()
        captured: dict = {}

        def fake_group(*a, **k):
            captured["group_kw"] = k
            return {
                "lift": 0.01, "lift_se": 0.001, "error": None,
                "n_candidates": 1, "expressions": ["ts_mean(close, 5)"],
            }

        def fake_per(*a, **k):
            captured["per_kw"] = k
            return [{
                "expression": "ts_mean(close, 5)",
                "lift": 0.008, "lift_se": 0.001,
                "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
            }]

        def fake_upsert(rows, *, market, **kw):
            captured["upsert_meta"] = kw.get("meta") or {}
            return {"added_active": 1, "added_probation": 0, "rejected": 0, "errors": []}

        mp.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)
        mp.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_per)
        mp.setattr(
            "factorzen.discovery.factor_library.upsert_lift_admissions",
            fake_upsert, raising=False,
        )

        meta = _session_end_auto_lift(
            state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
            market="ashare", library_root="/tmp/lib", seed=1,
            materialize_candidate=mat,
            active_factor_dfs={"base": _panel(100)},
            ret_df=_panel(100).rename({"factor_value": "ret"}),
            horizon=DEFAULT_HORIZON,
        )

        hs = holdout["trade_date"].min()
        he = holdout["trade_date"].max()
        gctx = captured["group_kw"]["ctx"]
        assert gctx is not None
        assert _as_ymd(gctx.admission_start) == _as_ymd(hs)
        assert _as_ymd(gctx.admission_end) == _as_ymd(he)
        assert gctx.horizon == DEFAULT_HORIZON

        pctx = captured["per_kw"]["ctx"]
        assert _as_ymd(pctx.admission_start) == _as_ymd(hs)
        assert pctx.horizon == DEFAULT_HORIZON

        assert _as_ymd(meta.get("admission_start")) == _as_ymd(hs)
        assert _as_ymd(meta.get("admission_end")) == _as_ymd(he)
        assert meta.get("horizon") == DEFAULT_HORIZON

        um = captured["upsert_meta"]
        assert um.get("horizon") == DEFAULT_HORIZON

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_team_hook_admission_window_in_ctx_and_meta(mp)

    # -- 原 test_team_hook_injected_materializer_skips_default --
    def _section_1_test_team_hook_injected_materializer_skips_default(mp):
        from factorzen.agents.team_orchestrator import _session_end_auto_lift

        state = _state_with_lift_queue(["ts_mean(close, 5)"])
        daily, holdout, mat = _holdout_and_mat()
        calls: list[str] = []

        def boom_default(*a, **k):
            calls.append("default")
            raise AssertionError("不应构造 _default_materializer")

        def boom_prepped(*a, **k):
            calls.append("prepped")
            raise AssertionError("注入路径不应构造 _materializer_from_prepped")

        mp.setattr(
            "factorzen.discovery.lift_test._default_materializer", boom_default,
        )
        mp.setattr(
            "factorzen.discovery.lift_test._materializer_from_prepped", boom_prepped,
        )

        def fake_group(*a, **k):
            return {
                "lift": 0.0, "lift_se": 0.1, "error": None,
                "expressions": ["ts_mean(close, 5)"],
            }

        mp.setattr("factorzen.discovery.lift_test.run_group_lift", fake_group)

        meta = _session_end_auto_lift(
            state, daily=daily, holdout_df=holdout, profile=None, ctx=_FakeCtx(),
            market="ashare", library_root="/tmp/lib", seed=1,
            materialize_candidate=mat,
            active_factor_dfs={"base": _panel(100)},
            horizon=1,
        )
        assert calls == [], f"注入 materialize 时不应构造默认 mat，got {calls}"
        assert meta.get("lift_error") is None

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_team_hook_injected_materializer_skips_default(mp)

    # -- 原 test_cli_rebuild_wires_daily_and_record_horizon --
    def _section_2_test_cli_rebuild_wires_daily_and_record_horizon(tmp_path, mp):
        import factorzen.cli.main as cli_main
        import factorzen.discovery.factor_library as fl
        import factorzen.discovery.lift_test as lt_mod
        from factorzen.cli.main import build_parser
        from factorzen.discovery.factor_library import FactorRecord, _save_library

        lib_root = tmp_path / "lib"
        lib_root.mkdir()
        _save_library(
            "ashare",
            [
                FactorRecord(
                    expression="rank(vol)", market="ashare", status="active",
                    admission_track="lift", ic_train=0.01, holdout_ic=0.0,
                    lift=0.01, lift_se=0.001, lift_second_half=0.005,
                    horizon=10,  # 与全局默认 5 不同
                    added_at="2026-07-02", updated_at="2026-07-02",
                ),
            ],
            root=str(lib_root),
        )

        captured: list[dict] = []

        def fake_lift(cands, **kw):
            captured.append(kw)
            expr = cands[0]["expression"] if cands else "x"
            return [{
                "expression": expr,
                "lift": 0.006, "lift_se": 0.001,
                "lift_second_half": 0.003, "baseline": 0.05, "passed": True,
            }]

        patch_cli_lift_pre_gates(mp)
        mp.setattr(lt_mod, "run_lift_tests", fake_lift)
        mp.setattr(
            cli_main, "_prepare_agent_mining_data",
            lambda args: (_fake_daily_full(), None, {}),
        )
        mp.setattr(fl, "collect_source_expressions", lambda market: [])
        mp.setattr(
            fl, "build_library_evaluator",
            lambda *a, **k: (lambda exprs: [], None),
        )
        # rebuild 落库根指向 tmp
        orig_rebuild = fl.rebuild

        def rebuild_to_tmp(*a, **kw):
            kw.setdefault("root", str(lib_root))
            return orig_rebuild(*a, **kw)

        mp.setattr(fl, "rebuild", rebuild_to_tmp)

        args = build_parser().parse_args([
            "factor-library", "rebuild",
            "--market", "ashare",
            "--start", "20200101",
            "--end", "20201231",
        ])
        # 强制 root：rebuild 默认 DEFAULT_ROOT；上面 mp 已 setdefault root
        # 但 CLI 未传 root——依赖 mp 包装
        rc = cli_main._cmd_factor_library_rebuild(args)
        assert rc == 0, "lift_review_error 应为 None 且 exit 0"
        assert captured, "默认 runner 应调用 run_lift_tests"
        ctx = captured[0].get("ctx")
        assert ctx is not None, "应传 ctx"
        assert ctx.horizon == 10, f"复审 horizon 应取 rec.horizon=10，got {ctx.horizon}"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cli_rebuild_wires_daily_and_record_horizon(_tp2, mp)

    # -- 原 test_cli_rebuild_missing_daily_still_errors --
    def _section_3_test_cli_rebuild_missing_daily_still_errors(mp, capsys):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser

        mp.setattr(
            cli_main, "_prepare_agent_mining_data",
            lambda args: (None, None, {}),
        )
        args = build_parser().parse_args([
            "factor-library", "rebuild",
            "--market", "ashare",
            "--start", "20200101",
            "--end", "20201231",
        ])
        rc = cli_main._cmd_factor_library_rebuild(args)
        assert rc == 1
        err = capsys.readouterr().err
        assert "挖掘帧为空" in err

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_cli_rebuild_missing_daily_still_errors(mp, capsys)

    # -- 原 test_lift_test_admission_from_manifest --
    def _section_4_test_lift_test_admission_from_manifest(tmp_path, mp):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser

        run_dir = _write_gray_session(tmp_path, holdout_start="20200901")
        captured: list = []
        _patch_lift_test_capture(mp, captured)

        args = build_parser().parse_args([
            "factor-library", "lift-test",
            "--session", str(run_dir),
            "--market", "ashare",
            "--start", "20200101",
            "--end", "20201231",
            "--library-root", str(tmp_path / "lib"),
        ])
        rc = cli_main._cmd_factor_library_lift_test(args)
        assert rc == 0
        assert captured
        ctx = captured[0].get("ctx")
        assert ctx is not None
        assert _as_ymd(ctx.admission_start) == "20200901"

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_lift_test_admission_from_manifest(_tp4, mp)


# ── 2. hook 注入优先 ─────────────────────────────────────────────────────────


# ── 3. CLI rebuild 接通（rec.horizon 进 ctx）──────────────────────────────────


# ── 4. CLI rebuild 缺 daily 回归 ─────────────────────────────────────────────


# ── 5. lift-test 窗口推导 ────────────────────────────────────────────────────


def _patch_lift_test_capture(monkeypatch, captured: list):
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod

    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_fake_daily_full(), None, {}),
    )

    def fake_lift(gray, **kw):
        captured.append(kw)
        return [{
            "expression": "rank(close)",
            "lift": 0.005, "lift_se": 0.001,
            "lift_second_half": 0.004, "baseline": 0.02, "passed": True,
            "candidate_rank_ic": 0.025,
        }]

    patch_cli_lift_pre_gates(monkeypatch)
    monkeypatch.setattr(lt_mod, "run_lift_tests", fake_lift)


def test_lift_test_admission_flag_overrides_manifest(tmp_path, monkeypatch):
    """--admission-start 旗标覆盖 manifest 推导。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path, holdout_start="20200901")
    captured: list = []
    _patch_lift_test_capture(monkeypatch, captured)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
        "--admission-start", "20201015",
        "--admission-end", "20201130",
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    ctx = captured[0]["ctx"]
    assert _as_ymd(ctx.admission_start) == "20201015"
    assert _as_ymd(ctx.admission_end) == "20201130"


def test_lift_test_no_window_warns(tmp_path, monkeypatch, capsys):
    """旗标与 manifest 皆无 holdout → admission_start=None + stderr 警告。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    run_dir = _write_gray_session(tmp_path, holdout_start=None)
    captured: list = []
    _patch_lift_test_capture(monkeypatch, captured)

    args = build_parser().parse_args([
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101",
        "--end", "20201231",
        "--library-root", str(tmp_path / "lib"),
    ])
    rc = cli_main._cmd_factor_library_lift_test(args)
    assert rc == 0
    ctx = captured[0]["ctx"]
    assert ctx.admission_start is None
    err = capsys.readouterr().err
    assert "未裁剪到 holdout" in err or "无独立性保证" in err
