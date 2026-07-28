"""帧瘦身 P1/P2/P3：列白名单、fwd_returns 收窄、session prepped 复用（全 mock）。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

# ── fixtures ────────────────────────────────────────────────────────────────


def _mock_daily(n_stocks: int = 20, n_days: int = 60, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days: list[date] = []
    d = date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        code = f"{i:06d}.SZ"
        px = 10.0
        for dd in days:
            px = float(max(px * (1 + rng.standard_normal() * 0.02), 0.1))
            vol = float(abs(rng.standard_normal()) * 1e5 + 1e4)
            rows.append({
                "trade_date": dd,
                "ts_code": code,
                "open": px * 0.99,
                "high": px * 1.01,
                "low": px * 0.98,
                "close": px,
                "pre_close": px * 0.995,
                "change": px * 0.005,
                "pct_chg": 0.5,
                "vol": vol,
                "amount": vol * px,
                "close_adj": px,
                "open_adj": px * 0.99,
                "high_adj": px * 1.01,
                "low_adj": px * 0.98,
                "pe": 12.0,
                "pe_ttm": 11.0,
                "pb": 1.5,
                "ps": 2.0,
                "ps_ttm": 1.8,
                "dv_ratio": 0.01,
                "dv_ttm": 0.012,
                "total_mv": 1e6,
                "circ_mv": 8e5,
                "turnover_rate": 1.0,
                "turnover_rate_f": 1.2,
                "volume_ratio": 1.0,
                "total_share": 1e5,
                "float_share": 8e4,
                "free_share": 7e4,
            })
    return pl.DataFrame(rows)


def _signal_factor_df(daily: pl.DataFrame) -> pl.DataFrame:
    df = daily.sort(["ts_code", "trade_date"]).with_columns(
        (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("fwd")
    )
    return df.select(["trade_date", "ts_code", pl.col("fwd").alias("factor_value")]).drop_nulls()


# ── P2: DataBundle.fwd_returns 收窄 ─────────────────────────────────────────


def test_p2_fwd_returns_slim_suite():
    """test_p2_fwd_returns_columns_are_keys_plus_fwd_only；窄 fwd 与「全宽 fwd 语义」下 quick_fitness / score 数值全等。"""
    # -- 原 test_p2_fwd_returns_columns_are_keys_plus_fwd_only --
    def _section_0_test_p2_fwd_returns_columns_are_keys_plus_fwd_only():
        from factorzen.discovery.scoring import DataBundle

        daily = _mock_daily()
        b = DataBundle.build(daily)
        cols = set(b.fwd_returns.columns)
        assert "trade_date" in cols and "ts_code" in cols
        assert "fwd_ret_1d" in cols
        # 默认 4 horizon 列仍保留（ic_overfit / multi-horizon 语义）
        for h in (1, 5, 10, 20):
            assert f"fwd_ret_{h}d" in cols
        # 不得残留全宽 mining 列
        for dead in ("close", "close_adj", "vol", "pe", "pe_ttm", "open", "amount", "change"):
            assert dead not in cols, f"fwd_returns 不应含 {dead}"
        assert len(b.fwd_returns.columns) == 2 + 4  # keys + 4 horizons

    _section_0_test_p2_fwd_returns_columns_are_keys_plus_fwd_only()

    # -- 原 test_p2_quick_fitness_score_holdout_numeric_parity --
    def _section_1_test_p2_quick_fitness_score_holdout_numeric_parity():
        from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.scoring import DataBundle, quick_fitness, score_candidate

        daily = _mock_daily()
        fac = _signal_factor_df(daily)

        # 窄路径（改后 DataBundle.build）
        narrow = DataBundle.build(daily)
        # 手工全宽对照（模拟改前：compute_fwd_returns 原样返回）
        wide_fwd = compute_fwd_returns(
            daily.sort(["ts_code", "trade_date"]),
            price_col="close_adj",
        )
        dates = sorted(daily["trade_date"].unique().to_list())
        cut = dates[min(int(len(dates) * 0.7), len(dates) - 1)]
        train_end = cut.strftime("%Y%m%d") if hasattr(cut, "strftime") else str(cut)
        wide = DataBundle(daily=daily.sort(["ts_code", "trade_date"]),
                          fwd_returns=wide_fwd, train_end=train_end)

        for seg in ("train", "valid"):
            a = quick_fitness(fac, narrow, segment=seg)  # type: ignore[arg-type]
            b = quick_fitness(fac, wide, segment=seg)  # type: ignore[arg-type]
            assert a["ic_mean"] == pytest.approx(b["ic_mean"], abs=0.0, rel=0.0)
            assert a["ir"] == pytest.approx(b["ir"], abs=0.0, rel=0.0)
            assert a["tstat"] == pytest.approx(b["tstat"], abs=0.0, rel=0.0)
            assert a["n"] == b["n"]

        node = parse_expr("close")
        sc_n = score_candidate(fac, node, narrow, pool={})
        sc_w = score_candidate(fac, node, wide, pool={})
        assert sc_n["fitness"] == pytest.approx(sc_w["fitness"], abs=0.0, rel=0.0)
        assert sc_n["ic_train"] == pytest.approx(sc_w["ic_train"], abs=0.0, rel=0.0)
        assert sc_n["tstat_train"] == pytest.approx(sc_w["tstat_train"], abs=0.0, rel=0.0)

    _section_1_test_p2_quick_fitness_score_holdout_numeric_parity()


# ── P1: prepare 白名单 ──────────────────────────────────────────────────────


def _fake_ctx_factory(daily: pl.DataFrame, basic: pl.DataFrame):
    class _FakeCtx:
        expanded_start = "20190101"

        def __init__(self, **kw):
            pass

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return basic.lazy()

    return _FakeCtx


def _prep_inputs():
    d = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    codes = ["000001.SZ", "000002.SZ"]
    rows_d, rows_b = [], []
    for c in codes:
        for dd in d:
            rows_d.append({
                "trade_date": dd, "ts_code": c,
                "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
                "pre_close": 10.0, "change": 0.5, "pct_chg": 5.0,
                "vol": 1e5, "amount": 1e6,
                "close_adj": 10.5, "open_adj": 10.0, "high_adj": 11.0, "low_adj": 9.0,
            })
            rows_b.append({
                "trade_date": dd, "ts_code": c,
                "pe": 12.0, "pe_ttm": 11.0, "pb": 1.5, "ps": 2.0, "ps_ttm": 1.8,
                "dv_ratio": 0.01, "dv_ttm": 0.012,
                "total_mv": 1e6, "circ_mv": 8e5,
                "turnover_rate": 1.0, "turnover_rate_f": 1.2, "volume_ratio": 1.0,
                "total_share": 1e5, "float_share": 8e4, "free_share": 7e4,
            })
    return pl.DataFrame(rows_d), pl.DataFrame(rows_b)


def test_p1_slim_whitelist_suite(monkeypatch):
    """test_p1_slim_whitelist_column_snapshot；test_p1_slim_false_keeps_dead_weight；跨叶子族表达式在 slim/fat 上 factor_value 全等（公共列语义）。"""
    # -- 原 test_p1_slim_whitelist_column_snapshot --
    def _section_0_test_p1_slim_whitelist_column_snapshot(mp):
        import factorzen.daily.data.context as ctx_mod
        import factorzen.discovery.preparation as prep
        from factorzen.core.feature_schema import BASIC_FEATURES

        daily, basic = _prep_inputs()
        mp.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
        # 跳过 attach 链（无真实数据）
        mp.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
        mp.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
        mp.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

        out = prep.prepare_mining_daily("20240102", "20240104", slim=True)
        cols = set(out.columns)

        # 死重不得进帧
        for dead in ("change", "pct_chg", "pe", "ps", "dv_ratio", "total_share", "free_share"):
            assert dead not in cols, f"slim 帧不应含死重列 {dead}"

        # BASIC 叶子应在
        for leaf in BASIC_FEATURES:
            assert leaf in cols, f"BASIC 叶 {leaf} 应保留"

        # raw OHLC + 派生输入
        for keep in ("open", "high", "low", "close", "pre_close", "vol", "amount",
                     "close_adj", "open_adj", "high_adj", "low_adj"):
            assert keep in cols

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_p1_slim_whitelist_column_snapshot(mp)

    # -- 原 test_p1_slim_false_keeps_dead_weight --
    def _section_1_test_p1_slim_false_keeps_dead_weight(mp):
        import factorzen.daily.data.context as ctx_mod
        import factorzen.discovery.preparation as prep

        daily, basic = _prep_inputs()
        mp.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
        mp.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
        mp.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
        mp.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

        fat = prep.prepare_mining_daily("20240102", "20240104", slim=False)
        cols = set(fat.columns)
        for keep in ("change", "pct_chg", "pe", "ps", "dv_ratio", "total_share", "free_share"):
            assert keep in cols, f"slim=False 应保留旧帧列 {keep}"

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_p1_slim_false_keeps_dead_weight(mp)

    # -- 原 test_p1_cross_family_expr_numeric_parity_slim_vs_fat --
    def _section_2_test_p1_cross_family_expr_numeric_parity_slim_vs_fat(mp):
        import factorzen.daily.data.context as ctx_mod
        import factorzen.discovery.preparation as prep
        from factorzen.discovery.evaluation import _factor_df_from_prepped, _preprocess_daily
        from factorzen.discovery.expression import parse_expr

        daily, basic = _prep_inputs()
        mp.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
        mp.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
        mp.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
        mp.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

        slim_df = prep.prepare_mining_daily("20240102", "20240104", slim=True)
        fat_df = prep.prepare_mining_daily("20240102", "20240104", slim=False)

        # 跨族：价量 leaf + BASIC leaf
        expr = "mul(rank(close), rank(pe_ttm))"
        node = parse_expr(expr)
        a = _factor_df_from_prepped(node, _preprocess_daily(slim_df))
        b = _factor_df_from_prepped(node, _preprocess_daily(fat_df))
        joined = a.join(b, on=["trade_date", "ts_code"], how="inner", suffix="_fat")
        assert joined.height > 0
        diff = (joined["factor_value"] - joined["factor_value_fat"]).abs().max()
        assert diff is None or float(diff) == 0.0 or diff == 0.0

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_p1_cross_family_expr_numeric_parity_slim_vs_fat(mp)


# ── P3: prepped 注入 + scout 失效 + 调用次数 ────────────────────────────────


def test_p3_prepped_reuse_suite(monkeypatch):
    """test_p3_prepped_inject_factor_value_parity；scout 注入新 ix_* 列后，缓存 prepped 必须重建，评估才能看到新叶。；session 级单次 prep：leaf_health/budgets/lib_pool/evaluate 共用 → 调用次数 ≤2。；test_p3_make_lift_context_reuses_prepped"""
    # -- 原 test_p3_prepped_inject_factor_value_parity --
    def _section_0_test_p3_prepped_inject_factor_value_parity():
        from factorzen.discovery.evaluation import (
            _factor_df_from_prepped,
            _preprocess_daily,
            evaluate_expressions,
        )
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.scoring import DataBundle

        daily = _mock_daily()
        bundle = DataBundle.build(daily)
        prepped = _preprocess_daily(daily)

        expr = "ts_mean(close, 5)"
        # 内建 prep
        out_a = evaluate_expressions([expr], daily, bundle)
        # 注入 prepped
        out_b = evaluate_expressions([expr], daily, bundle, prepped=prepped)

        assert out_a[0]["ic_train"] == pytest.approx(out_b[0]["ic_train"], abs=0.0, rel=0.0)
        assert out_a[0]["ir_train"] == pytest.approx(out_b[0]["ir_train"], abs=0.0, rel=0.0)
        assert out_a[0]["n_train"] == out_b[0]["n_train"]

        # factor_value 逐行全等
        node = parse_expr(expr)
        fa = _factor_df_from_prepped(node, _preprocess_daily(daily))
        fb = _factor_df_from_prepped(node, prepped)
        j = fa.join(fb, on=["trade_date", "ts_code"], suffix="_b")
        assert float((j["factor_value"] - j["factor_value_b"]).abs().max()) == 0.0

    _section_0_test_p3_prepped_inject_factor_value_parity()

    # -- 原 test_p3_scout_inject_invalidates_prepped --
    def _section_1_test_p3_scout_inject_invalidates_prepped():
        from factorzen.discovery.evaluation import _preprocess_daily, evaluate_expressions
        from factorzen.discovery.scoring import DataBundle

        daily = _mock_daily()
        bundle = DataBundle.build(daily)
        prepped_old = _preprocess_daily(daily)

        # 模拟 scout 注入假 ix 列
        daily_new = daily.with_columns(pl.lit(1.23).alias("ix_fake_scout"))
        # 旧 prepped 缺列 → 求值失败或全 null
        out_stale = evaluate_expressions(
            ["rank(ix_fake_scout)"], daily_new, bundle, prepped=prepped_old,
        )
        # leaf 不在默认 LEAF_FEATURES → parse 可能失败；扩展 leaf_map 测列缺失
        from factorzen.discovery.expression import evaluate_materialized, parse_expr
        from factorzen.discovery.operators import LEAF_FEATURES

        lm = {**LEAF_FEATURES, "ix_fake_scout": "ix_fake_scout"}
        node = parse_expr("rank(ix_fake_scout)", lm)
        with pytest.raises((pl.exceptions.ColumnNotFoundError, KeyError, ValueError)):
            # 旧 prepped 无 ix_fake_scout 列 → evaluate_materialized 应失败
            evaluate_materialized(node, prepped_old, lm)

        prepped_new = _preprocess_daily(daily_new)
        series = evaluate_materialized(node, prepped_new, lm)
        assert series.drop_nulls().len() > 0  # 新 prepped 可见假叶并成功求值

        # 本测重点：prepped 重建契约（旧缓存缺列、新缓存含列）
        assert "ix_fake_scout" in prepped_new.columns
        assert "ix_fake_scout" not in prepped_old.columns
        assert out_stale is not None  # 引用防 lint

    _section_1_test_p3_scout_inject_invalidates_prepped()

    # -- 原 test_p3_session_preprocess_call_count --
    def _section_2_test_p3_session_preprocess_call_count(mp):
        import factorzen.discovery.evaluation as ev_mod
        from factorzen.discovery.evaluation import (
            _preprocess_daily as real_prep,
        )
        from factorzen.discovery.evaluation import (
            evaluate_expressions,
            make_health_check,
        )
        from factorzen.discovery.scoring import DataBundle

        daily = _mock_daily(n_stocks=8, n_days=40)
        bundle = DataBundle.build(daily)

        calls: list[int] = []
        orig = real_prep

        def counting_prep(df, profile=None):
            calls.append(1)
            return orig(df, profile)

        mp.setattr(ev_mod, "_preprocess_daily", counting_prep)

        # 模拟 session：一次 prep + 复用
        session_prepped = counting_prep(daily, None)
        # leaf health / budgets 用同一帧（不调 prep）
        _ = session_prepped.columns
        # health 注入 prepped
        health = make_health_check(daily, prepped=session_prepped)
        assert health("ts_mean(close, 3)") is None or isinstance(health("ts_mean(close, 3)"), (str, type(None)))
        # evaluate 注入 prepped
        out = evaluate_expressions(
            ["ts_mean(close, 5)", "rank(vol)"], daily, bundle, prepped=session_prepped,
        )
        assert all(r["compile_ok"] for r in out)
        # 不应因 evaluate/health 再增 prep（仅 session 那一次）
        assert sum(calls) <= 2, f"_preprocess_daily 调用过多: {sum(calls)}"

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_p3_session_preprocess_call_count(mp)

    # -- 原 test_p3_make_lift_context_reuses_prepped --
    def _section_3_test_p3_make_lift_context_reuses_prepped(mp):
        import factorzen.discovery.evaluation as ev_mod
        from factorzen.discovery.evaluation import _preprocess_daily
        from factorzen.discovery.lift_test import make_lift_context

        daily = _mock_daily(n_stocks=5, n_days=20)
        prepped = _preprocess_daily(daily)

        calls = {"n": 0}
        orig = ev_mod._preprocess_daily

        def counting(df, profile=None):
            calls["n"] += 1
            return orig(df, profile)

        mp.setattr(ev_mod, "_preprocess_daily", counting)
        ctx = make_lift_context("ashare", daily, prepped=prepped)
        assert calls["n"] == 0
        assert "ret_1d" in ctx.prepped.columns or "close_adj" in ctx.prepped.columns
        # 不传 prepped 才 prep
        _ = make_lift_context("ashare", daily)
        assert calls["n"] == 1

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_p3_make_lift_context_reuses_prepped(mp)


# ── P5: 单副本纪律 ─────────────────────────────────────────────────────────


def test_p5_bundle_keys_suite(monkeypatch):
    """DataBundle.build 长驻 daily 仅键列（消灭 sort 全宽幽灵副本）。；键窄 daily 不改 quick_fitness 数值（消费面只读 fwd_returns）。；test_p5_narrow_holdout_price_frame；node_guardrails(prepped=) 不对 warmup 再调 _preprocess_daily。"""
    # -- 原 test_p5_bundle_daily_is_keys_only --
    def _section_0_test_p5_bundle_daily_is_keys_only():
        from factorzen.discovery.scoring import DataBundle

        daily = _mock_daily()
        b = DataBundle.build(daily)
        assert set(b.daily.columns) == {"trade_date", "ts_code"}
        assert b.daily.height == daily.height
        # fwd 仍完整可用
        assert "fwd_ret_1d" in b.fwd_returns.columns
        assert b.train_end is not None

    _section_0_test_p5_bundle_daily_is_keys_only()

    # -- 原 test_p5_bundle_keys_only_quick_fitness_parity --
    def _section_1_test_p5_bundle_keys_only_quick_fitness_parity():
        from factorzen.discovery.scoring import DataBundle, quick_fitness

        daily = _mock_daily()
        fac = _signal_factor_df(daily)
        b = DataBundle.build(daily)
        # 手工构造「旧」全宽 daily 字段的 bundle（仅测字段无关）
        wide_daily = daily.sort(["ts_code", "trade_date"])
        b_wide = DataBundle(
            daily=wide_daily, fwd_returns=b.fwd_returns, train_end=b.train_end,
        )
        for seg in ("train", "valid"):
            a = quick_fitness(fac, b, segment=seg)  # type: ignore[arg-type]
            c = quick_fitness(fac, b_wide, segment=seg)  # type: ignore[arg-type]
            assert a["ic_mean"] == pytest.approx(c["ic_mean"], abs=0.0, rel=0.0)
            assert a["ir"] == pytest.approx(c["ir"], abs=0.0, rel=0.0)
            assert a["n"] == c["n"]

    _section_1_test_p5_bundle_keys_only_quick_fitness_parity()

    # -- 原 test_p5_narrow_holdout_price_frame --
    def _section_2_test_p5_narrow_holdout_price_frame():
        from factorzen.agents.team_orchestrator import _narrow_holdout_price_frame

        daily = _mock_daily(n_stocks=4, n_days=10)
        narrow = _narrow_holdout_price_frame(daily)
        # 保留 close/open 价列供 holdout IC + exec 口径（open_adj）
        assert set(narrow.columns) == {
            "trade_date", "ts_code", "close_adj", "close", "open_adj", "open",
        }
        assert narrow.height == daily.height
        # 不得带全宽叶子/成交额等（P5 峰值）
        assert "vol" not in narrow.columns and "amount" not in narrow.columns

    _section_2_test_p5_narrow_holdout_price_frame()

    # -- 原 test_p5_guardrails_prepped_skips_warmup_reprep --
    def _section_3_test_p5_guardrails_prepped_skips_warmup_reprep(mp):
        from factorzen.agents.nodes import node_guardrails
        from factorzen.agents.state import AgentState, AttemptRecord
        from factorzen.agents.team_orchestrator import _narrow_holdout_price_frame
        from factorzen.discovery.evaluation import _preprocess_daily
        from factorzen.discovery.scoring import DataBundle
        from factorzen.validation.holdout import split_holdout
        from factorzen.validation.multiple_testing import TrialLedger

        daily = _mock_daily(n_stocks=10, n_days=40)
        mining_df, holdout_df, _ = split_holdout(daily, holdout_ratio=0.2)
        holdout_n = _narrow_holdout_price_frame(holdout_df)
        bundle = DataBundle.build(mining_df)
        prepped = _preprocess_daily(daily)

        state = AgentState(seed=1)
        state.iteration = 0
        state.attempts = [
            AttemptRecord(
                iteration=0, hypothesis="h", expression="rank(close)",
                compile_ok=True, ic_train=0.05, passed_guardrails=False,
                critic_verdict=None, error=None, ir_train=0.8, turnover=0.1, n_train=20,
            )
        ]
        ledger = TrialLedger()

        calls: list[int] = []
        orig = _preprocess_daily

        def counting(df, profile=None):
            calls.append(df.height)
            return orig(df, profile)

        # node_guardrails 函数内 from-import：patch evaluation 模块即可
        mp.setattr(
            "factorzen.discovery.evaluation._preprocess_daily", counting,
        )

        node_guardrails(
            state, daily=mining_df, holdout_df=holdout_n, bundle=bundle,
            ledger=ledger, top_k=5, warmup_daily=prepped, prepped=prepped,
            objective="raw", lib_pool={},
        )
        # hold/warmup 全帧高度不应出现在 prep 调用里（prepped 注入）
        assert daily.height not in calls, f"warmup 被重复 prep: {calls}"
        # 允许 mining 段 train residual 的 prep（height == mining）
        assert all(h == mining_df.height for h in calls), calls

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_p5_guardrails_prepped_skips_warmup_reprep(mp)


# ── P4c: ts_code Categorical ────────────────────────────────────────────────


def test_p4c_categorical_keys_suite(monkeypatch, tmp_path):
    """test_p4c_prepare_categorical_keys_explicit；默认 None + 小帧 → 不自动 Categorical（阈值 4M）。；Categorical on/off 同表达式 factor_value 与 IC 逐值相等。；库相关 scatter：因子帧 Categorical × stock_map Utf8 可 join 且 present 全覆盖。；落盘 alpha 截面 ts_code 仍为 Utf8。"""
    # -- 原 test_p4c_prepare_categorical_keys_explicit --
    def _section_0_test_p4c_prepare_categorical_keys_explicit(mp):
        import factorzen.daily.data.context as ctx_mod
        import factorzen.discovery.preparation as prep

        daily, basic = _prep_inputs()
        mp.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
        mp.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
        mp.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
        mp.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

        on = prep.prepare_mining_daily("20240102", "20240104", slim=True, categorical_keys=True)
        off = prep.prepare_mining_daily("20240102", "20240104", slim=True, categorical_keys=False)
        assert on.schema["ts_code"] == pl.Categorical
        assert off.schema["ts_code"] in (pl.Utf8, pl.String)

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_p4c_prepare_categorical_keys_explicit(mp)

    # -- 原 test_p4c_threshold_auto_off_for_small_frames --
    def _section_1_test_p4c_threshold_auto_off_for_small_frames(mp):
        import factorzen.daily.data.context as ctx_mod
        import factorzen.discovery.preparation as prep

        daily, basic = _prep_inputs()
        mp.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
        mp.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
        mp.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
        mp.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

        out = prep.prepare_mining_daily("20240102", "20240104", slim=True, categorical_keys=None)
        assert out.height < prep.KEYS_CATEGORICAL_ROWS_THRESHOLD
        assert out.schema["ts_code"] in (pl.Utf8, pl.String)

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_p4c_threshold_auto_off_for_small_frames(mp)

    # -- 原 test_p4c_categorical_factor_value_and_ic_parity --
    def _section_2_test_p4c_categorical_factor_value_and_ic_parity():
        from factorzen.discovery.evaluation import (
            _factor_df_from_prepped,
            _preprocess_daily,
            evaluate_expressions,
        )
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.scoring import DataBundle

        daily_u = _mock_daily(n_stocks=12, n_days=50)
        daily_c = daily_u.with_columns(pl.col("ts_code").cast(pl.Categorical))
        bundle_u = DataBundle.build(daily_u)
        bundle_c = DataBundle.build(daily_c)
        expr = "ts_mean(close, 5)"

        out_u = evaluate_expressions([expr], daily_u, bundle_u)
        out_c = evaluate_expressions([expr], daily_c, bundle_c)
        assert out_u[0]["compile_ok"] and out_c[0]["compile_ok"]
        assert out_u[0]["ic_train"] == pytest.approx(out_c[0]["ic_train"], abs=0.0, rel=0.0)
        assert out_u[0]["ir_train"] == pytest.approx(out_c[0]["ir_train"], abs=0.0, rel=0.0)
        assert out_u[0]["n_train"] == out_c[0]["n_train"]

        node = parse_expr(expr)
        fu = _factor_df_from_prepped(node, _preprocess_daily(daily_u))
        fc = _factor_df_from_prepped(node, _preprocess_daily(daily_c))
        # join 前把 cat 侧对齐
        fc_j = fc.with_columns(pl.col("ts_code").cast(pl.Utf8))
        j = fu.join(fc_j, on=["trade_date", "ts_code"], suffix="_c")
        assert j.height == fu.height
        assert float((j["factor_value"] - j["factor_value_c"]).abs().max()) == 0.0

    _section_2_test_p4c_categorical_factor_value_and_ic_parity()

    # -- 原 test_p4c_scatter_join_with_categorical_keys --
    def _section_3_test_p4c_scatter_join_with_categorical_keys():
        from factorzen.discovery.scoring import (
            _align_join_key,
            _scatter_candidate_to_panel,
            build_library_corr_panel,
        )

        daily = _mock_daily(n_stocks=8, n_days=30)
        daily_c = daily.with_columns(pl.col("ts_code").cast(pl.Categorical))
        fac = (
            daily_c.sort(["ts_code", "trade_date"])
            .select([
                "trade_date", "ts_code",
                pl.col("close_adj").alias("factor_value"),
            ])
        )
        pool = {"lib_a": fac}
        panel = build_library_corr_panel(pool)
        assert panel is not None
        # present=None 新契约（corr panel 瘦身）：掩码经 present_block 从 ~isnan 推导
        assert panel.present_block(0, len(panel.dates)).sum() == fac.height  # 键全覆盖，无 join 丢失
        _cand_v, cand_p = _scatter_candidate_to_panel(fac, panel)
        assert int(cand_p.sum()) == fac.height

        # align helper：小 Utf8 帧 cast 到 Categorical
        small = pl.DataFrame({"ts_code": ["000001.SZ"], "x": [1]})
        aligned = _align_join_key(small, "ts_code", fac)
        assert aligned.schema["ts_code"] == pl.Categorical
        j = fac.select(["ts_code"]).unique().join(aligned, on="ts_code", how="inner")
        assert j.height >= 1

    _section_3_test_p4c_scatter_join_with_categorical_keys()

    # -- 原 test_p4c_export_alpha_casts_utf8 --
    def _section_4_test_p4c_export_alpha_casts_utf8(tmp_path, mp):
        from factorzen.discovery import export as export_mod

        daily = _mock_daily(n_stocks=3, n_days=5).with_columns(
            pl.col("ts_code").cast(pl.Categorical)
        )
        # stub alpha_cross_section
        mp.setattr(
            export_mod,
            "alpha_cross_section",
            lambda *a, **k: daily.select([
                "ts_code", pl.lit(1.0).alias("alpha"),
            ]).unique(subset=["ts_code"]),
        )
        out = tmp_path / "alpha.parquet"
        export_mod.export_alpha_cross_section("rank(close)", object(), "20220103", str(out))
        loaded = pl.read_parquet(out)
        assert loaded.schema["ts_code"] in (pl.Utf8, pl.String)

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_p4c_export_alpha_casts_utf8(_tp4, mp)


def test_evaluate_materialized_prunes_unused_columns():
    """内存关键:求值入口裁到键+所引用叶列——with_columns 全列复制(polars 实测)
    在全宽帧×深嵌套下产生 10G+ 尖峰(全 A OOM 根因,memlog 实锤)。
    宽帧(多余列)与窄帧求值逐值相等,且多余列不进 work 帧。"""
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.discovery.expression import evaluate_materialized, parse_expr

    rng = np.random.default_rng(7)
    days = [dt.date(2021, 1, 4) + dt.timedelta(days=i) for i in range(30)]
    rows = []
    for c in ["000001.SZ", "000002.SZ", "000003.SZ"]:
        for d in days:
            rows.append({"trade_date": d, "ts_code": c,
                         "close_adj": float(rng.uniform(8, 15)),
                         "vol": float(rng.uniform(1e5, 1e6))})
    base = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
    # 加 30 个无关列模拟全宽帧
    wide = base.with_columns([pl.lit(float(i)).alias(f"junk_{i}") for i in range(30)])

    node = parse_expr("rank(ts_mean(close, 5))")
    s_narrow = evaluate_materialized(node, base)
    s_wide = evaluate_materialized(node, wide)
    assert s_wide.equals(s_narrow), "宽帧与窄帧求值必须逐值相等"


# ── P6: keep_leaves 列白名单裁剪（装配出口只留白名单叶列）────────────────────


def _p6_days(n: int = 12) -> list[date]:
    out: list[date] = []
    d = date(2024, 1, 2)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _p6_intraday_names() -> list[str]:
    from factorzen.core.feature_schema import INTRADAY_FEATURES

    return sorted(INTRADAY_FEATURES)


def _p6_sources(n_days: int = 12) -> dict[str, pl.DataFrame]:
    """全 attach 源的合成注入帧（daily / basic / fina / holder / flows / 事件 / 日内）。"""
    rng = np.random.default_rng(11)
    days = _p6_days(n_days)
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    ix_names = _p6_intraday_names()

    rows_d, rows_b, rows_mf, rows_hk, rows_mg, rows_tl, rows_ix = [], [], [], [], [], [], []
    for c in codes:
        px = 10.0
        for dd in days:
            px = float(max(px * (1 + rng.standard_normal() * 0.02), 0.5))
            vol = float(abs(rng.standard_normal()) * 1e5 + 1e4)
            rows_d.append({
                "trade_date": dd, "ts_code": c,
                "open": px * 0.99, "high": px * 1.02, "low": px * 0.97, "close": px,
                "pre_close": px * 0.995, "change": px * 0.005, "pct_chg": 0.5,
                "vol": vol, "amount": vol * px,
                "close_adj": px, "open_adj": px * 0.99,
                "high_adj": px * 1.02, "low_adj": px * 0.97,
            })
            rows_b.append({
                "trade_date": dd, "ts_code": c,
                "pe": 12.0, "pe_ttm": 11.0, "pb": 1.5, "ps": 2.0, "ps_ttm": 1.8,
                "dv_ratio": 0.01, "dv_ttm": 0.012,
                "total_mv": 1e6, "circ_mv": float(8e5 + rng.uniform(0, 1e5)),
                "turnover_rate": 1.0, "turnover_rate_f": 1.2, "volume_ratio": 1.0,
                "total_share": 1e5, "float_share": 8e4, "free_share": 7e4,
            })
            rows_mf.append({"ts_code": c, "trade_date": dd,
                            "net_mf_amount": float(rng.standard_normal() * 1e3)})
            rows_hk.append({"ts_code": c, "trade_date": dd,
                            "ratio": float(abs(rng.standard_normal()))})
            rows_mg.append({"ts_code": c, "trade_date": dd,
                            "rzye": float(abs(rng.standard_normal()) * 1e8 + 1e7),
                            "rzmre": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                            "rqyl": float(abs(rng.standard_normal()) * 1e4)})
            rows_tl.append({"ts_code": c, "trade_date": dd,
                            "net_amount": float(rng.standard_normal() * 1e3),
                            "amount": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
            ix = {"trade_date": dd, "ts_code": c}
            for j, name in enumerate(ix_names):
                ix[name] = float(rng.standard_normal() + j * 0.01)
            rows_ix.append(ix)

    fina = pl.DataFrame([
        {"ts_code": c, "end_date": date(2023, 9, 30), "ann_date": date(2023, 10, 20),
         "roe": 0.11 + i, "roa": 0.05 + i, "grossprofit_margin": 0.30 + i,
         "netprofit_margin": 0.12 + i, "debt_to_assets": 0.40 + i,
         "or_yoy": 0.08 + i, "netprofit_yoy": 0.09 + i, "assets_yoy": 0.07 + i}
        for i, c in enumerate(codes)
    ])
    holder = pl.DataFrame([
        {"ts_code": c, "end_date": ed, "ann_date": ad, "holder_num": float(hn)}
        for c in codes
        for ed, ad, hn in (
            (date(2023, 6, 30), date(2023, 7, 15), 10000),
            (date(2023, 9, 30), date(2023, 10, 20), 12000),
        )
    ])
    forecast = pl.DataFrame([
        {"ts_code": c, "ann_date": date(2024, 1, 3), "end_date": date(2023, 12, 31),
         "type": "预增", "p_change_min": 30.0, "p_change_max": 50.0}
        for c in codes
    ])
    express = pl.DataFrame([
        {"ts_code": c, "ann_date": date(2024, 1, 4), "end_date": date(2023, 12, 31),
         "yoy_net_profit": 1.0e8, "n_income": 1.3e8}
        for c in codes
    ])
    return {
        "daily": pl.DataFrame(rows_d),
        "basic": pl.DataFrame(rows_b),
        "fina": fina,
        "holder": holder,
        "moneyflow": pl.DataFrame(rows_mf),
        "hk_hold": pl.DataFrame(rows_hk),
        "margin_detail": pl.DataFrame(rows_mg),
        "top_list": pl.DataFrame(rows_tl),
        "forecast": forecast,
        "express": express,
        "intraday": pl.DataFrame(rows_ix),
    }


def _p6_patch_all(mp, src, *, spy: dict | None = None) -> None:
    """把 prepare_mining_daily 的全部数据源换成合成注入帧（``keep`` 原样透传）。"""
    import factorzen.daily.data.context as ctx_mod
    from factorzen.daily.data import events as ev_mod
    from factorzen.daily.data import flows as fl_mod
    from factorzen.daily.data import intraday as ix_mod
    from factorzen.daily.data import pit as pit_mod

    mp.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(src["daily"], src["basic"]))

    real_fund, real_hold = pit_mod.attach_fundamentals, pit_mod.attach_holders
    real_flows = fl_mod.attach_flows
    real_fc, real_ex = ev_mod.attach_forecast, ev_mod.attach_express
    real_ix = ix_mod.attach_intraday

    def _note(name, keep):
        if spy is not None:
            spy.setdefault(name, []).append(keep)

    def _fund(d, fina_df=None, **kw):
        _note("fundamentals", kw.get("keep"))
        return real_fund(d, src["fina"], **kw)

    def _hold(d, holder_df=None, **kw):
        _note("holders", kw.get("keep"))
        return real_hold(d, src["holder"], **kw)

    def _flows(d, **kw):
        _note("flows", kw.get("keep"))
        kw.pop("injected", None)
        return real_flows(d, injected={
            "moneyflow": src["moneyflow"], "hk_hold": src["hk_hold"],
            "margin_detail": src["margin_detail"], "top_list": src["top_list"],
        }, **kw)

    def _fc(d, forecast_df=None, **kw):
        _note("forecast", kw.get("keep"))
        return real_fc(d, src["forecast"], **kw)

    def _ex(d, express_df=None, **kw):
        _note("express", kw.get("keep"))
        return real_ex(d, src["express"], **kw)

    def _ix(d, **kw):
        _note("intraday", kw.get("keep"))
        kw.pop("injected", None)
        return real_ix(d, injected=src["intraday"], **kw)

    mp.setattr(pit_mod, "attach_fundamentals", _fund)
    mp.setattr(pit_mod, "attach_holders", _hold)
    mp.setattr(fl_mod, "attach_flows", _flows)
    mp.setattr(ev_mod, "attach_forecast", _fc)
    mp.setattr(ev_mod, "attach_express", _ex)
    mp.setattr(ix_mod, "attach_intraday", _ix)


def _p6_prepare(mp, src, *, keep_leaves=None, spy=None):
    import factorzen.discovery.preparation as prep

    _p6_patch_all(mp, src, spy=spy)
    return prep.prepare_mining_daily(
        "20240102", "20240119", keep_leaves=keep_leaves, intraday=True,
    )


def test_p6_derived_deps_table_is_exact():
    """``DERIVED_LEAF_RAW_DEPS`` 与 ``add_derived_columns`` 实际引用的 raw 列逐条相等。

    判别力（不是恒真）：
    - 充分性：只喂「keys ∪ 表内全部依赖」的最小帧，``add_derived_columns`` 必须跑通、
      六个派生叶都有值（漏列 → ColumnNotFoundError 或全 null）。
    - 必要性：逐个抠掉表里声明的某列必须炸（多写一列 → 不炸 → 测试红）。
    - 强制核心列集必须覆盖全部依赖，否则最激进的裁剪会让下游 prep 直接崩。
    """
    from factorzen.discovery.derived import add_derived_columns
    from factorzen.discovery.preparation import (
        DERIVED_LEAF_RAW_DEPS,
        MINING_MANDATORY_COLS,
    )

    assert set(DERIVED_LEAF_RAW_DEPS) == {
        "vwap", "log_vol", "ret_1d", "amplitude", "intraday_ret", "overnight_ret",
    }
    all_deps: set[str] = set()
    for v in DERIVED_LEAF_RAW_DEPS.values():
        all_deps |= set(v)
    assert all_deps <= set(MINING_MANDATORY_COLS), (
        f"派生依赖 {sorted(all_deps - set(MINING_MANDATORY_COLS))} 不在强制核心列集里"
    )

    days = _p6_days(6)
    rows = []
    for i, c in enumerate(["000001.SZ", "000002.SZ"]):
        for j, dd in enumerate(days):
            px = 10.0 + i + j * 0.3
            rows.append({
                "trade_date": dd, "ts_code": c,
                "open": px * 0.99, "high": px * 1.02, "low": px * 0.97,
                "close": px, "pre_close": px * 0.995,
                "vol": 1e5 + j, "amount": 1e6 + j, "close_adj": px,
            })
    minimal = pl.DataFrame(rows).select(["trade_date", "ts_code", *sorted(all_deps)])
    out = add_derived_columns(minimal.sort(["ts_code", "trade_date"]))
    for leaf in DERIVED_LEAF_RAW_DEPS:
        assert leaf in out.columns, f"最小依赖帧上派生叶 {leaf} 未生成"
        assert out[leaf].drop_nulls().len() > 0, f"派生叶 {leaf} 在最小依赖帧上全 null"

    for leaf, deps in DERIVED_LEAF_RAW_DEPS.items():
        for dep in deps:
            stripped = minimal.drop(dep).sort(["ts_code", "trade_date"])
            try:
                add_derived_columns(stripped)
            except pl.exceptions.ColumnNotFoundError:
                continue
            raise AssertionError(
                f"依赖表声称 {leaf} 需要 {dep}，但抠掉它 add_derived_columns 照样跑通"
            )


def test_p6_keep_leaves_none_is_zero_regression():
    """``keep_leaves=None``（默认）出口列集/行数/取值与「白名单覆盖全部叶」完全一致。

    列集用显式 golden 硬编码——拿被测值当期望的恒真断言在本仓翻过车。
    """
    from factorzen.core.feature_schema import LEAF_FEATURES

    src = _p6_sources()
    with pytest.MonkeyPatch.context() as mp:
        base = _p6_prepare(mp, src, keep_leaves=None)
    with pytest.MonkeyPatch.context() as mp:
        full = _p6_prepare(mp, src, keep_leaves=set(LEAF_FEATURES))

    golden = {
        "trade_date", "ts_code",
        "open", "high", "low", "close", "pre_close", "vol", "amount",
        "open_adj", "high_adj", "low_adj", "close_adj",
        # daily_basic 10 叶
        "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv",
        "turnover_rate", "turnover_rate_f", "volume_ratio", "float_share",
        # 基本面 8 + 股东 2
        "roe", "roa", "grossprofit_margin", "netprofit_margin", "debt_to_assets",
        "or_yoy", "netprofit_yoy", "assets_yoy", "holder_num", "holder_num_chg",
        # 资金流 2 + 两融 4 + 龙虎榜 2
        "net_mf_amount", "north_ratio",
        "margin_ratio", "margin_buy_ratio", "margin_balance", "short_balance",
        "top_list_net_buy", "top_list_flag",
        # 事件 3 + 1
        "fc_type_score", "fc_surprise", "fc_flag", "express_yoy",
    } | set(_p6_intraday_names())

    assert set(base.columns) == golden, (
        f"缺 {sorted(golden - set(base.columns))} / 多 {sorted(set(base.columns) - golden)}"
    )
    assert set(full.columns) == set(base.columns)
    assert full.height == base.height == src["daily"].height
    key = ["trade_date", "ts_code"]
    a, b = base.sort(key), full.sort(key)
    for col in sorted(golden - set(key)):
        assert a[col].equals(b[col]), f"列 {col} 在 None / 全叶白名单下不逐值相等"


def test_p6_trim_preserves_evaluate_materialized_values():
    """裁列后同一表达式的求值结果与全帧逐值一致（含真实 OOM 复现表达式）。"""
    from factorzen.discovery.evaluation import _factor_df_from_prepped, _preprocess_daily
    from factorzen.discovery.expression import feature_names, parse_expr

    src = _p6_sources(n_days=16)
    cases = [
        "sub(grossprofit_margin, delay(grossprofit_margin, 1))",   # 真实 OOM 复现目标
        "ts_mean(vwap, 3)",                                        # 派生叶 → raw 依赖
        "div(margin_ratio, add(amplitude, 1))",                    # 跨源 + 派生
        "rank(i_rv)",                                              # 日内叶
    ]
    keep: set[str] = set()
    for e in cases:
        keep |= feature_names(parse_expr(e))

    with pytest.MonkeyPatch.context() as mp:
        full = _p6_prepare(mp, src, keep_leaves=None)
    with pytest.MonkeyPatch.context() as mp:
        trim = _p6_prepare(mp, src, keep_leaves=keep)

    assert trim.height == full.height
    assert len(trim.columns) < len(full.columns), "白名单未裁掉任何列 → 测试无判别力"

    p_full = _preprocess_daily(full)
    p_trim = _preprocess_daily(trim)
    for e in cases:
        node = parse_expr(e)
        a = _factor_df_from_prepped(node, p_trim).sort(["trade_date", "ts_code"])
        b = _factor_df_from_prepped(node, p_full).sort(["trade_date", "ts_code"])
        assert a.height == b.height > 0, e
        assert a["factor_value"].equals(b["factor_value"]), f"{e} 裁列后数值漂移"


def test_p6_empty_intersection_skips_whole_source():
    """某源与白名单交集为空 → 整个 join 跳过（源不被调用），行数不变、列不出现。"""
    src = _p6_sources()
    spy: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        out = _p6_prepare(mp, src, keep_leaves={"close", "vol"}, spy=spy)
    with pytest.MonkeyPatch.context() as mp:
        full = _p6_prepare(mp, src, keep_leaves=None)

    assert out.height == full.height, "跳过 join 不得改变行数"
    for name in ("fundamentals", "holders", "flows", "forecast", "express", "intraday"):
        assert name not in spy, f"白名单不含 {name} 叶，该源仍被 attach"
    dead = {
        "roe", "holder_num", "net_mf_amount", "north_ratio", "margin_ratio",
        "top_list_flag", "fc_flag", "express_yoy", "i_rv",
        "pe_ttm", "total_mv", "circ_mv",   # daily_basic 交集为空 → 整源跳过
    }
    assert not (dead & set(out.columns)), sorted(dead & set(out.columns))
    for c in ("open", "high", "low", "close", "pre_close", "vol", "amount",
              "open_adj", "high_adj", "low_adj", "close_adj"):
        assert c in out.columns, f"强制核心列 {c} 被裁（下游 prep 会直接崩）"


def test_p6_cross_source_dep_margin_ratio_keeps_circ_mv():
    """``margin_ratio`` 的分母 ``circ_mv`` 是 basic 叶：只要 margin_ratio 也必须保留它。

    判别力：漏了 circ_mv 时 ``_attach_margin`` 会**静默**走 denom_cols 为空的分支、
    根本不产出 margin_ratio 列，不报错——断言必须落在值上。
    """
    src = _p6_sources()
    with pytest.MonkeyPatch.context() as mp:
        trim = _p6_prepare(mp, src, keep_leaves={"margin_ratio"})
    with pytest.MonkeyPatch.context() as mp:
        full = _p6_prepare(mp, src, keep_leaves=None)

    assert "circ_mv" in trim.columns, "margin_ratio 的跨源依赖 circ_mv 被裁掉了"
    assert "margin_ratio" in trim.columns
    key = ["trade_date", "ts_code"]
    a, b = trim.sort(key), full.sort(key)
    assert a["margin_ratio"].drop_nulls().len() > 0, "margin_ratio 全 null → 分母被裁"
    assert a["margin_ratio"].equals(b["margin_ratio"])
    assert "margin_buy_ratio" not in trim.columns
    assert "short_balance" not in trim.columns


def test_p6_attach_intraday_column_subset():
    """``attach_intraday(keep=...)`` 只 join 白名单内的 i_* 列，数值与全量 attach 一致。"""
    from factorzen.daily.data.intraday import attach_intraday

    src = _p6_sources()
    daily = src["daily"].select(["trade_date", "ts_code", "close"])
    subset = ["i_amihud", "i_rv"]

    full = attach_intraday(daily, injected=src["intraday"])
    part = attach_intraday(daily, injected=src["intraday"], keep=subset)

    assert set(part.columns) == {"trade_date", "ts_code", "close", *subset}
    assert set(full.columns) >= set(_p6_intraday_names())
    key = ["trade_date", "ts_code"]
    a, b = part.sort(key), full.sort(key)
    for c in subset:
        assert a[c].equals(b[c]), f"{c} 列子集 attach 后数值漂移"


def test_p6_cli_rebuild_passes_keep_leaves(tmp_path):
    """``factor-library rebuild`` 把本次要评估的表达式叶并集透传到装配层。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    _save_library(
        "ashare",
        [FactorRecord(expression="rank(pe_ttm)", market="ashare", status="active"),
         FactorRecord(expression="ts_mean(i_rv, 5)", market="ashare",
                      status="correlated")],
        root=str(tmp_path),
    )

    seen: dict = {}

    def _spy_prepare(args, *a, **kw):
        seen["keep_leaves"] = getattr(args, "keep_leaves", None)
        return pl.DataFrame({"trade_date": [date(2024, 1, 2)]}), None, {}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli_main, "_prepare_agent_mining_data", _spy_prepare)
        mp.setattr(fl, "collect_source_expressions", lambda market: [])
        mp.setattr(fl, "build_library_evaluator",
                   lambda *a, **k: ((lambda exprs: []), None))
        mp.setattr(cli_main, "_lift_admission_str", lambda v: None)
        mp.setattr(fl, "DEFAULT_ROOT", str(tmp_path), raising=False)

        orig = fl.rebuild

        def _rebuild(*a, **kw):
            kw.setdefault("root", str(tmp_path))
            return orig(*a, **kw)

        mp.setattr(fl, "rebuild", _rebuild)
        args = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi300", "--start", "20200101", "--end", "20201231",
            "--only", "rank(pe_ttm)",
        ])
        cli_main._cmd_factor_library_rebuild(args)

    keep = seen.get("keep_leaves")
    assert keep is not None, "rebuild 未计算白名单"
    # only 目标 ∪ 库内全部记录（含 correlated——去相关/复审都会碰到）的叶并集
    assert set(keep) == {"pe_ttm", "i_rv"}, sorted(keep)


def test_p6_prepare_agent_mining_data_defaults_to_no_trim():
    """未设 ``args.keep_leaves`` 的入口（mine / team / pool-prebuild）一律不裁列。"""
    import argparse

    import factorzen.cli.main as cli_main
    import factorzen.pipelines.factor_mine as fm

    seen: dict = {}

    def _fake_prepare(*a, **kw):
        seen["keep_leaves"] = kw.get("keep_leaves", "<absent>")
        return pl.DataFrame({"trade_date": [date(2024, 1, 2)]})

    ns = argparse.Namespace(market="ashare", start="20240102", end="20240104",
                            universe="csi300")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fm, "prepare_mining_daily", _fake_prepare)
        cli_main._prepare_agent_mining_data(ns)
    assert seen["keep_leaves"] in (None, "<absent>")

    ns2 = argparse.Namespace(market="ashare", start="20240102", end="20240104",
                             universe="csi300", keep_leaves=["close", "vol"])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fm, "prepare_mining_daily", _fake_prepare)
        cli_main._prepare_agent_mining_data(ns2)
    assert set(seen["keep_leaves"]) == {"close", "vol"}


# ── P7：rebuild 帧生命周期收敛（prepped 单例 + daily 及早释放）────────────────


def _p7_lib_record(expr: str, **kw):
    from factorzen.discovery.factor_library import FactorRecord

    base = dict(
        expression=expr, market="ashare", status="active", admission_track="lift",
        ic_train=0.05, holdout_ic=0.04, n_train=60, horizon=5,
    )
    base.update(kw)
    return FactorRecord(**base)  # type: ignore[arg-type]


def test_p7_evaluator_external_prepped_golden_parity():
    """``build_library_evaluator(prepped=...)`` 与内部自 prep 路径**逐值等价**。

    判别力：拿内部自 prep 路径当 golden，逐字段比 evaluate 输出的全部指标 +
    compact 矩阵逐元素；同时断言外部注入路径**一次都不调** ``_preprocess_daily``
    （否则「复用」是假的，内存不会降）。
    """
    import factorzen.discovery.evaluation as ev
    from factorzen.discovery.evaluation import _preprocess_daily
    from factorzen.discovery.factor_library import build_library_evaluator

    daily = _mock_daily(n_stocks=40, n_days=120, seed=5)
    dates = sorted(daily["trade_date"].unique().to_list())
    eval_start = dates[25].strftime("%Y%m%d")
    exprs = ["rank(close)", "ts_mean(close, 5)", "sub(vwap, delay(vwap, 1))"]

    ev_a, compact_a = build_library_evaluator(daily, eval_start=eval_start)
    rows_a = ev_a(exprs)

    prepped = _preprocess_daily(daily, None).sort(["ts_code", "trade_date"])
    calls: list[int] = []
    real = ev._preprocess_daily
    with pytest.MonkeyPatch.context() as mp:
        def _counting(*a, **k):
            calls.append(1)
            return real(*a, **k)

        mp.setattr(ev, "_preprocess_daily", _counting)
        ev_b, compact_b = build_library_evaluator(
            None, eval_start=eval_start, prepped=prepped,
        )
        rows_b = ev_b(exprs)

    assert calls == [], f"注入 prepped 后仍 prep 了 {len(calls)} 次"
    assert len(rows_a) == len(rows_b) > 0
    by_a = {r["expression"]: r for r in rows_a}
    by_b = {r["expression"]: r for r in rows_b}
    assert set(by_a) == set(by_b)
    # 恒真防护：golden 侧必须真的算出了非平凡指标
    assert any(r["ic_train"] not in (None, 0.0) for r in rows_a)
    for e, ra in by_a.items():
        rb = by_b[e]
        assert set(ra) == set(rb), e
        for k, va in ra.items():
            vb = rb[k]
            if isinstance(va, float) and va != va:
                assert isinstance(vb, float) and vb != vb, f"{e}.{k}: {va} vs {vb}"
            else:
                assert va == vb, f"{e}.{k}: {va} vs {vb}"

    for e in exprs:
        ma, mb = compact_a(e), compact_b(e)
        assert (ma is None) == (mb is None), e
        if ma is not None:
            assert ma.shape == mb.shape and ma.dtype == mb.dtype, e
            assert np.array_equal(ma, mb, equal_nan=True), f"compact 矩阵漂移: {e}"


def test_p7_cli_rebuild_preps_frame_once(tmp_path):
    """``factor-library rebuild`` 全链路只 ``_preprocess_daily`` 一次。

    三个消费方（evaluator / 复审物化器 / rebuild 内 lift 复审 ctx）共用同一份
    prepped。判别力：修复前这条链会 prep 3 次（外加 evaluate 每批一次），
    三份全宽帧叠加正是 OOM 成因。走 ``--only``（fresh=False）保证库内 active
    留存 → 复审物化器真的被调到，三个消费方全覆盖。
    """
    import factorzen.cli.main as cli_main
    import factorzen.discovery.evaluation as ev
    from factorzen.cli.main import build_parser
    from factorzen.discovery import factor_library as fl
    from factorzen.discovery.factor_library import _save_library

    daily = _mock_daily(n_stocks=40, n_days=120, seed=11)
    dates = sorted(daily["trade_date"].unique().to_list())
    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    _save_library(
        "ashare",
        [_p7_lib_record("rank(close)"), _p7_lib_record("rank(vol)")],
        root=str(tmp_path),
    )

    mat_calls: list[str] = []
    calls: list[str] = []
    real = ev._preprocess_daily
    with pytest.MonkeyPatch.context() as mp:
        def _counting(*a, **k):
            calls.append("prep")
            return real(*a, **k)

        mp.setattr(ev, "_preprocess_daily", _counting)
        mp.setattr(cli_main, "_prepare_agent_mining_data",
                   lambda args: (daily, None, {}))
        mp.setattr(fl, "collect_source_expressions", lambda market: ["rank(close)"])
        mp.setattr(fl, "DEFAULT_ROOT", str(tmp_path), raising=False)

        orig_rebuild = fl.rebuild

        def _rebuild(*a, **kw):
            kw.setdefault("root", str(tmp_path))
            mat = kw.get("materialize")
            if mat is not None:
                def _spy_mat(expr, _m=mat):
                    mat_calls.append(expr)
                    return _m(expr)

                kw["materialize"] = _spy_mat
            return orig_rebuild(*a, **kw)

        mp.setattr(fl, "rebuild", _rebuild)
        args = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi300", "--start", start, "--end", end,
            "--only", "rank(close)",
        ])
        args.library_root = str(tmp_path)
        cli_main._cmd_factor_library_rebuild(args)

    # 恒真防护：复审物化器必须真的被调到，否则「只 prep 一次」是因为没走到那条路
    assert mat_calls, "复审物化器未被调用，本用例覆盖不到第三个消费方"
    assert len(calls) == 1, f"rebuild 链路 prep 了 {len(calls)} 次（应恰好 1 次）"


def test_p7_rebuild_materializer_reuses_prepped_values_match():
    """复审物化器复用 CLI 单例 prepped 后，物化结果与「自己 prep」旧路径逐值一致。"""
    from factorzen.discovery.evaluation import _preprocess_daily
    from factorzen.discovery.lift_test import (
        _default_materializer,
        _materializer_from_prepped,
    )

    daily = _mock_daily(n_stocks=40, n_days=90, seed=13)
    prepped = _preprocess_daily(daily, None).sort(["ts_code", "trade_date"])
    reuse = _materializer_from_prepped(prepped, None)
    legacy = _default_materializer(daily, None)
    for e in ("rank(close)", "ts_mean(vol, 5)", "sub(vwap, delay(vwap, 1))"):
        a = reuse(e).sort(["ts_code", "trade_date"])
        b = legacy(e).sort(["ts_code", "trade_date"])
        assert a.height == b.height > 0, e
        assert a["factor_value"].to_list() == b["factor_value"].to_list(), e


def test_p7_rebuild_prepped_arg_matches_daily_path(tmp_path):
    """``fl.rebuild(prepped=...)`` 与 ``daily=...`` 旧路径产出的库记录逐值一致。"""
    import factorzen.discovery.evaluation as ev
    from factorzen.discovery.evaluation import _preprocess_daily
    from factorzen.discovery.factor_library import (
        _save_library,
        build_library_evaluator,
        load_library,
        rebuild,
    )

    daily = _mock_daily(n_stocks=40, n_days=120, seed=17)
    dates = sorted(daily["trade_date"].unique().to_list())
    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    prepped = _preprocess_daily(daily, None).sort(["ts_code", "trade_date"])

    def _run(root, **rebuild_kw):
        _save_library("ashare", [_p7_lib_record("rank(close)")], root=str(root))
        evaluate, compact = build_library_evaluator(
            daily, eval_start=start, prepped=rebuild_kw.get("prepped"),
        )
        rebuild("ashare", sources=["rank(close)"], eval_window=(start, end),
                universe="csi300", horizon=5, evaluate=evaluate,
                compact_materialize=compact, git_sha="x", now="2026-07-28",
                root=str(root), fresh=False, **rebuild_kw)
        return {r.expression: r for r in load_library("ashare", root=str(root))}

    a_root, b_root = tmp_path / "a", tmp_path / "b"
    a_root.mkdir()
    b_root.mkdir()
    got_a = _run(a_root, daily=daily)

    calls: list[str] = []
    real = ev._preprocess_daily
    with pytest.MonkeyPatch.context() as mp:
        def _counting(*a, **k):
            calls.append("prep")
            return real(*a, **k)

        mp.setattr(ev, "_preprocess_daily", _counting)
        got_b = _run(b_root, prepped=prepped)

    assert calls == [], f"注入 prepped 后 rebuild 仍 prep 了 {len(calls)} 次"
    assert set(got_a) == set(got_b) and got_a
    for e, ra in got_a.items():
        da, db = ra.__dict__.copy(), got_b[e].__dict__.copy()
        for k in ("added_at", "updated_at"):
            da.pop(k, None)
            db.pop(k, None)
        assert da == db, e
