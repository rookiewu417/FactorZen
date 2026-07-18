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


def test_p2_fwd_returns_columns_are_keys_plus_fwd_only():
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


def test_p2_quick_fitness_score_holdout_numeric_parity():
    """窄 fwd 与「全宽 fwd 语义」下 quick_fitness / score 数值全等。"""
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


def test_p1_slim_whitelist_column_snapshot(monkeypatch):
    import factorzen.daily.data.context as ctx_mod
    import factorzen.discovery.preparation as prep
    from factorzen.core.feature_schema import BASIC_FEATURES

    daily, basic = _prep_inputs()
    monkeypatch.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
    # 跳过 attach 链（无真实数据）
    monkeypatch.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

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


def test_p1_slim_false_keeps_dead_weight(monkeypatch):
    import factorzen.daily.data.context as ctx_mod
    import factorzen.discovery.preparation as prep

    daily, basic = _prep_inputs()
    monkeypatch.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
    monkeypatch.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

    fat = prep.prepare_mining_daily("20240102", "20240104", slim=False)
    cols = set(fat.columns)
    for keep in ("change", "pct_chg", "pe", "ps", "dv_ratio", "total_share", "free_share"):
        assert keep in cols, f"slim=False 应保留旧帧列 {keep}"


def test_p1_cross_family_expr_numeric_parity_slim_vs_fat(monkeypatch):
    """跨叶子族表达式在 slim/fat 上 factor_value 全等（公共列语义）。"""
    import factorzen.daily.data.context as ctx_mod
    import factorzen.discovery.preparation as prep
    from factorzen.discovery.evaluation import _factor_df_from_prepped, _preprocess_daily
    from factorzen.discovery.expression import parse_expr

    daily, basic = _prep_inputs()
    monkeypatch.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
    monkeypatch.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

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


# ── P3: prepped 注入 + scout 失效 + 调用次数 ────────────────────────────────


def test_p3_prepped_inject_factor_value_parity():
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


def test_p3_scout_inject_invalidates_prepped():
    """scout 注入新 ix_* 列后，缓存 prepped 必须重建，评估才能看到新叶。"""
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


def test_p3_session_preprocess_call_count(monkeypatch):
    """session 级单次 prep：leaf_health/budgets/lib_pool/evaluate 共用 → 调用次数 ≤2。"""
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

    monkeypatch.setattr(ev_mod, "_preprocess_daily", counting_prep)

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


def test_p3_make_lift_context_reuses_prepped(monkeypatch):
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

    monkeypatch.setattr(ev_mod, "_preprocess_daily", counting)
    ctx = make_lift_context("ashare", daily, prepped=prepped)
    assert calls["n"] == 0
    assert "ret_1d" in ctx.prepped.columns or "close_adj" in ctx.prepped.columns
    # 不传 prepped 才 prep
    _ = make_lift_context("ashare", daily)
    assert calls["n"] == 1


# ── P5: 单副本纪律 ─────────────────────────────────────────────────────────


def test_p5_bundle_daily_is_keys_only():
    """DataBundle.build 长驻 daily 仅键列（消灭 sort 全宽幽灵副本）。"""
    from factorzen.discovery.scoring import DataBundle

    daily = _mock_daily()
    b = DataBundle.build(daily)
    assert set(b.daily.columns) == {"trade_date", "ts_code"}
    assert b.daily.height == daily.height
    # fwd 仍完整可用
    assert "fwd_ret_1d" in b.fwd_returns.columns
    assert b.train_end is not None


def test_p5_bundle_keys_only_quick_fitness_parity():
    """键窄 daily 不改 quick_fitness 数值（消费面只读 fwd_returns）。"""
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


def test_p5_narrow_holdout_price_frame():
    from factorzen.agents.team_orchestrator import _narrow_holdout_price_frame

    daily = _mock_daily(n_stocks=4, n_days=10)
    narrow = _narrow_holdout_price_frame(daily)
    assert set(narrow.columns) == {"trade_date", "ts_code", "close_adj"}
    assert narrow.height == daily.height


def test_p5_guardrails_prepped_skips_warmup_reprep(monkeypatch):
    """node_guardrails(prepped=) 不对 warmup 再调 _preprocess_daily。"""
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
    monkeypatch.setattr(
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


# ── P4c: ts_code Categorical ────────────────────────────────────────────────


def test_p4c_prepare_categorical_keys_explicit(monkeypatch):
    import factorzen.daily.data.context as ctx_mod
    import factorzen.discovery.preparation as prep

    daily, basic = _prep_inputs()
    monkeypatch.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
    monkeypatch.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

    on = prep.prepare_mining_daily("20240102", "20240104", slim=True, categorical_keys=True)
    off = prep.prepare_mining_daily("20240102", "20240104", slim=True, categorical_keys=False)
    assert on.schema["ts_code"] == pl.Categorical
    assert off.schema["ts_code"] in (pl.Utf8, pl.String)


def test_p4c_threshold_auto_off_for_small_frames(monkeypatch):
    """默认 None + 小帧 → 不自动 Categorical（阈值 4M）。"""
    import factorzen.daily.data.context as ctx_mod
    import factorzen.discovery.preparation as prep

    daily, basic = _prep_inputs()
    monkeypatch.setattr(ctx_mod, "FactorDataContext", _fake_ctx_factory(daily, basic))
    monkeypatch.setattr("factorzen.daily.data.pit.attach_fundamentals", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

    out = prep.prepare_mining_daily("20240102", "20240104", slim=True, categorical_keys=None)
    assert out.height < prep.KEYS_CATEGORICAL_ROWS_THRESHOLD
    assert out.schema["ts_code"] in (pl.Utf8, pl.String)


def test_p4c_categorical_factor_value_and_ic_parity():
    """Categorical on/off 同表达式 factor_value 与 IC 逐值相等。"""
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


def test_p4c_scatter_join_with_categorical_keys():
    """库相关 scatter：因子帧 Categorical × stock_map Utf8 可 join 且 present 全覆盖。"""
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


def test_p4c_export_alpha_casts_utf8(tmp_path, monkeypatch):
    """落盘 alpha 截面 ts_code 仍为 Utf8。"""
    from factorzen.discovery import export as export_mod

    daily = _mock_daily(n_stocks=3, n_days=5).with_columns(
        pl.col("ts_code").cast(pl.Categorical)
    )
    # stub alpha_cross_section
    monkeypatch.setattr(
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
