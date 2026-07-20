"""Merged discovery tests: test_scoring_bundle.py

test_discovery_scoring.py：DataBundle 切分、quick_fitness、相关与 t-stat 门禁、复杂度惩罚
test_discovery_derived.py：add_derived_columns 派生列数值正确，pre_close=0 时安全出 null
test_run_mine_joins_daily_basic.py：run_mine 须 join daily_basic，否则 BASIC_FEATURES 叶子失效（F5）
"""

from __future__ import annotations

from datetime import (
    date,
    timedelta,
)

import numpy as np
import polars as pl
import pytest

# ==== 来自 test_discovery_scoring.py ====
# tests/test_discovery_scoring.py

def _daily(seed=1, n_stocks=40, n_days=120):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)

def _signal_factor_df(daily: pl.DataFrame) -> pl.DataFrame:
    """构造与次日收益正相关的因子（用于验证 IC 为正）。"""
    df = daily.sort(["ts_code", "trade_date"]).with_columns(
        (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("fwd"))
    return df.select(["trade_date", "ts_code", pl.col("fwd").alias("factor_value")]).drop_nulls()

def _noisy_signal_factor_df(daily: pl.DataFrame, noise: float = 0.6, seed: int = 3) -> pl.DataFrame:
    """与次日收益正相关但含噪的因子：日频 IC 为正但 <1、逐日波动（IR/t-stat 有限且不相等）。"""
    rng = np.random.default_rng(seed)
    df = (daily.sort(["ts_code", "trade_date"])
          .with_columns((pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("fwd"))
          .drop_nulls())
    vals = df["fwd"].to_numpy() + rng.standard_normal(df.height) * noise
    return df.select(["trade_date", "ts_code"]).with_columns(pl.Series("factor_value", vals))

def test_databundle_split():
    from factorzen.discovery.scoring import DataBundle
    b = DataBundle.build(_daily(), train_ratio=0.7)
    assert b.train_end is not None
    assert "fwd_ret_1d" in b.fwd_returns.columns

def test_quick_fitness_positive_for_signal():
    from factorzen.discovery.scoring import DataBundle, quick_fitness
    daily = _daily()
    b = DataBundle.build(daily, train_ratio=0.7)
    fac = _signal_factor_df(daily)
    res = quick_fitness(fac, b, segment="train")
    assert res["ic_mean"] > 0.05
    assert res["n"] > 0

def test_max_correlation_self_is_one():
    from factorzen.discovery.scoring import max_correlation
    daily = _daily()
    fac = _signal_factor_df(daily).rename({"factor_value": "factor_clean"})
    corr = max_correlation(fac.rename({"factor_clean": "factor_value"}),
                           {"self": fac})
    assert corr > 0.99

def test_max_correlation_pairwise_ignores_degenerate_pool_factor():
    """R3 复现：池里混入一个退化(截面常数)因子，不应把候选与真实高相关因子的相关性抹成 0。

    历史 bug：max_correlation 把候选 + 全池一次性 inner-join 交给 compute_factor_correlation，
    任一池因子截面 std==0 就丢掉整条截面 → count=0 → 所有真实相关一起被抹成 0.0。
    pairwise 修法：候选对池中每个因子单独算，退化因子只影响它自己那一对。
    """
    from factorzen.discovery.scoring import max_correlation
    daily = _daily()
    good = _signal_factor_df(daily).rename({"factor_value": "factor_clean"})  # 好池因子
    # 退化：同一 (trade_date, ts_code) 键上的常数因子，截面 std==0
    degenerate = good.with_columns(pl.lit(1.0).alias("factor_clean"))
    cand = _signal_factor_df(daily)  # 候选 == good（完全相关）
    corr = max_correlation(cand, {"good": good, "degenerate": degenerate})
    assert corr > 0.99  # 修前因退化因子污染整表返回 0.0


def test_fitness_sort_key_is_tstat_not_raw_ir():
    """R2：排序键由裸 IR 换成 t-stat。fitness 现在跟 t-stat 走，且 t-stat≠IR（换的是键而非恒等）。"""
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.scoring import DataBundle, score_candidate
    daily = _daily(n_stocks=40, n_days=120)
    b = DataBundle.build(daily, train_ratio=0.7)
    fac = _noisy_signal_factor_df(daily)
    sc = score_candidate(fac, parse_expr("close"), b, pool={}, gamma=0.002)
    assert sc["tstat_train"] != 0.0
    # fitness == t-stat − 复杂度惩罚（pool 空 → mc=0）；若仍用 ir 则会与此不符（因 t-stat≠ir）
    assert sc["fitness"] == pytest.approx(sc["tstat_train"] - 0.002 * sc["complexity"], abs=1e-9)
    assert abs(sc["tstat_train"] - sc["ir_train"]) > 1e-6

def test_fitness_low_sample_tstat_gate_kills_ir_illusion():
    """R2 核心：n<=4 时 HAC t-stat=0 → 低样本候选 fitness 不再吃 raw IR 的虚高。"""
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.scoring import DataBundle, quick_fitness, score_candidate
    daily = _daily(n_stocks=40, n_days=6)          # train 段仅 4 个有效 IC 日
    b = DataBundle.build(daily, train_ratio=0.5)
    fac = _noisy_signal_factor_df(daily)
    train = quick_fitness(fac, b, segment="train")
    sc = score_candidate(fac, parse_expr("close"), b, pool={}, gamma=0.002)
    assert train["n"] <= 4                          # 低样本
    assert sc["tstat_train"] == 0.0                 # t-stat 的 n>4 门槛未过
    assert sc["fitness"] <= 1e-9                    # 只剩复杂度惩罚，raw IR 被无视

def test_score_penalizes_complexity():
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.scoring import DataBundle, score_candidate
    daily = _daily()
    b = DataBundle.build(daily)
    fac = _signal_factor_df(daily)
    simple = score_candidate(fac, parse_expr("close"), b, pool={}, gamma=0.01)
    # 复杂表达式（节点更多）在相同 IC 下 fitness 更低
    assert simple["complexity"] == 1
    # 相同因子值(IC 相同) + 更复杂的 node → complexity 更大 → fitness 更低（纯复杂度惩罚）
    complex_score = score_candidate(fac, parse_expr("ts_mean(close, 5)"), b, pool={}, gamma=0.01)
    assert complex_score["complexity"] > simple["complexity"]
    assert complex_score["fitness"] < simple["fitness"]

def test_quick_fitness_uses_horizon_1_only(monkeypatch):
    """挖掘 quick_fitness 只算 1d IC；5/10/20d 无人消费（审计 Wave2 项 3）。"""
    from factorzen.discovery import scoring as scoring_mod
    from factorzen.discovery.scoring import DataBundle, quick_fitness

    daily = _daily()
    b = DataBundle.build(daily)
    fac = _signal_factor_df(daily)

    seen: list = []
    _orig = scoring_mod.compute_rank_ic

    def _wrap(*args, **kwargs):
        seen.append(kwargs.get("horizons"))
        return _orig(*args, **kwargs)

    monkeypatch.setattr(scoring_mod, "compute_rank_ic", _wrap)
    res = quick_fitness(fac, b, segment="train")
    assert seen == [[1]]
    assert res["n"] > 0
    # 与显式 1d 主 IC 一致：再跑无 mock 对照
    monkeypatch.setattr(scoring_mod, "compute_rank_ic", _orig)
    res2 = quick_fitness(fac, b, segment="train")
    assert res["ic_mean"] == pytest.approx(res2["ic_mean"], abs=1e-12)
    assert res["ir"] == pytest.approx(res2["ir"], abs=1e-12)
    assert res["tstat"] == pytest.approx(res2["tstat"], abs=1e-12)
    assert res["n"] == res2["n"]

# ==== 来自 test_discovery_derived.py ====
def _mask_df() -> pl.DataFrame:
    # 单股 3 天,已排序;含派生所需全部列
    return pl.DataFrame({
        "trade_date": [1, 2, 3],
        "ts_code": ["A", "A", "A"],
        "open": [10.0, 11.0, 12.0],
        "high": [11.0, 12.0, 13.0],
        "low": [9.0, 10.0, 11.0],
        "close": [10.5, 11.5, 12.5],
        "close_adj": [10.5, 11.5, 12.5],
        "pre_close": [10.0, 10.5, 11.5],
        "vol": [1e5, 1e5, 1e5],
        "amount": [1e6, 1e6, 1e6],
    }).sort(["ts_code", "trade_date"])

def test_add_derived_columns_values():
    from factorzen.discovery.derived import add_derived_columns
    out = add_derived_columns(_mask_df())
    for col in ["vwap", "log_vol", "ret_1d", "amplitude", "intraday_ret", "overnight_ret"]:
        assert col in out.columns
    row0 = out.row(0, named=True)
    assert abs(row0["amplitude"] - (11.0 - 9.0) / 10.0) < 1e-9          # (high-low)/pre_close
    assert abs(row0["intraday_ret"] - (10.5 / 10.0 - 1.0)) < 1e-9        # close/open-1
    assert abs(row0["overnight_ret"] - (10.0 / 10.0 - 1.0)) < 1e-9       # open/pre_close-1

def test_add_derived_columns_safe_when_pre_close_zero():
    from factorzen.discovery.derived import add_derived_columns
    df = _mask_df().with_columns(
        pl.when(pl.col("trade_date") == 1).then(0.0)
        .otherwise(pl.col("pre_close")).alias("pre_close"))
    out = add_derived_columns(df)
    assert out.row(0, named=True)["overnight_ret"] is None  # 分母 0 → null,不崩

# ==== 来自 test_run_mine_joins_daily_basic.py ====
def test_run_mine_joins_daily_basic_into_frame(monkeypatch):
    import factorzen.daily.data.context as ctx_mod
    import factorzen.pipelines.factor_mine as fm

    d = [date(2024, 1, 1), date(2024, 1, 2)]
    daily = pl.DataFrame({
        "trade_date": d * 2,
        "ts_code": ["A.SZ", "A.SZ", "B.SZ", "B.SZ"],
        "close": [10.0, 11.0, 20.0, 21.0], "close_adj": [10.0, 11.0, 20.0, 21.0],
        "open": [10.0, 11.0, 20.0, 21.0], "high": [10.0, 11.0, 20.0, 21.0],
        "low": [10.0, 11.0, 20.0, 21.0], "vol": [1e5, 1e5, 1e5, 1e5],
        "amount": [1e6, 1e6, 1e6, 1e6],
    })
    basic = pl.DataFrame({
        "trade_date": d * 2,
        "ts_code": ["A.SZ", "A.SZ", "B.SZ", "B.SZ"],
        "total_mv": [5e5, 5e5, 8e5, 8e5], "pb": [1.5, 1.5, 2.0, 2.0],
    })

    class _FakeCtx:
        def __init__(self, **kw):
            pass

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return basic.lazy()

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    captured: dict = {}

    def _fake_run_session(frame, **kw):
        captured["frame"] = frame
        return {"candidates": [], "session_dir": "x", "n_trials": 0, "n_scored": 0}

    monkeypatch.setattr(fm, "run_session", _fake_run_session)

    fm.run_mine(start="20240101", end="20240102", n_trials=1)

    cols = set(captured["frame"].columns)
    assert "total_mv" in cols and "pb" in cols, (
        f"run_mine 传给 run_session 的帧应含 daily_basic 基本面列（否则 BASIC_FEATURES 死叶子），实得 {cols}"
    )

def test_prepare_mining_daily_default_warmup_covers_search_space(monkeypatch):
    """默认预热前缀 = search_space_max_lookback()（覆盖搜索空间最大回看），不再是会误拒
    长窗口/深嵌套因子的旧默认 60。FactorDataContext 收到的 lookback_days 即证据。"""
    import factorzen.daily.data.context as ctx_mod
    import factorzen.pipelines.factor_mine as fm
    from factorzen.discovery.search.random_search import search_space_max_lookback

    captured: dict = {}
    empty = pl.DataFrame({"trade_date": [], "ts_code": []})

    class _FakeCtx:
        def __init__(self, **kw):
            captured["lookback_days"] = kw.get("lookback_days")

        @property
        def daily(self):
            return empty.lazy()

        @property
        def daily_basic(self):
            return empty.lazy()

    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)

    fm.prepare_mining_daily("20240101", "20240201")

    assert captured["lookback_days"] == search_space_max_lookback()

