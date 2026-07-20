"""合并自: test_combine_pipeline.py, test_combination_methods.py
目标: test_combine_methods.py

--- 来源 test_combine_pipeline.py ---
test_combine_from_session.py：mine→combine 端到端接线:因子库 session → 物化 → 四方法 OOS 组合。
test_combine_net_of_cost.py：`_evaluate_oos` 的带成本净收益列。
test_greedy_decorrelate_parity.py：_greedy_decorrelate 决策 parity：紧凑矩阵加速不得改变 kept/dropped 决策。

--- 来源 test_combination_methods.py ---
test_combination.py：S6 防回归：验证多因子合成方法。
test_combination_lgbm.py：LightGBM 组合器的测试:可学习性 / 确定性 / 泄漏探针 / 缺值处理。
test_combination_cv.py：Purged & embargoed walk-forward CV 切分协议的测试。
test_combination_importance.py：因子重要性 explain 的测试:gain / shap(可选) / 缺 shap 回退。
test_combination_robustness.py：组合层健壮性:因子库常含空因子/覆盖异质,组合器须丢弃退化因子 + 外连接容缺,
"""

from __future__ import annotations

import builtins
import datetime as dt
import importlib.util
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factorzen.pipelines import factor_combine
from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.experiment import (
    _COST_PER_SIDE,
    _evaluate_oos,
    _top_bucket_turnover_series,
)
from factorzen.research.combination.importance import explain
from factorzen.research.combination.methods import equal_weight
from factorzen.research.combination.models import (
    LGBMCombiner,
    build_panel,
    combine_lgbm,
)
from factorzen.research.combination.oos import combine_oos
from factorzen.research.combination.pipeline import (
    instantiate_factor,
    prepare_return_frame,
)


# ==== 来自 test_combine_pipeline.py ====
# ==== 来自 test_combine_from_session.py ====
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
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "close_adj": px,
                         "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)

def _session(tmp_path, exprs):
    sess = tmp_path / "sess"
    sess.mkdir()
    pl.DataFrame({"rank": list(range(1, len(exprs) + 1)),
                  "expression": exprs, "passed": [True] * len(exprs)}
                 ).write_csv(sess / "candidates.csv")
    return str(sess)

def test_combine_from_session_end_to_end(tmp_path, monkeypatch):
    """因子库(≥2 因子)→ 物化 + 收益面板 + 四方法 OOS 对比,返回 comparison。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily",
                        lambda *a, **k: _daily())
    session = _session(tmp_path, ["rank(close)", "ts_mean(vol,5)", "neg(rank(ts_std(close,10)))"])
    res = factor_combine.combine_from_session(
        session_dir=session, start="20230103", end="20231231", universe=None,
        horizon=5, train_days=60, test_days=15, out_dir=str(tmp_path / "out"))
    comp = res["comparison"]
    methods = set(comp["method"].to_list())
    assert {"equal_weight", "ic_weighted", "max_ir"} <= methods   # 至少线性三法都跑了
    assert comp.height >= 3

def test_combine_from_session_needs_two_factors(tmp_path, monkeypatch):
    """因子库不足 2 个 → 明确报错(组合至少需两个,不静默产垃圾)。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    session = _session(tmp_path, ["rank(close)"])
    with pytest.raises(ValueError, match="不足 2 个"):
        factor_combine.combine_from_session(
            session_dir=session, start="20230103", end="20231231", out_dir=str(tmp_path / "o"))

def test_combine_from_session_passed_only_filters(tmp_path, monkeypatch):
    """默认只取 passed=True 的库因子;过滤后不足 2 个则报错。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    sess = tmp_path / "s2"
    sess.mkdir()
    pl.DataFrame({"rank": [1, 2, 3], "expression": ["rank(close)", "rank(vol)", "rank(high)"],
                  "passed": [True, False, False]}).write_csv(sess / "candidates.csv")
    with pytest.raises(ValueError, match="不足 2 个"):
        factor_combine.combine_from_session(
            session_dir=str(sess), start="20230103", end="20231231", out_dir=str(tmp_path / "o"))

# ── 任务 C：多 session 合并去重 + 贪心去相关 ──────────────────────────────────
def _session_with_ic(tmp_path, name, rows):
    """rows: list[(expression, holdout_ic)] → 写含 holdout_ic 列的 candidates.csv。"""
    sess = tmp_path / name
    sess.mkdir()
    pl.DataFrame({"rank": list(range(1, len(rows) + 1)),
                  "expression": [e for e, _ in rows],
                  "holdout_ic": [ic for _, ic in rows],
                  "passed": [True] * len(rows)}).write_csv(sess / "candidates.csv")
    return str(sess)

def test_combine_merges_and_dedups_across_sessions(tmp_path, monkeypatch):
    """两个 session 各含同一表达式（规范形相同）→ 合并后只出现一次。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    s1 = _session_with_ic(tmp_path, "s1", [("rank(close)", 0.05), ("ts_mean(vol,5)", 0.04)])
    # rank( close ) 空格差异 → parse_expr 规范化后与 rank(close) 相同
    s2 = _session_with_ic(tmp_path, "s2", [("rank( close )", 0.05),
                                           ("neg(rank(ts_std(close,10)))", 0.03)])
    res = factor_combine.combine_from_session(
        session_dirs=[s1, s2], start="20230103", end="20231231", horizon=5,
        train_days=60, test_days=15, decorr_threshold=1.0, out_dir=str(tmp_path / "o"))
    used = res["factors_used"]
    assert used.count("rank(close)") == 1, f"规范形重复未去重: {used}"
    assert "ts_mean(vol, 5)" in used and "neg(rank(ts_std(close, 10)))" in used

def test_combine_decorr_drops_near_duplicate(tmp_path, monkeypatch):
    """构造高相关对（ts_mean(close,20) 与 ts_mean(close,21)）→ 仅 |holdout_ic| 高者存活，
    被剔者记入 dropped_correlated。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    sess = _session_with_ic(tmp_path, "s", [
        ("ts_mean(close,20)", 0.08),   # |ic| 高 → 存活
        ("ts_mean(close,21)", 0.03),   # 与上近亲 → 被剔
        ("rank(neg(vol))", 0.05),      # 独立 → 存活
    ])
    res = factor_combine.combine_from_session(
        session_dirs=[sess], start="20230103", end="20231231", horizon=5,
        train_days=60, test_days=15, decorr_threshold=0.7, out_dir=str(tmp_path / "o"))
    dropped = [d["identity"] for d in res["dropped_correlated"]]
    assert "ts_mean(close, 21)" in dropped, f"高相关近亲未被剔: {res['dropped_correlated']}"
    assert "ts_mean(close, 20)" not in dropped, "|holdout_ic| 高者不应被剔"
    used = res["factors_used"]
    assert "ts_mean(close, 20)" in used and "ts_mean(close, 21)" not in used

def test_combine_decorr_threshold_one_keeps_all(tmp_path, monkeypatch):
    """decorr_threshold=1.0 → 逃生口，无剔除。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    sess = _session_with_ic(tmp_path, "s", [("ts_mean(close,20)", 0.08), ("ts_mean(close,21)", 0.03)])
    res = factor_combine.combine_from_session(
        session_dirs=[sess], start="20230103", end="20231231", horizon=5,
        train_days=60, test_days=15, decorr_threshold=1.0, out_dir=str(tmp_path / "o"))
    assert res["dropped_correlated"] == []
    assert len(res["factors_used"]) == 2

def test_combine_decorr_below_two_errors(tmp_path, monkeypatch):
    """去相关后 < 2 个因子 → 报错（组合至少需两个）。"""
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", lambda *a, **k: _daily())
    sess = _session_with_ic(tmp_path, "s", [("ts_mean(close,20)", 0.08), ("ts_mean(close,21)", 0.03)])
    with pytest.raises(ValueError, match="不足 2 个"):
        factor_combine.combine_from_session(
            session_dirs=[sess], start="20230103", end="20231231", horizon=5,
            train_days=60, test_days=15, decorr_threshold=0.7, out_dir=str(tmp_path / "o"))

# ==== 来自 test_combine_net_of_cost.py ====
N_GROUPS = 5

def _panel(day_specs: list[list[tuple[str, float, float]]]):
    """day_specs[i] = 第 i 日的 [(ts_code, factor_value, ret), ...]。"""
    rows_f, rows_r = [], []
    for i, day in enumerate(day_specs):
        d = f"2024-01-{i + 1:02d}"
        for code, fv, rv in day:
            rows_f.append({"trade_date": d, "ts_code": code, "factor_value": fv})
            rows_r.append({"trade_date": d, "ts_code": code, "ret": rv})
    return pl.DataFrame(rows_f), pl.DataFrame(rows_r)

def _stable_day(d_idx: int, top_codes: list[str]):
    """10 只票：`top_codes` 是分数最高的 2 只（top 1/5 桶），收益 +1%；其余 0。

    分数与收益都写死 ⇒ 每日 spread = top均值 − bottom均值 = 0.01 − 0.0 = 0.01，
    与被测函数无关，可用作独立 ground-truth。
    """
    day = []
    others = [f"S{j}" for j in range(10) if f"S{j}" not in top_codes]
    for c in top_codes:
        day.append((c, 10.0, 0.01))
    # bottom 2 只分数最低、收益 0；中间 6 只分数居中、收益 0
    for rank, c in enumerate(others):
        day.append((c, float(rank), 0.0))
    return day

def test_zero_turnover_net_equals_gross():
    """持仓完全不变 ⇒ 换手 0 ⇒ 净 spread 必须**逐位等于**毛 spread。"""
    days = [_stable_day(i, ["S0", "S1"]) for i in range(6)]
    combined, ret_df = _panel(days)
    out = _evaluate_oos(combined, ret_df, n_groups=N_GROUPS)

    assert out["turnover"] == pytest.approx(0.0)
    assert out["net_spread_10bp"] == pytest.approx(out["top_bottom_spread"])
    # 独立 ground-truth：spread 由构造决定 = 0.01
    assert out["top_bottom_spread"] == pytest.approx(0.01, abs=1e-12)

def test_full_turnover_charges_expected_fee():
    """每期 top 桶**整桶换掉** ⇒ 换手 1.0 ⇒ 净 = 毛 − 4×1.0×10bp。

    期望值由构造独立给出（毛 0.01、换手 1.0），不引用被测函数的中间量。
    """
    tops = [["S0", "S1"], ["S2", "S3"], ["S4", "S5"], ["S6", "S7"]]
    combined, ret_df = _panel([_stable_day(i, t) for i, t in enumerate(tops)])
    out = _evaluate_oos(combined, ret_df, n_groups=N_GROUPS)

    assert out["turnover"] == pytest.approx(1.0)
    # 4 天：第 0 天不扣费（无前一期），后 3 天各扣 4×1.0×0.001
    fee_per_day = 4.0 * 1.0 * _COST_PER_SIDE
    expected = 0.01 - (3 * fee_per_day) / 4
    assert out["net_spread_10bp"] == pytest.approx(expected, abs=1e-12)
    assert out["net_spread_10bp"] < out["top_bottom_spread"]

def test_half_turnover_is_between():
    """换手 0.5（2 只里换 1 只）⇒ 费用恰为全换的一半。"""
    tops = [["S0", "S1"], ["S1", "S2"], ["S2", "S3"], ["S3", "S4"]]
    combined, ret_df = _panel([_stable_day(i, t) for i, t in enumerate(tops)])
    out = _evaluate_oos(combined, ret_df, n_groups=N_GROUPS)

    assert out["turnover"] == pytest.approx(0.5)
    fee_per_day = 4.0 * 0.5 * _COST_PER_SIDE
    expected = 0.01 - (3 * fee_per_day) / 4
    assert out["net_spread_10bp"] == pytest.approx(expected, abs=1e-12)

def test_turnover_series_matches_aggregate():
    """逐期序列的均值必须等于聚合版——两者是同一口径的两种取法。

    净收益必须逐期扣费再平均（换手与收益可能相关），
    但**均值口径**上二者应一致，此断言守住这个不变量。
    """
    import numpy as np

    from factorzen.research.combination.experiment import _top_bucket_turnover

    tops = [frozenset({"a", "b"}), frozenset({"b", "c"}),
            frozenset({"c", "d"}), frozenset({"c", "d"})]
    series = _top_bucket_turnover_series(tops)
    assert series == pytest.approx([0.5, 0.5, 0.0])
    assert float(np.mean(series)) == pytest.approx(_top_bucket_turnover(tops))

def test_empty_panel_has_net_keys():
    """空面板也须带上新键，否则消费方 `r['net_spread_10bp']` 会 KeyError。"""
    empty = pl.DataFrame(
        {"trade_date": [], "ts_code": [], "factor_value": []},
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "factor_value": pl.Float64},
    )
    ret_df = pl.DataFrame(
        {"trade_date": [], "ts_code": [], "ret": []},
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "ret": pl.Float64},
    )
    out = _evaluate_oos(empty, ret_df)
    assert out["net_spread_10bp"] == 0.0
    assert out["net_sharpe_10bp"] == 0.0

def test_net_sharpe_zero_when_no_variance():
    """净收益方差为 0（恒定）⇒ SR 定义为 0，不得抛除零或返 inf/nan。"""
    days = [_stable_day(i, ["S0", "S1"]) for i in range(6)]
    combined, ret_df = _panel(days)
    out = _evaluate_oos(combined, ret_df, n_groups=N_GROUPS)
    assert out["net_sharpe_10bp"] == 0.0

# ==== 来自 test_greedy_decorrelate_parity.py ====
def _panel_from_matrix(
    mat: np.ndarray,
    *,
    start: date = date(2020, 1, 2),
    stock_prefix: str = "",
    drop_dates: set[int] | None = None,
    drop_stocks: set[int] | None = None,
) -> pl.DataFrame:
    """(D×S) 矩阵 → [trade_date, ts_code, factor_value] 面板；可选缺日期/缺股票。"""
    d_n, s_n = mat.shape
    trade_dates: list[date] = []
    ts_codes: list[str] = []
    values: list[float] = []
    for di in range(d_n):
        if drop_dates and di in drop_dates:
            continue
        dt = start + timedelta(days=di)
        for si in range(s_n):
            if drop_stocks and si in drop_stocks:
                continue
            v = mat[di, si]
            trade_dates.append(dt)
            ts_codes.append(f"{stock_prefix}{si:06d}.SH")
            # 用 nan 而非 None，避免前段全缺时 schema 推断成 Null
            values.append(float(v) if np.isfinite(v) else float("nan"))
    return pl.DataFrame({
        "trade_date": trade_dates,
        "ts_code": ts_codes,
        "factor_value": values,
    })

def _synth_suite(seed: int, *, n_factors: int = 10, n_days: int = 60, n_stocks: int = 40):
    """构造一组合成因子：近亲≈0.95、边界≈0.70、全常数退化、含 NaN 块。"""
    rng = np.random.default_rng(seed)
    mats: list[np.ndarray] = []
    exprs: list[str] = []

    # f0: 基准独立信号
    base = rng.standard_normal((n_days, n_stocks))
    mats.append(base)
    exprs.append(f"f0_base_s{seed}")

    # f1: 近亲（corr≈0.95）— 与 f0 同向高相关
    twin = 0.95 * base + 0.05 * rng.standard_normal((n_days, n_stocks))
    mats.append(twin)
    exprs.append(f"f1_near_twin_s{seed}")

    # f2: 边界相关（目标 |corr| ∈ [0.69, 0.71] 附近，与 f0）
    orth = rng.standard_normal((n_days, n_stocks))
    alpha = 0.70
    boundary = alpha * base + np.sqrt(max(1.0 - alpha * alpha, 0.0)) * orth
    mats.append(boundary)
    exprs.append(f"f2_boundary_s{seed}")

    # f3: 全常数退化
    mats.append(np.full((n_days, n_stocks), 3.14))
    exprs.append(f"f3_const_s{seed}")

    # f4: 含 NaN 块（前 1/3 日全缺 + 随机点缺）
    nan_block = rng.standard_normal((n_days, n_stocks))
    nan_block[: n_days // 3, :] = np.nan
    mask = rng.random((n_days, n_stocks)) < 0.1
    nan_block[mask] = np.nan
    mats.append(nan_block)
    exprs.append(f"f4_nan_block_s{seed}")

    # 其余独立噪声，补满 n_factors
    k = 5
    while len(mats) < n_factors:
        mats.append(rng.standard_normal((n_days, n_stocks)))
        exprs.append(f"f{k}_noise_s{seed}")
        k += 1

    materialized = [(e, _panel_from_matrix(m)) for e, m in zip(exprs, mats, strict=False)]
    return materialized

def _assert_decision_parity(new_kept, new_dropped, ref_kept, ref_dropped, *, atol: float = 1e-9):
    assert [e for e, _ in new_kept] == [e for e, _ in ref_kept], (
        f"kept 表达式序列不一致\n new={[e for e,_ in new_kept]}\n ref={[e for e,_ in ref_kept]}"
    )
    # kept 必须是原面板对象（下游写 parquet）
    for (ne, nf), (re, rf) in zip(new_kept, ref_kept, strict=False):
        assert ne == re
        assert nf is rf or nf.equals(rf), f"kept 面板被替换: {ne}"

    assert len(new_dropped) == len(ref_dropped)
    for nd, rd in zip(new_dropped, ref_dropped, strict=False):
        assert nd["identity"] == rd["identity"]
        assert nd["corr_with"] == rd["corr_with"], (
            f"corr_with 不一致 for {nd['identity']}: "
            f"new={nd['corr_with']} ref={rd['corr_with']}"
        )
        assert abs(float(nd["corr"]) - float(rd["corr"])) <= atol, (
            f"corr 超容差 for {nd['identity']}: "
            f"new={nd['corr']} ref={rd['corr']} |diff|={abs(float(nd['corr'])-float(rd['corr']))}"
        )

# ── 1. 决策 parity（核心）────────────────────────────────────────────────────

def test_greedy_decorrelate_decision_parity_three_seeds():
    """随机 3 组合成面板：新旧 kept 序 + dropped(expression/corr_with) 完全一致，corr≤1e-9。"""
    from factorzen.pipelines.factor_combine import (
        _greedy_decorrelate,
        _greedy_decorrelate_reference,
    )

    threshold = 0.7
    for seed in (0, 7, 42):
        mats = _synth_suite(seed)
        new_k, new_d = _greedy_decorrelate(mats, threshold)
        ref_k, ref_d = _greedy_decorrelate_reference(mats, threshold)
        _assert_decision_parity(new_k, new_d, ref_k, ref_d)

# ── 2. 逃生口 threshold=1.0 ──────────────────────────────────────────────────

def test_greedy_decorrelate_threshold_one_keeps_all():
    """threshold=1.0 → >1.0 恒 False → 全 kept、dropped 空。"""
    from factorzen.pipelines.factor_combine import (
        _greedy_decorrelate,
        _greedy_decorrelate_reference,
    )

    mats = _synth_suite(1)
    new_k, new_d = _greedy_decorrelate(mats, 1.0)
    ref_k, ref_d = _greedy_decorrelate_reference(mats, 1.0)
    assert new_d == [] and ref_d == []
    assert [e for e, _ in new_k] == [e for e, _ in mats]
    assert [e for e, _ in ref_k] == [e for e, _ in mats]
    _assert_decision_parity(new_k, new_d, ref_k, ref_d)

# ── 3. 异质覆盖（缺日期 / 缺股票）───────────────────────────────────────────

def test_greedy_decorrelate_heterogeneous_coverage_parity():
    """一因子只有半段日期、另一因子缺部分股票：不崩且与旧实现一致。"""
    from factorzen.pipelines.factor_combine import (
        _greedy_decorrelate,
        _greedy_decorrelate_reference,
    )

    rng = np.random.default_rng(99)
    n_days, n_stocks = 60, 40
    a = rng.standard_normal((n_days, n_stocks))
    b = 0.96 * a + 0.04 * rng.standard_normal((n_days, n_stocks))  # 近亲
    c = rng.standard_normal((n_days, n_stocks))

    mats = [
        ("half_dates", _panel_from_matrix(a, drop_dates=set(range(n_days // 2, n_days)))),
        ("full_near", _panel_from_matrix(b)),
        ("missing_stocks", _panel_from_matrix(c, drop_stocks=set(range(0, 10)))),
        ("noise", _panel_from_matrix(rng.standard_normal((n_days, n_stocks)))),
    ]
    new_k, new_d = _greedy_decorrelate(mats, 0.7)
    ref_k, ref_d = _greedy_decorrelate_reference(mats, 0.7)
    _assert_decision_parity(new_k, new_d, ref_k, ref_d)

# ── 4. 缩尺性能冒烟（打印 + 宽松注释，CI 不依赖时序）────────────────────────

def test_greedy_decorrelate_scaled_perf_smoke():
    """~20 因子 × 250 日 × 100 股：打印 A/B 耗时；新实现预期 ≪ 旧（不硬断言防抖动）。"""
    from factorzen.pipelines.factor_combine import (
        _greedy_decorrelate,
        _greedy_decorrelate_reference,
    )

    rng = np.random.default_rng(123)
    n_factors, n_days, n_stocks = 20, 250, 100
    mats = []
    base = rng.standard_normal((n_days, n_stocks))
    for i in range(n_factors):
        if i == 1:
            m = 0.93 * base + 0.07 * rng.standard_normal((n_days, n_stocks))
        else:
            m = rng.standard_normal((n_days, n_stocks))
        mats.append((f"perf_f{i}", _panel_from_matrix(m)))

    t0 = time.perf_counter()
    ref_k, ref_d = _greedy_decorrelate_reference(mats, 0.7)
    t_ref = time.perf_counter() - t0

    t1 = time.perf_counter()
    new_k, new_d = _greedy_decorrelate(mats, 0.7)
    t_new = time.perf_counter() - t1

    _assert_decision_parity(new_k, new_d, ref_k, ref_d)
    # 打印供人工观察；目标约 <1/5，但不写入硬断言以免 CI 抖动。
    print(
        f"\n[decorr A/B] n={n_factors}×{n_days}×{n_stocks} "
        f"ref={t_ref:.3f}s new={t_new:.3f}s speedup={t_ref / max(t_new, 1e-9):.1f}x"
    )
    # 宽松护栏：仅在旧实现足够慢时检查加速（本地/CI 都极快则跳过）
    if t_ref > 1.0:
        assert t_new < t_ref / 5.0, (
            f"加速不足: ref={t_ref:.3f}s new={t_new:.3f}s (期望 <1/5)"
        )


# ==== 来自 test_combination_methods.py ====
# ==== 来自 test_combination.py ====
class _DummyFactor:
    required_data = ["daily"]
    lookback_days = 3

def test_instantiate_factor_builds_instance_from_registry_class():
    factor = instantiate_factor("dummy", registry_getter=lambda _name: _DummyFactor)

    assert isinstance(factor, _DummyFactor)
    assert factor.required_data == ["daily"]
    assert factor.lookback_days == 3

def test_prepare_return_frame_adds_ret_and_forward_returns():
    price_df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "ts_code": ["000001.SZ"] * 3,
            "close": [100.0, 110.0, 121.0],
        }
    )

    out = prepare_return_frame(price_df, horizons=[1])

    assert "ret" in out.columns
    assert "fwd_ret_1d" in out.columns
    assert out["ret"].to_list() == pytest.approx([None, 0.10, 0.10])
    assert out["fwd_ret_1d"].to_list() == pytest.approx([0.10, 0.10, None])

def _make_factor_ret(
    n_dates: int = 100,
    n_stocks: int = 50,
    n_factors: int = 3,
    seed: int = 0,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """合成多个弱相关因子 + 前向收益。"""
    rng = np.random.default_rng(seed)
    dates = [f"2024-{(i // 28 + 1):02d}-{(i % 28 + 1):02d}" for i in range(n_dates)]
    stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]

    factor_dfs: dict[str, pl.DataFrame] = {}
    for fi in range(n_factors):
        rows = []
        for d in dates:
            vals = rng.standard_normal(n_stocks)
            for i, s in enumerate(stocks):
                rows.append({"trade_date": d, "ts_code": s, "factor_value": float(vals[i])})
        df = pl.DataFrame(rows).with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d"))
        factor_dfs[f"factor_{fi}"] = df

    # 前向收益：弱正 IC ≈ 0.05 with factor_0
    ret_rows = []
    f0_map: dict[tuple, float] = {}
    f0 = factor_dfs["factor_0"]
    for row in f0.iter_rows(named=True):
        f0_map[(str(row["trade_date"]), row["ts_code"])] = row["factor_value"]

    for d in dates:
        rets = rng.normal(0, 0.02, n_stocks)
        for i, s in enumerate(stocks):
            signal = f0_map.get((d, s), 0.0)
            rets[i] += 0.003 * signal
        for i, s in enumerate(stocks):
            ret_rows.append({"trade_date": d, "ts_code": s, "ret": float(rets[i])})

    ret_df = pl.DataFrame(ret_rows).with_columns(
        pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    return factor_dfs, ret_df

class TestEqualWeight:

    def test_no_nan(self):
        """等权合成结果不含 null/nan。"""
        factor_dfs, _ = _make_factor_ret()
        result = equal_weight(factor_dfs)
        assert result["factor_value"].drop_nulls().len() == len(result)
        assert result["factor_value"].is_nan().sum() == 0

    def test_cross_sectional_mean_near_zero(self):
        """等权合成后截面均值接近 0（z-score 均值属性）。"""
        factor_dfs, _ = _make_factor_ret()
        result = equal_weight(factor_dfs)
        mean_per_date = result.group_by("trade_date").agg(
            pl.col("factor_value").mean().alias("cs_mean")
        )
        assert mean_per_date["cs_mean"].abs().mean() < 0.1

# ==== 来自 test_combination_lgbm.py ====
def _panel__lgbm(n_days=200, n_stocks=50, seed=0):
    """ret = 0.8*fa - 0.4*fb + 噪声:fa 正贡献强、fb 负贡献。"""
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.8 * fa - 0.4 * fb + rng.standard_normal(n_stocks) * 0.3
        for s in range(n_stocks):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    return {"fa": pl.DataFrame(ra), "fb": pl.DataFrame(rb)}, pl.DataFrame(rr), dates

def _oos_rank_ic(combined: pl.DataFrame, ret_df: pl.DataFrame) -> float:
    m = combined.join(
        ret_df.with_columns(pl.col("trade_date").cast(pl.Utf8)),
        on=["trade_date", "ts_code"],
        how="inner",
    )
    ics = []
    for _d, g in m.group_by("trade_date"):
        if len(g) < 10:
            continue
        f = g["factor_value"].to_numpy()
        r = g["ret"].to_numpy()
        fr = f.argsort().argsort().astype(float)
        rr = r.argsort().argsort().astype(float)
        ic = float(np.corrcoef(fr, rr)[0, 1])
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics))

def test_lgbm_learns_signal():
    factor_dfs, ret_df, _ = _panel__lgbm()
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    out = combine_lgbm(factor_dfs, ret_df, cv, min_child_samples=20, n_estimators=80)
    assert _oos_rank_ic(out, ret_df) > 0.15

def test_lgbm_deterministic():
    factor_dfs, ret_df, _ = _panel__lgbm(n_days=120, n_stocks=30)
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    a = combine_lgbm(factor_dfs, ret_df, cv, seed=7, n_estimators=50).sort(
        ["trade_date", "ts_code"]
    )
    b = combine_lgbm(factor_dfs, ret_df, cv, seed=7, n_estimators=50).sort(
        ["trade_date", "ts_code"]
    )
    assert_frame_equal(a, b)

def test_lgbm_no_lookahead():
    """泄漏探针:扰动 cutoff 后收益,cutoff 前 OOS 预测逐行不变。"""
    factor_dfs, ret_df, dates = _panel__lgbm(n_days=120, n_stocks=30)
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    base = combine_lgbm(factor_dfs, ret_df, cv, seed=1, n_estimators=50)
    cutoff = dates[99]
    tampered_ret = ret_df.with_columns(
        pl.when(pl.col("trade_date") > cutoff)
        .then(pl.col("ret") * -3.0)
        .otherwise(pl.col("ret"))
        .alias("ret")
    )
    tampered = combine_lgbm(factor_dfs, tampered_ret, cv, seed=1, n_estimators=50)
    b = base.filter(pl.col("trade_date") <= cutoff).sort(["trade_date", "ts_code"])
    t = tampered.filter(pl.col("trade_date") <= cutoff).sort(["trade_date", "ts_code"])
    assert_frame_equal(b, t)

def test_build_panel_inner_join_and_ret():
    factor_dfs, ret_df, _ = _panel__lgbm(n_days=30, n_stocks=20)
    panel = build_panel(factor_dfs, ret_df)
    assert set(panel.columns) >= {"trade_date", "ts_code", "fa", "fb", "ret"}
    assert panel.height > 0

@pytest.mark.filterwarnings("ignore:build_panel")
def test_lgbm_drops_all_null_factor_and_continues():
    """一个因子全缺 → 丢弃它、用其余因子照常组合(健壮性:不因坏因子崩整个 run)。"""
    factor_dfs, ret_df, _ = _panel__lgbm(n_days=80, n_stocks=30)
    factor_dfs["fa"] = factor_dfs["fa"].with_columns(
        pl.lit(None, dtype=pl.Float64).alias("factor_value")
    )
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    out = combine_lgbm(factor_dfs, ret_df, cv, n_estimators=20)
    assert out.height > 0  # fb 仍在 → 正常产出

# ==== 来自 test_combination_cv.py ====
def _dates(n: int) -> list[str]:
    """n 个唯一升序交易日串(等长 8 字符,可字典序/整数比较)。"""
    return [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n)]

def test_purge_gap_holds():
    dates = _dates(100)
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    folds = cv.split(dates)
    assert len(folds) == 3  # (40|20)(60|20)(80|20)
    for tr, te in folds:
        # train 末尾与 test 首之间至少隔 purge_days+1(防前向标签重叠泄漏)
        assert dates.index(te[0]) - dates.index(tr[-1]) >= 5 + 1

def test_test_segments_contiguous_non_overlapping():
    dates = _dates(100)
    folds = PurgedWalkForwardCV(40, 20, 5).split(dates)
    flat = [d for _, te in folds for d in te]
    assert flat == dates[40:100]  # test 并集连续覆盖 train_days..end,无重叠

def test_max_train_before_min_test():
    dates = _dates(100)
    for tr, te in PurgedWalkForwardCV(40, 20, 5).split(dates):
        assert max(int(d) for d in tr) < min(int(d) for d in te)

def test_expanding_train_grows_and_nests():
    dates = _dates(100)
    folds = PurgedWalkForwardCV(40, 20, 5, expanding=True).split(dates)
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes)  # 单调不减
    assert set(folds[0][0]).issubset(set(folds[1][0]))  # 后折 train 含前折

def test_rolling_window_fixed_size():
    dates = _dates(100)
    folds = PurgedWalkForwardCV(40, 20, 5, expanding=False).split(dates)
    sizes = [len(tr) for tr, _ in folds]
    assert max(sizes) - min(sizes) <= 1  # 滚动窗训练集基本定长

def test_embargo_widens_gap():
    dates = _dates(100)
    f0 = PurgedWalkForwardCV(40, 20, 5, embargo_days=0).split(dates)
    f5 = PurgedWalkForwardCV(40, 20, 5, embargo_days=5).split(dates)
    # embargo 使 train 末尾更早,间隔更大
    assert dates.index(f5[0][0][-1]) < dates.index(f0[0][0][-1])

def test_insufficient_dates_raises():
    with pytest.raises(ValueError):
        PurgedWalkForwardCV(40, 20, 5).split(_dates(30))

# ==== 来自 test_combination_importance.py ====
_HAS_SHAP = importlib.util.find_spec("shap") is not None

def _fitted():
    """在 ret=0.8*fa-0.4*fb 合成数据上 fit,fa 贡献强于 fb。"""
    rng = np.random.default_rng(0)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(150)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(40)
        fb = rng.standard_normal(40)
        ret = 0.8 * fa - 0.4 * fb + rng.standard_normal(40) * 0.3
        for s in range(40):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    factor_dfs = {"fa": pl.DataFrame(ra), "fb": pl.DataFrame(rb)}
    panel = build_panel(factor_dfs, pl.DataFrame(rr))
    combiner = LGBMCombiner(min_child_samples=20, n_estimators=60, seed=0)
    combiner.fit(panel.select(["fa", "fb"]), panel["ret"])
    return combiner, panel.select(["fa", "fb"])

def test_explain_gain():
    c, x = _fitted()
    out = explain(c, x, method="gain")
    assert set(out.columns) == {"factor", "importance", "method"}
    assert (out["method"] == "gain").all()
    d = dict(zip(out["factor"].to_list(), out["importance"].to_list(), strict=True))
    assert d["fa"] > d["fb"]

@pytest.mark.skipif(not _HAS_SHAP, reason="shap 未安装")
def test_explain_auto_uses_shap_when_available():
    c, x = _fitted()
    out = explain(c, x, method="auto")
    assert (out["method"] == "shap").all()
    d = dict(zip(out["factor"].to_list(), out["importance"].to_list(), strict=True))
    assert d["fa"] > d["fb"]

def test_explain_auto_falls_back_to_gain_without_shap(monkeypatch):
    c, x = _fitted()
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "shap":
            raise ImportError("模拟 shap 未安装")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = explain(c, x, method="auto")
    assert (out["method"] == "gain").all()

def test_explain_unknown_method_raises():
    c, x = _fitted()
    with pytest.raises(ValueError):
        explain(c, x, method="nonsense")

# ==== 来自 test_combination_robustness.py ====
def _panel__robustness(n_days=120, n_stocks=40, seed=0, stocks=None):
    """ret = 0.7*fa - 0.3*fb + 噪声。stocks 指定则只在这些股票上出因子(制造覆盖异质)。"""
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    codes = [f"{s:04d}.SZ" for s in range(n_stocks)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.7 * fa - 0.3 * fb + rng.standard_normal(n_stocks) * 0.3
        for s in range(n_stocks):
            rr.append({"trade_date": d, "ts_code": codes[s], "ret": float(ret[s])})
            ra.append({"trade_date": d, "ts_code": codes[s], "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": codes[s], "factor_value": float(fb[s])})
    return {"fa": pl.DataFrame(ra), "fb": pl.DataFrame(rb)}, pl.DataFrame(rr), codes

def test_lgbm_drops_empty_factor_instead_of_crashing():
    """库里混入一个物化为空(0 行)的因子(陈旧基本面 ts_skew 退化)→ 丢弃它、用其余因子照常组合。

    复现:实测 csi300 池化库里 `ts_skew(netprofit_margin,60)` 物化 0 行 → inner join 塌空
    → 「训练面板为空」崩。修后应丢弃空因子、正常产出。
    """
    factor_dfs, ret_df, _ = _panel__robustness()
    factor_dfs["empty"] = pl.DataFrame(
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "factor_value": pl.Float64}
    )
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    out = combine_lgbm(factor_dfs, ret_df, cv, n_estimators=30, min_child_samples=20)
    assert out.height > 0

def test_lgbm_survives_heterogeneous_coverage():
    """两因子覆盖的股票**不相交** → inner join 全空(train 与 test 都空)→ 原实现崩。

    复现:实测覆盖率异质(部分因子 ~71k 行 vs 满 144k)→ 某 test 折 inner join 空
    → lightgbm「Input data must be 2 dimensional and non empty」崩。
    修后:外连接取并集、缺失特征交给 LGBM 原生 NaN 处理 → 覆盖并集、不崩。
    """
    fa_dfs, ret_df, codes = _panel__robustness(n_stocks=40, seed=1)
    half = len(codes) // 2
    left, right = set(codes[:half]), set(codes[half:])
    fa = fa_dfs["fa"].filter(pl.col("ts_code").is_in(left))
    fb = fa_dfs["fb"].filter(pl.col("ts_code").is_in(right))
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    out = combine_lgbm({"fa": fa, "fb": fb}, ret_df, cv, n_estimators=30, min_child_samples=20)
    assert out.height > 0
    covered = set(out["ts_code"].to_list())
    assert covered & left and covered & right  # 并集覆盖两组,不是只剩交集

def test_lgbm_all_null_factors_raise_clear_error():
    """所有因子都全缺 → 去除退化因子后 0 个有效 → 明确报错(不静默产垃圾)。"""
    factor_dfs, ret_df, _ = _panel__robustness(n_days=80)
    for k in list(factor_dfs):
        factor_dfs[k] = factor_dfs[k].with_columns(
            pl.lit(None, dtype=pl.Float64).alias("factor_value")
        )
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    with pytest.raises(ValueError):
        combine_lgbm(factor_dfs, ret_df, cv, n_estimators=20)

def test_combine_oos_survives_heterogeneous_coverage():
    """线性路径同样容缺:覆盖不相交的两因子外连接 + 缺失补 0(中性)→ 覆盖并集不崩。"""
    fa_dfs, ret_df, codes = _panel__robustness(n_stocks=40, seed=2)
    half = len(codes) // 2
    fa = fa_dfs["fa"].filter(pl.col("ts_code").is_in(set(codes[:half])))
    fb = fa_dfs["fb"].filter(pl.col("ts_code").is_in(set(codes[half:])))
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    out = combine_oos({"fa": fa, "fb": fb}, ret_df, cv, method="ic_weighted")
    assert out.height > 0
    covered = set(out["ts_code"].to_list())
    assert covered & set(codes[:half]) and covered & set(codes[half:])

def test_duplicate_join_keys_do_not_explode_rows():
    """因子面板含少量重复 (trade_date, ts_code) 时，链式 outer join 不得笛卡尔积爆炸。

    **实测根因（2026-07-19）**：`_zscore_and_merge` 把 k 个因子链式 full join，
    每次遇重复键行数相乘。生产物化产物里几乎每个因子面板都有 3 行重复
    （2026-06-30 的 6 只 603xxx，每键 4 条），重复率仅 0.0006%——
    但 62 个因子链式 join 下按 4^n 放大：逐步打点实测

        join #5: 6786 行 → join #10: 1,051,266 行（5 次 join 涨 155 倍）

    最终 anon-rss 打满 23GB 被 OOM killer 杀。P1-① 阶段 2 的四次尝试全折在这里。

    键唯一性是链式 join 的**前提**，不该假设成立而不校验——上游任何一处重复
    都会被指数放大成 OOM。
    """
    from factorzen.research.combination.methods import _zscore_and_merge

    n_days, n_stocks, k = 6, 20, 8
    dates = [f"2025010{i + 1}" for i in range(n_days)]
    codes = [f"{s:04d}.SZ" for s in range(n_stocks)]
    rng = np.random.default_rng(3)

    factor_dfs = {}
    for j in range(k):
        rows = [
            {"trade_date": d, "ts_code": c, "factor_value": float(rng.standard_normal())}
            for d in dates
            for c in codes
        ]
        # 每个因子在同一个 (日, 股) 上多出 3 条重复（模拟生产的 4 条/键）
        dup_key = (dates[0], codes[0])
        rows += [
            {"trade_date": dup_key[0], "ts_code": dup_key[1],
             "factor_value": float(rng.standard_normal())}
            for _ in range(3)
        ]
        factor_dfs[f"f{j}"] = pl.DataFrame(rows)

    merged, names = _zscore_and_merge(factor_dfs)

    assert len(names) == k
    # 唯一键数就是行数上界；未防御时这里是 4**8 = 65536 量级
    assert merged.height == n_days * n_stocks, (
        f"链式 join 把 {k} 个因子的重复键放大到 {merged.height} 行"
        f"（应为 {n_days * n_stocks}）"
    )
    assert merged.select(["trade_date", "ts_code"]).unique().height == merged.height

def test_duplicate_join_keys_emit_warning():
    """重复键必须告警——静默去重会掩盖上游数据缺陷（本例真实来源仍未查清）。"""
    from factorzen.research.combination.methods import _zscore_and_merge

    base = [
        {"trade_date": "20250101", "ts_code": "0001.SZ", "factor_value": 1.0},
        {"trade_date": "20250101", "ts_code": "0002.SZ", "factor_value": 2.0},
    ]
    dfs = {
        "clean": pl.DataFrame(base),
        "dirty": pl.DataFrame([
            *base,
            {"trade_date": "20250101", "ts_code": "0001.SZ", "factor_value": 9.0},
        ]),
    }
    with pytest.warns(UserWarning, match="重复"):
        merged, _ = _zscore_and_merge(dfs)
    assert merged.height == 2


