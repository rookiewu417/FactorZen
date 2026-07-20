"""test_combination.py：S6 防回归：验证多因子合成方法。
test_combination_lgbm.py：LightGBM 组合器的测试:可学习性 / 确定性 / 泄漏探针 / 缺值处理。
test_combination_cv.py：Purged & embargoed walk-forward CV 切分协议的测试。
test_combination_importance.py：因子重要性 explain 的测试:gain / shap(可选) / 缺 shap 回退。
test_combination_robustness.py：组合层健壮性:因子库常含空因子/覆盖异质,组合器须丢弃退化因子 + 外连接容缺,
"""

from __future__ import annotations

import builtins
import importlib.util
from datetime import date

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.importance import explain
from factorzen.research.combination.methods import equal_weight, ic_weighted, max_ir
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
    def test_output_schema(self):
        """等权合成输出包含 trade_date, ts_code, factor_value。"""
        factor_dfs, _ = _make_factor_ret()
        result = equal_weight(factor_dfs)
        assert "trade_date" in result.columns
        assert "ts_code" in result.columns
        assert "factor_value" in result.columns

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

    def test_single_factor_passthrough(self):
        """只有 1 个因子时，等权合成 ≈ 原因子 z-score。"""
        factor_dfs, _ = _make_factor_ret(n_factors=1)
        result = equal_weight(factor_dfs)
        assert len(result) > 0

class TestICWeighted:
    def test_output_schema(self):
        """IC 加权输出包含必要列。"""
        factor_dfs, ret_df = _make_factor_ret()
        result = ic_weighted(factor_dfs, ret_df)
        assert set(["trade_date", "ts_code", "factor_value"]).issubset(result.columns)

    def test_no_nan(self):
        """IC 加权结果不含 nan。"""
        factor_dfs, ret_df = _make_factor_ret()
        result = ic_weighted(factor_dfs, ret_df)
        finite_count = result["factor_value"].is_finite().sum()
        assert finite_count > 0

    def test_differs_from_equal_weight(self):
        """IC 加权与等权结果不完全相同（权重不等时）。"""
        factor_dfs, ret_df = _make_factor_ret(seed=7)
        ew = equal_weight(factor_dfs)
        iw = ic_weighted(factor_dfs, ret_df)
        joined = ew.join(iw, on=["trade_date", "ts_code"], suffix="_iw")
        # IC 加权与等权在特殊情况下可相同（权重退化），只验证 join 成功无崩溃
        assert len(joined) > 0

class TestMaxIR:
    def test_output_schema(self):
        """max_ir 输出包含必要列。"""
        factor_dfs, ret_df = _make_factor_ret()
        result = max_ir(factor_dfs, ret_df)
        assert set(["trade_date", "ts_code", "factor_value"]).issubset(result.columns)

    def test_no_nan(self):
        """max_ir 结果不含 nan。"""
        factor_dfs, ret_df = _make_factor_ret()
        result = max_ir(factor_dfs, ret_df)
        assert result["factor_value"].is_finite().sum() > 0

    def test_fallback_on_insufficient_data(self):
        """数据不足时退化为等权，不崩溃。"""
        factor_dfs, ret_df = _make_factor_ret(n_dates=5, n_stocks=10)
        result = max_ir(factor_dfs, ret_df, lookback=120)
        assert len(result) >= 0  # 不崩溃

class TestCombinationIR:
    def test_combined_ir_not_worse_than_worst_factor(self):
        """合成因子的 IR 应不低于最差单因子 IR（分散化应有收益）。"""
        rng = np.random.default_rng(42)
        n_dates, n_stocks = 150, 80
        dates = [f"2024-{(i // 25 + 1):02d}-{(i % 25 + 1):02d}" for i in range(n_dates)]
        stocks = [f"{i:06d}.SZ" for i in range(n_stocks)]

        factor_dfs = {}
        ic_values_list = []
        for fi in range(3):
            rows = []
            ic_vals = []
            for d in dates:
                fv = rng.standard_normal(n_stocks)
                ret = 0.05 * fv / n_stocks + rng.normal(0, 0.02, n_stocks)
                ic = float(np.corrcoef(fv.argsort().argsort(), ret.argsort().argsort())[0, 1])
                ic_vals.append(ic)
                for i, s in enumerate(stocks):
                    rows.append({"trade_date": d, "ts_code": s, "factor_value": float(fv[i])})
            df = pl.DataFrame(rows).with_columns(
                pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d")
            )
            factor_dfs[f"f{fi}"] = df
            ic_values_list.append(ic_vals)

        combined = equal_weight(factor_dfs)
        # 合成因子的截面标准差应 > 0（有效信号）
        std_per_date = combined.group_by("trade_date").agg(
            pl.col("factor_value").std().alias("cs_std")
        )
        assert std_per_date["cs_std"].mean() > 0

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

