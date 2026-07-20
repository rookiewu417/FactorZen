"""test_combination_oos.py：估权/应用拆分 + 滚动 OOS 组合器(含泄漏探针)的测试。
test_combination_parity.py：组合管道数值 parity 硬约束：同 seed 同输入，comparison + combined 必须与 golden 一致。
test_combination_experiment.py：四方法 OOS 对比实验驱动的测试。
test_signed_combination_weights.py：P1-①：组合层允许负权——让权重自己处理符号。
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.experiment import run_combination_experiment
from factorzen.research.combination.methods import (
    apply_weights,
    equal_weight,
    estimate_ic_weights,
    ic_weighted,
)
from factorzen.research.combination.oos import combine_oos


# ==== 来自 test_combination_oos.py ====
def _panel__oos(n_days=120, n_stocks=30, seed=0):
    """两因子面板:a 与 ret 强相关(IC 高),b 弱相关。"""
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.6 * fa + 0.1 * fb + rng.standard_normal(n_stocks) * 0.5
        for s in range(n_stocks):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    return {"a": pl.DataFrame(ra), "b": pl.DataFrame(rb)}, pl.DataFrame(rr), dates

def _single_day(vals_a, vals_b):
    codes = [str(i) for i in range(len(vals_a))]
    dfa = pl.DataFrame({"trade_date": ["d"] * len(codes), "ts_code": codes, "factor_value": vals_a})
    dfb = pl.DataFrame({"trade_date": ["d"] * len(codes), "ts_code": codes, "factor_value": vals_b})
    return dfa, dfb

# ── 拆分:apply / estimate ───────────────────────────────
def test_apply_and_ic_weights_suite():
    """test_apply_weights_ground_truth；test_equal_weight_is_mean；test_estimate_ic_weights_favors_high_ic；test_estimate_ic_weights_all_negative_degenerates_equal；薄包装等价:ic_weighted == apply_weights(estimate_ic_weights)。"""
    # -- 原 test_apply_weights_ground_truth --
    def _section_0_test_apply_weights_ground_truth():
        dfa, dfb = _single_day([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        out = apply_weights({"a": dfa, "b": dfb}, {"a": 0.7, "b": 0.3}).sort("ts_code")
        # z_a=[-1,0,1] z_b=[1,0,-1] → 0.7*z_a+0.3*z_b = [-0.4,0,0.4]
        assert out["factor_value"].to_list() == pytest.approx([-0.4, 0.0, 0.4])

    _section_0_test_apply_weights_ground_truth()

    # -- 原 test_equal_weight_is_mean --
    def _section_1_test_equal_weight_is_mean():
        dfa, dfb = _single_day([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        out = equal_weight({"a": dfa, "b": dfb}).sort("ts_code")
        assert out["factor_value"].to_list() == pytest.approx([0.0, 0.0, 0.0])

    _section_1_test_equal_weight_is_mean()

    # -- 原 test_estimate_ic_weights_favors_high_ic --
    def _section_2_test_estimate_ic_weights_favors_high_ic():
        factor_dfs, ret_df, _ = _panel__oos()
        w = estimate_ic_weights(factor_dfs, ret_df, ic_window=-1)
        assert w["a"] > w["b"]
        assert sum(w.values()) == pytest.approx(1.0)

    _section_2_test_estimate_ic_weights_favors_high_ic()

    # -- 原 test_estimate_ic_weights_all_negative_degenerates_equal --
    def _section_3_test_estimate_ic_weights_all_negative_degenerates_equal():
        _, ret_df, _ = _panel__oos()
        # 因子 = -ret → IC=-1 全负 → max(0,ic)=0 → 退化等权
        neg = ret_df.rename({"ret": "factor_value"}).with_columns(
            (pl.col("factor_value") * -1).alias("factor_value")
        )
        w = estimate_ic_weights({"a": neg, "b": neg}, ret_df, ic_window=-1)
        assert w == pytest.approx({"a": 0.5, "b": 0.5})

    _section_3_test_estimate_ic_weights_all_negative_degenerates_equal()

    # -- 原 test_ic_weighted_equals_estimate_then_apply --
    def _section_4_test_ic_weighted_equals_estimate_then_apply():
        factor_dfs, ret_df, _ = _panel__oos()
        direct = ic_weighted(factor_dfs, ret_df, ic_window=60).sort(["trade_date", "ts_code"])
        w = estimate_ic_weights(factor_dfs, ret_df, ic_window=60)
        via = apply_weights(factor_dfs, w).sort(["trade_date", "ts_code"])
        assert_frame_equal(direct, via)

    _section_4_test_ic_weighted_equals_estimate_then_apply()


# ── OOS 组合器 ──────────────────────────────────────────
def test_combine_oos_suite():
    """test_combine_oos_covers_test_and_has_fold_id；泄漏探针:扰动 cutoff 之后的收益,cutoff 之前的 OOS 组合值必须逐行不变。；等权同样过泄漏探针(天然 OOS,但走同一接口)。；test_combine_oos_unknown_method_raises"""
    # -- 原 test_combine_oos_covers_test_and_has_fold_id --
    def _section_0_test_combine_oos_covers_test_and_has_fold_id():
        factor_dfs, ret_df, dates = _panel__oos(n_days=120)
        cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
        out = combine_oos(factor_dfs, ret_df, cv, method="ic_weighted")
        assert set(out.columns) >= {"trade_date", "ts_code", "factor_value", "fold_id"}
        assert set(out["trade_date"].to_list()) == set(dates[40:120])

    _section_0_test_combine_oos_covers_test_and_has_fold_id()

    # -- 原 test_combine_oos_no_lookahead --
    def _section_1_test_combine_oos_no_lookahead():
        factor_dfs, ret_df, dates = _panel__oos(n_days=120)
        cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
        base = combine_oos(factor_dfs, ret_df, cv, method="ic_weighted")
        cutoff = dates[79]
        tampered_ret = ret_df.with_columns(
            pl.when(pl.col("trade_date") > cutoff)
            .then(pl.col("ret") * -3.0)
            .otherwise(pl.col("ret"))
            .alias("ret")
        )
        tampered = combine_oos(factor_dfs, tampered_ret, cv, method="ic_weighted")
        b = base.filter(pl.col("trade_date") <= cutoff).sort(["trade_date", "ts_code"])
        t = tampered.filter(pl.col("trade_date") <= cutoff).sort(["trade_date", "ts_code"])
        assert_frame_equal(b, t)

    _section_1_test_combine_oos_no_lookahead()

    # -- 原 test_combine_oos_equal_weight_also_probes_clean --
    def _section_2_test_combine_oos_equal_weight_also_probes_clean():
        factor_dfs, ret_df, dates = _panel__oos(n_days=120)
        cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
        out = combine_oos(factor_dfs, ret_df, cv, method="equal_weight")
        assert set(out["trade_date"].to_list()) == set(dates[40:120])

    _section_2_test_combine_oos_equal_weight_also_probes_clean()

    # -- 原 test_combine_oos_unknown_method_raises --
    def _section_3_test_combine_oos_unknown_method_raises():
        factor_dfs, ret_df, _ = _panel__oos(n_days=80)
        cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
        with pytest.raises(ValueError):
            combine_oos(factor_dfs, ret_df, cv, method="nonsense")

    _section_3_test_combine_oos_unknown_method_raises()


# ==== 来自 test_combination_parity.py ====
_N_DAYS = 180
_N_STOCKS = 50
_N_FACTORS = 4
_DATA_SEED = 42
_MODEL_SEED = 7
_CV = dict(train_days=60, test_days=10, purge_days=5, embargo_days=0, expanding=True)
_METHODS = ["equal_weight", "ic_weighted", "max_ir", "lgbm"]

# ── golden：优化前用同一参数跑 run_combination_experiment 落盘（2026-07-13）──
# comparison.csv 逐方法指标；combined_* 用 sum/mean/std + 排序后前 5 值做指纹。
_GOLDEN_COMPARISON = {
    "equal_weight": {
        "rank_ic_mean": 0.22705722288915567,
        "icir": 1.9549690788119853,
        "ic_positive_ratio": 0.975,
        "top_bottom_spread": 0.5416715606200107,
        "max_drawdown": -0.5932966429552202,
        "n_periods": 120,
    },
    "ic_weighted": {
        "rank_ic_mean": 0.7868787515006003,
        "icir": 12.781262665891607,
        "ic_positive_ratio": 1.0,
        "top_bottom_spread": 1.9144851919070378,
        "max_drawdown": 0.0,
        "n_periods": 120,
    },
    "max_ir": {
        "rank_ic_mean": 0.7877326930772308,
        "icir": 12.924104188520449,
        "ic_positive_ratio": 1.0,
        "top_bottom_spread": 1.9122830758231355,
        "max_drawdown": 0.0,
        "n_periods": 120,
    },
    "lgbm": {
        "rank_ic_mean": 0.8544945978391356,
        "icir": 22.973456950349078,
        "ic_positive_ratio": 1.0,
        "top_bottom_spread": 2.0666241033701023,
        "max_drawdown": 0.0,
        "n_periods": 120,
    },
}

_GOLDEN_COMBINED_FINGERPRINT = {
    "equal_weight": {
        "n": 6000,
        "sum": -6.661338147750939e-16,
        "mean": -1.1102230246251566e-19,
        "std": 0.4915416959585447,
        "first5": [
            -0.0051138671387954415,
            -0.0034369983566984695,
            0.06133959083482021,
            -0.8345838684886397,
            0.031493171830069645,
        ],
    },
    "ic_weighted": {
        "n": 6000,
        "sum": -2.042810365310288e-14,
        "mean": -3.4046839421838132e-18,
        "std": 0.9808905812215484,
        "first5": [
            1.2240512411242044,
            0.6377352724524825,
            1.9730751614278443,
            -1.6972954962036835,
            -0.3577384581142817,
        ],
    },
    "max_ir": {
        "n": 6000,
        "sum": -2.842170943040401e-14,
        "mean": -4.736951571734001e-18,
        "std": 0.9779522201855253,
        "first5": [
            1.196732960925405,
            0.6285331652515013,
            1.9323814560345678,
            -1.6834530310370717,
            -0.3561577747105275,
        ],
    },
    "lgbm": {
        "n": 6000,
        "sum": 0.583085155229595,
        "mean": 9.71808592049325e-05,
        "std": 0.2553867479215032,
        "first5": [
            0.37599384330020985,
            0.2179679552980397,
            0.502547313084538,
            -0.2545213683387086,
            -0.08949804390612773,
        ],
    },
}

_TOL = 1e-10

def _make_panel(
    n_days: int = _N_DAYS,
    n_stocks: int = _N_STOCKS,
    n_factors: int = _N_FACTORS,
    seed: int = _DATA_SEED,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = [f"D{i:04d}" for i in range(n_days)]
    factor_dfs: dict[str, pl.DataFrame] = {}
    for fi in range(n_factors):
        rows = []
        for d in dates:
            vals = rng.standard_normal(n_stocks)
            for s in range(n_stocks):
                rows.append(
                    {
                        "trade_date": d,
                        "ts_code": f"{s:04d}.SZ",
                        "factor_value": float(vals[s]),
                    }
                )
        factor_dfs[f"f{fi}"] = pl.DataFrame(rows)
    joined = (
        factor_dfs["f0"]
        .rename({"factor_value": "a"})
        .join(
            factor_dfs["f1"].rename({"factor_value": "b"}),
            on=["trade_date", "ts_code"],
        )
    )
    noise = rng.standard_normal(joined.height) * 0.4
    ret = 0.7 * joined["a"].to_numpy() - 0.3 * joined["b"].to_numpy() + noise
    ret_df = joined.select(["trade_date", "ts_code"]).with_columns(pl.Series("ret", ret))
    return factor_dfs, ret_df

def _run(tmp_path, run_id: str = "parity"):
    factor_dfs, ret_df = _make_panel()
    cv = PurgedWalkForwardCV(**_CV)
    return run_combination_experiment(
        factor_dfs,
        ret_df,
        cv=cv,
        methods=list(_METHODS),
        seed=_MODEL_SEED,
        out_dir=str(tmp_path),
        run_id=run_id,
    )

def test_combination_parity_and_experiment_suite(tmp_path):
    """comparison.csv 每个方法每个指标与 golden 在 1e-10 内一致。；combined_*.parquet 排序后的 factor_value 指纹与 golden 一致（数值语义未改）。；test_run_experiment_produces_artifacts；test_run_experiment_default_methods"""
    # -- 原 test_combination_parity_comparison_csv --
    def _section_0_test_combination_parity_comparison_csv(tmp_path):
        res = _run(tmp_path)
        comp = pl.read_csv(res["run_dir"] + "/comparison.csv")
        by_method = {r["method"]: r for r in comp.iter_rows(named=True)}
        assert set(by_method) == set(_GOLDEN_COMPARISON)
        for method, gold in _GOLDEN_COMPARISON.items():
            row = by_method[method]
            for key, expected in gold.items():
                got = row[key]
                if isinstance(expected, float):
                    assert got == pytest.approx(expected, abs=_TOL, rel=0.0), (
                        f"{method}.{key}: got={got!r} expected={expected!r}"
                    )
                else:
                    assert got == expected, f"{method}.{key}: got={got!r} expected={expected!r}"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_combination_parity_comparison_csv(_tp0)

    # -- 原 test_combination_parity_combined_values --
    def _section_1_test_combination_parity_combined_values(tmp_path):
        res = _run(tmp_path)
        for method, gold in _GOLDEN_COMBINED_FINGERPRINT.items():
            cdf = pl.read_parquet(f"{res['run_dir']}/combined_{method}.parquet").sort(
                ["trade_date", "ts_code", "fold_id"]
            )
            arr = cdf["factor_value"].to_numpy().astype(float)
            assert len(arr) == gold["n"]
            assert float(np.nansum(arr)) == pytest.approx(gold["sum"], abs=_TOL, rel=0.0)
            assert float(np.nanmean(arr)) == pytest.approx(gold["mean"], abs=_TOL, rel=0.0)
            assert float(np.nanstd(arr)) == pytest.approx(gold["std"], abs=_TOL, rel=0.0)
            for i, (g, e) in enumerate(zip(arr[:5], gold["first5"], strict=True)):
                assert float(g) == pytest.approx(e, abs=_TOL, rel=0.0), (
                    f"{method}.first5[{i}]: got={g!r} expected={e!r}"
                )

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_combination_parity_combined_values(_tp1)

    # -- 原 test_run_experiment_produces_artifacts --
    def _section_2_test_run_experiment_produces_artifacts(tmp_path):
        factor_dfs, ret_df, _ = _panel__experiment()
        cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
        res = run_combination_experiment(
            factor_dfs,
            ret_df,
            cv=cv,
            methods=["equal_weight", "ic_weighted", "lgbm"],
            seed=0,
            out_dir=str(tmp_path),
            run_id="exp1",
        )
        run_dir = tmp_path / "exp1"
        assert res["run_dir"] == str(run_dir)
        for fname in ("comparison.csv", "manifest.json", "report.md", "importance.csv"):
            assert (run_dir / fname).exists(), f"缺 {fname}"

        comp = pl.read_csv(run_dir / "comparison.csv")
        assert set(comp["method"].to_list()) == {"equal_weight", "ic_weighted", "lgbm"}
        assert {"rank_ic_mean", "icir", "top_bottom_spread", "max_drawdown"} <= set(comp.columns)
        # 强信号:各方法 OOS RankIC 为正
        assert (comp["rank_ic_mean"] > 0).all()

        mani = json.loads((run_dir / "manifest.json").read_text())
        assert mani["seed"] == 0
        assert mani["cv"]["train_days"] == 60
        assert mani["cv"]["purge_days"] == 5
        assert set(mani["factors"]) == {"fa", "fb"}

        report = (run_dir / "report.md").read_text()
        assert "equal_weight" in report and "lgbm" in report

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_run_experiment_produces_artifacts(_tp2)

    # -- 原 test_run_experiment_default_methods --
    def _section_3_test_run_experiment_default_methods(tmp_path):
        factor_dfs, ret_df, _ = _panel__experiment(n_days=140, n_stocks=30)
        cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
        res = run_combination_experiment(
            factor_dfs, ret_df, cv=cv, out_dir=str(tmp_path), run_id="exp2"
        )
        comp = pl.read_csv(res["run_dir"] + "/comparison.csv")
        # 默认四方法。*_signed（允许负权）可选但**不进默认**——真实库 OOS 实测更差，
        # 见 methods._solve_max_ir_weights 的对照表。
        assert set(comp["method"].to_list()) == {
            "equal_weight",
            "ic_weighted",
            "max_ir",
            "lgbm",
        }

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_run_experiment_default_methods(_tp3)


# ==== 来自 test_combination_experiment.py ====
def _panel__experiment(n_days=150, n_stocks=40, seed=0):
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


# ── 换手：组合层部署判据（IC 高不代表实盘优，报告早就承认该看换手）────────────

def test_evaluate_oos_turnover_suite():
    """组合评估必须给出换手率，否则无法据 IC 下部署结论。；打分次序完全不变 → 换手 0（判别力：与上一条构成对照）。；无有效日 → turnover 0（形态一致，下游 no KeyError）。"""
    # -- 原 test_evaluate_oos_reports_turnover --
    def _section_0_test_evaluate_oos_reports_turnover():
        from factorzen.research.combination.experiment import _evaluate_oos

        dates = [f"2024010{i}" for i in range(1, 6)]
        codes = [f"{i:06d}.SZ" for i in range(10)]
        # 每天把打分整体反转 → top 分位成分每期全换 → 换手应≈1.0
        rows = []
        for di, d in enumerate(dates):
            for ci, c in enumerate(codes):
                v = float(ci) if di % 2 == 0 else float(-ci)
                rows.append({"trade_date": d, "ts_code": c, "factor_value": v})
        combined = pl.DataFrame(rows)
        ret = pl.DataFrame([
            {"trade_date": d, "ts_code": c, "ret": 0.001 * ci}
            for d in dates for ci, c in enumerate(codes)
        ])

        m = _evaluate_oos(combined, ret, n_groups=5)
        assert "turnover" in m, f"缺 turnover 键：{sorted(m)}"
        assert m["turnover"] > 0.9, m["turnover"]

    _section_0_test_evaluate_oos_reports_turnover()

    # -- 原 test_evaluate_oos_turnover_zero_when_ranking_static --
    def _section_1_test_evaluate_oos_turnover_zero_when_ranking_static():
        from factorzen.research.combination.experiment import _evaluate_oos

        dates = [f"2024010{i}" for i in range(1, 6)]
        codes = [f"{i:06d}.SZ" for i in range(10)]
        combined = pl.DataFrame([
            {"trade_date": d, "ts_code": c, "factor_value": float(ci)}
            for d in dates for ci, c in enumerate(codes)
        ])
        ret = pl.DataFrame([
            {"trade_date": d, "ts_code": c, "ret": 0.001 * ci}
            for d in dates for ci, c in enumerate(codes)
        ])

        m = _evaluate_oos(combined, ret, n_groups=5)
        assert m["turnover"] == 0.0, m["turnover"]

    _section_1_test_evaluate_oos_turnover_zero_when_ranking_static()

    # -- 原 test_evaluate_oos_turnover_empty_is_zero --
    def _section_2_test_evaluate_oos_turnover_empty_is_zero():
        from factorzen.research.combination.experiment import _evaluate_oos

        empty = pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8,
                                     "factor_value": pl.Float64})
        ret = pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8,
                                   "ret": pl.Float64})
        m = _evaluate_oos(empty, ret)
        assert m["turnover"] == 0.0

    _section_2_test_evaluate_oos_turnover_empty_is_zero()


# ==== 来自 test_signed_combination_weights.py ====
_DATES = [f"2024-{m:02d}-{d:02d}" for m in (1, 2, 3) for d in range(1, 21)]
_CODES = [f"{i:06d}.SZ" for i in range(60)]

def _mk(values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": np.repeat(_DATES, len(_CODES)),
        "ts_code": np.tile(_CODES, len(_DATES)),
        "factor_value": values.reshape(-1).astype(float),
    })

def _ret(values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": np.repeat(_DATES, len(_CODES)),
        "ts_code": np.tile(_CODES, len(_DATES)),
        "ret": values.reshape(-1).astype(float),
    })

def _scenario(seed: int = 0):
    """构造 IC 符号**已知**的三因子场景（ground truth 由构造给定，非由被测实现给出）：

    - ``pos``：与收益同向 → IC 明显为正
    - ``neg``：与收益反向 → IC 明显为负
    - ``noise``：纯噪声 → IC ≈ 0
    """
    rng = np.random.default_rng(seed)
    shape = (len(_DATES), len(_CODES))
    signal = rng.standard_normal(shape)
    noise = rng.standard_normal(shape)
    ret = signal + 0.5 * rng.standard_normal(shape)
    return (
        {
            "pos": _mk(signal + 0.3 * rng.standard_normal(shape)),
            "neg": _mk(-signal + 0.3 * rng.standard_normal(shape)),
            "noise": _mk(noise),
        },
        _ret(ret),
    )

# ── 1. 现状锚：三个方法都给不出负权（本轮改动的前提）────────────────────────

def test_signed_weighting_suite():
    """默认口径下 IC 加权 / max_IR **都不会**给负权——这正是「换方法」解决不了问题的原因。；ground truth 由**构造**给定：pos 正权、neg 负权。；负权下必须用 L1 归一化：Σ|w| = 1。；正负近乎抵消（Σw≈0）时不得爆炸——这正是 Σw 归一化会炸的场景。；max_IR 闭式解 w ∝ Σ⁻¹μ 的**手算**对拍：构造让最优权重必为负的 μ/Σ。；机制测试：IC 符号**稳定且估准**时，允许负权的合成因子 IC 更高。；新增参数必须带默认值且默认行为逐位不变（A股基线零回归底线）。；退化守卫：全零 IC 时（Σ|w|≈0）退化等权，不得除零。；`ic_weighted_signed` / `max_ir_signed` 必须能进 OOS 协议的方法分派。；异常契约不变：未知方法名仍抛 ValueError（解析外部输入只抛一类）。"""
    # -- 原 test_default_estimators_are_long_only --
    def _section_0_test_default_estimators_are_long_only():
        from factorzen.research.combination.methods import (
            estimate_ic_weights,
            estimate_max_ir_weights,
        )

        dfs, ret = _scenario()
        w_ic = estimate_ic_weights(dfs, ret)
        assert all(v >= 0.0 for v in w_ic.values()), "默认 IC 加权不应给负权（现语义）"
        assert w_ic["neg"] == pytest.approx(0.0, abs=1e-12), "负 IC 因子应被裁到 0"

        w_ir = estimate_max_ir_weights(dfs, ret)
        assert w_ir is not None
        assert all(v >= 0.0 for v in w_ir.values()), "默认 max_IR 不应给负权（现语义）"

    _section_0_test_default_estimators_are_long_only()

    # -- 原 test_ic_weights_signed_follows_known_ic_sign --
    def _section_1_test_ic_weights_signed_follows_known_ic_sign():
        from factorzen.research.combination.methods import estimate_ic_weights

        dfs, ret = _scenario()
        w = estimate_ic_weights(dfs, ret, allow_negative=True)
        assert w["pos"] > 0.0, f"同向因子应得正权，得 {w['pos']}"
        assert w["neg"] < 0.0, f"反向因子应得负权，得 {w['neg']}"
        assert abs(w["noise"]) < abs(w["pos"]), "噪声因子权重量级应远小于信号因子"

    _section_1_test_ic_weights_signed_follows_known_ic_sign()

    # -- 原 test_ic_weights_signed_l1_normalized --
    def _section_2_test_ic_weights_signed_l1_normalized():
        from factorzen.research.combination.methods import estimate_ic_weights

        dfs, ret = _scenario()
        w = estimate_ic_weights(dfs, ret, allow_negative=True)
        assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)

    _section_2_test_ic_weights_signed_l1_normalized()

    # -- 原 test_l1_normalization_survives_near_cancelling_weights --
    def _section_3_test_l1_normalization_survives_near_cancelling_weights():
        from factorzen.research.combination.methods import estimate_ic_weights

        rng = np.random.default_rng(5)
        shape = (len(_DATES), len(_CODES))
        signal = rng.standard_normal(shape)
        ret = signal + 0.5 * rng.standard_normal(shape)
        # 两条对称的反向因子 → mean IC 近似 +c 与 −c，Σw ≈ 0
        dfs = {"a": _mk(signal), "b": _mk(-signal)}
        w = estimate_ic_weights(dfs, _ret(ret), allow_negative=True)
        assert all(np.isfinite(v) for v in w.values())
        assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)
        assert max(abs(v) for v in w.values()) <= 1.0 + 1e-9, "权重被放大 = 归一化炸了"

    _section_3_test_l1_normalization_survives_near_cancelling_weights()

    # -- 原 test_max_ir_signed_closed_form_ground_truth --
    def _section_4_test_max_ir_signed_closed_form_ground_truth():
        from factorzen.research.combination.methods import _solve_max_ir_weights

        mu = np.array([0.05, 0.01])
        sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
        w = _solve_max_ir_weights(mu, sigma, allow_negative=True)
        # 手算纯闭式解：Σ⁻¹ = 1/(1-0.81) · [[1,-0.9],[-0.9,1]]，再 L1 归一化
        want = np.linalg.inv(sigma) @ mu
        want = want / np.abs(want).sum()
        assert want[1] < 0, "前提不成立：构造的场景应让第二个权重为负"
        # rtol 放到 1e-5：实现对 Σ 加了文档化的 1e-6 岭正则，与纯闭式解的偏差正是该量级。
        # 这里要测的是「是否还是那个闭式解」，不是「是否逐位复刻实现的正则项」。
        np.testing.assert_allclose(w, want, rtol=1e-5)
        assert w[1] < 0, "允许负权后第二个权重必须真为负（裁剪会让它变 0）"

    _section_4_test_max_ir_signed_closed_form_ground_truth()

    # -- 原 test_signed_weighting_beats_clipped_when_negative_ic_factor_present --
    def _section_5_test_signed_weighting_beats_clipped_when_negative_ic_factor_present():
        from factorzen.research.combination.methods import (
            _rank_ic_numpy,
            apply_weights,
            estimate_ic_weights,
        )

        dfs, ret = _scenario(seed=3)

        def _composite_ic(weights):
            comp = apply_weights(dfs, weights)
            j = comp.join(ret, on=["trade_date", "ts_code"], how="inner")
            ics = []
            for _, g in j.group_by("trade_date"):
                v = _rank_ic_numpy(
                    g["factor_value"].to_numpy(), g["ret"].to_numpy()
                )
                if v is not None:
                    ics.append(v)
            return float(np.mean(ics))

        ic_clipped = _composite_ic(estimate_ic_weights(dfs, ret))
        ic_signed = _composite_ic(estimate_ic_weights(dfs, ret, allow_negative=True))
        assert ic_signed > ic_clipped, (
            f"允许负权未能提升合成 IC：signed={ic_signed:.4f} vs clipped={ic_clipped:.4f}"
        )

    _section_5_test_signed_weighting_beats_clipped_when_negative_ic_factor_present()

    # -- 原 test_default_behavior_bitwise_unchanged --
    def _section_6_test_default_behavior_bitwise_unchanged():
        from factorzen.research.combination.methods import (
            estimate_ic_weights,
            estimate_max_ir_weights,
        )

        dfs, ret = _scenario(seed=7)
        w_ic = estimate_ic_weights(dfs, ret)
        w_ic2 = estimate_ic_weights(dfs, ret, allow_negative=False)
        assert w_ic == w_ic2
        assert sum(w_ic.values()) == pytest.approx(1.0, abs=1e-9), "默认仍是 Σw=1"

        w_ir = estimate_max_ir_weights(dfs, ret)
        w_ir2 = estimate_max_ir_weights(dfs, ret, allow_negative=False)
        assert w_ir == w_ir2

    _section_6_test_default_behavior_bitwise_unchanged()

    # -- 原 test_all_zero_ic_falls_back_to_equal_weights --
    def _section_7_test_all_zero_ic_falls_back_to_equal_weights():
        from factorzen.research.combination.methods import estimate_ic_weights

        shape = (len(_DATES), len(_CODES))
        const = np.ones(shape)
        dfs = {"a": _mk(const), "b": _mk(const)}
        rng = np.random.default_rng(1)
        w = estimate_ic_weights(dfs, _ret(rng.standard_normal(shape)), allow_negative=True)
        assert all(np.isfinite(v) for v in w.values())
        assert w["a"] == pytest.approx(0.5)
        assert w["b"] == pytest.approx(0.5)

    _section_7_test_all_zero_ic_falls_back_to_equal_weights()

    # -- 原 test_oos_dispatch_supports_signed_methods --
    def _section_8_test_oos_dispatch_supports_signed_methods():
        from factorzen.research.combination.oos import _estimate_fold

        dfs, ret = _scenario()
        for method in ("ic_weighted_signed", "max_ir_signed"):
            w = _estimate_fold(method, dfs, dfs, ret, {})
            assert set(w) == set(dfs), f"{method} 权重键不全"
            assert all(np.isfinite(v) for v in w.values()), f"{method} 产出非有限权重"
        # 有负 IC 因子在场时，signed 方法应真的给出负权（否则等于没接上）
        w = _estimate_fold("ic_weighted_signed", dfs, dfs, ret, {})
        assert w["neg"] < 0.0, "signed 方法没给出负权 = 分派接到了 clipped 实现"

    _section_8_test_oos_dispatch_supports_signed_methods()

    # -- 原 test_unknown_method_still_raises_valueerror --
    def _section_9_test_unknown_method_still_raises_valueerror():
        from factorzen.research.combination.oos import _estimate_fold

        dfs, ret = _scenario()
        with pytest.raises(ValueError, match="未知 method"):
            _estimate_fold("no_such_method", dfs, dfs, ret, {})

    _section_9_test_unknown_method_still_raises_valueerror()


# ── 2. 允许负权后：符号必须跟着 IC 符号走 ────────────────────────────────────


# ── 3. 判别力核心：允许负权确实能救回负 IC 因子 ──────────────────────────────


# ── 4. 零回归锚：默认行为逐位不变 ────────────────────────────────────────────


# ── 5. OOS 协议分派：新方法名接得上 ──────────────────────────────────────────


