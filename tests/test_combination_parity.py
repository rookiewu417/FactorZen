"""组合管道数值 parity 硬约束：同 seed 同输入，comparison + combined 必须与 golden 一致。

本文件在性能优化前锁定基线；任何 research/combination 改动后必须仍绿。
容差 1e-10（浮点累加）；lgbm 在 deterministic + seed 下也应 bit-stable。
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.experiment import run_combination_experiment

# ── 固定合成数据参数（改这些会改 golden，勿动）──────────────────────────
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


def test_combination_parity_comparison_csv(tmp_path):
    """comparison.csv 每个方法每个指标与 golden 在 1e-10 内一致。"""
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


def test_combination_parity_combined_values(tmp_path):
    """combined_*.parquet 排序后的 factor_value 指纹与 golden 一致（数值语义未改）。"""
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
