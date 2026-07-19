"""四方法 OOS 对比实验驱动的测试。"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.experiment import run_combination_experiment


def _panel(n_days=150, n_stocks=40, seed=0):
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


def test_run_experiment_produces_artifacts(tmp_path):
    factor_dfs, ret_df, _ = _panel()
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


def test_run_experiment_default_methods(tmp_path):
    factor_dfs, ret_df, _ = _panel(n_days=140, n_stocks=30)
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


# ── 换手：组合层部署判据（IC 高不代表实盘优，报告早就承认该看换手）────────────


def test_evaluate_oos_reports_turnover():
    """组合评估必须给出换手率，否则无法据 IC 下部署结论。

    因子是**截面打分**不是权重，故换手定义为「相邻期 top 分位成分的变动率」
    ——直接对应「按打分选股、每期换仓」要付的交易成本。
    """
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


def test_evaluate_oos_turnover_zero_when_ranking_static():
    """打分次序完全不变 → 换手 0（判别力：与上一条构成对照）。"""
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


def test_evaluate_oos_turnover_empty_is_zero():
    """无有效日 → turnover 0（形态一致，下游 no KeyError）。"""
    from factorzen.research.combination.experiment import _evaluate_oos

    empty = pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8,
                                 "factor_value": pl.Float64})
    ret = pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8,
                               "ret": pl.Float64})
    m = _evaluate_oos(empty, ret)
    assert m["turnover"] == 0.0
