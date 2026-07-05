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
    # 默认四方法
    assert set(comp["method"].to_list()) == {
        "equal_weight",
        "ic_weighted",
        "max_ir",
        "lgbm",
    }
