from types import SimpleNamespace

import polars as pl

from factorzen.llm.snapshot import build_factor_snapshot


def test_snapshot_keeps_only_compact_metrics():
    ic_result = SimpleNamespace(
        factor_name="momentum_20d",
        ic_mean=0.0234,
        ic_std=0.071,
        ir=0.33,
        ic_positive_ratio=0.58,
        n_periods=120,
        ic_tstat=1.82,
        ic_pvalue=0.071,
        decay={1: 0.0234, 5: 0.018, 10: 0.011, 20: 0.004},
        multi_period={5: {"ic_mean": 0.018, "ir": 0.22}},
        oos_ic={"train": 0.026, "test": 0.012},
        ic_series=pl.DataFrame({"trade_date": ["2025-01-01"], "ic": [0.1]}),
    )
    bt_result = SimpleNamespace(
        strategy_name="quantile_long_short",
        summary_stats={
            "long_short": {"ann_ret": 0.082, "sharpe": 0.91, "max_dd": -0.12}
        },
    )
    to_result = SimpleNamespace(avg_turnover=0.42)

    snapshot = build_factor_snapshot(
        factor_name="momentum_20d",
        factor_description="20日价格动量",
        frequency="daily",
        date_range="2025-01-01 ~ 2026-05-13",
        universe="csi300",
        ic_result=ic_result,
        bt_result=bt_result,
        to_result=to_result,
        walk_forward_summary={"status": "ok", "n_folds": 3, "oos_sharpe_mean": 0.31},
        quality_report={"warnings": ["coverage below target"]},
        backtest_direction={"direction": "normal", "reason": "IC非负，保持原方向"},
    )

    assert snapshot["factor"]["name"] == "momentum_20d"
    assert snapshot["ic"]["mean"] == 0.0234
    assert snapshot["backtest"]["ls_ann_ret"] == 0.082
    assert snapshot["quality"]["warnings"] == ["coverage below target"]
    assert "ic_series" not in str(snapshot)
    assert "positions" not in str(snapshot)
    assert "trades" not in str(snapshot)
