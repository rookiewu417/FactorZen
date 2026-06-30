import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl


def _mock(n_stocks=12, n_days=290, seed=3):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    daily = pl.DataFrame([{"trade_date": dd, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
                          for c in codes for dd in days])
    db = pl.DataFrame([{"trade_date": dd, "ts_code": c,
                        "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                        "pb": float(abs(rng.standard_normal()) + 1.5),
                        "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15)}
                       for c in codes for dd in days])
    stocks = pl.DataFrame({"ts_code": codes,
                           "industry": [["银行", "医药", "电子"][i % 3] for i in range(n_stocks)]})
    return daily, db, stocks, days[260].strftime("%Y%m%d"), days[-1].strftime("%Y%m%d")


def test_run_risk_build_writes_artifacts(tmp_path: Path):
    from factorzen.pipelines.risk_build import run_risk_build
    daily, db, stocks, start, end = _mock()
    res = run_risk_build(daily, db, stocks, start, end, out_dir=str(tmp_path), run_id="t1")
    run_dir = Path(res["run_dir"])
    for f in ["exposures.parquet", "factor_covariance.parquet", "specific_risk.parquet",
              "factor_returns.parquet", "risk_summary.csv", "manifest.json"]:
        assert (run_dir / f).exists(), f
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert 0.0 <= manifest["r_squared"] <= 1.0
    assert "factor_names" in manifest
    # 产物可读非空（避免 build 返回空 result 时 false-pass）
    assert manifest["factor_names"], "factor_names 不应为空"
    exp_df = pl.read_parquet(run_dir / "exposures.parquet")
    assert exp_df.height > 0, "exposures.parquet 应有数据行"
    sr_df = pl.read_parquet(run_dir / "specific_risk.parquet")
    assert sr_df.height > 0, "specific_risk.parquet 应有数据行"

    # ── risk_summary.csv 内容验证（spec §7：5 类信息）──
    csv_df = pl.read_csv(run_dir / "risk_summary.csv")
    assert set(csv_df.columns) == {"section", "metric", "value"}, \
        f"CSV 列应为 section/metric/value，实际: {csv_df.columns}"
    sections = set(csv_df["section"].to_list())

    # §1 因子波动
    assert "factor_vol" in sections, "CSV 应含 factor_vol section"
    fvol_rows = csv_df.filter(pl.col("section") == "factor_vol")
    assert fvol_rows.height > 0, "factor_vol 应有行"
    assert (fvol_rows["value"] >= 0).all(), "factor_vol 值应非负"

    # §2 特质风险分布
    assert "specific_risk" in sections, "CSV 应含 specific_risk section"
    sr_metrics = set(csv_df.filter(pl.col("section") == "specific_risk")["metric"].to_list())
    for m in ("mean", "median", "p25", "p75", "max"):
        assert m in sr_metrics, f"specific_risk section 缺少 metric={m}"

    # §3 R²
    assert "r_squared" in sections, "CSV 应含 r_squared section"
    r2_val = csv_df.filter(
        (pl.col("section") == "r_squared") & (pl.col("metric") == "r_squared")
    )["value"].to_list()
    assert len(r2_val) == 1 and 0.0 <= r2_val[0] <= 1.0, f"r_squared 值应在 [0,1]，实际: {r2_val}"

    # §4 风格暴露
    assert "style_exposure" in sections, "CSV 应含 style_exposure section"
    se_metrics = csv_df.filter(pl.col("section") == "style_exposure")["metric"].to_list()
    assert any(m.endswith("_mean") for m in se_metrics), "style_exposure 应含 *_mean 指标"
    assert any(m.endswith("_std")  for m in se_metrics), "style_exposure 应含 *_std 指标"

    # §5 等权组合风险分解
    assert "decomp" in sections, "CSV 应含 decomp section"
    decomp_metrics = set(csv_df.filter(pl.col("section") == "decomp")["metric"].to_list())
    for m in ("total_risk", "factor_risk", "specific_risk", "factor_pct", "specific_pct"):
        assert m in decomp_metrics, f"decomp section 缺少 metric={m}"
    pcts = csv_df.filter(
        (pl.col("section") == "decomp") & pl.col("metric").is_in(["factor_pct", "specific_pct"])
    )["value"].to_list()
    assert abs(sum(pcts) - 1.0) < 1e-3, f"factor_pct + specific_pct 应约等于 1，实际: {pcts}"
