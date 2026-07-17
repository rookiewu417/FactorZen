import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import factorzen.risk.exposures as exposures_module


@pytest.fixture(autouse=True)
def _pit_industry_unavailable_by_default(monkeypatch):
    """run_risk_build 内部经 RiskModel.build() 循环调用 compute_exposures：默认不
    触达真实 Tushare，PIT 历史行业数据视为不可用，走 stocks.industry 降级路径。

    与 test_risk_model.py/test_risk_exposures.py 保持一致的隔离方式（同一份
    fixture在此重复一遍，因为项目当前没有共享 conftest.py）。没有这层隔离时，
    本文件 _mock() 用的合成代码（000000.SZ..000011.SZ）会与真实 Tushare 股票
    代码部分撞号，混入真实行业分类后，12 只股票的行业类别数接近甚至超过样本量，
    导致截面回归退化失败——这个失败只在本地有真实 Tushare 缓存/token 时才会
    触发，干净 CI 环境不会复现，但仍是需要显式隔离的真实测试污染风险。
    """
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)
    yield


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


def test_run_risk_build_manifest_has_reproducibility_fields(tmp_path: Path):
    """manifest.json 应含 command/git_dirty/pixi_lock_sha256/schema_version（复用 core.experiment 的
    build_manifest_base，而非各自手写精简版 manifest）。"""
    from factorzen.pipelines.risk_build import run_risk_build
    daily, db, stocks, start, end = _mock()
    res = run_risk_build(daily, db, stocks, start, end, out_dir=str(tmp_path), run_id="repro1")
    manifest = json.loads((Path(res["run_dir"]) / "manifest.json").read_text())

    assert manifest["schema_version"] == "1"
    assert isinstance(manifest["git_dirty"], bool)
    assert isinstance(manifest["pixi_lock_sha256"], str) and manifest["pixi_lock_sha256"]
    assert isinstance(manifest["command"], list) and manifest["command"]
    assert manifest.get("git_sha")
    # 原有字段不应回归丢失
    assert manifest["run_id"] == "repro1"
    assert "duration_seconds" in manifest


def test_run_risk_build_manifest_command_override(tmp_path: Path):
    """显式传 command 时应原样记录，供复现当时具体怎么跑的。"""
    from factorzen.pipelines.risk_build import run_risk_build
    daily, db, stocks, start, end = _mock()
    res = run_risk_build(daily, db, stocks, start, end, out_dir=str(tmp_path), run_id="repro2",
                         command=["fz", "risk", "build", "--start", start, "--end", end])
    manifest = json.loads((Path(res["run_dir"]) / "manifest.json").read_text())
    assert manifest["command"] == ["fz", "risk", "build", "--start", start, "--end", end]


def _ymd_to_date(s: str) -> dt.date:
    return dt.datetime.strptime(s.replace("-", ""), "%Y%m%d").date()


def test_risk_lookback_start_covers_max_rolling_window():
    """lookback 起始日须往前推足够覆盖最长滚动风格因子窗（momentum rolling_sum(252)、
    growth shift(252)）。252 交易日在 A 股约需 380 日历日，默认应有余量。"""
    from factorzen.pipelines.risk_build import risk_lookback_start

    lb = risk_lookback_start("20240101")
    assert lb < "20240101"
    gap_days = (dt.date(2024, 1, 1) - _ymd_to_date(lb)).days
    assert gap_days >= 380, f"lookback 应≥380 日历日以覆盖 252 交易日，实际 {gap_days}"
    # 支持 YYYY-MM-DD 输入
    assert risk_lookback_start("2024-01-01") == lb


def test_load_risk_inputs_fetches_lookback_so_build_keeps_all_style_factors():
    """生产回归测试：CLI 若只 fetch [start,end]，窗口首日滚动风格因子全空、
    build 退化为 4 风格因子并跳过大量交易日。load_risk_inputs 须补 lookback 历史，
    使 8 个风格因子在窗口首日即齐全、无交易日被丢弃。"""
    from factorzen.pipelines.risk_build import load_risk_inputs
    from factorzen.risk import RiskModel

    daily_all, db_all, stocks, start, end = _mock(n_days=290)  # start=days[260]，前有 260 天历史
    codes = stocks["ts_code"].to_list()

    class _SliceLoader:
        """模拟真实 loader：fetch_daily(s,e) 只返回 [s,e] 切片。"""

        def fetch_daily(self, s: str, e: str) -> pl.DataFrame:
            return daily_all.filter(
                (pl.col("trade_date") >= _ymd_to_date(s)) & (pl.col("trade_date") <= _ymd_to_date(e))
            )

        def fetch_daily_basic(self, s: str, e: str) -> pl.DataFrame:
            return db_all.filter(
                (pl.col("trade_date") >= _ymd_to_date(s)) & (pl.col("trade_date") <= _ymd_to_date(e))
            )

    daily, daily_basic = load_risk_inputs(_SliceLoader(), start, end, codes)

    # 补了 lookback：窗口首日之前应有 ≥252 交易日历史用于滚动因子预热
    hist_days = daily.filter(pl.col("trade_date") < _ymd_to_date(start))["trade_date"].n_unique()
    assert hist_days >= 252, f"应补≥252 交易日 lookback，实际 {hist_days}"

    res = RiskModel().build(daily, daily_basic, stocks, start, end)
    names = set(res.factor_names)
    # momentum/volatility/growth = 252/60/252 长窗滚动因子，无 lookback 时窗口内全空
    rolling = {"momentum", "volatility", "growth"}
    assert rolling <= names, f"补 lookback 后长窗滚动风格因子应齐全，实际缺 {rolling - names}"
    assert res.n_dropped_dates == 0, "窗口内因子集应稳定，不应有交易日被跳过"


def test_build_no_lookback_still_runs_with_union_styles():
    """无 lookback 时：W1 一次物化后滚动因子在窗口后段出现，W2 并集固定列集不再因
    因子名漂移丢日。factor_names 会含后期出现的滚动因子；早期日对应列填 0。
    正确的 lookback 仍由 load_risk_inputs 保障（见上一测）。"""
    from factorzen.risk import RiskModel

    daily_all, db_all, stocks, _, _ = _mock(n_days=290)
    all_days = daily_all.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    start, end = all_days[0].strftime("%Y%m%d"), all_days[-1].strftime("%Y%m%d")

    res = RiskModel().build(daily_all, db_all, stocks, start, end)
    # W2 后不应因风格列中途出现而 mismatch 丢日
    assert res.n_factor_mismatch == 0
    assert res.n_valid_dates > 0
    names = set(res.factor_names)
    # 窗口够长时后段滚动因子会进入并集
    assert "size" in names and "value" in names
