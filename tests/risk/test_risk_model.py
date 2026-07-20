"""合并自: test_risk_pipeline.py, test_risk_model_cov.py
目标: test_risk_model.py

--- 来源 test_risk_pipeline.py ---
test_risk_build_pipeline.py：run_risk_build 产物/manifest/lookback 与无 lookback 回归
test_risk_research_reuse.py：research 风格面板复用与单独 build 的 PIT 等价

--- 来源 test_risk_model_cov.py ---
test_risk_model.py：RiskModel 端到端 R²、残差对齐、predict/decompose 方差守恒
test_risk_covariance.py：因子协方差/特质风险/特征向量调整的数学性质
test_risk_industry.py：行业哑元 one-hot 与行业裸名排序
test_risk_factor_set_names.py：行业因子集并集 + reindex 缺列 0，中途新行业不丢日
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import factorzen.risk.exposures as exposures_module
from factorzen.risk.exposures import (
    ExposureMatrix,
    materialize_style_panel,
    reindex_exposure,
    standardize_style_panel,
)
from factorzen.risk.model import RiskModel, RiskModelResult


# ==== 来自 test_risk_pipeline.py ====
# ==== 来自 test_risk_build_pipeline.py ====
@pytest.fixture(autouse=True)
def _pit_industry_unavailable_by_default__risk_pipeline(monkeypatch):
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

# ==== 来自 test_risk_research_reuse.py ====
@pytest.fixture(autouse=True)
def _pit_off(monkeypatch):
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)
    yield


def _mock_long(n_stocks=10, n_days=120, seed=5):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    daily = pl.DataFrame([
        {"trade_date": dd, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
        for c in codes for dd in days
    ])
    db = pl.DataFrame([
        {
            "trade_date": dd, "ts_code": c,
            "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
            "pb": float(abs(rng.standard_normal()) + 1.5),
            "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15),
            "turnover_rate": float(abs(rng.standard_normal()) * 2 + 1),
        }
        for c in codes for dd in days
    ])
    stocks = pl.DataFrame({
        "ts_code": codes,
        "industry": [["银行", "医药", "电子"][i % 3] for i in range(n_stocks)],
    })
    return daily, db, stocks, days


def test_research_style_reuse_matches_standalone_build():
    """全窗 raw 物化 → 按 ≤d + universe 标准化 再 build，≡ 单独 build(start,d)。"""
    daily, db, stocks, days = _mock_long()
    start_d = days[60]
    rebal_d = days[100]
    start = start_d.strftime("%Y%m%d")
    d_str = rebal_d.strftime("%Y%m%d")
    codes = stocks["ts_code"].to_list()

    # standalone
    daily_d = daily.filter(
        (pl.col("trade_date") <= rebal_d) & pl.col("ts_code").is_in(codes)
    )
    db_d = db.filter(
        (pl.col("trade_date") <= rebal_d) & pl.col("ts_code").is_in(codes)
    )
    standalone = RiskModel().build(daily_d, db_d, stocks, start, d_str)

    # research reuse path
    raw = materialize_style_panel(daily, db, standardize=False)
    style_d = standardize_style_panel(
        raw.filter(
            (pl.col("trade_date") <= rebal_d) & pl.col("ts_code").is_in(codes)
        )
    )
    reused = RiskModel().build(
        daily_d, db_d, stocks, start, d_str, style_panel=style_d
    )

    assert standalone.n_valid_dates == reused.n_valid_dates
    assert standalone.factor_names == reused.factor_names
    assert abs(standalone.r_squared - reused.r_squared) < 1e-10

    # 协方差逐值
    np.testing.assert_allclose(
        standalone.factor_covariance, reused.factor_covariance, atol=1e-12
    )
    # 末日期暴露
    np.testing.assert_allclose(
        standalone.factor_exposures.matrix,
        reused.factor_exposures.matrix,
        atol=1e-12,
    )
    assert standalone.factor_exposures.codes == reused.factor_exposures.codes

# ==== 来自 test_risk_model_cov.py ====
# ==== 来自 test_risk_model.py ====
# tests/test_risk_model.py

@pytest.fixture(autouse=True)
def _pit_industry_unavailable_by_default__risk_model_cov(monkeypatch):
    """RiskModel.build() 内部循环调用 compute_exposures：默认不触达真实 Tushare，
    PIT 历史行业数据视为不可用，走现有 stocks.industry 降级路径（行为与改造
    PIT 行业暴露之前完全一致）。"""
    monkeypatch.setattr(exposures_module, "fetch_index_member_all", lambda: None)
    monkeypatch.setattr(exposures_module, "_pit_industry_warned", False)
    yield

def _toy_result():
    """手搓一个 RiskModelResult，绕开截面回归，做确定性 predict/decompose 验证。"""
    codes = ["A", "B", "C"]
    factor_names = ["size", "value"]
    X = np.array([[1.0, 0.5], [0.8, -0.3], [-0.2, 1.1]])  # (3 stocks, 2 factors)
    F = np.array([[0.04, 0.01], [0.01, 0.09]])             # (2,2) 因子协方差
    D = np.array([0.10, 0.15, 0.20])                       # (3,) 特质风险（std）
    exp = ExposureMatrix(codes=codes, factor_names=factor_names, matrix=X)
    return RiskModelResult(factor_exposures=exp, factor_covariance=F,
                           specific_risk=D, factor_names=factor_names)

def test_decompose_risk_variance_conservation():
    """factor_risk² + specific_risk² ≈ total_risk²（方差可加）。"""
    result = _toy_result()
    w = np.array([0.5, 0.3, 0.2])
    d = RiskModel().decompose_risk(w, result)
    assert {"total_risk", "factor_risk", "specific_risk"} <= set(d)
    assert math.isclose(d["factor_risk"]**2 + d["specific_risk"]**2,
                        d["total_risk"]**2, rel_tol=1e-9)
    # 每个因子名都有一个贡献键
    assert "size" in d and "value" in d

    # Fix 1: 跨函数验证 decompose.total_risk == predict_risk（F 用错/转置错会被抓）
    assert math.isclose(
        d["total_risk"], RiskModel().predict_risk(w, result), rel_tol=1e-9
    ), "decompose_risk 的 total_risk 与 predict_risk 不一致，两者公式不同步"

    # Fix 3: per-factor 贡献语义验证
    # MCR 分解：risk_contrib_k = Xw[k]*(F@Xw)[k] / total_var * total_std * sqrt(252)
    # 数学上：sum_k(risk_contrib_k) = factor_var / total_var * total_risk
    #                               = factor_risk² / total_risk
    # （这是加权 MCR 分解，非欧拉分解到 total_risk）
    per_factor_sum = sum(d[n] for n in result.factor_names)
    expected_factor_sum = d["factor_risk"] ** 2 / d["total_risk"]
    assert math.isclose(per_factor_sum, expected_factor_sum, rel_tol=1e-9), (
        f"per-factor 贡献之和 {per_factor_sum} != factor_risk²/total_risk {expected_factor_sum}"
    )
    # 至少一个因子贡献非零（避免退化）
    assert any(abs(d[n]) > 1e-12 for n in result.factor_names), (
        "所有因子贡献均为零，分解结果退化"
    )

def test_build_end_to_end_r_squared_in_range():
    """端到端 build（mock 数据，n_days≥280 让 momentum 有值）→ R²∈[0,1]。"""
    rng = np.random.default_rng(7)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < 290:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(12)]
    daily = pl.DataFrame([{"trade_date": dd, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
                          for c in codes for dd in days])
    db = pl.DataFrame([{"trade_date": dd, "ts_code": c,
                        "total_mv": float(abs(rng.standard_normal()) * 1e9 + 5e9),
                        "pb": float(abs(rng.standard_normal()) + 1.5),
                        "pe_ttm": float(abs(rng.standard_normal()) * 10 + 15)}
                       for c in codes for dd in days])
    stocks = pl.DataFrame({"ts_code": codes,
                           "industry": [["银行", "医药", "电子"][i % 3] for i in range(12)]})
    start = days[260].strftime("%Y%m%d")
    end = days[-1].strftime("%Y%m%d")
    result = RiskModel().build(daily, db, stocks, start, end)
    assert 0.0 <= result.r_squared <= 1.0
    assert result.factor_covariance.shape[0] == result.factor_covariance.shape[1]
    assert len(result.factor_names) > 0

    # Fix 4: 因子协方差矩阵半正定（核心数学约束）
    assert np.linalg.eigvalsh(result.factor_covariance).min() >= -1e-8, (
        "factor_covariance 不是半正定矩阵，风险模型协方差估计有误"
    )

def test_build_residual_matrix_mid_window_gap_no_misalignment():
    """回归测试：股票在窗口第3期(非首尾)缺失时，重建残差矩阵不能把缺口前的残差
    整体右移一位。

    历史实现"取最后 T_valid 个，右对齐"拼接残差：股票 B 在 5 期窗口里只有 4 期
    数据（第3期停牌缺失），右对齐会把第1、2期残差错位推到第2、3行，
    且把本应是第1期残差的第0行错误置为 NaN（真正的缺口在第2行，反而被掩盖）。
    正确实现必须按真实交易日索引对齐：第1、2、4、5期残差落在各自正确的行，
    第3期（缺口）显式为 NaN。
    """
    from factorzen.risk.model import _build_residual_matrix

    days = [dt.date(2023, 1, 2 + i) for i in range(5)]  # d1..d5，5 期窗口
    residual_dict = {
        "A": [(days[0], 0.1), (days[1], 0.2), (days[2], 0.3), (days[3], 0.4), (days[4], 0.5)],
        # B 第3期(days[2])缺失（停牌/无收益等），非首尾
        "B": [(days[0], 1.1), (days[1], 1.2), (days[3], 1.4), (days[4], 1.5)],
    }
    codes = ["A", "B"]

    matrix = _build_residual_matrix(residual_dict, codes, days)

    assert matrix.shape == (5, 2)
    # A 全勤：5 期精确对应，不受 B 缺口影响
    np.testing.assert_allclose(matrix[:, 0], [0.1, 0.2, 0.3, 0.4, 0.5])
    # B：第1、2、4、5期落在各自正确的行，未被缺口向后挤压一位
    assert math.isclose(matrix[0, 1], 1.1, rel_tol=1e-12)
    assert math.isclose(matrix[1, 1], 1.2, rel_tol=1e-12)
    assert math.isclose(matrix[3, 1], 1.4, rel_tol=1e-12)
    assert math.isclose(matrix[4, 1], 1.5, rel_tol=1e-12)
    # 第3期(缺口本身)应为 NaN，而不是被其他期残差顶替
    assert math.isnan(matrix[2, 1]), f"缺口行应为 NaN，实际: {matrix[2, 1]}"

# ==== 来自 test_risk_covariance.py ====
def test_factor_covariance_symmetric_psd():
    from factorzen.risk.covariance import estimate_factor_covariance

    rng = np.random.default_rng(0)
    fr = rng.standard_normal((120, 5))  # (T=120, K=5)
    cov = estimate_factor_covariance(fr, half_life=60, nw_lags=2)
    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T, atol=1e-10)  # 对称
    assert np.linalg.eigvalsh(cov).min() >= -1e-8  # 半正定

def test_specific_risk_positive():
    from factorzen.risk.covariance import estimate_specific_risk

    rng = np.random.default_rng(0)
    resid = rng.standard_normal((120, 8))  # (T=120, N=8)
    sr = estimate_specific_risk(resid, half_life=60, shrinkage=0.3)
    assert sr.shape == (8,)
    assert (sr > 0).all()  # 特质风险全正

def test_eigenvector_adjustment_symmetric_same_shape():
    from factorzen.risk.covariance import eigenvector_adjustment

    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    cov = a @ a.T  # 半正定对称
    adj = eigenvector_adjustment(cov, n_simulations=200, seed=1)
    assert adj.shape == (4, 4)
    # eigenvector_adjustment 内部执行 (A + A.T) / 2，对称性达机器精度，与
    # factor_covariance 对称断言保持一致
    assert np.allclose(adj, adj.T, atol=1e-10)

def test_covariance_too_short_returns_identity():
    from factorzen.risk.covariance import estimate_factor_covariance

    cov = estimate_factor_covariance(np.zeros((1, 3)), half_life=60)
    assert cov.shape == (3, 3)
    # T < 2 时 estimate_factor_covariance 明确返回单位阵（见 covariance.py L40-42）
    assert np.allclose(cov, np.eye(3))

# ==== 来自 test_risk_industry.py ====
def make_stocks(n_stocks=8):
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    industries = ["银行", "医药", "电子", "食品饮料"]
    return pl.DataFrame({
        "ts_code": codes,
        "industry": [industries[i % len(industries)] for i in range(n_stocks)],
    })

def test_industry_dummies_one_hot_per_stock():
    from factorzen.risk.industry_factors import get_industry_dummies
    dummies = get_industry_dummies(make_stocks())
    ind_cols = [c for c in dummies.columns if c.startswith("ind_")]
    assert len(ind_cols) == 4  # 4 个唯一行业
    # 每只股票恰属一个行业：ind_* 列之和 == 1
    row_sums = dummies.select(ind_cols).sum_horizontal()
    assert row_sums.to_list() == [1.0] * dummies.height

def test_industry_names_sorted_bare():
    from factorzen.risk.industry_factors import get_industry_names
    names = get_industry_names(make_stocks())
    assert names == sorted(names)
    assert set(names) == {"银行", "医药", "电子", "食品饮料"}
    assert all(not n.startswith("ind_") for n in names)  # 裸名，无前缀

def test_industry_dummies_missing_col_raises():
    import pytest

    from factorzen.risk.industry_factors import get_industry_dummies
    with pytest.raises(ValueError):
        get_industry_dummies(pl.DataFrame({"ts_code": ["000001.SZ"]}))

# ==== 来自 test_risk_factor_set_names.py ====
def _make_daily(dates, codes):
    rng = np.random.default_rng(3)
    return pl.DataFrame([
        {"trade_date": d, "ts_code": c, "pct_chg": float(rng.standard_normal() * 2)}
        for d in dates for c in codes
    ])

def test_reindex_exposure_fills_missing_industry_with_zero():
    """缺列填 0、列序对齐固定全集。"""
    exp = ExposureMatrix(
        codes=["A", "B"],
        factor_names=["size", "ind_A"],
        matrix=np.array([[1.0, 1.0], [0.5, 0.0]]),
    )
    fixed = ["size", "ind_A", "ind_C"]
    out = reindex_exposure(exp, fixed)
    assert out.factor_names == fixed
    assert out.matrix.shape == (2, 3)
    np.testing.assert_allclose(out.matrix[:, 0], [1.0, 0.5])
    np.testing.assert_allclose(out.matrix[:, 1], [1.0, 0.0])
    np.testing.assert_allclose(out.matrix[:, 2], [0.0, 0.0])

def test_industry_mid_window_appearance_kept_with_zero_fill():
    """合成场景：某行业中途出现——旧逻辑丢日、新逻辑保留且缺列 0。

    前几日仅 ind_A/ind_B 非零，末日 ind_C 出现；面板并集含三列，早期 ind_C=0。
    """
    dates = [dt.date(2024, 1, i) for i in range(2, 8)]  # 6 个交易日
    codes = [f"{i:06d}.SZ" for i in range(6)]
    daily = _make_daily(dates, codes)
    db = pl.DataFrame([
        {"trade_date": d, "ts_code": c, "total_mv": 5e9, "pb": 1.5, "pe_ttm": 15.0}
        for d in dates for c in codes
    ])
    stocks = pl.DataFrame({"ts_code": codes, "industry": ["银行"] * 6})
    rng = np.random.default_rng(0)

    style_panel = pl.DataFrame({
        "trade_date": [d for d in dates for _ in codes],
        "ts_code": codes * len(dates),
        "size": rng.standard_normal(len(dates) * len(codes)).tolist(),
    })
    # 行业面板：全窗并集含 ind_C；早期 ind_C=0，末日 ind_C=1
    industry_panel = pl.DataFrame([
        {
            "trade_date": d,
            "ts_code": c,
            "ind_A": 0.0 if d == dates[-1] else 1.0,
            "ind_B": 0.0,
            "ind_C": 1.0 if d == dates[-1] else 0.0,
        }
        for d in dates for c in codes
    ])

    result = RiskModel().build(
        daily, db, stocks, "20240102", "20240107",
        style_panel=style_panel,
        industry_panel=industry_panel,
        industry_names=["ind_A", "ind_B", "ind_C"],
    )

    assert result.n_factor_mismatch == 0, (
        f"行业中途出现不应再触发 factor_mismatch，实得 {result.n_factor_mismatch}"
    )
    assert result.n_valid_dates == len(dates), (
        f"全部 {len(dates)} 日应保留，实得 n_valid={result.n_valid_dates}"
    )
    assert "ind_C" in result.factor_names
    assert "ind_C" in result.factor_returns.columns
    # 早期日也有 ind_C 因子收益列（对应暴露为 0）
    assert result.factor_returns.filter(pl.col("trade_date") == dates[0]).height == 1

def test_n_factor_mismatch_visible_on_result():
    """退化可见性：n_factor_mismatch / n_valid_dates 字段存在且默认可读。"""
    dates = [dt.date(2024, 1, i) for i in range(2, 6)]
    codes = [f"{i:06d}.SZ" for i in range(8)]
    daily = _make_daily(dates, codes)
    db = pl.DataFrame([
        {"trade_date": d, "ts_code": c, "total_mv": 5e9, "pb": 1.5, "pe_ttm": 15.0}
        for d in dates for c in codes
    ])
    stocks = pl.DataFrame({
        "ts_code": codes,
        "industry": (["银行", "医药"] * 4),
    })
    rng = np.random.default_rng(2)
    style_panel = pl.DataFrame({
        "trade_date": [d for d in dates for _ in codes],
        "ts_code": codes * len(dates),
        "size": rng.standard_normal(len(dates) * len(codes)).tolist(),
        "value": rng.standard_normal(len(dates) * len(codes)).tolist(),
    })
    industry_panel = pl.DataFrame([
        {
            "trade_date": d,
            "ts_code": c,
            "ind_银行": 1.0 if stocks.filter(pl.col("ts_code") == c)["industry"][0] == "银行" else 0.0,
            "ind_医药": 1.0 if stocks.filter(pl.col("ts_code") == c)["industry"][0] == "医药" else 0.0,
        }
        for d in dates for c in codes
    ])
    result = RiskModel().build(
        daily, db, stocks, "20240102", "20240105",
        style_panel=style_panel,
        industry_panel=industry_panel,
    )
    assert hasattr(result, "n_factor_mismatch")
    assert hasattr(result, "n_valid_dates")
    assert result.n_factor_mismatch == 0
    assert result.n_valid_dates == len(dates)

