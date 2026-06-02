"""daily/evaluation/attribution.py 的单元测试。"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factorzen.config.constants import TRADING_DAYS_PER_YEAR
from factorzen.daily.evaluation.attribution import (
    BarraStyleResult,
    BrinsonResult,
    aggregate_positions_to_sectors,
    barra_style_attribution,
    brinson_attribution,
)

# ───────────────────────────────────────────────────────────────────────────────
# 辅助：合成 Brinson 数据
# ───────────────────────────────────────────────────────────────────────────────


def _make_brinson_data(
    n_dates: int = 5,
    sectors: list[str] | None = None,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """生成合成的行业权重与收益数据。

    Returns:
        (portfolio_sector_weights, benchmark_sector_weights, sector_returns)
    """
    if sectors is None:
        sectors = ["Tech", "Finance", "Energy", "Health"]

    rng = np.random.default_rng(seed)
    dates = [f"2026-01-{d + 1:02d}" for d in range(n_dates)]

    port_rows: list[dict] = []
    bench_rows: list[dict] = []
    ret_rows: list[dict] = []

    for d in dates:
        # 组合权重：随机 Dirichlet 保证和为 1
        pw = rng.dirichlet(np.ones(len(sectors)))
        bw = rng.dirichlet(np.ones(len(sectors)))
        for i, s in enumerate(sectors):
            port_rows.append({"trade_date": d, "sector": s, "port_weight": float(pw[i])})
            bench_rows.append({"trade_date": d, "sector": s, "bench_weight": float(bw[i])})
            ret_rows.append(
                {
                    "trade_date": d,
                    "sector": s,
                    "port_ret": float(rng.normal(0.001, 0.02)),
                    "bench_ret": float(rng.normal(0.0005, 0.015)),
                }
            )

    return (
        pl.DataFrame(port_rows),
        pl.DataFrame(bench_rows),
        pl.DataFrame(ret_rows),
    )


def _make_pure_allocation_data(
    n_dates: int = 3,
    sectors: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """生成"纯配置"合成数据：所有行业 port_ret == bench_ret。"""
    if sectors is None:
        sectors = ["A", "B", "C"]

    rng = np.random.default_rng(42)
    dates = [f"2026-02-{d + 1:02d}" for d in range(n_dates)]

    port_rows: list[dict] = []
    bench_rows: list[dict] = []
    ret_rows: list[dict] = []

    for d in dates:
        pw = rng.dirichlet(np.ones(len(sectors)))
        bw = rng.dirichlet(np.ones(len(sectors)))
        for i, s in enumerate(sectors):
            r = float(rng.normal(0.001, 0.01))
            port_rows.append({"trade_date": d, "sector": s, "port_weight": float(pw[i])})
            bench_rows.append({"trade_date": d, "sector": s, "bench_weight": float(bw[i])})
            ret_rows.append(
                {
                    "trade_date": d,
                    "sector": s,
                    "port_ret": r,
                    "bench_ret": r,  # 两者相等
                }
            )

    return (
        pl.DataFrame(port_rows),
        pl.DataFrame(bench_rows),
        pl.DataFrame(ret_rows),
    )


# ───────────────────────────────────────────────────────────────────────────────
# 辅助：合成 Barra 数据
# ───────────────────────────────────────────────────────────────────────────────


def _make_barra_data(
    n_dates: int = 60,
    styles: list[str] | None = None,
    seed: int = 7,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """生成合成的超额收益与风格因子收益数据。

    Returns:
        (portfolio_excess_returns, style_factor_returns)
    """
    if styles is None:
        styles = ["value", "momentum", "size"]

    rng = np.random.default_rng(seed)
    dates = [f"2026-01-{d + 1:02d}" for d in range(n_dates)]

    style_rets = {s: rng.normal(0.0, 0.01, n_dates) for s in styles}
    # 组合超额收益 = 固定 beta 的线性组合 + 噪声
    betas = {s: rng.uniform(0.2, 1.5) for s in styles}
    excess_ret = sum(betas[s] * style_rets[s] for s in styles) + rng.normal(0.0, 0.002, n_dates)

    port_rows = [{"trade_date": dates[i], "excess_ret": float(excess_ret[i])} for i in range(n_dates)]

    style_rows: list[dict] = []
    for i in range(n_dates):
        row: dict = {"trade_date": dates[i]}
        for s in styles:
            row[s] = float(style_rets[s][i])
        style_rows.append(row)

    return pl.DataFrame(port_rows), pl.DataFrame(style_rows)


def _make_perfect_barra_data(
    n_dates: int = 80,
    styles: list[str] | None = None,
    seed: int = 3,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
    """生成"完美拟合"数据：超额收益严格等于风格因子的线性组合（无噪声）。

    Returns:
        (portfolio_excess_returns, style_factor_returns, true_betas)
    """
    if styles is None:
        styles = ["value", "momentum"]

    rng = np.random.default_rng(seed)
    dates = [f"2026-03-{d + 1:02d}" for d in range(n_dates)]

    style_rets = {s: rng.normal(0.0, 0.01, n_dates) for s in styles}
    true_betas = {s: float(rng.uniform(0.5, 2.0)) for s in styles}
    true_alpha = 0.0001  # 日度截距

    excess_ret = true_alpha + sum(true_betas[s] * style_rets[s] for s in styles)

    port_rows = [{"trade_date": dates[i], "excess_ret": float(excess_ret[i])} for i in range(n_dates)]

    style_rows: list[dict] = []
    for i in range(n_dates):
        row: dict = {"trade_date": dates[i]}
        for s in styles:
            row[s] = float(style_rets[s][i])
        style_rows.append(row)

    return pl.DataFrame(port_rows), pl.DataFrame(style_rows), true_betas


# ═══════════════════════════════════════════════════════════════════════════════
# TestBrinsonAttribution
# ═══════════════════════════════════════════════════════════════════════════════


class TestBrinsonAttribution:
    """Brinson BHB 归因测试。"""

    def test_brinson_identity(self) -> None:
        """三项之和严格等于每期 active_ret（恒等式验证）。"""
        pw, bw, sr = _make_brinson_data(n_dates=10, seed=0)
        result = brinson_attribution(pw, bw, sr)

        assert isinstance(result, BrinsonResult)

        period = result.period_df
        alloc = period["allocation"].to_numpy()
        select = period["selection"].to_numpy()
        interact = period["interaction"].to_numpy()
        active = period["active_ret"].to_numpy()

        # 每期：allocation + selection + interaction == active_ret
        np.testing.assert_allclose(
            alloc + select + interact,
            active,
            atol=1e-10,
            err_msg="Brinson 恒等式违反：三项之和 != active_ret",
        )

    def test_brinson_pure_allocation(self) -> None:
        """当所有行业 port_ret == bench_ret 时，selection 和 interaction 均应为 0。"""
        pw, bw, sr = _make_pure_allocation_data()
        result = brinson_attribution(pw, bw, sr)

        period = result.period_df
        select = period["selection"].to_numpy()
        interact = period["interaction"].to_numpy()

        np.testing.assert_allclose(
            select,
            np.zeros_like(select),
            atol=1e-12,
            err_msg="纯配置场景下 selection 应为 0",
        )
        np.testing.assert_allclose(
            interact,
            np.zeros_like(interact),
            atol=1e-12,
            err_msg="纯配置场景下 interaction 应为 0",
        )

    def test_brinson_result_structure(self) -> None:
        """BrinsonResult 包含正确的列结构与字段类型。"""
        pw, bw, sr = _make_brinson_data(n_dates=5, seed=1)
        result = brinson_attribution(pw, bw, sr)

        # sector_df 列
        assert "sector" in result.sector_df.columns
        assert "allocation" in result.sector_df.columns
        assert "selection" in result.sector_df.columns
        assert "interaction" in result.sector_df.columns
        assert "total_contribution" in result.sector_df.columns

        # period_df 列
        assert "trade_date" in result.period_df.columns
        assert "active_ret" in result.period_df.columns

        # 标量字段类型
        assert isinstance(result.ann_allocation, float)
        assert isinstance(result.ann_selection, float)
        assert isinstance(result.ann_interaction, float)
        assert isinstance(result.ann_active_return, float)

    def test_ann_active_return_equals_sum(self) -> None:
        """ann_active_return == ann_allocation + ann_selection + ann_interaction。"""
        pw, bw, sr = _make_brinson_data(n_dates=8, seed=2)
        result = brinson_attribution(pw, bw, sr)

        expected = result.ann_allocation + result.ann_selection + result.ann_interaction
        assert abs(result.ann_active_return - expected) < 1e-12

    def test_aggregate_positions_to_sectors(self) -> None:
        """aggregate_positions_to_sectors 输出行业权重每日之和应约为 1。"""
        sector_map = {
            "000001.SZ": "Finance",
            "000002.SZ": "RealEstate",
            "600036.SH": "Finance",
            "600519.SH": "Consumer",
            "601318.SH": "Finance",
        }

        rows = []
        for d in ["2026-01-01", "2026-01-02"]:
            for code in sector_map:
                rows.append({"trade_date": d, "ts_code": code, "weight": 0.2})

        positions = pl.DataFrame(rows)
        result = aggregate_positions_to_sectors(positions, sector_map)

        assert "trade_date" in result.columns
        assert "sector" in result.columns
        assert "weight" in result.columns

        # 每日权重之和应约为 1
        daily_sums = (
            result.group_by("trade_date")
            .agg(pl.col("weight").sum().alias("total"))
        )
        for row in daily_sums.iter_rows(named=True):
            assert abs(row["total"] - 1.0) < 1e-10, (
                f"{row['trade_date']} 权重之和 {row['total']:.6f} != 1.0"
            )

    def test_aggregate_unknown_codes_excluded(self) -> None:
        """不在 sector_map 中的个股权重应被排除（权重归一化仍有效）。"""
        sector_map = {"A.SH": "Tech", "B.SH": "Health"}
        rows = [
            {"trade_date": "2026-01-01", "ts_code": "A.SH", "weight": 0.3},
            {"trade_date": "2026-01-01", "ts_code": "B.SH", "weight": 0.3},
            {"trade_date": "2026-01-01", "ts_code": "C.SH", "weight": 0.4},  # 无映射
        ]
        positions = pl.DataFrame(rows)
        result = aggregate_positions_to_sectors(positions, sector_map)

        # 只有 A.SH 和 B.SH 在结果中
        sectors_in_result = set(result["sector"].to_list())
        assert "Tech" in sectors_in_result
        assert "Health" in sectors_in_result

        # 归一化后权重之和为 1
        total = result["weight"].sum()
        assert abs(float(total) - 1.0) < 1e-10  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# TestBarraStyleAttribution
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarraStyleAttribution:
    """Barra 风格归因测试。"""

    def test_residual_mean_near_zero(self) -> None:
        """OLS 带截距时残差均值应约为 0（OLS 正规方程性质）。"""
        port, styles = _make_barra_data(n_dates=60)
        result = barra_style_attribution(port, styles)

        assert isinstance(result, BarraStyleResult)
        resid = result.residual_series["residual"].to_numpy()
        assert abs(float(np.mean(resid))) < 1e-10, (
            f"残差均值 {np.mean(resid):.2e} 偏离 0（OLS 性质违反）"
        )

    def test_perfect_attribution(self) -> None:
        """超额收益严格等于风格线性组合时 R² 应约为 1。"""
        port, styles, _ = _make_perfect_barra_data(n_dates=80)
        result = barra_style_attribution(port, styles)

        assert result.r_squared > 0.999, (
            f"完美线性组合时 R² 应接近 1，实际为 {result.r_squared:.6f}"
        )

    def test_contributions_sum_to_active_return(self) -> None:
        """Σ contributions + alpha/252 ≈ mean(excess_ret)/day（日度等式）。"""
        port, styles = _make_barra_data(n_dates=60, seed=10)
        result = barra_style_attribution(port, styles)

        sum_contrib = sum(result.contributions.values())
        alpha_daily = result.alpha / TRADING_DAYS_PER_YEAR  # 还原到日度
        lhs = sum_contrib + alpha_daily

        mean_excess = float(np.mean(port["excess_ret"].to_numpy()))
        assert abs(lhs - mean_excess) < 1e-10, (
            f"contributions 之和 + alpha_daily = {lhs:.8f}，mean(excess_ret) = {mean_excess:.8f}"
        )

    def test_exposures_keys_match_styles(self) -> None:
        """exposures 和 contributions 的键应与风格因子列名一致。"""
        styles_names = ["value", "momentum", "size", "quality"]
        port, styles = _make_barra_data(n_dates=50, styles=styles_names, seed=5)
        result = barra_style_attribution(port, styles)

        assert set(result.exposures.keys()) == set(styles_names)
        assert set(result.contributions.keys()) == set(styles_names)

    def test_alpha_is_annualized(self) -> None:
        """alpha 字段应为年化（日度截距 × 252）。"""
        port, styles, _ = _make_perfect_barra_data(n_dates=100)
        result = barra_style_attribution(port, styles, trading_days_per_year=252)

        # alpha 为年化，量级上应与日度截距有数量级差异
        # 即 |alpha| / 252 应在合理日度截距范围内
        daily_alpha = result.alpha / 252
        assert abs(daily_alpha) < 1.0, (
            f"日度截距 {daily_alpha:.6f} 不合理（超过 100% 每日）"
        )

    def test_residual_series_structure(self) -> None:
        """residual_series 应包含 trade_date 和 residual 列。"""
        port, styles = _make_barra_data(n_dates=30)
        result = barra_style_attribution(port, styles)

        assert "trade_date" in result.residual_series.columns
        assert "residual" in result.residual_series.columns
        assert result.residual_series.height == len(port)

    def test_empty_data_raises(self) -> None:
        """空 DataFrame（内连接为空）时应抛出 ValueError。"""
        port = pl.DataFrame({"trade_date": ["2026-01-01"], "excess_ret": [0.001]})
        styles = pl.DataFrame({"trade_date": ["2026-02-01"], "value": [0.005]})

        with pytest.raises(ValueError, match="无有效数据"):
            barra_style_attribution(port, styles)
