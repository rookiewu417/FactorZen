"""Sector-stratified IC — 行业分层 IC。"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SectorICResult:
    """行业分层 IC 结果。

    Attributes:
        factor_name: 因子名称
        sector_ic_df: 行业 IC DataFrame (sector, ic)
        low_sample_warnings: 低样本量警告列表
    """

    factor_name: str = ""
    sector_ic_df: pl.DataFrame = field(default_factory=pl.DataFrame)
    low_sample_warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Sector IC: {self.factor_name}"]
        if not self.sector_ic_df.is_empty():
            for row in self.sector_ic_df.iter_rows(named=True):
                lines.append(f"  {row['sector']}: IC={row['ic']:.4f}")
        if self.low_sample_warnings:
            lines.append("  Warnings:")
            for w in self.low_sample_warnings:
                lines.append(f"    {w}")
        return "\n".join(lines)


def compute_sector_ic(
    factor_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    ret_col: str = "fwd_ret",
    sector_col: str = "sector",
    return_object: bool = False,
    min_samples: int = 30,
) -> pl.DataFrame | SectorICResult:
    """按行业分组计算 Rank IC。

    Args:
        factor_df: DataFrame，列: trade_date, ts_code, {factor_col}, {ret_col}, {sector_col}
        factor_col: 因子列名
        ret_col: 收益列名
        sector_col: 行业列名
        return_object: True 时返回 SectorICResult 对象
        min_samples: 触发低样本警告的阈值

    Returns:
        pl.DataFrame (sector, ic) 或 SectorICResult
    """
    valid_df = factor_df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )

    warnings: list[str] = []
    sector_counts = factor_df.group_by(sector_col).agg(pl.len().alias("_n"))
    for row in sector_counts.iter_rows(named=True):
        if row["_n"] < min_samples:
            warnings.append(
                f"Sector '{row[sector_col]}' has only {row['_n']} samples (< {min_samples})"
            )

    if valid_df.is_empty():
        result_df = pl.DataFrame({"sector": [], "ic": []})
    else:
        ranked = valid_df.with_columns(
            [
                pl.col(factor_col)
                .rank(method="average")
                .over([sector_col, "trade_date"])
                .alias("_factor_rank"),
                pl.col(ret_col)
                .rank(method="average")
                .over([sector_col, "trade_date"])
                .alias("_ret_rank"),
            ]
        )
        result_df = (
            ranked.group_by([sector_col, "trade_date"])
            .agg(
                [
                    pl.corr("_factor_rank", "_ret_rank").alias("ic"),
                    pl.len().alias("_n"),
                ]
            )
            .filter(pl.col("_n") >= 2)
            .filter(pl.col("ic").is_not_null() & pl.col("ic").is_finite())
            .drop("_n")
            .group_by(sector_col)
            .agg(pl.col("ic").mean())
            .rename({sector_col: "sector"} if sector_col != "sector" else {})
            .sort("sector")
        )

    if return_object:
        return SectorICResult(
            factor_name=factor_col,
            sector_ic_df=result_df,
            low_sample_warnings=warnings,
        )
    return result_df
