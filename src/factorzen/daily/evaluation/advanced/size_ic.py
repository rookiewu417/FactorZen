"""Size-stratified IC — 市值分层 IC。"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SizeICResult:
    """市值分层 IC 结果。

    Attributes:
        factor_name: 因子名称
        buckets: 市值分桶关键词典 {bucket_name: ic_mean}
        summary: 文本摘要
    """

    factor_name: str = ""
    buckets: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def __str__(self) -> str:
        lines = [f"Size IC: {self.factor_name}"]
        for name, ic in self.buckets.items():
            lines.append(f"  {name}: IC={ic:.4f}")
        return "\n".join(lines)


def compute_size_ic(
    factor_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    ret_col: str = "fwd_ret",
    cap_col: str = "market_cap",
    n_buckets: int = 3,
    return_object: bool = False,
) -> pl.DataFrame | SizeICResult:
    """按市值分组计算 Rank IC。

    Args:
        factor_df: DataFrame，列: trade_date, ts_code, {factor_col}, {ret_col}, {cap_col}
        factor_col: 因子列名
        ret_col: 收益列名
        cap_col: 市值列名
        n_buckets: 分桶数（默认 3: Large/Mid/Small）
        return_object: True 时返回 SizeICResult 对象

    Returns:
        pl.DataFrame (cap_bucket, ic) 或 SizeICResult
    """
    # 按市值排序分桶
    df = (
        factor_df.with_columns(
            pl.col(cap_col).rank("ordinal", descending=False).over("trade_date").alias("_cap_rank")
        )
        .with_columns(
            ((pl.col("_cap_rank") - 1) * n_buckets // pl.col("_cap_rank").max().over("trade_date"))
            .cast(pl.Int32)
            .alias("cap_bucket")
        )
        .drop("_cap_rank")
    )

    # bucket labels
    if n_buckets == 2:
        labels = {0: "Small", 1: "Large"}
    elif n_buckets == 3:
        labels = {0: "Small", 1: "Mid", 2: "Large"}
    else:
        labels = {i: f"Bucket{i}" for i in range(n_buckets)}

    valid_df = df.filter(
        pl.col(factor_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(ret_col).is_finite()
    )

    ic_rows: list[dict] = []
    buckets_dict: dict[str, float] = {}

    if valid_df.is_empty():
        result_df = pl.DataFrame({"cap_bucket": [], "ic": []})
    else:
        ranked = valid_df.with_columns(
            [
                pl.col(factor_col)
                .rank(method="average")
                .over(["cap_bucket", "trade_date"])
                .alias("_factor_rank"),
                pl.col(ret_col)
                .rank(method="average")
                .over(["cap_bucket", "trade_date"])
                .alias("_ret_rank"),
            ]
        )
        bucket_ic_df = (
            ranked.group_by(["cap_bucket", "trade_date"])
            .agg(
                [
                    pl.corr("_factor_rank", "_ret_rank").alias("ic"),
                    pl.len().alias("_n"),
                ]
            )
            .filter(pl.col("_n") >= 2)
            .filter(pl.col("ic").is_not_null() & pl.col("ic").is_finite())
            .drop("_n")
            .group_by("cap_bucket")
            .agg(pl.col("ic").mean())
            .sort("cap_bucket")
        )

        for row in bucket_ic_df.iter_rows(named=True):
            label = labels.get(row["cap_bucket"], f"Bucket{row['cap_bucket']}")
            ic_rows.append({"cap_bucket": label, "ic": row["ic"]})
            buckets_dict[label] = row["ic"]

        result_df = pl.DataFrame(ic_rows)

    if return_object:
        lines = [f"Size IC: {factor_col}"]
        for name, ic in buckets_dict.items():
            lines.append(f"  {name}: IC={ic:.4f}")
        return SizeICResult(
            factor_name=factor_col,
            buckets=buckets_dict,
            summary="\n".join(lines),
        )
    return result_df
