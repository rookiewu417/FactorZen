"""Monthly asset growth factor.

Asset growth = (total_assets_t - total_assets_{t-4q}) / abs(total_assets_{t-4q}).
High asset growth predicts lower future returns (asset growth anomaly, Cooper et al. 2008).
Uses PIT alignment to avoid look-ahead bias.
"""

import polars as pl

from common.logger import get_logger
from common.storage import scan_parquet
from daily.data.context import FactorDataContext
from daily.data.pit import pit_align
from daily.factors.base import LFTFactor

logger = get_logger(__name__)


class AssetGrowthMonthly(LFTFactor):
    name = "asset_growth"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5
    description = "YoY total asset growth rate (Cooper et al. 2008 asset growth anomaly)"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        empty = pl.DataFrame(
            schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64}
        )
        try:
            fina_lf = scan_parquet("finance")
            fina_df = (
                fina_lf
                .filter(pl.col("end_date").is_not_null())
                .select(["ts_code", "end_date", "ann_date", "total_assets"])
                .filter(pl.col("total_assets").is_not_null() & (pl.col("total_assets") > 0))
                .collect()
            )
        except Exception as e:
            logger.warning(f"finance data load failed: {e}, returning empty")
            return empty

        if fina_df.is_empty():
            return empty

        # Compute YoY growth: need current and prior-year same quarter.
        # Sort by (ts_code, end_date) then shift by 4 rows (4 quarters back).
        fina_sorted = fina_df.sort(["ts_code", "end_date"])
        fina_with_prior = fina_sorted.with_columns(
            pl.col("total_assets").shift(4).over("ts_code").alias("_prior_assets")
        ).filter(
            pl.col("_prior_assets").is_not_null() & (pl.col("_prior_assets") > 0)
        ).with_columns(
            ((pl.col("total_assets") - pl.col("_prior_assets")) / pl.col("_prior_assets").abs())
            .alias("asset_growth_yoy")
        )

        pit_input = fina_with_prior.select(
            ["ts_code", "end_date", "ann_date", "asset_growth_yoy"]
        )

        snapshot_dates = ctx.snapshot_dates
        pit_df = pit_align(pit_input, snapshot_dates)

        if pit_df.is_empty():
            return empty

        result = (
            pit_df
            .select([
                pl.col("snapshot_date").alias("trade_date"),
                pl.col("ts_code"),
                pl.col("asset_growth_yoy").alias("factor_value"),
            ])
            .filter(pl.col("factor_value").is_not_null())
        )
        return result


AssetGrowthMonthly()
