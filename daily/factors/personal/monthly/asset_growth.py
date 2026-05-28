"""Monthly asset growth factor.

Uses Tushare fina_indicator's assets_yoy field (YoY total asset growth rate, %).
High asset growth predicts lower future returns (asset growth anomaly, Cooper et al. 2008).
Uses PIT alignment to avoid look-ahead bias.
"""

import polars as pl

from common.logger import get_logger
from common.storage import scan_parquet
from daily.data.context import FactorDataContext
from daily.data.pit import pit_align
from daily.factors.base import DailyFactor

logger = get_logger(__name__)


class AssetGrowthMonthly(DailyFactor):
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
            schema = fina_lf.collect_schema()
            if "assets_yoy" not in schema:
                logger.warning(
                    "finance data missing 'assets_yoy' column; re-fetch with assets_yoy field"
                )
                return empty
            fina_df = (
                fina_lf.filter(pl.col("end_date").is_not_null())
                .select(["ts_code", "end_date", "ann_date", "assets_yoy"])
                .filter(pl.col("assets_yoy").is_not_null())
                .collect()
            )
        except Exception as e:
            logger.warning(f"finance data load failed: {e}, returning empty")
            return empty

        if fina_df.is_empty():
            return empty

        pit_input = fina_df.select(["ts_code", "end_date", "ann_date", "assets_yoy"])

        snapshot_dates = ctx.snapshot_dates
        pit_df = pit_align(pit_input, snapshot_dates)

        if pit_df.is_empty():
            return empty

        result = pit_df.select(
            [
                pl.col("snapshot_date").alias("trade_date"),
                pl.col("ts_code"),
                pl.col("assets_yoy").alias("factor_value"),
            ]
        ).filter(pl.col("factor_value").is_not_null())
        return result


AssetGrowthMonthly()
