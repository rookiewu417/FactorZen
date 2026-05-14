"""月频 ROE TTM 因子。使用 PIT 对齐确保无未来信息。"""

import polars as pl

from common.logger import get_logger
from common.storage import scan_parquet
from daily.data.context import FactorDataContext
from daily.data.pit import pit_align
from daily.factors.base import LFTFactor

logger = get_logger(__name__)


class RoeTtmMonthly(LFTFactor):
    name = "roe_ttm"
    category = "monthly"
    frequency = "monthly"
    required_data = ["daily_basic"]
    lookback_days = 5
    description = "月频 ROE TTM（PIT 对齐），每月末截面"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        # 1. 加载财务数据
        try:
            fina_lf = scan_parquet("finance")
            fina_df = (
                fina_lf
                .filter(pl.col("end_date").is_not_null())
                .select(["ts_code", "end_date", "ann_date", "roe"])
                .collect()
            )
        except Exception as e:
            logger.warning(f"财务数据加载失败: {e}，返回空结果")
            return pl.DataFrame(schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64})

        if fina_df.is_empty():
            logger.warning("财务数据为空，返回空结果")
            return pl.DataFrame(schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64})

        # 2. PIT 对齐到月频快照日
        snapshot_dates = ctx.snapshot_dates
        pit_df = pit_align(fina_df, snapshot_dates)

        if pit_df.is_empty():
            return pl.DataFrame(schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64})

        # 3. 提取 roe 作为因子值
        result = (
            pit_df
            .select([
                pl.col("snapshot_date").alias("trade_date"),
                pl.col("ts_code"),
                pl.col("roe").alias("factor_value"),
            ])
            .filter(pl.col("factor_value").is_not_null())
        )
        return result


RoeTtmMonthly()
