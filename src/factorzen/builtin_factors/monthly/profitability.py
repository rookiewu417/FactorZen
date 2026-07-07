"""月频 ROE（YTD 口径）因子。使用 PIT 对齐确保无未来信息。

诚实标注：本因子直接取 Tushare fina_indicator 的 ``roe``——它是**报告期 YTD 累计** ROE
（年报=全年、一季报=单季、中报=半年），并非 TTM 滚动。PIT 对齐后同一快照日截面上不同
公司的最新报告期不同，累计窗口不可比（排的是「披露进度×窗口长度」而非盈利能力）。真正的
TTM（income 单季净利润滚动 4 季 / 平均净资产）留作后续研究线；故命名为 roe_ytd 而非 roe_ttm。
"""

import polars as pl

from factorzen.core.logger import get_logger
from factorzen.core.storage import scan_parquet
from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.data.pit import pit_align
from factorzen.daily.factors.base import DailyFactor

logger = get_logger(__name__)


class RoeYtdMonthly(DailyFactor):
    name = "roe_ytd"
    category = "monthly"
    frequency = "monthly"
    # compute 读 finance parquet；pipeline 还需 ctx.daily 算前向收益（否则 raise）。
    # 不读 daily_basic。clean 环境 finance 需先 loader.fetch_finance 拉取（follow-up）。
    required_data = ["finance", "daily"]
    lookback_days = 5
    description = "月频 ROE（fina_indicator YTD 累计口径，非 TTM；截面披露进度不可比），每月末截面"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        # 1. 加载财务数据
        try:
            fina_lf = scan_parquet("finance")
            fina_df = (
                fina_lf.filter(pl.col("end_date").is_not_null())
                .select(["ts_code", "end_date", "ann_date", "roe"])
                .collect()
            )
        except Exception as e:
            logger.warning(f"财务数据加载失败: {e}，返回空结果")
            return pl.DataFrame(
                schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64}
            )

        if fina_df.is_empty():
            logger.warning("财务数据为空，返回空结果")
            return pl.DataFrame(
                schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64}
            )

        # 2. PIT 对齐到月频快照日
        snapshot_dates = ctx.snapshot_dates
        pit_df = pit_align(fina_df, snapshot_dates)

        if pit_df.is_empty():
            return pl.DataFrame(
                schema={"trade_date": pl.Date, "ts_code": pl.Utf8, "factor_value": pl.Float64}
            )

        # 3. 提取 roe 作为因子值
        result = pit_df.select(
            [
                pl.col("snapshot_date").alias("trade_date"),
                pl.col("ts_code"),
                pl.col("roe").alias("factor_value"),
            ]
        ).filter(pl.col("factor_value").is_not_null())
        return result


RoeYtdMonthly()
