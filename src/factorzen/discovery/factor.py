"""把表达式包装成标准 DailyFactor，可被 registry/评估管线无缝消费。"""
from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import polars as pl

from factorzen.daily.factors.base import DailyFactor
from factorzen.discovery.derived import add_derived_columns
from factorzen.discovery.expression import evaluate_materialized, feature_names, parse_expr
from factorzen.discovery.operators import (
    BASIC_FEATURES,
    FLOW_FEATURES,
    FUNDAMENTAL_FEATURES,
    HOLDER_FEATURES,
    MARGIN_FEATURES,
)

_PRICE_COLS = ["open", "high", "low", "close", "open_adj", "high_adj",
               "low_adj", "close_adj", "vol", "amount"]


class ExpressionFactor(DailyFactor):
    """表达式因子：可直接实例化（传 expression），或被子类用类属性覆盖 expression 后实例化。"""

    required_data: ClassVar[list[str]] = ["daily", "daily_basic"]
    expression: str = ""       # 子类可用类属性覆盖
    mined_name: str = ""
    lookback_days: int = 60

    def __init__(self, expression: str | None = None, mined_name: str | None = None,
                 lookback_days: int | None = None) -> None:
        # 不加 @dataclass：支持「直接传参」与「子类用类属性提供 expression」两种构造方式
        if expression is not None:
            self.expression = expression
        if mined_name is not None:
            self.mined_name = mined_name
        if lookback_days is not None:
            self.lookback_days = lookback_days
        if not self.expression:
            raise ValueError("ExpressionFactor 需要非空 expression")
        self.node = parse_expr(self.expression)
        if not getattr(self, "name", ""):
            self.name = self.mined_name or f"mined_{abs(hash(self.expression)) % (10**8)}"
        self.description = f"mined: {self.expression}"
        self._feats = feature_names(self.node)

    def compute(self, ctx) -> pl.DataFrame:
        daily = ctx.daily.collect()
        # 停牌掩码：vol==0 行的价量列置 null，避免污染时序算子
        daily = daily.with_columns([
            pl.when(pl.col("vol") > 0).then(pl.col(c)).otherwise(None).alias(c)
            for c in _PRICE_COLS if c in daily.columns
        ])
        # 仅在表达式引用基本面叶子时 join daily_basic
        if self._feats & BASIC_FEATURES:
            basic = ctx.daily_basic.collect()
            if not basic.is_empty():
                daily = daily.join(basic, on=["trade_date", "ts_code"], how="left")
        # 仅在引用财报叶子(roe/assets_yoy)时 attach PIT 对齐的基本面——与挖掘路径
        # prepare_mining_daily 共用 attach_fundamentals，保证同一因子两条路逐值一致（陷阱#2）。
        if self._feats & FUNDAMENTAL_FEATURES:
            from factorzen.daily.data.pit import attach_fundamentals
            daily = attach_fundamentals(daily)
        # 股东户数（ann_date PIT，与 fina 同款 pit_align）
        if self._feats & HOLDER_FEATURES:
            from factorzen.daily.data.pit import attach_holders
            daily = attach_holders(daily)
        # 仅在引用资金流/北向/两融/龙虎榜叶子时 attach（日频 join，与挖掘路径共用 attach_flows，防漂移）
        if self._feats & FLOW_FEATURES:
            # margin_ratio 需 circ_mv（万元）；prepare_mining 已 join daily_basic，
            # 物化路径若未因 BASIC 叶子 join 过，在此补 join，避免比值全 null（双路径）。
            if (self._feats & MARGIN_FEATURES) and "circ_mv" not in daily.columns:
                basic = ctx.daily_basic.collect()
                if not basic.is_empty():
                    daily = daily.join(basic, on=["trade_date", "ts_code"], how="left")
            from factorzen.daily.data.flows import attach_flows
            daily = attach_flows(daily)
        # 排序必须在依赖行序的派生列（shift/over）之前完成，否则 ret_1d 等会用到
        # 乱序的「上一行」当成「前一交易日」算出错误结果（与 mining_session.py 保持一致）
        df = daily.sort(["ts_code", "trade_date"])
        # 派生列（与 mining_session.py 共用 add_derived_columns，消除双路径漂移）
        df = add_derived_columns(df)
        df = df.with_columns(evaluate_materialized(self.node, df).alias("factor_value"))
        start = datetime.strptime(ctx.start, "%Y%m%d").date()
        return (
            df.filter(pl.col("trade_date") >= start)
            .select(["trade_date", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
        )
