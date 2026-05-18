"""因子计算的数据上下文。根据因子声明的 required_data 自动加载数据。"""

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from common.calendar import prev_trade_date
from common.storage import load_parquet


@dataclass
class FactorDataContext:
    start: str  # "YYYYMMDD"
    end: str  # "YYYYMMDD"
    required_data: list[str] = field(default_factory=lambda: ["daily"])
    lookback_days: int = 20
    universe: list[str] | None = None  # 股票池 ts_code 列表
    snapshot_mode: str = "daily"  # "daily" | "weekly" | "monthly"

    # 私有：惰性缓存
    _daily: pl.LazyFrame | None = field(default=None, repr=False)
    _daily_basic: pl.LazyFrame | None = field(default=None, repr=False)
    _weekly_snapshot: pl.LazyFrame | None = field(default=None, repr=False)
    _monthly_snapshot: pl.LazyFrame | None = field(default=None, repr=False)
    _weekly_basic: pl.LazyFrame | None = field(default=None, repr=False)
    _monthly_basic: pl.LazyFrame | None = field(default=None, repr=False)

    @property
    def expanded_start(self) -> str:
        """返回考虑 lookback_days 后扩展的起始日期（YYYYMMDD 格式）。"""
        return prev_trade_date(self.start, self.lookback_days).strftime("%Y%m%d")

    @property
    def daily(self) -> pl.LazyFrame:
        """惰性加载日线行情（含扩展区间），自动 join 复权因子列，按需过滤 universe。

        返回的 LazyFrame 包含原始列 + close_adj / open_adj / high_adj / low_adj。
        若 adj_factor 数据不存在，close_adj 等列回退为原始价格（无中断）。
        """
        if "daily" not in self.required_data:
            raise ValueError("daily data not declared in required_data")
        if self._daily is None:
            lf = load_parquet("daily", start=self.expanded_start, end=self.end)
            if self.universe:
                lf = lf.filter(pl.col("ts_code").is_in(self.universe))

            # 尝试 join 复权因子
            try:
                adj_lf = load_parquet("adj_factor", start=self.expanded_start, end=self.end)
                adj_lf = adj_lf.select(["ts_code", "trade_date", "adj_factor"])
                lf = lf.join(adj_lf, on=["ts_code", "trade_date"], how="left")
                for col in ("close", "open", "high", "low"):
                    lf = lf.with_columns((pl.col(col) * pl.col("adj_factor")).alias(f"{col}_adj"))
                lf = lf.drop("adj_factor")
            except Exception:
                # adj_factor 未落盘时优雅回退
                for col in ("close", "open", "high", "low"):
                    lf = lf.with_columns(pl.col(col).alias(f"{col}_adj"))

            self._daily = lf
        return self._daily

    @property
    def daily_basic(self) -> pl.LazyFrame:
        """惰性加载每日估值数据（含扩展区间），按需过滤 universe。"""
        if "daily_basic" not in self.required_data:
            raise ValueError("daily_basic data not declared in required_data")
        if self._daily_basic is None:
            lf = load_parquet("daily_basic", start=self.expanded_start, end=self.end)
            if self.universe:
                lf = lf.filter(pl.col("ts_code").is_in(self.universe))
            self._daily_basic = lf
        return self._daily_basic

    @property
    def snapshot_dates(self) -> list[date]:
        """根据 snapshot_mode 返回快照日期列表。"""
        from common.calendar import (
            get_monthly_snapshot_dates,
            get_trade_dates,
            get_weekly_snapshot_dates,
        )

        if self.snapshot_mode == "weekly":
            return get_weekly_snapshot_dates(self.start, self.end)
        elif self.snapshot_mode == "monthly":
            return get_monthly_snapshot_dates(self.start, self.end)
        else:
            return get_trade_dates(self.start, self.end)

    @property
    def weekly(self) -> pl.LazyFrame:
        """日线数据下采样到周频快照。"""
        if self._weekly_snapshot is None:
            lf = self.daily  # 复用已有懒加载链
            snap = self.snapshot_dates
            lf = lf.filter(pl.col("trade_date").is_in(snap))
            self._weekly_snapshot = lf
        return self._weekly_snapshot

    @property
    def monthly(self) -> pl.LazyFrame:
        """日线数据下采样到月频快照。"""
        if self._monthly_snapshot is None:
            lf = self.daily
            snap = self.snapshot_dates
            lf = lf.filter(pl.col("trade_date").is_in(snap))
            self._monthly_snapshot = lf
        return self._monthly_snapshot

    @property
    def weekly_basic(self) -> pl.LazyFrame:
        """daily_basic 下采样到周频快照。"""
        if self._weekly_basic is None:
            lf = self.daily_basic
            snap = self.snapshot_dates
            lf = lf.filter(pl.col("trade_date").is_in(snap))
            self._weekly_basic = lf
        return self._weekly_basic

    @property
    def monthly_basic(self) -> pl.LazyFrame:
        """daily_basic 下采样到月频快照。"""
        if self._monthly_basic is None:
            lf = self.daily_basic
            snap = self.snapshot_dates
            lf = lf.filter(pl.col("trade_date").is_in(snap))
            self._monthly_basic = lf
        return self._monthly_basic

    def load_all(self) -> None:
        """强制加载所有声明的数据。"""
        for data_type in self.required_data:
            if data_type == "daily":
                _ = self.daily
            elif data_type == "daily_basic":
                _ = self.daily_basic
        if self.snapshot_mode == "weekly":
            _ = self.weekly
        elif self.snapshot_mode == "monthly":
            _ = self.monthly
