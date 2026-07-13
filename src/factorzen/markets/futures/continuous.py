"""主力连续合约拼接 + 乘法后复权（本 Phase 核心难点，纯函数、离线可测）。

输入原始 ``fut_daily``（全合约日线）与 ``fut_mapping``（逐日主力映射），输出品种级
后复权连续序列（``ts_code`` = 连续码，如 ``CU.SHF``）。

**展期跳变处理硬契约**（见 ground-truth 测试 ``tests/test_futures_continuous.py``）：
- 乘法后复权（earliest-anchored）：首段 ``adj_factor=1.0``；每个展期日按
  ``roll_step = 旧主力前日收盘 / 新主力前日收盘`` 累乘，``adj_factor`` 沿品种 ``cum_prod``；
  OHLC × ``adj_factor`` 得复权价，使复权后展期日 ``ret`` = 新主力自身当日收益、``ts_*``
  算子跨展期无跳变。
- 推导：设复权 close ``S[d]=R[d]·c[d]``（R=主力原始 close，c=复权系数）。要求展期日
  ``S[d]/S[d-1] = B_d/B_{d-1}``（新主力 B 自身收益）⟹ ``c[d]/c[d-1] = A_{d-1}/B_{d-1}``
  （A=旧主力）；非展期日 c 不变（同合约，原始收益即真实收益）。
- earliest-anchored（而非前复权 latest-anchored）：历史复权值不随新增数据变，对增量缓存友好。
- **PIT**：主力切换以 mapping 为准（当日盘后可得）；``roll_step`` 只用展期日**前一日**收盘，
  PIT 安全。窗口首日相对窗口外的展期无法侦测（无窗口外 mapping）→ 视作新段起点（c=1），
  不用窗口外未来信息。
- **量列不复权**：``vol``/``amount``/``oi`` 保持每合约原始值（换合约=真实特征）。价格代理
  ``vwap`` 的复权在 ``FuturesFactorSet.derived_columns`` 用 ``adj_factor`` 处理。
"""
from __future__ import annotations

import polars as pl

_OUT_COLS = [
    "ts_code", "trade_date", "open", "high", "low", "close",
    "vol", "amount", "oi", "adj_factor", "mapping_ts_code",
]
_PRICE_COLS = ("open", "high", "low", "close")
_QTY_COLS = ("vol", "amount", "oi")


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "ts_code": pl.String, "trade_date": pl.Date,
        "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
        "vol": pl.Float64, "amount": pl.Float64, "oi": pl.Float64,
        "adj_factor": pl.Float64, "mapping_ts_code": pl.String,
    })


def build_continuous(
    mapping: pl.DataFrame, daily: pl.DataFrame, fut_codes: set[str]
) -> pl.DataFrame:
    """按主力映射拼品种级后复权连续序列。

    Args:
        mapping: 列 ``ts_code``(连续码)/``trade_date``/``mapping_ts_code``(实际合约)。
        daily: 原始 fut_daily，列至少含 ``ts_code``(实际合约)/``trade_date``/OHLC/``vol``/
            ``amount``/``oi``（可含 Tushare 连续镜像行，会被 join 语义自然排除）。
        fut_codes: 主力连续码过滤集（品种字母，如 ``{"CU","A"}``）；连续码 base∈该集才保留，
            滤掉 ``L`` 后缀次主力连续。

    Returns:
        列 ``ts_code``(连续码)/``trade_date``/复权 OHLC/原始 ``vol``/``amount``/``oi``/
        ``adj_factor``/``mapping_ts_code``，按 (ts_code, trade_date) 排序。
    """
    if mapping.is_empty() or daily.is_empty():
        return _empty()

    # 主力连续过滤：连续码 base（'.' 前）∈ fut_codes（滤掉 L 后缀次主力）
    codes = list(fut_codes)
    m = (
        mapping.with_columns(
            pl.col("ts_code").str.split(".").list.first().alias("_base")
        )
        .filter(pl.col("_base").is_in(codes))
        .select("ts_code", "trade_date", "mapping_ts_code")
        .unique()
        .sort(["ts_code", "trade_date"])
    )
    if m.is_empty():
        return _empty()

    # 主力合约当日行情：daily(ts_code=mapping_ts_code, trade_date) 左连
    keep = ["ts_code", "trade_date", *[c for c in (*_PRICE_COLS, *_QTY_COLS) if c in daily.columns]]
    d_main = daily.select(keep).rename({"ts_code": "mapping_ts_code"})
    cont = m.join(d_main, on=["mapping_ts_code", "trade_date"], how="left").sort(
        ["ts_code", "trade_date"]
    )

    # 品种内前一日：日期 / 旧主力代码 / 旧主力前日收盘(=连续序列前一行 close，因前一日主力=旧主力)
    cont = cont.with_columns(
        pl.col("trade_date").shift(1).over("ts_code").alias("_prev_date"),
        pl.col("mapping_ts_code").shift(1).over("ts_code").alias("_prev_main"),
        pl.col("close").shift(1).over("ts_code").alias("_old_prev_close"),
    ).with_columns(
        (pl.col("mapping_ts_code") != pl.col("_prev_main")).fill_null(False).alias("_is_roll")
    )

    # 新主力前日收盘：daily(ts_code=当前 mapping_ts_code, trade_date=_prev_date) 左连
    new_prev = (
        daily.select("ts_code", "trade_date", "close")
        .rename({"ts_code": "mapping_ts_code", "trade_date": "_prev_date", "close": "_new_prev_close"})
    )
    cont = cont.join(new_prev, on=["mapping_ts_code", "_prev_date"], how="left")

    # roll_step：展期日且旧/新前日收盘均有效 → 旧/新；否则 1.0（含首日、非展期日、缺报价退化）
    cont = cont.with_columns(
        pl.when(
            pl.col("_is_roll")
            & pl.col("_old_prev_close").is_not_null()
            & pl.col("_new_prev_close").is_not_null()
            & (pl.col("_new_prev_close").abs() > 1e-12)
        )
        .then(pl.col("_old_prev_close") / pl.col("_new_prev_close"))
        .otherwise(1.0)
        .alias("_roll_step")
    )

    # adj_factor = 品种内 roll_step 的累乘（earliest-anchored：首段=1）
    cont = cont.sort(["ts_code", "trade_date"]).with_columns(
        pl.col("_roll_step").cum_prod().over("ts_code").alias("adj_factor")
    )

    # 价格列复权，量列保持原始
    cont = cont.with_columns(
        [(pl.col(c) * pl.col("adj_factor")).alias(c) for c in _PRICE_COLS]
    )
    for c in _QTY_COLS:
        if c not in cont.columns:
            cont = cont.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return cont.select(_OUT_COLS).sort(["ts_code", "trade_date"])
