"""日频信号（资金流 / 北向持股）对齐:直接按交易日 join。

与基本面不同,资金流/北向本身就是**日频 point-in-time**,当日数据当日可得,无需
PIT 季度对齐(pit_align)。挖掘(`prepare_mining_daily`)与物化(`ExpressionFactor.compute`)
两条路都调 `attach_flows`,保证同一因子逐值一致(陷阱#2)。

叶子有效期(2026-07 缓存审计):
- ``net_mf_amount``(moneyflow):2016-06 起逐日完整(≈244 交易日/年)。
- ``north_ratio``(hk_hold 北向持股占比):**有效起点 2016-06-29**(此前 API 返回空,非 bug);
  缓存存在 **2017-2018 空档**(需增量回补,Tushare 限速);2019+ 逐日完整(≈236 交易日/年,
  因港股通假期少于 A 股日历)。**2024-08-16 后港交所停止逐日披露个股北向持股**——缓存里
  2024-09 起覆盖股票数骤降(约 1/5)、连续重复值占比升高(2024上 0.245 → 2024下 0.380),属
  降频/部分披露。此后缺失日 join 得 null(``attach_flows`` 只 left-join,**不 ffill 伪造**,
  PIT 诚实);用 ``north_ratio`` 的因子在 2024-09 后有效截面变薄,评估时会自然缩样。
"""
from __future__ import annotations

import polars as pl

# 叶子名 → (缓存分区, 源列名)。北向 ratio 重命名为 north_ratio,避免与通用名冲突。
_FLOW_SOURCES: dict[str, tuple[str, str]] = {
    "net_mf_amount": ("moneyflow", "net_mf_amount"),  # 主力净流入额(万元)
    "north_ratio": ("hk_hold", "ratio"),              # 北向持股占比(%)
}


def attach_flows(daily: pl.DataFrame, *, injected: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    """把资金流/北向日频信号按 (trade_date, ts_code) join 进日线帧,作为叶子列。

    缺数据 / 读取失败 → 原样返回,缺的叶子补 null(表达式引用到时得 null 而非 KeyError)。
    ``injected``:``{分区名: DataFrame}`` 供测试注入,绕过 parquet 读取。
    """
    from factorzen.discovery.operators import FLOW_FEATURES

    if daily.is_empty() or "trade_date" not in daily.columns:
        return daily
    injected = injected or {}
    by_part: dict[str, list[tuple[str, str]]] = {}
    for leaf in sorted(FLOW_FEATURES):
        part, col = _FLOW_SOURCES[leaf]
        by_part.setdefault(part, []).append((leaf, col))

    for part, pairs in by_part.items():
        src_cols = [c for _, c in pairs]
        df = injected.get(part)
        if df is None:
            df = _load_flow(part, src_cols)
        if df is None or df.is_empty():
            continue
        sel = df.select(["ts_code", "trade_date", *[c for c in src_cols if c in df.columns]])
        sel = sel.rename({col: leaf for leaf, col in pairs if col in sel.columns})
        sel = _align_trade_date(sel, daily)
        daily = daily.join(sel, on=["trade_date", "ts_code"], how="left")
    return _ensure_flow_cols(daily)


def _load_flow(part: str, cols: list[str]) -> pl.DataFrame | None:
    from factorzen.core.storage import scan_parquet
    try:
        lf = scan_parquet(part)
        names = lf.collect_schema().names()
        have = [c for c in cols if c in names]
        if not have:
            return None
        return lf.select(["ts_code", "trade_date", *have]).collect()
    except Exception:
        return None


def _align_trade_date(sel: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """把 flow 帧的 trade_date 类型对齐到 daily(通常都是 Date;注入的 String 转 Date)。"""
    if sel["trade_date"].dtype == daily["trade_date"].dtype:
        return sel
    if sel["trade_date"].dtype == pl.Utf8:
        return sel.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d", strict=False))
    return sel


def _ensure_flow_cols(daily: pl.DataFrame) -> pl.DataFrame:
    from factorzen.discovery.operators import FLOW_FEATURES
    missing = [c for c in sorted(FLOW_FEATURES) if c not in daily.columns]
    if missing:
        daily = daily.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
    return daily
