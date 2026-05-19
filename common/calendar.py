"""A 股交易日历查询。

基于 Tushare ``trade_cal`` 接口（SSE 交易所），首次拉取后缓存到
``data/cache/trade_cal.parquet``，7 天过期后自动刷新。

使用示例::

    from common.calendar import is_trade_date, prev_trade_date, next_trade_date

    is_trade_date("20260320")          # True / False
    prev_trade_date(date(2026, 3, 22)) # 前一个交易日
    next_trade_date("20260320", n=3)   # 后 3 个交易日
"""

from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from config.settings import DATA_CACHE
from config.tushare_config import CACHE_EXPIRE_DAYS, ensure_token

# ── 常量 ─────────────────────────────────────────────────
_CAL_FILE: Path = DATA_CACHE / "trade_cal.parquet"


# ── 内部缓存管理 ────────────────────────────────────────


def _is_cache_valid() -> bool:
    """判断本地缓存是否存在且未过期。"""
    if not _CAL_FILE.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(_CAL_FILE.stat().st_mtime)
    return age.days < CACHE_EXPIRE_DAYS


def _fetch_from_tushare() -> pl.DataFrame:
    """通过 Tushare trade_cal 接口拉取全量交易日历（SSE，1990 至今）。

    仅在缓存失效时调用，避免模块级 Tushare 初始化。
    """
    import tushare as ts

    ts.set_token(ensure_token())
    pro = ts.pro_api()

    df = pro.trade_cal(exchange="SSE", start_date="19900101", end_date="20301231")
    if df is None or df.empty:
        raise RuntimeError("Tushare trade_cal 返回空数据，请检查网络或积分权限。")

    # 转换字段类型
    out = pl.from_pandas(df).with_columns(
        pl.col("cal_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d"),
        pl.col("is_open").cast(pl.Int8),
    )

    # 确保缓存目录存在
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    out.write_parquet(_CAL_FILE)
    return out


def _load_calendar() -> pl.DataFrame:
    """加载交易日历（优先缓存，失效则刷新）。"""
    if _is_cache_valid():
        return pl.read_parquet(_CAL_FILE)
    return _fetch_from_tushare()


# ── 公开 API ────────────────────────────────────────────


def get_trade_calendar(start: str | None = None, end: str | None = None) -> pl.DataFrame:
    """返回交易日历 DataFrame。

    Parameters
    ----------
    start : str, optional
        起始日期，格式 ``YYYYMMDD``。不传则返回全量。
    end : str, optional
        结束日期，格式 ``YYYYMMDD``。不传则返回全量。

    Returns
    -------
    pl.DataFrame
        字段: ``cal_date`` (Date), ``is_open`` (Int8, 1=交易日),
        ``pretrade_date`` (Utf8)。
    """
    cal = _load_calendar()
    if start:
        cal = cal.filter(pl.col("cal_date") >= datetime.strptime(start, "%Y%m%d").date())
    if end:
        cal = cal.filter(pl.col("cal_date") <= datetime.strptime(end, "%Y%m%d").date())
    return cal.sort("cal_date")


def is_trade_date(d: date | str) -> bool:
    """判断 *d* 是否为交易日。

    Parameters
    ----------
    d : date | str
        支持 ``date`` 对象或 ``YYYYMMDD`` 格式字符串。

    Returns
    -------
    bool
    """
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y%m%d").date()
    cal = _load_calendar()
    row = cal.filter(pl.col("cal_date") == d)
    if row.is_empty():
        return False
    return row["is_open"].item() == 1


def prev_trade_date(d: date | str, n: int = 1) -> date:
    """返回 *d* 之前的第 *n* 个交易日。

    Parameters
    ----------
    d : date | str
    n : int, default 1
        向前推 n 个交易日。

    Returns
    -------
    date
    """
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y%m%d").date()
    cal = _load_calendar()
    trade_dates = (
        cal.filter(pl.col("is_open") == 1, pl.col("cal_date") < d)
        .sort("cal_date", descending=True)
        .limit(n)
    )
    if len(trade_dates) < n:
        raise ValueError(f"不足 {n} 个前序交易日（从 {d} 往前）")
    return trade_dates["cal_date"].to_list()[n - 1]


def next_trade_date(d: date | str, n: int = 1) -> date:
    """返回 *d* 之后的第 *n* 个交易日。

    Parameters
    ----------
    d : date | str
    n : int, default 1
        向后推 n 个交易日。

    Returns
    -------
    date
    """
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y%m%d").date()
    cal = _load_calendar()
    trade_dates = (
        cal.filter(pl.col("is_open") == 1, pl.col("cal_date") > d).sort("cal_date").limit(n)
    )
    if len(trade_dates) < n:
        raise ValueError(f"不足 {n} 个后续交易日（从 {d} 往后）")
    return trade_dates["cal_date"].to_list()[n - 1]


def get_trade_dates(start: str, end: str) -> list[date]:
    """返回 [*start*, *end*] 区间内所有交易日列表。

    Parameters
    ----------
    start : str, format ``YYYYMMDD``
    end : str, format ``YYYYMMDD``

    Returns
    -------
    list[date]
    """
    cal = get_trade_calendar(start, end)
    return cal.filter(pl.col("is_open") == 1)["cal_date"].to_list()


def get_trading_sessions() -> list[tuple[time, time]]:
    """A 股交易时段。

    Returns
    -------
    list[tuple[time, time]]
        ``[(9:30, 11:30), (13:00, 15:00)]`` — MFT 模块使用。
    """
    return [(time(9, 30), time(11, 30)), (time(13, 0), time(15, 0))]


def get_weekly_snapshot_dates(start: str, end: str) -> list[date]:
    """返回 [start, end] 区间内每周最后一个交易日列表。

    规则: ISO 周编号分组，取每周最大值作为快照日期。
    """
    trades = get_trade_dates(start, end)
    if not trades:
        return []

    # 用 (year, iso_week) 分组，取每组最大值
    groups: dict[tuple[int, int], date] = {}
    for d in trades:
        iso = d.isocalendar()
        key = (iso[0], iso[1])
        if key not in groups or d > groups[key]:
            groups[key] = d

    return sorted(groups.values())


def get_monthly_snapshot_dates(start: str, end: str) -> list[date]:
    """返回 [start, end] 区间内每月最后一个交易日列表。

    规则: 按年-月分组，取每月最大值作为快照日期。
    """
    trades = get_trade_dates(start, end)
    if not trades:
        return []

    groups: dict[tuple[int, int], date] = {}
    for d in trades:
        key = (d.year, d.month)
        if key not in groups or d > groups[key]:
            groups[key] = d

    return sorted(groups.values())
