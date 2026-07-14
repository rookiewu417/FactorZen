"""股票池构建与过滤系统。

提供预设股票池与可组合的过滤链：
- ``get_universe(date_str, universe_name)`` — 从 6 种预设池中选取
- ``create_universe(date_str, base, filters)`` — 自定义过滤链
- 5 个独立过滤器可单独使用

使用示例::

    from factorzen.core.universe import get_universe, create_universe

    # 预设日频股票池
    daily = get_universe("20260513", "daily_default")

    # 自定义过滤：全A + 剔除ST + 剔除次新
    custom = create_universe("20260513", filters=["st", "new_listing"])
"""

from __future__ import annotations

import calendar
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from factorzen.config.settings import DATA_CACHE
from factorzen.core.loader import fetch_namechange, fetch_stock_basic
from factorzen.core.logger import get_logger

logger = get_logger(__name__)

# 动态过滤池：过滤条件（停牌/涨跌停/流动性）本身逐日变化，不属于成分 membership 语义。
# 挖掘请用基础池（all_a / csi300 / csi500 / csi800）。
_DYNAMIC_UNIVERSES = frozenset(
    {"daily_default", "intraday_default", "lft_default", "mft_default"}
)


# ══════════════════════════════════════════════════════════
# 基础数据
# ══════════════════════════════════════════════════════════


def get_stock_basic(use_cache: bool = True) -> pl.DataFrame:
    """获取全量 A 股股票列表。

    Returns
    -------
    pl.DataFrame
        列: ts_code, symbol, name, area, industry, market, list_date, delist_date。
    """
    return fetch_stock_basic()


# ══════════════════════════════════════════════════════════
# 预设股票池
# ══════════════════════════════════════════════════════════

_UNIVERSE_REGISTRY: dict[str, str] = {
    "all_a": "全 A 股（仅上市状态，无额外过滤）",
    "csi300": "沪深 300 成分股",
    "csi500": "中证 500 成分股",
    "csi800": "沪深 300 + 中证 500",
    "daily_default": "全A → 过滤 ST/次新/停牌/涨跌停",
    "intraday_default": "daily_default → 流动性过滤（日成交额 >= 1000 万）",
    "lft_default": "兼容别名：daily_default",
    "mft_default": "兼容别名：intraday_default",
}

# universe_name → Tushare index_code
_INDEX_CODE_MAP: dict[str, str] = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
}

_INDEX_MEMBER_MEMORY_CACHE: dict[tuple[str, str, str], tuple[str, ...]] = {}


# ══════════════════════════════════════════════════════════
# 指数成分股加载
# ══════════════════════════════════════════════════════════


def _members_as_of(df: pl.DataFrame, date_str: str) -> list[str]:
    """从指数成分股原始数据中按 ``trade_date`` 精确截取 ``date_str`` 当天有效的成分股。

    取 ``trade_date <= date_str`` 中**最近一个 trade_date** 对应的 ``con_code``
    集合，而不是整批/整月数据的并集——避免在调样生效日（6月/12月中旬等）前就
    看到尚未生效的新成分（look-ahead bias）。

    Parameters
    ----------
    df : pl.DataFrame
        Tushare ``index_weight`` 原始返回（或其月度缓存），需含 ``trade_date``、
        ``con_code`` 列。
    date_str : str
        交易日 ``"YYYYMMDD"``。

    Returns
    -------
    list[str]
        ``date_str`` 当天有效的 ``con_code`` 列表；若无 ``trade_date <= date_str``
        的记录（或缺少必需列）则为空列表。
    """
    if "con_code" not in df.columns or "trade_date" not in df.columns:
        return []

    eligible = df.filter(pl.col("trade_date").cast(pl.Utf8) <= date_str)
    if eligible.is_empty():
        return []

    latest_trade_date = eligible["trade_date"].cast(pl.Utf8).max()
    return (
        eligible.filter(pl.col("trade_date").cast(pl.Utf8) == latest_trade_date)["con_code"]
        .drop_nulls()
        .to_list()
    )


def _load_index_members(index_code: str, date_str: str) -> list[str]:
    """从 Tushare ``index_weight`` 加载指数成分股，按月缓存、按日精确截取。

    Parameters
    ----------
    index_code : str
        Tushare 指数代码，如 ``"000300.SH"``。
    date_str : str
        日期 ``"YYYYMMDD"``，用于确定拉取月份，并精确截取该日生效的成分股。

    Returns
    -------
    list[str]
        ``ts_code`` 列表（成分股代码，如 ``"000001.SZ"``），按 ``trade_date <=
        date_str`` 截取自最近一次调样后的集合，不包含尚未生效的未来调样结果。

    Raises
    ------
    Exception
        Tushare API 调用失败时直接抛出，由调用方处理降级。
    """
    from factorzen.core.loader import _retry, init_tushare

    # 计算当月第一天及最后一天（月度缓存粒度，减少 Tushare 调用次数）
    dt = datetime.strptime(date_str, "%Y%m%d")
    year_month = date_str[:6]
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    start_date = f"{year_month}01"
    end_date = f"{year_month}{last_day:02d}"

    safe_name = index_code.replace(".", "_")
    cache_file = DATA_CACHE / f"index_member_{safe_name}_{year_month}.parquet"
    # 内存缓存按精确 date_str 区分（而非 year_month）：同一月内调样前后成分不同，
    # 若按月共享会导致调样生效日前后的查询互相串用过期/超前结果。
    memory_key = (str(DATA_CACHE), index_code, date_str)

    cached_members = _INDEX_MEMBER_MEMORY_CACHE.get(memory_key)
    if cached_members is not None:
        logger.info(f"[index_member] {index_code} {date_str} 内存缓存命中")
        return list(cached_members)

    if cache_file.exists():
        logger.info(f"[index_member] {index_code} {year_month} 缓存命中")
        cached_df = _read_index_member_cache(cache_file)
        members = _members_as_of(cached_df, date_str)
        _INDEX_MEMBER_MEMORY_CACHE[memory_key] = tuple(members)
        return members

    # 从 Tushare 拉取（按月范围，原始数据含 trade_date，供按日精确截取复用）
    pro = init_tushare()
    try:
        df_pd = _retry(
            pro.index_weight,
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        cached = _load_latest_cached_index_members(index_code, date_str)
        if cached:
            logger.warning(
                f"[index_member] {index_code} {year_month} 拉取失败，使用最近可用成分股缓存"
            )
            _INDEX_MEMBER_MEMORY_CACHE[memory_key] = tuple(cached)
            return cached
        raise

    if df_pd is None or df_pd.empty:
        cached = _load_latest_cached_index_members(index_code, date_str)
        if cached:
            logger.warning(
                f"[index_member] {index_code} {year_month} 无成分股数据，使用最近可用成分股缓存"
            )
            _INDEX_MEMBER_MEMORY_CACHE[memory_key] = tuple(cached)
            return cached
        logger.warning(f"[index_member] {index_code} {year_month} 无成分股数据")
        _INDEX_MEMBER_MEMORY_CACHE[memory_key] = ()
        return []

    df = pl.from_pandas(df_pd)

    # 写入缓存（原始月度数据，含 trade_date，供后续按日精确截取复用）
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(cache_file))
    logger.info(f"[index_member] {index_code} {year_month}: {len(df)} 条原始记录，已缓存")

    members = _members_as_of(df, date_str)
    if not members:
        # 当月数据非空，但没有任何 trade_date<=date_str 的记录（如当月首个快照
        # 本身就晚于查询日）：不能当成"该指数当月无成分股"直接返回空列表，须
        # 与拉取异常/拉取结果整体为空这两个分支一致，尝试回退到历史月份缓存。
        cached = _load_latest_cached_index_members(index_code, date_str)
        if cached:
            logger.warning(
                f"[index_member] {index_code} {year_month} 当月数据无 "
                f"trade_date<={date_str} 的记录，使用最近可用成分股缓存"
            )
            _INDEX_MEMBER_MEMORY_CACHE[memory_key] = tuple(cached)
            return cached
    logger.info(f"[index_member] {index_code} {date_str}: 截取 {len(members)} 只成分股")
    _INDEX_MEMBER_MEMORY_CACHE[memory_key] = tuple(members)
    return members


def _read_index_member_cache(cache_file: Path) -> pl.DataFrame:
    """读取月度成分股缓存文件的原始数据（含 trade_date，供按日精确截取复用）。"""
    return pl.read_parquet(cache_file)


def _load_latest_cached_index_members(index_code: str, date_str: str) -> list[str]:
    """当月数据不可用时，回退到最近一个有缓存的历史月份。

    按 ``trade_date <= date_str`` 截取该历史月份中最近一次调样后的成分股
    （而非整月并集），与 ``_load_index_members`` 的截取口径保持一致。
    """
    year_month = date_str[:6]
    safe_name = index_code.replace(".", "_")
    prefix = f"index_member_{safe_name}_"
    candidates: list[tuple[str, Path]] = []
    for path in DATA_CACHE.glob(f"{prefix}*.parquet"):
        month = path.stem.removeprefix(prefix)
        if len(month) == 6 and month.isdigit() and month <= year_month:
            candidates.append((month, path))

    for month, path in sorted(candidates, reverse=True):
        members = _members_as_of(_read_index_member_cache(path), date_str)
        if members:
            logger.info(f"[index_member] {index_code} {year_month} 回退到 {month} 缓存")
            return members
    return []


def get_universe(
    date_str: str,
    universe_name: str = "all_a",
) -> pl.DataFrame:
    """获取指定日期的预设股票池。

    Parameters
    ----------
    date_str : str
        日期，格式 ``"YYYYMMDD"``。
    universe_name : str, default ``"all_a"``
        股票池名称，可选值:

        - ``"all_a"``: 全 A 股（仅上市状态，无额外过滤）
        - ``"csi300"``: 沪深 300 成分股（Tushare 动态拉取）
        - ``"csi500"``: 中证 500 成分股（Tushare 动态拉取）
        - ``"csi800"``: 沪深 300 + 中证 500（csi300 ∪ csi500）
        - ``"daily_default"``: 全A → 过滤 ST/次新/停牌/涨跌停
        - ``"intraday_default"``: daily_default → 流动性过滤
        - ``"lft_default"`` / ``"mft_default"``: 旧命名兼容别名

    Returns
    -------
    pl.DataFrame
        列: ts_code, symbol, name, area, industry, market, list_date。
    """
    if universe_name not in _UNIVERSE_REGISTRY:
        valid = ", ".join(_UNIVERSE_REGISTRY.keys())
        raise ValueError(f"未知 universe_name: '{universe_name}'。可选: {valid}")

    logger.info(f"[universe] 获取股票池: {universe_name} ({date_str})")

    # --- all_a: PIT 全 A 股基础池 ---
    # fetch_stock_basic 默认拉全量（L+D+P），PIT 过滤保留在 date_str 时实际在市股票
    all_a = get_stock_basic()

    snapshot = datetime.strptime(date_str, "%Y%m%d").date()
    all_a = all_a.filter(
        (pl.col("list_date").is_not_null())
        & (pl.col("list_date").cast(pl.Date) <= pl.lit(snapshot))
        & (
            pl.col("delist_date").is_null()
            | (pl.col("delist_date").cast(pl.Date) > pl.lit(snapshot))
        )
    )

    if universe_name == "all_a":
        return all_a

    # --- CSI 指数成分股 ---
    if universe_name in ("csi300", "csi500", "csi800"):
        try:
            members: set[str] = set()
            for uname in ("csi300", "csi500"):
                if universe_name in (uname, "csi800"):
                    code = _INDEX_CODE_MAP[uname]
                    fresh = _load_index_members(code, date_str)
                    members.update(fresh)

            logger.info(f"[universe] {universe_name}: 加载 {len(members)} 只成分股")

            result = all_a.filter(pl.col("ts_code").is_in(list(members)))
            return result

        except Exception as e:
            logger.warning(f"[universe] {universe_name} 指数成分股加载失败 ({e})，降级为全 A 股")
            return all_a

    if universe_name == "lft_default":
        universe_name = "daily_default"
    elif universe_name == "mft_default":
        universe_name = "intraday_default"

    # --- daily_default ---
    if universe_name == "daily_default":
        result = all_a
        result = filter_st(result, date_str)
        result = filter_new_listing(result, date_str)
        result = filter_suspended(result, date_str)
        result = filter_limit(result, date_str)
        return result

    # --- intraday_default ---
    if universe_name == "intraday_default":
        result = get_universe(date_str, "daily_default")
        result = filter_liquidity(result, date_str)
        return result

    # 不应到达此处
    return all_a


# ══════════════════════════════════════════════════════════
# 逐日 PIT membership（消除期末成分幸存偏差）
# ══════════════════════════════════════════════════════════


def _year_months_in_range(start: str, end: str) -> list[str]:
    """返回 [start, end] 覆盖的自然月列表，格式 ``YYYYMM``。"""
    y, m = int(start[:4]), int(start[4:6])
    y_end, m_end = int(end[:4]), int(end[4:6])
    out: list[str] = []
    while (y, m) <= (y_end, m_end):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _month_asof_date(year_month: str, start: str, end: str) -> str:
    """某自然月用于拉取成分的 as-of 日：当月**第一天**，夹在 [start, end] 内。

    PIT 取舍：``_load_index_members`` 按 ``trade_date <= as-of`` 截取已生效成分。
    月初 as-of = 该月开盘时已生效的上次调样；月中生效的调整**滞后**到下月才反映
    ——宁滞后勿前视（月末 as-of 会让月初交易日提前看到月中调样，是前视）。
    """
    asof = f"{year_month}01"
    if asof > end:
        asof = end
    if asof < start:
        asof = start
    return asof


def membership_hash(membership: pl.DataFrame) -> str:
    """对 membership 表做内容 hash（排序后稳定），供 manifest 溯源。

    本任务只提供 hash 函数，不做 manifest 接线（后续任务接）。
    """
    if membership.is_empty():
        return hashlib.sha256(b"").hexdigest()
    sorted_df = (
        membership.select(["trade_date", "ts_code"])
        .unique()
        .sort(["trade_date", "ts_code"])
    )
    payload = "\n".join(f"{td}|{code}" for td, code in sorted_df.iter_rows())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_universe_membership(
    start: str,
    end: str,
    universe_name: str,
) -> pl.DataFrame:
    """逐日 PIT 成分 membership：``[trade_date(Utf8), ts_code]``。

    对命名指数池（csi300/csi500/csi800）按**自然月**取 ``_load_index_members``
    （月度 parquet 缓存命中则零网络），将该月成分展开到该月内、且落在
    ``[start, end]`` 的全部交易日。``csi800 = csi300 ∪ csi500`` 逐月并。

    ``all_a``：按 ``list_date`` / ``delist_date`` 构造上市区间后逐日展开
    （与 ``get_universe("all_a", date)`` 的 PIT 语义一致：``list_date <= d`` 且
    ``delist_date is null | delist_date > d``）。

    动态过滤池（``daily_default`` / ``intraday_default`` 及别名）**不支持**——
    其过滤条件（停牌/涨跌停/流动性）本身逐日变化，不属于成分 membership 语义；
    挖掘请改用基础池。

    不做的事（后续任务 / 其他路径）：
    - manifest ``membership_hash`` 接线；
    - lift/rebuild 消费路径的 membership（它们经 ``_prepare_agent_mining_data`` →
      ``prepare_mining_daily`` 间接受益）；
    - 动态过滤池的逐日语义。

    Parameters
    ----------
    start, end : str
        评估窗边界 ``"YYYYMMDD"``（membership 不覆盖预热段）。
    universe_name : str
        基础池名：``all_a`` / ``csi300`` / ``csi500`` / ``csi800``。

    Returns
    -------
    pl.DataFrame
        列 ``trade_date`` (Utf8 YYYYMMDD)、``ts_code``；可能为空表。

    Raises
    ------
    ValueError
        未知池名，或动态过滤池。
    """
    if universe_name in _DYNAMIC_UNIVERSES:
        raise ValueError(
            f"universe={universe_name!r} 是动态过滤池，不支持逐日 membership；"
            f"挖掘请用基础池 all_a / csi300 / csi500 / csi800"
        )
    if universe_name not in _UNIVERSE_REGISTRY:
        valid = ", ".join(_UNIVERSE_REGISTRY.keys())
        raise ValueError(f"未知 universe_name: '{universe_name}'。可选: {valid}")

    if start > end:
        raise ValueError(f"start ({start}) 不能晚于 end ({end})")

    if universe_name == "all_a":
        return _membership_all_a(start, end)

    if universe_name in ("csi300", "csi500", "csi800"):
        return _membership_index(start, end, universe_name)

    # 不应到达（动态池已拒）
    raise ValueError(
        f"universe={universe_name!r} 不支持逐日 membership；"
        f"请用基础池 all_a / csi300 / csi500 / csi800"
    )


def _membership_index(start: str, end: str, universe_name: str) -> pl.DataFrame:
    """指数池：按月拉成分 → 展开到该月交易日。"""
    from factorzen.core.calendar import get_trade_dates

    trade_dates = get_trade_dates(start, end)
    if not trade_dates:
        return pl.DataFrame(
            schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8}
        )

    # 交易日按月分桶
    by_month: dict[str, list[str]] = {}
    for d in trade_dates:
        ym = d.strftime("%Y%m")
        by_month.setdefault(ym, []).append(d.strftime("%Y%m%d"))

    index_names = (
        ("csi300", "csi500") if universe_name == "csi800" else (universe_name,)
    )
    parts: list[pl.DataFrame] = []
    for ym, day_strs in by_month.items():
        asof = _month_asof_date(ym, start, end)
        members: set[str] = set()
        for uname in index_names:
            code = _INDEX_CODE_MAP[uname]
            members.update(_load_index_members(code, asof))
        if not members:
            continue
        parts.append(
            pl.DataFrame({"trade_date": day_strs}).join(
                pl.DataFrame({"ts_code": sorted(members)}), how="cross"
            )
        )

    if not parts:
        return pl.DataFrame(
            schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8}
        )
    return pl.concat(parts).select(["trade_date", "ts_code"]).unique()


def _membership_all_a(start: str, end: str) -> pl.DataFrame:
    """全 A：按 list_date / delist_date 上市区间逐日展开。

    与 ``get_universe("all_a", date)`` 同口径（含退市日严格大于）。
    若基础数据无 delist_date 列，则仅按 list_date 过滤（退市不可得时的取舍）。
    """
    from factorzen.core.calendar import get_trade_dates

    trade_dates = get_trade_dates(start, end)
    if not trade_dates:
        return pl.DataFrame(
            schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8}
        )

    basic = get_stock_basic()
    start_d = datetime.strptime(start, "%Y%m%d").date()
    end_d = datetime.strptime(end, "%Y%m%d").date()

    # 窗口内可能在市的股票（粗滤，减小 cross join）
    stocks = basic.filter(
        pl.col("list_date").is_not_null()
        & (pl.col("list_date").cast(pl.Date) <= pl.lit(end_d))
    )
    if "delist_date" in stocks.columns:
        stocks = stocks.filter(
            pl.col("delist_date").is_null()
            | (pl.col("delist_date").cast(pl.Date) > pl.lit(start_d))
        )

    if stocks.is_empty():
        return pl.DataFrame(
            schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8}
        )

    td = pl.DataFrame(
        {
            "trade_date": [d.strftime("%Y%m%d") for d in trade_dates],
            "_td": trade_dates,
        }
    )
    sel_cols = ["ts_code", "list_date"]
    if "delist_date" in stocks.columns:
        sel_cols.append("delist_date")

    joined = stocks.select(sel_cols).join(td, how="cross")
    mask = pl.col("list_date").cast(pl.Date) <= pl.col("_td")
    if "delist_date" in joined.columns:
        mask = mask & (
            pl.col("delist_date").is_null()
            | (pl.col("delist_date").cast(pl.Date) > pl.col("_td"))
        )
    return (
        joined.filter(mask)
        .select(["trade_date", "ts_code"])
        .unique()
    )


# ══════════════════════════════════════════════════════════
# 自定义股票池
# ══════════════════════════════════════════════════════════


def create_universe(
    date_str: str,
    base: str = "all_a",
    filters: list[str] | None = None,
    min_days: int = 250,
    min_amount: float = 10_000_000,
) -> pl.DataFrame:
    """自定义股票池：指定基础池 + 过滤链。

    Parameters
    ----------
    date_str : str
        日期，格式 ``"YYYYMMDD"``。
    base : str, default ``"all_a"``
        基础池名称，同 ``get_universe`` 的 ``universe_name`` 参数。
    filters : list[str] | None, optional
        过滤链，按顺序依次应用。可选过滤项:

        - ``"st"``: 剔除 ST / *ST / PT
        - ``"new_listing"``: 剔除上市不足 ``min_days`` 天的次新股
        - ``"suspended"``: 剔除当日停牌股票
        - ``"limit"``: 剔除当日涨跌停股票
        - ``"liquidity"``: 剔除日成交额低于 ``min_amount`` 的股票

        为 ``None`` 或空列表时仅返回基础池，不做过滤。
    min_days : int, default 250
        ``"new_listing"`` 过滤的最小上市天数（自然日）。
    min_amount : float, default 10_000_000
        ``"liquidity"`` 过滤的最低日成交额（元）。

    Returns
    -------
    pl.DataFrame
    """
    result = get_universe(date_str, base)

    if not filters:
        logger.info(f"[universe] create_universe: base={base}, 无过滤链")
        return result

    filter_map = {
        "st": filter_st,
        "new_listing": lambda df, ds: filter_new_listing(df, ds, min_days=min_days),
        "suspended": filter_suspended,
        "limit": filter_limit,
        "liquidity": lambda df, ds: filter_liquidity(df, ds, min_amount=min_amount),
    }

    for f_name in filters:
        if f_name not in filter_map:
            logger.warning(f"[universe] 未知过滤项 '{f_name}'，跳过")
            continue
        func = filter_map[f_name]
        result = func(result, date_str)
        logger.debug(f"[universe] 应用过滤 '{f_name}' 后: {len(result)} 只")

    logger.info(
        f"[universe] create_universe: base={base}, filters={filters}, 最终 {len(result)} 只"
    )
    return result


# ══════════════════════════════════════════════════════════
# PIT ST 状态判定（基于 namechange 曾用名变更记录）
# ══════════════════════════════════════════════════════════

_namechange_unavailable_warned = False


def _safe_fetch_namechange() -> pl.DataFrame | None:
    """安全获取 namechange 全量数据。

    获取失败（无 token / 网络异常 / 接口异常等任意原因）时返回 ``None``，不向
    上抛出异常；仅在进程内警告一次，避免日志刷屏。

    Returns
    -------
    pl.DataFrame | None
        成功时返回 ``fetch_namechange()`` 结果；失败时返回 ``None``。
    """
    global _namechange_unavailable_warned
    try:
        return fetch_namechange()
    except Exception as e:
        if not _namechange_unavailable_warned:
            logger.warning(
                f"[st] namechange 数据获取失败（{e}），ST 判断降级为按当前 name "
                "字符串匹配（非 PIT 模式，本进程内仅警告一次）"
            )
            _namechange_unavailable_warned = True
        return None


def _is_st_asof(
    codes: list[str],
    trade_date: str,
    namechange_df: pl.DataFrame,
) -> set[str]:
    """基于 namechange 曾用名记录，判断给定股票在 ``trade_date`` 当天是否处于 ST 状态（PIT）。

    直接判断该记录对应的历史 ``name`` 是否含 "ST"/"PT"（与非 PIT 降级路径
    ``_st_codes_by_name`` 的判断口径一致，只是这里按 ``[start_date, end_date)``
    区间取历史某一时刻的 name，而非当前 name），按该区间判断（``end_date``
    为空表示状态持续至今）。

    2026-07-01 用真实 Tushare token 核对 ``change_reason`` 实际取值分布后发现：
    早期版本按 ``change_reason`` 关键词过滤（含"ST"且不含"撤销"/"摘星"）存在
    漏判——``change_reason="摘星"``（从 *ST 降级为 ST，仍是 ST 状态，不是摘帽）
    对应记录的 ``name`` 仍以 "ST" 开头（如 "ST沈机"），但 ``change_reason``
    字符串本身不含 "ST" 子串，导致这类记录被误判为非 ST。真实数据抽样统计：
    全量约 2.7%（269/10000）的 namechange 记录受此影响。改为直接检查 ``name``
    后不再有此类误判。

    Parameters
    ----------
    codes : list[str]
        待判断的股票代码列表。
    trade_date : str
        判断基准日期 ``"YYYYMMDD"``。
    namechange_df : pl.DataFrame
        ``fetch_namechange()`` 返回的全量曾用名变更记录，需含列
        ``ts_code, name, start_date, end_date``。

    Returns
    -------
    set[str]
        在 ``trade_date`` 当天处于 ST/\\*ST 状态的股票代码集合。
    """
    if namechange_df.is_empty() or not codes:
        return set()

    target_date = datetime.strptime(trade_date, "%Y%m%d").date()
    name = pl.col("name").fill_null("")

    st_records = (
        namechange_df.filter(pl.col("ts_code").is_in(codes))
        .with_columns(
            pl.col("start_date").cast(pl.Date),
            pl.col("end_date").cast(pl.Date),
        )
        .filter(
            name.str.contains("ST|PT")
            & pl.col("start_date").is_not_null()
            & (pl.col("start_date") <= pl.lit(target_date))
            & (pl.col("end_date").is_null() | (pl.col("end_date") > pl.lit(target_date)))
        )
    )
    return set(st_records["ts_code"].unique().to_list())


def _st_codes_by_name(name_source: pl.DataFrame, codes: list[str]) -> set[str]:
    """按『当前』``name`` 字段判断 ST/PT（非 PIT，日期无关）。

    仅作为 namechange 不可用时的降级方案，供 ``_resolve_st_codes`` 与
    ``build_is_st_by_date`` 共用。
    """
    if "name" not in name_source.columns or not codes:
        return set()
    return set(
        name_source.filter(
            pl.col("ts_code").is_in(codes) & pl.col("name").str.contains("ST|PT")
        )["ts_code"].to_list()
    )


def _resolve_st_codes(
    name_source: pl.DataFrame,
    codes: list[str],
    date_str: str,
) -> set[str]:
    """解析 ``codes`` 在 ``date_str`` 当天处于 ST 状态的子集。

    优先使用 ``namechange`` 曾用名记录做 PIT 正确判断；若获取失败，优雅降级
    为按 ``name_source`` 的 ``name`` 列是否含 ``"ST"``/``"PT"`` 判断（非 PIT，
    但保证离线可用、不崩溃）。降级时仅警告一次由 ``_safe_fetch_namechange``
    负责。

    Parameters
    ----------
    name_source : pl.DataFrame
        含 ``ts_code``、``name`` 列的股票池，仅在降级模式下使用。
    codes : list[str]
        待判断的股票代码全集。
    date_str : str
        判断基准日期 ``"YYYYMMDD"``。

    Returns
    -------
    set[str]
        处于 ST 状态的股票代码集合。
    """
    namechange_df = _safe_fetch_namechange()
    if namechange_df is not None:
        return _is_st_asof(codes, date_str, namechange_df)
    return _st_codes_by_name(name_source, codes)


def build_is_st_by_date(
    codes: list[str],
    trade_dates: list[date],
    name_source: pl.DataFrame | None = None,
) -> dict[date, set[str]]:
    """为整段回测窗口批量构建 ``execution_date -> 当日 ST 状态代码集合``。

    供 ``factorzen.daily.evaluation.backtest.run_strategy_backtest`` 的
    ``is_st_by_date`` 形参使用，让执行约束层能对 ST 股票收窄涨跌停阈值（见
    ``_get_board_limit``）。只拉取一次 namechange 全量数据（``fetch_namechange``
    内部走 7 天磁盘缓存），在内存中对每个交易日切片判断 PIT 状态，避免对每个
    交易日重复触发一次磁盘 I/O。

    Parameters
    ----------
    codes : list[str]
        回测涉及的全部股票代码。
    trade_dates : list[date]
        回测窗口内的全部交易日，须与调用方传给
        ``run_strategy_backtest`` 的 ``price_df`` 的 ``trade_date`` 列同为
        ``datetime.date`` 取值，才能在查表时正确命中。
    name_source : pl.DataFrame | None, optional
        含 ``ts_code``、``name`` 列的股票池；仅在 namechange 不可用时用于
        降级判断（按『当前』名称是否含 ST/PT，对所有交易日一致）。为
        ``None`` 且 namechange 不可用时返回空 dict，等价于
        ``is_st_by_date=None`` 的既有行为（一律按非 ST 阈值判断）。

    Returns
    -------
    dict[date, set[str]]
    """
    namechange_df = _safe_fetch_namechange()
    if namechange_df is not None:
        return {
            d: _is_st_asof(codes, d.strftime("%Y%m%d"), namechange_df) for d in trade_dates
        }
    if name_source is None:
        return {}
    fallback_codes = _st_codes_by_name(name_source, codes)
    if not fallback_codes:
        return {}
    return dict.fromkeys(trade_dates, fallback_codes)


# ══════════════════════════════════════════════════════════
# 过滤器
# ══════════════════════════════════════════════════════════


def filter_st(stocks: pl.DataFrame, date_str: str) -> pl.DataFrame:
    """剔除 ST / *ST / PT 股票。

    优先使用 ``namechange`` 曾用名记录做 PIT 正确的 ST 状态判断（即
    ``date_str`` 当天是否处于 ST/\\*ST 状态，而非按当前最新名称判断）；若获取
    失败（无 token / 网络异常等任意原因），优雅降级为按当前 ``name`` 字段是否
    含 ``"ST"``/``"PT"`` 判断（非 PIT，但保证离线可用、不崩溃）。

    Parameters
    ----------
    stocks : pl.DataFrame
        待过滤股票池，必须包含 ``ts_code``、``name`` 列。
    date_str : str
        判断基准日期 ``"YYYYMMDD"``，用于 PIT 切片。

    Returns
    -------
    pl.DataFrame
    """
    before = len(stocks)
    st_codes = _resolve_st_codes(stocks, stocks["ts_code"].to_list(), date_str)
    result = stocks.filter(~pl.col("ts_code").is_in(list(st_codes)))
    after = len(result)
    if before > after:
        logger.info(f"[filter_st] 剔除 {before - after} 只 ST/PT 股票")
    return result


def filter_new_listing(
    stocks: pl.DataFrame,
    date_str: str,
    min_days: int = 250,
) -> pl.DataFrame:
    """剔除上市不足 ``min_days`` 个自然日的次新股。

    Parameters
    ----------
    stocks : pl.DataFrame
        待过滤股票池，必须包含 ``list_date`` 列（``pl.Date`` 类型）。
    date_str : str
        基准日期 ``"YYYYMMDD"``。
    min_days : int, default 250
        最小上市天数（自然日）。

    Returns
    -------
    pl.DataFrame
    """
    target_date = datetime.strptime(date_str, "%Y%m%d").date()
    cutoff = target_date - timedelta(days=min_days)

    before = len(stocks)
    # list_date 已由 fetch_stock_basic 转换为 pl.Date
    result = stocks.filter(pl.col("list_date") <= cutoff)
    after = len(result)
    if before > after:
        logger.info(f"[filter_new_listing] 剔除 {before - after} 只次新股 (上市日期 > {cutoff})")
    return result


def filter_suspended(stocks: pl.DataFrame, date_str: str) -> pl.DataFrame:
    """剔除当日停牌股票。通过检查当日日线数据中 ``vol > 0`` 的股票。

    如果无法读取日线数据（数据未准备好等），优雅降级：不过滤。

    Parameters
    ----------
    stocks : pl.DataFrame
        待过滤股票池，必须包含 ``ts_code`` 列。
    date_str : str
        交易日 ``"YYYYMMDD"``。

    Returns
    -------
    pl.DataFrame
    """
    from factorzen.core.storage import load_parquet

    try:
        daily = load_parquet("daily", start=date_str, end=date_str).collect()
        if daily.is_empty():
            logger.warning(f"[filter_suspended] {date_str} 无日线数据，不过滤")
            return stocks

        active = daily.filter(pl.col("vol") > 0).select("ts_code").unique()
        before = len(stocks)
        result = stocks.join(active, on="ts_code", how="inner")
        after = len(result)
        if before > after:
            logger.info(f"[filter_suspended] 剔除 {before - after} 只停牌股票")
        return result

    except Exception as e:
        logger.warning(f"[filter_suspended] 读取日线失败 ({e})，优雅降级：不过滤")
        return stocks


def _get_board_limit(ts_code: str, is_st: bool = False) -> float:
    """按板块返回单边涨跌停幅度（不含1）。

    主板 9.8%（ST/\\*ST 收窄为 4.8%），创业板/科创板 19.8%，北交所 29.8%。

    Parameters
    ----------
    ts_code : str
        股票代码，如 ``"300001.SZ"``、``"688001.SH"``。
    is_st : bool, default False
        是否为 ST/\\*ST 股票。仅影响主板：真实涨跌幅限制为 5%，这里与其余
        板块阈值同样的容差处理方式保持一致（nominal - 0.2pp，对应
        9.8%/19.8%/29.8% 的构造方式），返回 4.8%。创业板/科创板 2020 年
        注册制改革后 ST 与非 ST 股票涨跌幅规则相同（统一 20%），北交所同理
        统一 30%，均不受此参数影响。

    Returns
    -------
    float
        涨跌停幅度小数（例如 0.098 表示 9.8%）。
    """
    code = ts_code.upper()
    if code.startswith("300") or code.startswith("301"):  # 创业板
        return 0.198
    if code.startswith("688") or code.startswith("689"):  # 科创板
        return 0.198
    if code.endswith(".BJ"):  # 北交所
        return 0.298
    if is_st:
        return 0.048  # 主板 ST/*ST：5% 真实限额 - 0.2pp 容差（与其余板块一致）
    return 0.098  # 主板


def filter_limit(stocks: pl.DataFrame, date_str: str) -> pl.DataFrame:
    """剔除当日涨跌停股票（按板块细化阈值）。

    使用 ``pct_chg`` 阈值判断（pct_chg 单位为百分比，如 9.8 表示 9.8%）：
    - 创业板/科创板：±19.8%
    - 北交所：±29.8%
    - 主板：±9.8%

    如果无法读取日线数据，优雅降级：不过滤。

    Parameters
    ----------
    stocks : pl.DataFrame
        待过滤股票池，必须包含 ``ts_code`` 列。
    date_str : str
        交易日 ``"YYYYMMDD"``。

    Returns
    -------
    pl.DataFrame
    """
    from factorzen.core.storage import load_parquet

    try:
        daily = load_parquet("daily", start=date_str, end=date_str).collect()
        if daily.is_empty():
            logger.warning(f"[filter_limit] {date_str} 无日线数据，不过滤")
            return stocks

        # 按板块 + ST 状态（PIT）构建每只股票的涨跌停阈值（pct_chg 单位为百分比）
        codes = daily["ts_code"].unique().to_list()
        st_codes = _resolve_st_codes(stocks, codes, date_str)
        limits = {code: _get_board_limit(code, is_st=(code in st_codes)) * 100 for code in codes}
        limit_df = pl.DataFrame(
            {
                "ts_code": list(limits.keys()),
                "_limit_pct": list(limits.values()),
            }
        )

        not_limit = (
            daily.join(limit_df, on="ts_code", how="left")
            # 浮点容差与 backtest.py（_apply_trade_constraints / 快路径）保持一致：
            # 创业板 open=11.98/pre_close=10.0 → pct_chg=19.7999...997（非字面量
            # 19.8），若不减 1e-9 则 abs(pct_chg) < limit_pct 为 True，涨停股被
            # 误判为「未到涨停」而漏过滤。
            .filter(pl.col("pct_chg").abs() < pl.col("_limit_pct") - 1e-9)
            .select("ts_code")
            .unique()
        )
        before = len(stocks)
        result = stocks.join(not_limit, on="ts_code", how="inner")
        after = len(result)
        if before > after:
            logger.info(f"[filter_limit] 剔除 {before - after} 只涨跌停股票")
        return result

    except Exception as e:
        logger.warning(f"[filter_limit] 读取日线失败 ({e})，优雅降级：不过滤")
        return stocks


def filter_liquidity(
    stocks: pl.DataFrame,
    date_str: str,
    min_amount: float = 10_000_000.0,
) -> pl.DataFrame:
    """剔除日成交额低于 ``min_amount`` 的股票（中频交易用，默认 1000 万）。

    如果无法读取日线数据，优雅降级：不过滤。

    Parameters
    ----------
    stocks : pl.DataFrame
        待过滤股票池，必须包含 ``ts_code`` 列。
    date_str : str
        交易日 ``"YYYYMMDD"``。
    min_amount : float, default 10_000_000.0
        最低日成交额（元）。

    Returns
    -------
    pl.DataFrame
    """
    from factorzen.core.storage import load_parquet

    try:
        daily = load_parquet("daily", start=date_str, end=date_str).collect()
        if daily.is_empty():
            logger.warning(f"[filter_liquidity] {date_str} 无日线数据，不过滤")
            return stocks

        # Tushare daily.amount 单位是千元，min_amount 语义是元 → 先换算再比较，
        # 否则等价于要求 1000×min_amount 元（默认 1000万→假门槛 100 亿，股票池塌缩）。
        liquid = (
            daily.filter(pl.col("amount") * 1000.0 >= min_amount).select("ts_code").unique()
        )
        before = len(stocks)
        result = stocks.join(liquid, on="ts_code", how="inner")
        after = len(result)
        if before > after:
            logger.info(
                f"[filter_liquidity] 剔除 {before - after} 只低流动性股票 "
                f"(amount < {min_amount:.0f})"
            )
        return result

    except Exception as e:
        logger.warning(f"[filter_liquidity] 读取日线失败 ({e})，优雅降级：不过滤")
        return stocks


# ══════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════


def get_index_members(index_code: str, date_str: str) -> pl.DataFrame:
    """获取指数成分股。

    先从 Tushare ``index_weight`` 拉取成分代码，再与全量股票信息 join
    以保留 ``name``、``industry`` 等字段。

    Parameters
    ----------
    index_code : str
        指数代码，如 ``"000300.SH"``（沪深 300）。
    date_str : str
        日期 ``"YYYYMMDD"``。

    Returns
    -------
    pl.DataFrame
        成分股信息，列: ts_code, symbol, name, area, industry, market, list_date。
    """
    try:
        codes = _load_index_members(index_code, date_str)
        if not codes:
            logger.warning(f"[universe] {index_code} {date_str[:6]} 无成分股，返回全市场")
            return get_universe(date_str, "all_a")

        all_a = get_stock_basic()
        result = all_a.filter(pl.col("ts_code").is_in(codes))
        return result

    except Exception as e:
        logger.warning(f"[universe] {index_code} 成分股加载失败 ({e})，返回全市场")
        return get_universe(date_str, "all_a")


# ══════════════════════════════════════════════════════════
# 股票池快照（含微观结构标记）
# ══════════════════════════════════════════════════════════


def get_universe_snapshot(
    date_str: str,
    universe_name: str = "all_a",
) -> pl.DataFrame:
    """获取指定日期股票池快照，附带微观结构布尔标记列。

    在 ``get_universe`` 返回的基础池上追加以下列，方便下游（风险模型、
    组合优化器等）直接消费，无需再重复调用各过滤函数。

    追加列
    ------
    - ``is_st``         : ``True`` 如果 ``date_str`` 当天处于 ST/\\*ST 状态
      （PIT，基于 namechange；namechange 不可用时降级为按当前 name 含
      ST/PT 判断）
    - ``is_suspended``  : ``True`` 如果当日成交量 == 0
    - ``is_limit_up``   : ``True`` 如果当日涨幅 >= 板块涨停阈值
    - ``is_limit_down`` : ``True`` 如果当日跌幅 <= -板块跌停阈值
    - ``is_new_listing``: ``True`` 如果上市不足 250 个自然日

    Parameters
    ----------
    date_str : str
        交易日 ``"YYYYMMDD"``。
    universe_name : str, default ``"all_a"``
        基础股票池名称，同 ``get_universe``。

    Returns
    -------
    pl.DataFrame
        在基础池列之上追加 5 列布尔标记。

    Notes
    -----
    当日线数据不可用时，``is_suspended`` / ``is_limit_up`` / ``is_limit_down``
    均填充为 ``False``（优雅降级）。
    """
    base = get_universe(date_str, universe_name)

    # ── is_st（PIT：优先 namechange，失败优雅降级为按当前 name 匹配）──
    st_codes = _resolve_st_codes(base, base["ts_code"].to_list(), date_str)
    base = base.with_columns(
        pl.col("ts_code").is_in(list(st_codes)).alias("is_st"),
    )

    # ── is_new_listing ──
    target_date = datetime.strptime(date_str, "%Y%m%d").date()
    cutoff = target_date - timedelta(days=250)
    base = base.with_columns(
        (pl.col("list_date") > cutoff).alias("is_new_listing"),
    )

    # ── 日线数据相关标记 ──
    from factorzen.core.storage import load_parquet

    try:
        daily = load_parquet("daily", start=date_str, end=date_str).collect()
    except Exception as e:
        logger.warning(f"[universe_snapshot] 读取日线失败 ({e})，相关标记填 False")
        daily = pl.DataFrame()

    if daily.is_empty():
        base = base.with_columns(
            pl.lit(False).alias("is_suspended"),
            pl.lit(False).alias("is_limit_up"),
            pl.lit(False).alias("is_limit_down"),
        )
        return base

    # 构建每只股票的涨跌停阈值（ST 主板复用上面已解析的 st_codes）
    codes = daily["ts_code"].unique().to_list()
    limits = {code: _get_board_limit(code, is_st=(code in st_codes)) * 100 for code in codes}
    limit_df = pl.DataFrame(
        {
            "ts_code": list(limits.keys()),
            "_limit_pct": list(limits.values()),
        }
    )

    daily_with_limit = daily.join(limit_df, on="ts_code", how="left")

    markers = daily_with_limit.select(
        [
            "ts_code",
            (pl.col("vol") == 0).fill_null(True).alias("is_suspended"),
            # 浮点容差与 filter_limit / backtest.py 保持一致，避免
            # 19.7999...997 这类浮点舍入误差导致涨跌停状态漏判。
            (pl.col("pct_chg") >= pl.col("_limit_pct") - 1e-9)
            .fill_null(False)
            .alias("is_limit_up"),
            (pl.col("pct_chg") <= -pl.col("_limit_pct") + 1e-9)
            .fill_null(False)
            .alias("is_limit_down"),
        ]
    )

    # A股停牌股当日 Tushare 不发日线行 → 左 join 后 markers 列为 null。
    # is_suspended 的 null 语义 = 无日线行 = 当日不可交易 → 判停牌（True）;
    # 只有 vol==0 的显式零量行才靠 markers 判——仅此漏了主流「无行」停牌。
    # 涨跌停无行时保持 False（无价无法判涨跌停）。
    base = base.join(markers, on="ts_code", how="left").with_columns(
        pl.col("is_suspended").fill_null(True),
        pl.col("is_limit_up").fill_null(False),
        pl.col("is_limit_down").fill_null(False),
    )

    return base
