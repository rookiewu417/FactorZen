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
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from factorzen.config.settings import DATA_CACHE
from factorzen.core.loader import fetch_stock_basic
from factorzen.core.logger import get_logger

logger = get_logger(__name__)


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


def _load_index_members(index_code: str, date_str: str) -> list[str]:
    """从 Tushare ``index_weight`` 加载指数成分股，按月缓存。

    Parameters
    ----------
    index_code : str
        Tushare 指数代码，如 ``"000300.SH"``。
    date_str : str
        日期 ``"YYYYMMDD"``，用于确定拉取月份。

    Returns
    -------
    list[str]
        ``ts_code`` 列表（成分股代码，如 ``"000001.SZ"``）。

    Raises
    ------
    Exception
        Tushare API 调用失败时直接抛出，由调用方处理降级。
    """
    from factorzen.core.loader import _retry, init_tushare

    # 计算当月第一天及最后一天
    dt = datetime.strptime(date_str, "%Y%m%d")
    year_month = date_str[:6]
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    start_date = f"{year_month}01"
    end_date = f"{year_month}{last_day:02d}"

    safe_name = index_code.replace(".", "_")
    cache_file = DATA_CACHE / f"index_member_{safe_name}_{year_month}.parquet"
    memory_key = (str(DATA_CACHE), index_code, year_month)

    cached_members = _INDEX_MEMBER_MEMORY_CACHE.get(memory_key)
    if cached_members is not None:
        logger.info(f"[index_member] {index_code} {year_month} 内存缓存命中")
        return list(cached_members)

    if cache_file.exists():
        logger.info(f"[index_member] {index_code} {year_month} 缓存命中")
        members = _read_index_member_cache(cache_file)
        _INDEX_MEMBER_MEMORY_CACHE[memory_key] = tuple(members)
        return members

    # 从 Tushare 拉取
    pro = init_tushare()
    try:
        df_pd = _retry(
            pro.index_weight,
            index_code=index_code,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        cached = _load_latest_cached_index_members(index_code, year_month)
        if cached:
            logger.warning(
                f"[index_member] {index_code} {year_month} 拉取失败，使用最近可用成分股缓存"
            )
            _INDEX_MEMBER_MEMORY_CACHE[memory_key] = tuple(cached)
            return cached
        raise

    if df_pd is None or df_pd.empty:
        cached = _load_latest_cached_index_members(index_code, year_month)
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

    # 写入缓存
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(cache_file))
    logger.info(f"[index_member] {index_code} {year_month}: {len(df)} 只成分股，已缓存")

    members = df["con_code"].drop_nulls().to_list()
    _INDEX_MEMBER_MEMORY_CACHE[memory_key] = tuple(members)
    return members


def _read_index_member_cache(cache_file: Path) -> list[str]:
    df = pl.read_parquet(cache_file)
    if "con_code" not in df.columns:
        return []
    return df["con_code"].drop_nulls().to_list()


def _load_latest_cached_index_members(index_code: str, year_month: str) -> list[str]:
    safe_name = index_code.replace(".", "_")
    prefix = f"index_member_{safe_name}_"
    candidates: list[tuple[str, Path]] = []
    for path in DATA_CACHE.glob(f"{prefix}*.parquet"):
        month = path.stem.removeprefix(prefix)
        if len(month) == 6 and month.isdigit() and month <= year_month:
            candidates.append((month, path))

    for month, path in sorted(candidates, reverse=True):
        members = _read_index_member_cache(path)
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
# 过滤器
# ══════════════════════════════════════════════════════════


def filter_st(stocks: pl.DataFrame, date_str: str) -> pl.DataFrame:
    """剔除 ST / *ST / PT 股票。基于 name 字段包含 ``"ST"`` 或 ``"PT"``。

    Parameters
    ----------
    stocks : pl.DataFrame
        待过滤股票池，必须包含 ``name`` 列。
    date_str : str
        日期（仅用于接口签名一致性，实际不依赖）。

    Returns
    -------
    pl.DataFrame
    """
    before = len(stocks)
    result = stocks.filter(~pl.col("name").str.contains("ST|PT"))
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


def _get_board_limit(ts_code: str) -> float:
    """按板块返回单边涨跌停幅度（不含1）。主板9.8%，创业板/科创板19.8%，北交所29.8%。

    Parameters
    ----------
    ts_code : str
        股票代码，如 ``"300001.SZ"``、``"688001.SH"``。

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
    return 0.098  # 主板（含ST的4.95%由调用方另外判断）


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

        # 按板块构建每只股票的涨跌停阈值（pct_chg 单位为百分比）
        codes = daily["ts_code"].unique().to_list()
        limits = {code: _get_board_limit(code) * 100 for code in codes}
        limit_df = pl.DataFrame(
            {
                "ts_code": list(limits.keys()),
                "_limit_pct": list(limits.values()),
            }
        )

        not_limit = (
            daily.join(limit_df, on="ts_code", how="left")
            .filter(pl.col("pct_chg").abs() < pl.col("_limit_pct"))
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

        liquid = daily.filter(pl.col("amount") >= min_amount).select("ts_code").unique()
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
    - ``is_st``         : ``True`` 如果股票名含 ST 或 PT
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

    # ── is_st ──
    base = base.with_columns(
        pl.col("name").str.contains("ST|PT").alias("is_st"),
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

    # 构建每只股票的涨跌停阈值
    codes = daily["ts_code"].unique().to_list()
    limits = {code: _get_board_limit(code) * 100 for code in codes}
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
            (pl.col("pct_chg") >= pl.col("_limit_pct")).fill_null(False).alias("is_limit_up"),
            (pl.col("pct_chg") <= -pl.col("_limit_pct")).fill_null(False).alias("is_limit_down"),
        ]
    )

    base = base.join(markers, on="ts_code", how="left").with_columns(
        pl.col("is_suspended").fill_null(False),
        pl.col("is_limit_up").fill_null(False),
        pl.col("is_limit_down").fill_null(False),
    )

    return base
