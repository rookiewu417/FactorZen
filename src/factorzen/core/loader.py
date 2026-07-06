"""Tushare 数据拉取桥接层。

这是整个项目中唯一与 Tushare 直接交互的模块。
所有 raw 层数据获取走这里，上层模块通过此模块获取数据，不直接调用 tushare。

设计原则：
- 分段拉取（日线按年，分钟按月，财报按季度）
- 缓存优先（每段先检查 cache，命中跳过）
- 限流（_rate_limit 确保每秒不超过 MAX_RPS 次）
- 重试（仅网络错误重试，参数/权限错误立即失败）
- pandas → polars 转换（拿到数据后立即转换）
"""

from datetime import datetime, timedelta
from typing import Any

import polars as pl
import tushare as ts

from factorzen.config.settings import DATA_CACHE, DATA_RAW
from factorzen.config.tushare_config import (
    CACHE_EXPIRE_DAYS,
    MAX_RETRIES,
    MAX_RPS,
    RETRY_DELAY,
    ensure_token,
)
from factorzen.core.logger import get_logger
from factorzen.core.storage import load_parquet, partition_exists, save_parquet

logger = get_logger(__name__)

# ── 模块级状态 ──────────────────────────────────────────
_pro: Any = None  # Tushare Pro API 单例
_last_call: float = 0.0  # 上次 API 调用时间戳


# ══════════════════════════════════════════════════════════
# 基础设施
# ══════════════════════════════════════════════════════════


def init_tushare() -> ts.pro_api:
    """初始化 Tushare Pro API 客户端（单例）。

    Returns:
        ts.pro_api: 初始化后的 Tushare Pro API 实例。
    """
    global _pro
    if _pro is None:
        ts.set_token(ensure_token())
        _pro = ts.pro_api()
        logger.info("Tushare Pro API 初始化完成")
    return _pro


def _rate_limit() -> None:
    """确保每次 API 调用间隔 >= 1/MAX_RPS 秒。"""
    global _last_call
    import time

    elapsed = time.time() - _last_call
    min_interval = 1.0 / MAX_RPS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call = time.time()


def _retry(func: Any, *args: Any, **kwargs: Any) -> Any:
    """带重试的 Tushare 调用。

    仅网络/超时类错误重试，参数/权限错误立即失败。
    空结果触发重试（最多 MAX_RETRIES 次）。

    Args:
        func: Tushare API 方法（如 pro.daily）。
        *args: 位置参数。
        **kwargs: 关键字参数。

    Returns:
        Tushare API 返回的原始结果（pandas DataFrame 或类似对象）。

    Raises:
        RuntimeError: 达到重试上限后仍失败。
        特定异常: 参数/权限错误立即抛出。
    """
    import time

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            _rate_limit()
            result = func(*args, **kwargs)
            # Tushare 返回 None 或空 pd.DataFrame 表示无数据
            if result is not None and not (hasattr(result, "empty") and result.empty):
                return result
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # 参数/权限错误不重试
            if any(x in msg for x in ("参数", "param", "权限", "token", "积分")):
                raise
            if attempt < MAX_RETRIES:
                # stk_mins 频率超限：等足 62s（跨越 2次/分钟 固定窗口边界）
                if "频率超限" in str(e) or "频率" in msg:
                    wait = 62.0
                else:
                    wait = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Tushare 调用失败 (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                time.sleep(wait)
    raise last_exc or RuntimeError("Tushare 返回空结果")


def _str_to_date(series: pl.Expr, fmt: str = "%Y%m%d") -> pl.Expr:
    """将字符串日期列转换为 pl.Date 类型。"""
    return series.str.strptime(pl.Date, fmt, strict=False)


# ══════════════════════════════════════════════════════════
# 数据拉取函数
# ══════════════════════════════════════════════════════════


def fetch_daily(
    start: str,
    end: str,
    ts_codes: list[str] | None = None,
) -> pl.DataFrame:
    """拉取日线行情。按年分段，自动缓存。

    Args:
        start: 起始日期 "YYYYMMDD"。
        end: 截止日期 "YYYYMMDD"。
        ts_codes: 股票代码列表。为 None 时拉取全市场。

    Returns:
        pl.DataFrame，包含列:
        trade_date, ts_code, open, high, low, close,
        pre_close, change, pct_chg, vol, amount。
    """
    pro = init_tushare()
    start_year = int(start[:4])
    end_year = int(end[:4])

    for year in range(start_year, end_year + 1):
        # 缓存检查：该年份是否已有数据（用每季度第一个月作为检查点）
        if all(partition_exists("daily", year, q_month) for q_month in (1, 4, 7, 10)):
            logger.info(f"[daily] {year} 已缓存，跳过")
            continue

        year_start = max(f"{year}0101", start)
        year_end = min(f"{year}1231", end)  # 不拉超出请求范围的未来日期
        parts: list[pl.DataFrame] = []

        if ts_codes is not None:
            # 逐股票拉取
            for code in ts_codes:
                try:
                    df_pd = _retry(
                        pro.daily, ts_code=code, start_date=year_start, end_date=year_end
                    )
                except Exception as e:
                    logger.error(f"[daily] {code} {year} 拉取失败: {e}")
                    continue
                if df_pd is not None and not df_pd.empty:
                    parts.append(pl.from_pandas(df_pd))
            if not parts:
                logger.warning(f"[daily] {year} 无数据（逐股模式）")
                continue
        else:
            # 全市场按日期逐日拉取（避免 API 行数限制）
            cal = fetch_trade_cal(year_start, year_end)
            trade_dates_list = (
                cal.filter(pl.col("is_open") == 1)
                .select(pl.col("cal_date").dt.strftime("%Y%m%d"))
                .to_series()
                .to_list()
            )
            for date_str in trade_dates_list:
                try:
                    df_pd = _retry(pro.daily, trade_date=date_str)
                except Exception as e:
                    logger.error(f"[daily] {date_str} 拉取失败: {e}")
                    continue
                if df_pd is not None and not df_pd.empty:
                    parts.append(pl.from_pandas(df_pd))
            if not parts:
                logger.warning(f"[daily] {year} 全市场无数据")
                continue

        # 合并并统一格式
        merged = (
            pl.concat(parts)
            .with_columns(_str_to_date(pl.col("trade_date")))
            .sort(["trade_date", "ts_code"])
        )

        # 选取标准字段（避免多余列干扰分区写入）
        std_cols = [
            "trade_date",
            "ts_code",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ]
        merged = merged.select([c for c in std_cols if c in merged.columns])

        save_parquet(merged, data_type="daily")
        logger.info(f"[daily] {year} 已保存 ({len(merged)} 行)")

    # 从缓存完整加载
    return load_parquet("daily", start=start, end=end).collect()


DAILY_BASIC_COLS: list[str] = [
    "trade_date", "ts_code", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
    "dv_ratio", "dv_ttm", "total_mv", "circ_mv",
    "turnover_rate", "turnover_rate_f", "volume_ratio",
    "total_share", "float_share", "free_share",
]


def fetch_daily_basic(
    start: str,
    end: str,
    ts_codes: list[str] | None = None,
) -> pl.DataFrame:
    """拉取每日估值指标。按年分段，自动缓存。

    Args:
        start: 起始日期 "YYYYMMDD"。
        end: 截止日期 "YYYYMMDD"。
        ts_codes: 股票代码列表。为 None 时拉取全市场。

    Returns:
        pl.DataFrame，包含列:
        trade_date, ts_code, pe, pe_ttm, pb, ps, ps_ttm,
        dv_ratio, dv_ttm, total_mv, circ_mv,
        turnover_rate, turnover_rate_f, volume_ratio,
        total_share, float_share, free_share。
    """
    pro = init_tushare()
    start_year = int(start[:4])
    end_year = int(end[:4])

    for year in range(start_year, end_year + 1):
        if all(partition_exists("daily_basic", year, q_month) for q_month in (1, 4, 7, 10)):
            logger.info(f"[daily_basic] {year} 已缓存，跳过")
            continue

        year_start = max(f"{year}0101", start)
        year_end = min(f"{year}1231", end)
        parts: list[pl.DataFrame] = []

        if ts_codes is not None:
            for code in ts_codes:
                try:
                    df_pd = _retry(
                        pro.daily_basic, ts_code=code, start_date=year_start, end_date=year_end
                    )
                except Exception as e:
                    logger.error(f"[daily_basic] {code} {year} 拉取失败: {e}")
                    continue
                if df_pd is not None and not df_pd.empty:
                    parts.append(pl.from_pandas(df_pd))
            if not parts:
                continue
        else:
            # 全市场按日期逐日拉取（避免 API 行数限制）
            cal = fetch_trade_cal(year_start, year_end)
            trade_dates_list = (
                cal.filter(pl.col("is_open") == 1)
                .select(pl.col("cal_date").dt.strftime("%Y%m%d"))
                .to_series()
                .to_list()
            )
            for date_str in trade_dates_list:
                try:
                    df_pd = _retry(pro.daily_basic, trade_date=date_str)
                except Exception as e:
                    logger.error(f"[daily_basic] {date_str} 拉取失败: {e}")
                    continue
                if df_pd is not None and not df_pd.empty:
                    parts.append(pl.from_pandas(df_pd))
            if not parts:
                logger.warning(f"[daily_basic] {year} 全市场无数据")
                continue

        merged = (
            pl.concat(parts)
            .with_columns(_str_to_date(pl.col("trade_date")))
            .sort(["trade_date", "ts_code"])
        )

        merged = merged.select([c for c in DAILY_BASIC_COLS if c in merged.columns])

        save_parquet(merged, data_type="daily_basic")
        logger.info(f"[daily_basic] {year} 已保存 ({len(merged)} 行)")

    return load_parquet("daily_basic", start=start, end=end).collect()


def fetch_minute(
    ts_code: str,
    freq: str,
    start: str,
    end: str,
    call_delay: float = 0.0,
) -> pl.DataFrame:
    """拉取分钟线数据。按月分段，自动缓存。

    Args:
        ts_code: 股票代码（单只）。
        freq: 频率，如 "1min" | "5min" | "15min" | "30min" | "60min"。
        start: 起始日期 "YYYYMMDD"。
        end: 截止日期 "YYYYMMDD"。
        call_delay: 每次 API 调用（成功或失败）后额外等待秒数，确保与下次调用间隔 >= call_delay。
                    stk_mins 2000积分限 2次/分钟，建议设为 62.0（跨越固定窗口边界）。

    Returns:
        pl.DataFrame，包含列:
        ts_code, trade_time, open, high, low, close, vol, amount。
    """
    import time as _time

    pro = init_tushare()
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    # freq 纳入分区命名空间：不同频率(1min/5min/…)分开缓存，否则缓存键只有
    # (year, month, ts_code)，先拉 5min 再请求 1min 会命中同分区被跳过、返回错频率数据。
    data_type = f"minute_{freq}"

    # 按月迭代
    current = start_dt.replace(day=1)
    while current <= end_dt:
        year = current.year
        month = current.month

        # 该月的最后一天
        if month == 12:
            next_month = current.replace(year=year + 1, month=1)
        else:
            next_month = current.replace(month=month + 1)
        month_end = next_month - timedelta(days=1)
        # 裁剪到请求范围
        seg_start = current.strftime("%Y%m%d")
        seg_end = min(month_end, end_dt).strftime("%Y%m%d")

        # 缓存检查（按 ts_code 粒度，支持多只股票追加写入同一分区）
        if partition_exists(data_type, year, month):
            try:
                _fp = DATA_RAW / data_type / f"year={year}" / f"month={month:02d}" / "data.parquet"
                _existing_codes = (
                    pl.read_parquet(str(_fp), columns=["ts_code"])["ts_code"].unique().to_list()
                )
                if ts_code in _existing_codes:
                    logger.info(f"[minute] {ts_code} {year}-{month:02d} 已缓存，跳过")
                    current = next_month
                    continue
            except Exception:
                pass  # 分区读取失败，继续拉取

        _seg_start_ts = _time.time()
        try:
            df_pd = _retry(
                pro.stk_mins,
                ts_code=ts_code,
                freq=freq,
                start_date=seg_start,
                end_date=seg_end,
            )
        except Exception as e:
            logger.error(f"[minute] {ts_code} {seg_start}~{seg_end} 拉取失败: {e}")
            current = next_month
            # 保证与下一次 API 调用间隔 >= call_delay（扣除已用时间）
            if call_delay > 0:
                _elapsed = _time.time() - _seg_start_ts
                _remaining = call_delay - _elapsed
                if _remaining > 0:
                    _time.sleep(_remaining)
            continue

        if df_pd is None or df_pd.empty:
            logger.warning(f"[minute] {ts_code} {seg_start}~{seg_end} 无数据")
            current = next_month
            continue

        df = (
            pl.from_pandas(df_pd)
            .with_columns(
                pl.col("trade_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            )
            .sort("trade_time")
        )

        # 确保 ts_code 列存在
        if "ts_code" not in df.columns:
            df = df.with_columns(pl.lit(ts_code).alias("ts_code"))

        std_cols = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
        df = df.select([c for c in std_cols if c in df.columns])

        save_parquet(df, data_type=data_type, date_col="trade_time")
        logger.info(f"[minute/{freq}] {ts_code} {year}-{month:02d} 已保存 ({len(df)} 行)")

        # stk_mins 有严格限速（如 2000积分：2次/分钟），确保间隔 >= call_delay
        if call_delay > 0:
            _elapsed = _time.time() - _seg_start_ts
            _remaining = call_delay - _elapsed
            if _remaining > 0:
                _time.sleep(_remaining)

        current = next_month

    return load_parquet(data_type, start=start, end=end, date_col="trade_time").collect()


_FINANCE_BATCH_SIZE = 50  # 每批股票数（Tushare fina_indicator 需要 ts_code，不支持全市场无参拉取）


def fetch_finance(
    api_name: str,
    start: str,
    end: str,
    ts_codes: list[str] | None = None,
    fields: str | None = None,
) -> pl.DataFrame:
    """拉取财务报表数据。按季度分段，自动缓存。

    Args:
        api_name: 接口名，如 "income" | "balancesheet" | "cashflow"
                  | "fina_indicator" | "forecast" | "express"。
        start: 起始日期 "YYYYMMDD"。
        end: 截止日期 "YYYYMMDD"。
        ts_codes: 股票代码列表。为 None 时自动拉取全市场（分批查询）。
        fields: 字段列表（逗号分隔）。为 None 时使用接口默认字段。

    Returns:
        pl.DataFrame，会计年度数据，date_col 为 "end_date"。
    """
    pro = init_tushare()
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")

    fin_api = getattr(pro, api_name)
    # 每个接口(income/cashflow/fina_indicator/…)列集不同，必须分命名空间缓存：
    # 否则共用 "finance" 分区，append concat 会 schema 冲突崩溃、或把不同接口数据串在一起。
    data_type = f"finance_{api_name}"

    # 全市场模式：先获取所有股票代码
    if ts_codes is None:
        stock_df = fetch_stock_basic()
        ts_codes_all: list[str] = stock_df["ts_code"].to_list()
    else:
        ts_codes_all = ts_codes

    # 按季度分段：一年 4 个季度
    quarters = [(1, 3), (4, 6), (7, 9), (10, 12)]
    current_year = start_dt.year
    end_year = end_dt.year

    for year in range(current_year, end_year + 1):
        for q_start_month, q_end_month in quarters:
            seg_start = datetime(year, q_start_month, 1)
            if q_end_month == 12:
                seg_end = datetime(year, q_end_month, 31)
            else:
                seg_end = datetime(year, q_end_month + 1, 1) - timedelta(days=1)

            if seg_end < start_dt or seg_start > end_dt:
                continue
            q_start_str = max(seg_start, start_dt).strftime("%Y%m%d")
            q_end_str = min(seg_end, end_dt).strftime("%Y%m%d")

            # 完整性检查用季末月(3/6/9/12)：数据以 end_date 落盘，end_date 月份是季末，
            # 而非季初月(1/4/7/10)——用季初月检查会永不命中、缓存形同虚设、每次全量重拉。
            if partition_exists(data_type, year, q_end_month):
                logger.info(f"[finance/{api_name}] {year} Q{q_start_month // 3 + 1} 已缓存，跳过")
                continue

            # 分批查询（Tushare fina_indicator 要求 ts_code 参数）
            parts: list[pl.DataFrame] = []
            n_batches = (len(ts_codes_all) + _FINANCE_BATCH_SIZE - 1) // _FINANCE_BATCH_SIZE
            for i in range(0, len(ts_codes_all), _FINANCE_BATCH_SIZE):
                batch = ts_codes_all[i : i + _FINANCE_BATCH_SIZE]
                kwargs: dict[str, Any] = {
                    "ts_code": ",".join(batch),
                    "start_date": q_start_str,
                    "end_date": q_end_str,
                }
                if fields is not None:
                    kwargs["fields"] = fields
                try:
                    df_pd = _retry(fin_api, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"[finance/{api_name}] batch {i // _FINANCE_BATCH_SIZE + 1}/{n_batches} 失败: {e}"
                    )
                    continue
                if df_pd is not None and not df_pd.empty:
                    parts.append(pl.from_pandas(df_pd))

            if not parts:
                logger.warning(f"[finance/{api_name}] {year} Q{q_start_month // 3 + 1} 无数据")
                continue

            # 统一数值列类型（不同批次可能返回 String/Float64 混合），对齐后再 concat
            str_cols = {"ts_code", "ann_date", "end_date"}
            aligned = []
            for p in parts:
                casts = {
                    c: pl.Float64
                    for c in p.columns
                    if c not in str_cols and p[c].dtype != pl.Float64
                }
                aligned.append(
                    p.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in casts])
                )
            df = (
                pl.concat(aligned)
                .with_columns(_str_to_date(pl.col("end_date")))
                .unique(subset=["ts_code", "end_date"], keep="first")
                .sort("end_date")
            )

            save_parquet(df, data_type=data_type, date_col="end_date")
            logger.info(
                f"[finance/{api_name}] {year} Q{q_start_month // 3 + 1} 已保存 ({len(df)} 行)"
            )

    return load_parquet(data_type, start=start, end=end, date_col="end_date").collect()


def fetch_stock_basic(list_status: str = "L,D,P") -> pl.DataFrame:
    """拉取全量股票基本信息，缓存 7 天。

    默认拉取全量（上市 + 退市 + 暂停），以支持 PIT 股票池历史回溯。
    调用方通过 list_date / delist_date 字段自行做 snapshot 过滤。

    Args:
        list_status: Tushare list_status 参数。
            "L" — 仅上市；"D" — 仅退市；"P" — 仅暂停；
            "L,D,P" — 全量（默认，用于 PIT 回溯）。

    Returns:
        pl.DataFrame，包含列:
        ts_code, symbol, name, area, industry, market, list_date, delist_date。
    """
    safe_status = list_status.replace(",", "_")
    cache_file = DATA_CACHE / f"stock_basic_{safe_status}.parquet"

    # 缓存检查
    if cache_file.exists():
        file_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if file_age < CACHE_EXPIRE_DAYS:
            logger.info(f"[stock_basic] 使用缓存（{file_age} 天前更新，status={list_status}）")
            return pl.read_parquet(cache_file)

    # 多 status 时逐个拉取再合并（Tushare 不支持逗号分隔批量）
    pro = init_tushare()
    fields_str = "ts_code,symbol,name,area,industry,market,list_date,delist_date"
    statuses = [s.strip() for s in list_status.split(",")]
    parts: list[pl.DataFrame] = []

    for st in statuses:
        try:
            df_pd = _retry(pro.stock_basic, list_status=st, fields=fields_str)
        except Exception as e:
            logger.error(f"[stock_basic] list_status={st} 拉取失败: {e}")
            continue
        if df_pd is not None and not df_pd.empty:
            parts.append(pl.from_pandas(df_pd))

    if not parts:
        logger.warning("[stock_basic] 无数据")
        if cache_file.exists():
            return pl.read_parquet(cache_file)
        return pl.DataFrame()

    df = (
        pl.concat(parts)
        .unique(subset=["ts_code"])
        .with_columns(
            _str_to_date(pl.col("list_date")),
            _str_to_date(pl.col("delist_date")),
        )
        .sort("ts_code")
    )

    # 写入缓存
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(cache_file))
    logger.info(f"[stock_basic] 已更新缓存 ({len(df)} 只，status={list_status}）")

    return df


def fetch_namechange() -> pl.DataFrame:
    """拉取全量股票曾用名变更记录（含 ST/\\*ST 状态变更历史），缓存 7 天。

    用于重建任意历史日期的 ST 状态时间线（PIT），参见
    ``factorzen.core.universe._is_st_asof``。

    已知坑：若调用时传入 start_date/end_date，Tushare ``namechange`` 接口底层
    按 ann_date 过滤，早期 ann_date 为空的记录会被静默丢弃。因此这里固定
    **不传日期参数**全量拉取，缓存到本地后由调用方自行按日期区间在本地切片。

    Returns:
        pl.DataFrame，包含列:
        ts_code, name, start_date, end_date, ann_date, change_reason。

    Raises:
        拉取失败且无可用本地缓存时，向上抛出底层异常，由调用方决定如何降级
        （参见 ``factorzen.core.universe._safe_fetch_namechange``）。
    """
    cache_file = DATA_CACHE / "namechange.parquet"

    if cache_file.exists():
        file_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if file_age < CACHE_EXPIRE_DAYS:
            logger.info(f"[namechange] 使用缓存（{file_age} 天前更新）")
            return pl.read_parquet(cache_file)

    pro = init_tushare()
    fields_str = "ts_code,name,start_date,end_date,ann_date,change_reason"

    try:
        df_pd = _retry(pro.namechange, fields=fields_str)
    except Exception as e:
        logger.error(f"[namechange] 拉取失败: {e}")
        if cache_file.exists():
            logger.warning("[namechange] 拉取失败，使用过期缓存")
            return pl.read_parquet(cache_file)
        raise

    if df_pd is None or df_pd.empty:
        logger.warning("[namechange] 无数据")
        if cache_file.exists():
            return pl.read_parquet(cache_file)
        return pl.DataFrame()

    df = (
        pl.from_pandas(df_pd)
        .with_columns(
            _str_to_date(pl.col("start_date")),
            _str_to_date(pl.col("end_date")),
            _str_to_date(pl.col("ann_date")),
        )
        .sort(["ts_code", "start_date"])
    )

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(cache_file))
    logger.info(f"[namechange] 已更新缓存 ({len(df)} 条)")

    return df


def fetch_adj_factor(
    start: str,
    end: str,
) -> pl.DataFrame:
    """拉取复权因子（后复权累乘因子）。按年分段，自动缓存。

    Args:
        start: 起始日期 "YYYYMMDD"。
        end: 截止日期 "YYYYMMDD"。

    Returns:
        pl.DataFrame，包含列: ts_code, trade_date, adj_factor。
    """
    pro = init_tushare()
    start_year = int(start[:4])
    end_year = int(end[:4])

    for year in range(start_year, end_year + 1):
        if all(partition_exists("adj_factor", year, q_month) for q_month in (1, 4, 7, 10)):
            logger.info(f"[adj_factor] {year} 已缓存，跳过")
            continue

        year_start = max(f"{year}0101", start)
        year_end = min(f"{year}1231", end)

        cal = fetch_trade_cal(year_start, year_end)
        trade_dates_list = (
            cal.filter(pl.col("is_open") == 1)
            .select(pl.col("cal_date").dt.strftime("%Y%m%d"))
            .to_series()
            .to_list()
        )

        parts: list[pl.DataFrame] = []
        for date_str in trade_dates_list:
            try:
                df_pd = _retry(pro.adj_factor, trade_date=date_str)
            except Exception as e:
                logger.error(f"[adj_factor] {date_str} 拉取失败: {e}")
                continue
            if df_pd is not None and not df_pd.empty:
                parts.append(pl.from_pandas(df_pd))

        if not parts:
            logger.warning(f"[adj_factor] {year} 无数据")
            continue

        merged = (
            pl.concat(parts)
            .with_columns(_str_to_date(pl.col("trade_date")))
            .select(["ts_code", "trade_date", "adj_factor"])
            .sort(["trade_date", "ts_code"])
        )

        save_parquet(merged, data_type="adj_factor")
        logger.info(f"[adj_factor] {year} 已保存 ({len(merged)} 行)")

    return load_parquet("adj_factor", start=start, end=end).collect()


def fetch_index_daily(index_code: str, start: str, end: str) -> pl.DataFrame:
    """拉取指数日线行情。按年分段，自动缓存。

    Args:
        index_code: 指数代码，如 "000300.SH"。
        start: 起始日期 "YYYYMMDD"。
        end: 截止日期 "YYYYMMDD"。

    Returns:
        pl.DataFrame，包含列:
        trade_date, ts_code, open, high, low, close, pre_close, change, pct_chg, vol, amount。
    """
    pro = init_tushare()
    start_year = int(start[:4])
    end_year = int(end[:4])
    _data_type = f"index_daily_{index_code.replace('.', '_')}"

    for year in range(start_year, end_year + 1):
        if all(partition_exists(_data_type, year, q_month) for q_month in (1, 4, 7, 10)):
            logger.info(f"[index_daily] {index_code} {year} 已缓存，跳过")
            continue

        year_start = max(f"{year}0101", start)
        year_end = min(f"{year}1231", end)

        try:
            df_pd = _retry(
                pro.index_daily,
                ts_code=index_code,
                start_date=year_start,
                end_date=year_end,
            )
        except Exception as e:
            logger.error(f"[index_daily] {index_code} {year} 拉取失败: {e}")
            continue

        if df_pd is None or df_pd.empty:
            logger.warning(f"[index_daily] {index_code} {year} 无数据")
            continue

        merged = (
            pl.from_pandas(df_pd)
            .with_columns(_str_to_date(pl.col("trade_date")))
            .sort(["trade_date", "ts_code"])
        )

        std_cols = [
            "trade_date",
            "ts_code",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ]
        merged = merged.select([c for c in std_cols if c in merged.columns])

        save_parquet(merged, data_type=_data_type)
        logger.info(f"[index_daily] {index_code} {year} 已保存 ({len(merged)} 行)")

    return load_parquet(_data_type, start=start, end=end).collect()


_INDEX_MEMBER_ALL_MEMORY_CACHE: dict[str, pl.DataFrame | None] = {}


def fetch_index_member_all() -> pl.DataFrame | None:
    """拉取申万一级行业历史成分（PIT，含 in_date/out_date），全市场。

    用于风险模型按 ``trade_date`` 做历史行业归属查找（PIT 行业暴露），替代
    "用当前分类污染历史窗口"的非 PIT 做法。``index_member_all`` 积分门槛较高，
    当前 token 是否有权限未知：任何原因（无权限/无 token/网络失败/字段不符等）
    失败都优雅降级，返回 ``None``（不抛异常），调用方应自行回退到非 PIT 的
    现有行为。

    全市场覆盖通过按申万一级行业（L1）分别拉取再合并实现：``index_member_all``
    不带过滤条件时单次调用会截断在固定行数（实测 3000 行 / 3000 只股票，覆盖
    不到全市场 5000+ 只股票的成分历史），必须按 ``l1_code`` 循环拉取才能覆盖
    全市场，故先用 ``index_classify(level="L1")`` 枚举一级行业代码。

    本地缓存：单文件 ``DATA_CACHE/index_member_all.parquet``，``CACHE_EXPIRE_DAYS``
    天后过期重新拉取（与 ``fetch_stock_basic`` 的缓存模式一致）；拉取失败时若
    存在（即使过期的）缓存，回退读取缓存而非直接返回 ``None``。此外还有一层
    进程内内存缓存（``_INDEX_MEMBER_ALL_MEMORY_CACHE``）：同一进程内重复调用
    直接复用第一次的结果，不再重复读盘/请求，避免 ``RiskModel.build()`` 对长
    窗口每个交易日都重新加载同一份全市场行业成分表。

    Returns:
        pl.DataFrame | None。失败且无可用缓存时返回 ``None``。成功时包含列：
        l1_code, l1_name, l2_code, l2_name, l3_code, l3_name, ts_code, name,
        in_date (pl.Date), out_date (pl.Date，仍在该行业则为 null), is_new。
    """
    if "value" in _INDEX_MEMBER_ALL_MEMORY_CACHE:
        return _INDEX_MEMBER_ALL_MEMORY_CACHE["value"]

    cache_file = DATA_CACHE / "index_member_all.parquet"

    if cache_file.exists():
        file_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if file_age < CACHE_EXPIRE_DAYS:
            logger.info(f"[index_member_all] 使用缓存（{file_age} 天前更新）")
            result = pl.read_parquet(cache_file)
            _INDEX_MEMBER_ALL_MEMORY_CACHE["value"] = result
            return result

    try:
        pro = init_tushare()

        l1_df = _retry(pro.index_classify, level="L1", src="SW2021")
        if l1_df is None or l1_df.empty:
            raise RuntimeError("index_classify(level=L1) 返回空，无法枚举一级行业")
        l1_codes = l1_df["index_code"].tolist()

        fields = (
            "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
            "ts_code,name,in_date,out_date,is_new"
        )
        parts: list[pl.DataFrame] = []
        for l1_code in l1_codes:
            df_pd = _retry(pro.index_member_all, l1_code=l1_code, fields=fields)
            if df_pd is not None and not df_pd.empty:
                parts.append(pl.from_pandas(df_pd))

        if not parts:
            raise RuntimeError("index_member_all 所有一级行业均无数据")

        df = pl.concat(parts, how="vertical_relaxed")
    except Exception as e:
        logger.warning(f"[index_member_all] 拉取失败（可能无权限/无 token/网络问题）: {e}")
        if cache_file.exists():
            logger.warning("[index_member_all] 回退到本地（可能过期的）缓存")
            result = pl.read_parquet(cache_file)
            _INDEX_MEMBER_ALL_MEMORY_CACHE["value"] = result
            return result
        _INDEX_MEMBER_ALL_MEMORY_CACHE["value"] = None
        return None

    df = df.unique(subset=["ts_code", "l1_code", "in_date"], keep="first").with_columns(
        _str_to_date(pl.col("in_date")),
        _str_to_date(pl.col("out_date")),
    )

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(cache_file))
    logger.info(f"[index_member_all] 已更新缓存 ({len(df)} 行，{df['ts_code'].n_unique()} 只股票)")

    _INDEX_MEMBER_ALL_MEMORY_CACHE["value"] = df
    return df


def fetch_trade_cal(start: str, end: str) -> pl.DataFrame:
    """拉取交易日历。

    用于 calendar.py 内部调用，不在此处额外缓存（calendar.py 自己管理缓存）。

    Args:
        start: 起始日期 "YYYYMMDD"。
        end: 截止日期 "YYYYMMDD"。

    Returns:
        pl.DataFrame，包含列:
        exchange, cal_date, is_open, pretrade_date。
    """
    pro = init_tushare()

    try:
        df_pd = _retry(
            pro.trade_cal,
            exchange="SSE",
            start_date=start,
            end_date=end,
        )
    except Exception as e:
        logger.error(f"[trade_cal] 拉取失败: {e}")
        raise

    if df_pd is None or df_pd.empty:
        logger.warning(f"[trade_cal] {start}~{end} 无数据")
        return pl.DataFrame()

    df = pl.from_pandas(df_pd).with_columns(_str_to_date(pl.col("cal_date"))).sort("cal_date")

    logger.info(f"[trade_cal] {start}~{end} 已拉取 ({len(df)} 条)")
    return df
