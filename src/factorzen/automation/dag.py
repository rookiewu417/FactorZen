"""日终任务 DAG 构建器。

使用 APScheduler BackgroundScheduler 在 16:30 (Asia/Shanghai) 触发日终流水线。
流水线步骤：
  fetch_daily → fetch_index → compute_factors → evaluate (per factor) → generate_report (per factor)

每步最多重试 3 次，指数退避（基础 60 秒）。
非交易日自动跳过整条链路。
"""

import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from factorzen.automation.jobs import (
    job_compute_factors,
    job_evaluate,
    job_fetch_daily,
    job_fetch_index,
    job_generate_report,
)
from factorzen.automation.state import run_record
from factorzen.config.settings import (
    SCHEDULER_CRON_HOUR,
    SCHEDULER_CRON_MINUTE,
    SCHEDULER_MAX_RETRIES,
    SCHEDULER_RETRY_BASE_SECONDS,
    SCHEDULER_TIMEZONE,
)
from factorzen.core.calendar import is_trade_date
from factorzen.core.logger import get_logger

logger = get_logger(__name__)


def _with_retry(func, *args, max_retries=SCHEDULER_MAX_RETRIES, base_seconds=SCHEDULER_RETRY_BASE_SECONDS, **kwargs):
    """带指数退避重试的函数调用包装器。

    Parameters
    ----------
    func : callable
        要调用的函数。
    *args :
        传递给 func 的位置参数。
    max_retries : int
        最大重试次数（不含首次调用）。
    base_seconds : int
        退避基数（秒），第 i 次重试等待 base_seconds * 2^(i-1)。
    **kwargs :
        传递给 func 的关键字参数。
    """
    for attempt in range(max_retries + 1):
        try:
            func(*args, **kwargs)
            return
        except Exception as exc:
            if attempt >= max_retries:
                logger.error("重试次数耗尽 (%s), 最终失败: %s", func.__name__, exc)
                raise
            wait = base_seconds * (2 ** attempt)
            logger.warning("第 %d 次重试失败 (%s), %ds 后重试: %s", attempt + 1, func.__name__, wait, exc)
            time.sleep(wait)


def run_daily_pipeline(date: str, factor_list: list[str], benchmark: str) -> None:
    """执行完整的日终流水线。

    Parameters
    ----------
    date : str
        交易日期，格式 YYYYMMDD。
    factor_list : list[str]
        需要计算/评估/报告的因子名列表。
    benchmark : str
        基准指数代码，如 "000300.SH"。
    """
    logger.info(f"[pipeline] 日终流水线开始: date={date} factors={factor_list}")

    with run_record("pipeline"):
        if not is_trade_date(date):
            logger.info("非交易日 %s，跳过流水线", date)
            return

        # 1. 拉取行情
        with run_record("fetch_daily"):
            _with_retry(job_fetch_daily, date)

        # 2. 拉取指数
        with run_record("fetch_index"):
            _with_retry(job_fetch_index, date)

        # 3. 计算因子
        with run_record("compute_factors"):
            _with_retry(job_compute_factors, date, factor_list)

        # 4. 逐因子评估 + 报告
        for factor_name in factor_list:
            with run_record(f"evaluate:{factor_name}"):
                _with_retry(job_evaluate, date, factor_name)
            with run_record(f"generate_report:{factor_name}"):
                _with_retry(job_generate_report, date, factor_name)

    logger.info(f"[pipeline] 日终流水线完成: date={date}")


def build_daily_dag(
    factor_list: list[str],
    benchmark: str = "000300.SH",
) -> BackgroundScheduler:
    """构建日终任务链 scheduler。

    触发时间: 每个交易日 16:30 Asia/Shanghai。
    非交易日自动跳过。

    Parameters
    ----------
    factor_list : list[str]
        需要每日处理的因子名列表。
    benchmark : str
        基准指数代码，默认 "000300.SH"。

    Returns
    -------
    BackgroundScheduler
        已配置但未启动的 APScheduler 调度器。
        调用方应调用 scheduler.start() 启动。
    """
    scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)

    def _trigger_today() -> None:
        date = datetime.now().strftime("%Y%m%d")
        run_daily_pipeline(date=date, factor_list=factor_list, benchmark=benchmark)

    scheduler.add_job(
        func=_trigger_today,
        trigger=CronTrigger(
            hour=SCHEDULER_CRON_HOUR,
            minute=SCHEDULER_CRON_MINUTE,
        ),
        id="daily_pipeline",
        name="日终因子流水线",
        replace_existing=True,
        misfire_grace_time=3600,  # 1 小时内补跑
        max_instances=1,
    )

    logger.info(
        f"[dag] 日终 DAG 构建完成: factors={factor_list}, benchmark={benchmark},"
        f" cron={SCHEDULER_CRON_HOUR}:{SCHEDULER_CRON_MINUTE:02d} {SCHEDULER_TIMEZONE}"
    )
    return scheduler
