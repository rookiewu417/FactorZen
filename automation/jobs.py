"""自动化任务原子函数。

每个 job 函数:
- 通过 get_logger 记录 start/end
- 捕获所有异常，记录后重新抛出
- 是纯函数，无全局状态
"""

import subprocess

from common.loader import fetch_daily, fetch_index_daily
from common.logger import get_logger
from config.constants import BENCHMARK_INDICES

logger = get_logger(__name__)


def job_fetch_daily(date: str) -> None:
    """拉取当日行情数据。

    Parameters
    ----------
    date : str
        交易日期，格式 YYYYMMDD。
    """
    logger.info(f"[job_fetch_daily] 开始拉取行情: {date}")
    try:
        fetch_daily(start=date, end=date)
        logger.info(f"[job_fetch_daily] 完成: {date}")
    except Exception as exc:
        logger.exception(f"[job_fetch_daily] 失败: {date} — {exc}")
        raise


def job_fetch_index(date: str) -> None:
    """拉取所有 benchmark 指数数据。

    Parameters
    ----------
    date : str
        交易日期，格式 YYYYMMDD。
    """
    logger.info(f"[job_fetch_index] 开始拉取指数: {date}")
    try:
        for index_code in BENCHMARK_INDICES:
            logger.info(f"[job_fetch_index] 拉取 {index_code}")
            fetch_index_daily(index_code=index_code, start=date, end=date)
        logger.info(f"[job_fetch_index] 完成: {date}")
    except Exception as exc:
        logger.exception(f"[job_fetch_index] 失败: {date} — {exc}")
        raise


def job_compute_factors(date: str, factor_list: list[str]) -> None:
    """计算指定因子列表。

    Parameters
    ----------
    date : str
        交易日期，格式 YYYYMMDD。
    factor_list : list[str]
        因子名称列表。
    """
    logger.info(f"[job_compute_factors] 开始计算因子: {factor_list} date={date}")
    try:
        from daily.factors.registry import get_factor

        for factor_name in factor_list:
            logger.info(f"[job_compute_factors] 计算因子: {factor_name}")
            factor_cls = get_factor(factor_name)
            factor_cls()  # 验证因子可实例化
            logger.info(f"[job_compute_factors] 因子 {factor_name} 实例化完成")
        logger.info(f"[job_compute_factors] 完成: {date}")
    except Exception as exc:
        logger.exception(f"[job_compute_factors] 失败: {date} {factor_list} — {exc}")
        raise


def job_evaluate(date: str, factor_name: str) -> None:
    """运行 IC + 回测 + 换手评估。

    通过 subprocess 调用 scripts/run_daily_single.py 以避免循环导入。

    Parameters
    ----------
    date : str
        交易日期，格式 YYYYMMDD。
    factor_name : str
        因子名称。
    """
    logger.info(f"[job_evaluate] 开始评估: factor={factor_name} date={date}")
    try:
        cmd = [
            "pixi",
            "run",
            "python",
            "scripts/run_daily_single.py",
            "--factor",
            factor_name,
            "--start",
            date,
            "--end",
            date,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"run_daily_single 退出码={result.returncode}\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
        logger.info(f"[job_evaluate] 完成: factor={factor_name} date={date}")
    except Exception as exc:
        logger.exception(f"[job_evaluate] 失败: factor={factor_name} date={date} — {exc}")
        raise


def job_generate_report(date: str, factor_name: str) -> None:
    """生成 HTML tear sheet 报告。

    通过 subprocess 调用 scripts/generate_report.py 以避免循环导入。

    Parameters
    ----------
    date : str
        交易日期，格式 YYYYMMDD。
    factor_name : str
        因子名称。
    """
    logger.info(f"[job_generate_report] 开始生成报告: factor={factor_name} date={date}")
    try:
        cmd = [
            "pixi",
            "run",
            "python",
            "scripts/generate_report.py",
            "--factor",
            factor_name,
            "--start",
            date,
            "--end",
            date,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"generate_report 退出码={result.returncode}\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
        logger.info(f"[job_generate_report] 完成: factor={factor_name} date={date}")
    except Exception as exc:
        logger.exception(
            f"[job_generate_report] 失败: factor={factor_name} date={date} — {exc}"
        )
        raise
