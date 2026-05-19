"""APScheduler 日终流水线入口。

用法:
  python scripts/run_scheduler.py                    # 守护进程模式（16:30 自动触发）
  python scripts/run_scheduler.py --once 20250513    # 单次运行（回填指定日期）
  python scripts/run_scheduler.py --factors momentum_20d reversal_5d  # 指定因子列表
"""

import argparse
import sys
import time

from automation.dag import build_daily_dag, run_daily_pipeline
from common.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# 默认因子列表（可通过 --factors 覆盖）
DEFAULT_FACTORS: list[str] = []
DEFAULT_BENCHMARK: str = "000300.SH"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="APScheduler 日终因子流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--once",
        metavar="DATE",
        help="单次运行模式：立即执行指定日期的流水线后退出（格式 YYYYMMDD）",
    )
    parser.add_argument(
        "--factors",
        nargs="*",
        default=DEFAULT_FACTORS,
        metavar="FACTOR",
        help="因子名称列表（空格分隔），默认为空列表",
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK,
        help=f"基准指数代码，默认 {DEFAULT_BENCHMARK}",
    )
    args = parser.parse_args()

    factor_list: list[str] = args.factors or DEFAULT_FACTORS
    benchmark: str = args.benchmark

    if args.once:
        # ── 单次运行模式 ──
        date: str = args.once
        logger.info(f"[scheduler] 单次运行模式: date={date} factors={factor_list}")
        try:
            run_daily_pipeline(date=date, factor_list=factor_list, benchmark=benchmark)
            logger.info("[scheduler] 单次运行完成，退出")
            sys.exit(0)
        except Exception as exc:
            logger.error(f"[scheduler] 单次运行失败: {exc}")
            sys.exit(1)
    else:
        # ── 守护进程模式 ──
        logger.info(f"[scheduler] 启动守护进程: factors={factor_list} benchmark={benchmark}")
        scheduler = build_daily_dag(factor_list=factor_list, benchmark=benchmark)
        scheduler.start()
        logger.info("[scheduler] APScheduler 已启动，按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("[scheduler] 收到终止信号，正在关闭 APScheduler...")
            scheduler.shutdown()
            logger.info("[scheduler] APScheduler 已关闭，退出")


if __name__ == "__main__":
    main()
