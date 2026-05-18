"""补全所需基础数据（daily_basic + finance）。

用法:
  pixi run python scripts/fetch_data.py
  pixi run python scripts/fetch_data.py --start 20240101 --end 20260513

拉取内容:
  - daily_basic: 每日估值指标（PE/PB/市值），月频价值因子、市值中性化依赖
  - finance/fina_indicator: 财务指标（ROE），盈利质量因子依赖

如果已缓存，会自动跳过，无需担心重复拉取。
"""

import argparse

from common.loader import fetch_daily_basic, fetch_finance
from common.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="补全基础数据")
    parser.add_argument(
        "--start",
        default="20240101",
        help="起始日期 YYYYMMDD（默认 20240101，财务数据需要至少 4 季度历史）",
    )
    parser.add_argument("--end", default="20260513", help="截止日期 YYYYMMDD")
    args = parser.parse_args()

    logger.info(f"=== 开始补全数据 [{args.start} ~ {args.end}] ===")

    # ── 1. 每日估值指标 ──────────────────────────────────────────────────
    logger.info("── 拉取 daily_basic（PE/PB/总市值）──")
    try:
        df = fetch_daily_basic(args.start, args.end)
        logger.info(f"daily_basic 完成：{len(df):,} 行，{df['trade_date'].n_unique()} 个交易日")
    except Exception as e:
        logger.error(f"daily_basic 拉取失败: {e}", exc_info=True)
        logger.warning("跳过 daily_basic，月频价值因子和市值中性化将不可用")

    # ── 2. 财务指标（ROE TTM 等）────────────────────────────────────────
    logger.info("── 拉取 finance/fina_indicator（ROE / 盈利质量）──")
    try:
        df = fetch_finance("fina_indicator", args.start, args.end)
        logger.info(f"fina_indicator 完成：{len(df):,} 行")
    except Exception as e:
        logger.error(f"fina_indicator 拉取失败: {e}", exc_info=True)
        logger.warning("跳过 fina_indicator，ROE 盈利质量因子将不可用")

    logger.info("=== 数据补全完成，可运行 generate_report.py ===")


if __name__ == "__main__":
    main()
