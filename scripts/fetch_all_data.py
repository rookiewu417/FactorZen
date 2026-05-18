"""Fetch all data needed for factor evaluation.

Usage:
    pixi run python scripts/fetch_all_data.py --start 20240101 --end 20260514
"""

import argparse

from common.loader import fetch_daily, fetch_daily_basic, fetch_finance
from common.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default="20260514")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-basic", action="store_true")
    parser.add_argument("--skip-finance", action="store_true")
    args = parser.parse_args()

    logger.info(f"Data fetch: {args.start} ~ {args.end}")

    if not args.skip_daily:
        logger.info("=== Fetching daily price data ===")
        try:
            df = fetch_daily(args.start, args.end)
            logger.info(f"daily: {len(df):,} rows, {df['ts_code'].n_unique()} stocks")
        except Exception as e:
            logger.error(f"daily fetch failed: {e}")

    if not args.skip_basic:
        logger.info("=== Fetching daily_basic (valuation) ===")
        try:
            df = fetch_daily_basic(args.start, args.end)
            logger.info(f"daily_basic: {len(df):,} rows")
        except Exception as e:
            logger.error(f"daily_basic fetch failed: {e}")

    if not args.skip_finance:
        logger.info("=== Fetching finance (fina_indicator) ===")
        try:
            # Start earlier to have enough lookback for YoY calculations
            fin_start = str(int(args.start[:4]) - 2) + args.start[4:]
            df = fetch_finance(
                "fina_indicator",
                fin_start,
                args.end,
                fields="ts_code,ann_date,end_date,roe,assets_yoy",
            )
            logger.info(f"fina_indicator: {len(df):,} rows")
        except Exception as e:
            logger.error(f"fina_indicator fetch failed: {e}")

    logger.info("Data fetch complete.")


if __name__ == "__main__":
    main()
