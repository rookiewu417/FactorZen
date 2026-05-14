"""LFT 多因子 IC 对比。用法: python scripts/run_lft_compare.py --factors momentum_20d,reversal_5d --start 20250101 --end 20250513"""

import argparse

import polars as pl

from common.logger import setup_logging, get_logger
from common.loader import fetch_daily
from common.universe import get_universe
from lft.data.context import FactorDataContext
from lft.factors.registry import get_factor
from lft.preprocessing.pipeline import quick_preprocess
from lft.evaluation.ic_analysis import compute_rank_ic, compute_fwd_returns
from lft.evaluation.correlation import compute_factor_correlation

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LFT 多因子对比")
    parser.add_argument("--factors", required=True, help="逗号分隔的因子名")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default="lft_default", help="股票池")
    args = parser.parse_args()

    factor_names = [f.strip() for f in args.factors.split(",")]
    logger.info(
        f"──── 多因子对比: {', '.join(factor_names)} | {args.start} ~ {args.end} ────"
    )

    # ── 验证因子存在 ──
    valid_factors = []
    for fname in factor_names:
        try:
            get_factor(fname)
            valid_factors.append(fname)
        except KeyError as e:
            logger.error(str(e))
            sys.exit(1)
    factor_names = valid_factors

    # ── 准备数据 ──
    try:
        fetch_daily(args.start, args.end)
    except Exception as e:
        logger.error(f"数据拉取失败: {e}")
        sys.exit(1)

    universe = get_universe(args.end, args.universe)
    if universe.is_empty():
        logger.error(f"股票池为空: {args.universe} ({args.end})")
        sys.exit(1)
    ts_codes = universe["ts_code"].to_list()
    logger.info(f"股票池: {len(ts_codes)} 只")

    # ── 计算前向收益（共用）──
    ctx = FactorDataContext(start=args.start, end=args.end, universe=ts_codes)
    daily = ctx.daily.collect()
    if daily.is_empty():
        logger.error("日线数据为空，无法计算收益")
        sys.exit(1)
    ret_df = daily.select(["trade_date", "ts_code", "close"]).sort(["ts_code", "trade_date"])
    ret_df = ret_df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1).alias("ret")
    )
    ret_df = compute_fwd_returns(ret_df, ret_col="ret")
    logger.info(f"前向收益计算完成 (horizons: 1/5/10/20d)")

    factor_dict = {}

    for fname in factor_names:
        logger.info(f"── 计算因子: {fname} ──")
        factor_cls = get_factor(fname)
        factor = factor_cls()

        ctx = FactorDataContext(
            start=args.start,
            end=args.end,
            required_data=factor.required_data,
            lookback_days=factor.lookback_days,
            universe=ts_codes,
        )
        try:
            factor_df = factor.compute(ctx)
        except Exception as e:
            logger.error(f"因子 {fname} 计算失败: {e}")
            sys.exit(1)

        validation = factor.validate(factor_df)
        logger.info(f"计算完成: {validation}")
        if factor_df.is_empty():
            logger.error(f"因子 {fname} 计算结果为空，退出")
            sys.exit(1)

        clean_df = quick_preprocess(factor_df, col="factor_value")
        factor_dict[fname] = clean_df

        ic_result = compute_rank_ic(clean_df, ret_df)
        ic_result.factor_name = fname
        logger.info(f"\n{ic_result.summary()}")

    # ── 因子相关性 ──
    if len(factor_names) >= 2:
        corr_result = compute_factor_correlation(factor_dict)
        logger.info(f"\n{corr_result.summary()}")


if __name__ == "__main__":
    main()
