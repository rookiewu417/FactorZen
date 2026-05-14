"""LFT 单因子完整评估。用法: python scripts/run_lft_single.py --factor momentum_20d --start 20250101 --end 20250513"""

import argparse
import sys
from pathlib import Path

# 添加项目根到 sys.path，确保 scripts/ 目录下运行时能导入 config/common/lft
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from common.calendar import get_trade_dates
from common.loader import fetch_daily
from common.logger import get_logger, setup_logging
from common.universe import get_universe
from config.settings import OUTPUT_LFT_FACTORS, OUTPUT_LFT_RESULTS
from daily.data.context import FactorDataContext
from daily.evaluation.backtest import run_stratified_backtest
from daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from daily.evaluation.turnover import compute_turnover
from daily.factors.registry import get_factor
from daily.preprocessing.pipeline import quick_preprocess

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LFT 单因子评估")
    parser.add_argument("--factor", required=True, help="因子名称")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default="lft_default", help="股票池")
    parser.add_argument("--frequency", default="daily", choices=["daily", "weekly", "monthly"], help="因子频率")
    args = parser.parse_args()

    # ── 1. 获取因子类 ──
    logger.info(f"──── 单因子评估: {args.factor} | {args.start} ~ {args.end} ────")
    try:
        factor_cls = get_factor(args.factor)
    except KeyError as e:
        logger.error(str(e))
        sys.exit(1)
    factor = factor_cls()
    logger.info(f"因子: {factor.name} | {factor.description}")

    # ── 2. 准备数据 ──
    trade_dates = get_trade_dates(args.start, args.end)
    logger.info(f"交易日数: {len(trade_dates)}")
    if len(trade_dates) < 30:
        logger.warning("交易日不足 30 天，IC 分析可能不稳定")

    # 确保日线数据已缓存
    try:
        fetch_daily(args.start, args.end)
    except Exception as e:
        logger.error(f"数据拉取失败: {e}")
        sys.exit(1)

    # ── 3. 股票池 ──
    universe = get_universe(args.end, args.universe)
    if universe.is_empty():
        logger.error(f"股票池为空: {args.universe} ({args.end})")
        sys.exit(1)
    ts_codes = universe["ts_code"].to_list()
    logger.info(f"股票池: {len(ts_codes)} 只")

    # ── 4. 计算因子 ──
    ctx = FactorDataContext(
        start=args.start,
        end=args.end,
        required_data=factor.required_data,
        lookback_days=factor.lookback_days,
        universe=ts_codes,
        snapshot_mode=args.frequency,
    )
    try:
        factor_df = factor.compute(ctx)
    except Exception as e:
        logger.error(f"因子计算失败: {e}")
        sys.exit(1)

    validation = factor.validate(factor_df)
    logger.info(f"因子计算完成: {validation}")
    if factor_df.is_empty():
        logger.error("因子计算结果为空，退出")
        sys.exit(1)
    if validation.get("coverage", 0) < 0.5:
        logger.warning("因子覆盖率不足 50%，结果可能不可靠")

    # ── 5. 预处理 ──
    clean_df = quick_preprocess(factor_df, col="factor_value")
    logger.info("预处理完成 (去极值 → 填充 → 标准化)")

    # ── 6. 计算前向收益 ──
    daily = ctx.daily.collect()
    if daily.is_empty():
        logger.error("日线数据为空，无法计算收益")
        sys.exit(1)
    ret_df = daily.select(["trade_date", "ts_code", "close"]).sort(["ts_code", "trade_date"])
    ret_df = ret_df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1).alias("ret")
    )
    ret_df = compute_fwd_returns(ret_df, ret_col="ret")
    logger.info("前向收益计算完成 (horizons: 1/5/10/20d)")

    # ── 7. IC 分析 ──
    ic_result = compute_rank_ic(clean_df, ret_df, frequency=args.frequency)
    ic_result.factor_name = factor.name
    logger.info(f"\n{ic_result.summary()}")

    # ── 8. 分层回测 ──
    bt_result = run_stratified_backtest(
        clean_df, ret_df.select(["trade_date", "ts_code", "ret"]),
        frequency=args.frequency, factor_name=factor.name,
    )
    logger.info(f"\n{bt_result.summary()}")

    # ── 9. 换手率 ──
    to_result = compute_turnover(clean_df, frequency=args.frequency)
    to_result.factor_name = factor.name
    logger.info(f"\n{to_result.summary()}")

    # ── 10. 落盘 ──
    OUTPUT_LFT_FACTORS.mkdir(parents=True, exist_ok=True)
    OUTPUT_LFT_RESULTS.mkdir(parents=True, exist_ok=True)

    factor_path = OUTPUT_LFT_FACTORS / f"{factor.name}_{args.start}_{args.end}.parquet"
    clean_df.write_parquet(str(factor_path))
    logger.info(f"因子已保存: {factor_path}")

    ic_result.ic_series.write_parquet(
        str(OUTPUT_LFT_RESULTS / f"{factor.name}_ic.parquet")
    )
    logger.info("完成!")


if __name__ == "__main__":
    main()
