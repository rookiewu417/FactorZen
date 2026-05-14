#!/usr/bin/env python
"""因子 Tear Sheet 报告生成器。

整合因子计算、基础评价、高级评价与 HTML 报告输出。

用法:
  pixi run python scripts/generate_report.py --factor momentum_20d --start 20250101 --end 20250513
  pixi run report -- --factor momentum_20d --start 20250101 --end 20250513
"""

import argparse

import polars as pl

from config.settings import OUTPUT_DIR, OUTPUT_LFT_REPORTS
from common.logger import setup_logging, get_logger
from common.loader import fetch_daily
from common.calendar import get_trade_dates
from common.universe import get_universe
from daily.data.context import FactorDataContext
from daily.factors.registry import get_factor
from daily.preprocessing.pipeline import quick_preprocess
from daily.evaluation.ic_analysis import compute_rank_ic, compute_fwd_returns
from daily.evaluation.backtest import run_stratified_backtest
from daily.evaluation.turnover import compute_turnover
from reporting.tear_sheet import generate_tear_sheet

setup_logging()
logger = get_logger(__name__)


def _run_advanced_evaluation(clean_df, ret_df, frequency):
    """运行高级评价模块，各模块互不依赖，单个失败不影响整体。

    Returns:
        dict: 高级评价结果，键对应 generate_tear_sheet 的 advanced_results。
    """
    advanced: dict = {}

    # ── IC Decay 增强分析 ──
    try:
        from daily.evaluation.advanced import compute_ic_decay
        advanced["decay_results"] = compute_ic_decay(
            clean_df, ret_df, factor_col="factor_clean"
        )
        logger.info(f"IC Decay: {len(advanced['decay_results'])} horizons")
    except ImportError as e:
        logger.warning(f"IC Decay 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"IC Decay 失败: {e}")

    # ── 单调性分析 ──
    try:
        from daily.evaluation.advanced import compute_monotonicity
        # 合并因子与 fwd_ret_1d 用于单调性
        mono_df = clean_df.join(
            ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
            on=["trade_date", "ts_code"], how="inner"
        )
        advanced["mono"] = compute_monotonicity(
            mono_df, factor_col="factor_clean", ret_col="fwd_ret_1d"
        )
        logger.info(f"单调性: score={advanced['mono'].monotonicity_score:.3f}")
    except ImportError as e:
        logger.warning(f"Monotonicity 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"单调性分析失败: {e}")

    # ── 排名自相关 ──
    try:
        from daily.evaluation.advanced import compute_rank_autocorr
        advanced["autocorr"] = compute_rank_autocorr(
            clean_df, factor_col="factor_clean", lags=[1, 5, 10]
        )
        logger.info(
            f"排名自相关: mean={advanced['autocorr'].mean_autocorr:.3f}, "
            f"half_life={advanced['autocorr'].half_life_est:.1f}"
        )
    except ImportError as e:
        logger.warning(f"Rank Autocorr 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"排名自相关失败: {e}")

    # ── 市场状态 IC ──
    try:
        from daily.evaluation.advanced import compute_market_regime_ic
        advanced["regime"] = compute_market_regime_ic(
            clean_df.join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
                on=["trade_date", "ts_code"], how="inner"
            ),
            factor_col="factor_clean",
            ret_col="fwd_ret_1d",
            regime_type="direction",
            return_object=True,
        )
        logger.info(f"市场状态 IC: {advanced['regime'].regime_type}")
    except ImportError as e:
        logger.warning(f"Market Regime 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"市场状态 IC 失败: {e}")

    return advanced if advanced else None


def main():
    parser = argparse.ArgumentParser(description="因子 Tear Sheet 报告生成")
    parser.add_argument("--factor", required=True, help="因子名称")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default="lft_default", help="股票池")
    parser.add_argument(
        "--frequency", default="daily",
        choices=["daily", "weekly", "monthly"], help="因子频率"
    )
    args = parser.parse_args()

    # ── 1. 获取因子类 ──
    logger.info(f"──── 因子报告生成: {args.factor} | {args.start} ~ {args.end} ────")
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
        clean_df, ret_df.select(["trade_date", "ts_code", "ret"]), frequency=args.frequency
    )
    bt_result.factor_name = factor.name
    logger.info(f"\n{bt_result.summary()}")

    # ── 9. 换手率 ──
    to_result = compute_turnover(clean_df, frequency=args.frequency)
    to_result.factor_name = factor.name
    logger.info(f"\n{to_result.summary()}")

    # ── 10. 高级评价 ──
    advanced_results = _run_advanced_evaluation(clean_df, ret_df, args.frequency)

    # ── 11. 生成 HTML 报告 ──
    date_range = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]} ~ {args.end[:4]}-{args.end[4:6]}-{args.end[6:]}"
    html = generate_tear_sheet(
        factor_name=factor.name,
        ic_result=ic_result,
        bt_result=bt_result,
        to_result=to_result,
        frequency=args.frequency,
        date_range=date_range,
        advanced_results=advanced_results,
        universe=args.universe,
    )

    # ── 12. 落盘 ──
    OUTPUT_LFT_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_LFT_REPORTS / f"{factor.name}_{args.start}_{args.end}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"报告已生成: {report_path}")
    logger.info("完成!")


if __name__ == "__main__":
    main()
