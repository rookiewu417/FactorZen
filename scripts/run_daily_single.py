"""日频单因子完整评估。用法: python scripts/run_daily_single.py --factor momentum_20d --start 20250101 --end 20250513"""

import argparse
import sys

import polars as pl

from common.calendar import get_trade_dates
from common.loader import fetch_daily
from common.logger import get_logger, setup_logging
from common.universe import get_universe
from config.settings import OUTPUT_DAILY_FACTORS, OUTPUT_DAILY_REPORTS, OUTPUT_DAILY_RESULTS
from daily.data.context import FactorDataContext
from daily.evaluation.backtest import run_stratified_backtest
from daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from daily.evaluation.turnover import compute_turnover
from daily.factors.registry import get_factor
from daily.preprocessing.pipeline import quick_preprocess
from reporting.tear_sheet import generate_tear_sheet

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="日频单因子评估")
    parser.add_argument("--factor", required=True, help="因子名称")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default="csi300", help="股票池")
    parser.add_argument(
        "--frequency", default="daily", choices=["daily", "weekly", "monthly"], help="因子频率"
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="基准指数代码（如 000300.SH），若指定则计算超额收益并生成 HTML 报告",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML 运行配置文件路径")
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子")
    args = parser.parse_args()

    # ── 0. 加载 YAML 配置（可选），CLI 参数优先级更高 ──
    run_config = None
    if args.config:
        from common.config_loader import load_run_config

        run_config = load_run_config(args.config)
        # CLI 默认值时，从 config 填充
        if args.universe == "csi300" and run_config.universe:
            args.universe = run_config.universe
        if args.benchmark is None and run_config.benchmark:
            args.benchmark = run_config.benchmark
        if args.seed is None and run_config.seed is not None:
            args.seed = run_config.seed

    # ── 0b. 设置全局随机种子（可选）──
    if args.seed is not None:
        from common.seed import set_global_seed

        set_global_seed(args.seed)
        logger.info(f"全局随机种子已设置: {args.seed}")

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
        clean_df,
        daily,
        frequency=args.frequency,
        factor_name=factor.name,
    )
    logger.info(f"\n{bt_result.summary()}")

    # ── 9. 换手率 ──
    to_result = compute_turnover(clean_df, frequency=args.frequency)
    to_result.factor_name = factor.name
    logger.info(f"\n{to_result.summary()}")

    # ── 10. 落盘 ──
    OUTPUT_DAILY_FACTORS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DAILY_RESULTS.mkdir(parents=True, exist_ok=True)

    factor_path = OUTPUT_DAILY_FACTORS / f"{factor.name}_{args.start}_{args.end}.parquet"
    clean_df.write_parquet(str(factor_path))
    logger.info(f"因子已保存: {factor_path}")

    ic_result.ic_series.write_parquet(
        str(OUTPUT_DAILY_RESULTS / f"{factor.name}_{args.start}_{args.end}_ic.parquet")
    )

    # ── 11. Benchmark 对比（可选）──
    benchmark_result = None
    if args.benchmark:
        try:
            from daily.evaluation.benchmark import compute_excess_return

            benchmark_result = compute_excess_return(
                bt_result.returns, args.benchmark, args.start, args.end
            )
            logger.info(f"Benchmark: {benchmark_result.summary()}")
        except Exception as e:
            logger.warning(f"Benchmark 计算失败（跳过）: {e}")

    # ── 12. HTML 报告（当 --benchmark 提供时生成，或始终生成）──
    date_range = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]} ~ {args.end[:4]}-{args.end[4:6]}-{args.end[6:]}"
    html = generate_tear_sheet(
        factor_name=factor.name,
        ic_result=ic_result,
        bt_result=bt_result,
        to_result=to_result,
        frequency=args.frequency,
        date_range=date_range,
        universe=args.universe,
        benchmark_result=benchmark_result,
        attribution_result=None,
    )
    OUTPUT_DAILY_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DAILY_REPORTS / f"{factor.name}_{args.start}_{args.end}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"报告已生成: {report_path}")
    logger.info("完成!")


if __name__ == "__main__":
    main()
