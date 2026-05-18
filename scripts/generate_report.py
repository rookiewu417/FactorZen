#!/usr/bin/env python
"""因子 Tear Sheet 报告生成器。

整合因子计算、基础评价、高级评价与 HTML 报告输出。

用法:
  pixi run report -- --factor momentum_20d --start 20250101 --end 20250513
  pixi run report -- --factor momentum_20d --start 20250101 --end 20250513 --reuse
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from common.calendar import get_trade_dates
from common.loader import fetch_daily
from common.logger import get_logger, setup_logging
from common.storage import load_parquet
from common.universe import get_universe
from config.settings import (
    OUTPUT_DAILY_FACTORS,
    OUTPUT_DAILY_REPORTS,
    OUTPUT_DAILY_RESULTS,
)
from daily.data.context import FactorDataContext
from daily.evaluation.backtest import BacktestResult, run_stratified_backtest
from daily.evaluation.ic_analysis import (
    ICAnalysisResult,
    compute_fwd_returns,
    compute_rank_ic,
)
from daily.evaluation.turnover import TurnoverResult, compute_turnover
from daily.factors.registry import get_factor
from daily.preprocessing.pipeline import quick_preprocess
from reporting.tear_sheet import generate_tear_sheet

setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 持久化辅助
# ---------------------------------------------------------------------------


def _meta_path(factor_name: str, start: str, end: str) -> "Path":
    OUTPUT_DAILY_RESULTS.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DAILY_RESULTS / f"{factor_name}_{start}_{end}_meta.json"


def _save_results(
    factor_name: str,
    start: str,
    end: str,
    clean_df: pl.DataFrame,
    ic_result: ICAnalysisResult,
    bt_result: BacktestResult,
    to_result: TurnoverResult,
) -> None:
    """将因子 DataFrame 和评价结果落盘到 output/daily/。"""
    OUTPUT_DAILY_FACTORS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DAILY_RESULTS.mkdir(parents=True, exist_ok=True)

    prefix = f"{factor_name}_{start}_{end}"

    clean_df.write_parquet(str(OUTPUT_DAILY_FACTORS / f"{prefix}.parquet"))
    ic_result.ic_series.write_parquet(str(OUTPUT_DAILY_RESULTS / f"{prefix}_ic.parquet"))
    bt_result.returns.write_parquet(str(OUTPUT_DAILY_RESULTS / f"{prefix}_bt_returns.parquet"))
    bt_result.nav.write_parquet(str(OUTPUT_DAILY_RESULTS / f"{prefix}_bt_nav.parquet"))
    bt_result.positions.write_parquet(str(OUTPUT_DAILY_RESULTS / f"{prefix}_bt_positions.parquet"))
    bt_result.trades.write_parquet(str(OUTPUT_DAILY_RESULTS / f"{prefix}_bt_trades.parquet"))
    to_result.daily_turnover.write_parquet(str(OUTPUT_DAILY_RESULTS / f"{prefix}_to_daily.parquet"))
    to_result.migration_matrix.write_parquet(
        str(OUTPUT_DAILY_RESULTS / f"{prefix}_to_matrix.parquet")
    )

    meta = {
        "factor_name": ic_result.factor_name,
        "frequency": ic_result.frequency,
        "ic_mean": ic_result.ic_mean,
        "ic_std": ic_result.ic_std,
        "ir": ic_result.ir,
        "ic_positive_ratio": ic_result.ic_positive_ratio,
        "n_periods": ic_result.n_periods,
        "ic_tstat": ic_result.ic_tstat,
        "ic_pvalue": ic_result.ic_pvalue,
        "decay": {str(k): v for k, v in ic_result.decay.items()},
        "multi_period": {str(k): v for k, v in ic_result.multi_period.items()},
        "oos_ic": ic_result.oos_ic,
        "bt_factor_name": bt_result.factor_name,
        "bt_strategy_name": bt_result.strategy_name,
        "bt_n_groups": bt_result.n_groups,
        "bt_summary_stats": {str(k): v for k, v in bt_result.summary_stats.items()},
        "bt_frequency": bt_result.frequency,
        "bt_config": bt_result.config,
        "bt_ret_definition": bt_result.ret_definition,
        "to_factor_name": to_result.factor_name,
        "to_avg_turnover": to_result.avg_turnover,
        "to_frequency": to_result.frequency,
    }
    _meta_path(factor_name, start, end).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"中间结果已落盘: output/daily/results/{prefix}_*.parquet")


def _load_results(
    factor_name: str, start: str, end: str
) -> "tuple[pl.DataFrame, ICAnalysisResult, BacktestResult, TurnoverResult] | None":
    """从磁盘加载已有的评价结果。若文件不存在返回 None。"""
    mp = _meta_path(factor_name, start, end)
    if not mp.exists():
        return None

    prefix = f"{factor_name}_{start}_{end}"
    ic_path = OUTPUT_DAILY_RESULTS / f"{prefix}_ic.parquet"
    bt_ret_path = OUTPUT_DAILY_RESULTS / f"{prefix}_bt_returns.parquet"
    bt_nav_path = OUTPUT_DAILY_RESULTS / f"{prefix}_bt_nav.parquet"
    bt_pos_path = OUTPUT_DAILY_RESULTS / f"{prefix}_bt_positions.parquet"
    bt_trades_path = OUTPUT_DAILY_RESULTS / f"{prefix}_bt_trades.parquet"
    to_daily_path = OUTPUT_DAILY_RESULTS / f"{prefix}_to_daily.parquet"
    to_mat_path = OUTPUT_DAILY_RESULTS / f"{prefix}_to_matrix.parquet"
    factor_path = OUTPUT_DAILY_FACTORS / f"{prefix}.parquet"

    for p in [
        ic_path,
        bt_ret_path,
        bt_nav_path,
        bt_pos_path,
        bt_trades_path,
        to_daily_path,
        to_mat_path,
        factor_path,
    ]:
        if not p.exists():
            logger.warning(f"--reuse: 缺少文件 {p.name}，退回重新计算")
            return None

    meta = json.loads(mp.read_text(encoding="utf-8"))

    clean_df = pl.read_parquet(str(factor_path))
    ic_result = ICAnalysisResult(
        factor_name=meta["factor_name"],
        ic_mean=meta["ic_mean"],
        ic_std=meta["ic_std"],
        ir=meta["ir"],
        ic_positive_ratio=meta["ic_positive_ratio"],
        n_periods=meta["n_periods"],
        ic_series=pl.read_parquet(str(ic_path)),
        decay={int(k): v for k, v in meta["decay"].items()},
        frequency=meta["frequency"],
        ic_tstat=meta["ic_tstat"],
        ic_pvalue=meta["ic_pvalue"],
        multi_period={int(k): v for k, v in meta["multi_period"].items()},
        oos_ic=meta["oos_ic"],
    )
    bt_result = BacktestResult(
        factor_name=meta["bt_factor_name"],
        strategy_name=meta.get("bt_strategy_name", "quantile_long_short"),
        n_groups=meta["bt_n_groups"],
        returns=pl.read_parquet(str(bt_ret_path)),
        nav=pl.read_parquet(str(bt_nav_path)),
        positions=pl.read_parquet(str(bt_pos_path)),
        trades=pl.read_parquet(str(bt_trades_path)),
        summary_stats={
            (int(k) if k.isdigit() else k): v for k, v in meta["bt_summary_stats"].items()
        },
        config=meta.get("bt_config", {}),
        frequency=meta["bt_frequency"],
        ret_definition=meta.get("bt_ret_definition", "open_to_close_with_overnight_carry"),
    )
    to_result = TurnoverResult(
        factor_name=meta["to_factor_name"],
        avg_turnover=meta["to_avg_turnover"],
        migration_matrix=pl.read_parquet(str(to_mat_path)),
        daily_turnover=pl.read_parquet(str(to_daily_path)),
        frequency=meta["to_frequency"],
    )
    logger.info(f"--reuse: 从磁盘加载 {prefix} 评价结果")
    return clean_df, ic_result, bt_result, to_result


# ---------------------------------------------------------------------------
# 高级评价
# ---------------------------------------------------------------------------


def _run_advanced_evaluation(clean_df, ret_df, frequency, start: str = "", end: str = ""):
    """运行高级评价模块，各模块互不依赖，单个失败不影响整体。"""
    advanced: dict = {}

    try:
        from daily.evaluation.advanced import compute_ic_decay

        advanced["decay_results"] = compute_ic_decay(clean_df, ret_df, factor_col="factor_clean")
        logger.info(f"IC Decay: {len(advanced['decay_results'])} horizons")
    except ImportError as e:
        logger.warning(f"IC Decay 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"IC Decay 失败: {e}")

    try:
        from daily.evaluation.advanced import compute_monotonicity

        mono_df = clean_df.join(
            ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
            on=["trade_date", "ts_code"],
            how="inner",
        )
        advanced["mono"] = compute_monotonicity(
            mono_df, factor_col="factor_clean", ret_col="fwd_ret_1d"
        )
        logger.info(f"单调性: score={advanced['mono'].monotonicity_score:.3f}")
    except ImportError as e:
        logger.warning(f"Monotonicity 模块不可用: {e}")
    except Exception as e:
        logger.warning(f"单调性分析失败: {e}")

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

    try:
        from daily.evaluation.advanced import compute_market_regime_ic

        advanced["regime"] = compute_market_regime_ic(
            clean_df.join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
                on=["trade_date", "ts_code"],
                how="inner",
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

    # ── 行业分层 IC ──
    try:
        from common.loader import fetch_stock_basic
        from daily.evaluation.advanced import compute_sector_ic

        stock_basic = (
            fetch_stock_basic().select(["ts_code", "industry"]).rename({"industry": "sector"})
        )
        sector_df = (
            clean_df.join(stock_basic, on="ts_code", how="left")
            .join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
                on=["trade_date", "ts_code"],
                how="inner",
            )
            .filter(pl.col("sector").is_not_null() & (pl.col("sector") != ""))
        )
        if not sector_df.is_empty():
            advanced["sector"] = compute_sector_ic(
                sector_df,
                factor_col="factor_clean",
                ret_col="fwd_ret_1d",
                sector_col="sector",
                return_object=True,
            )
            logger.info(f"行业 IC: {advanced['sector'].sector_ic_df.height} 个行业")
    except Exception as e:
        logger.warning(f"行业分层 IC 失败: {e}")

    # ── 市值分层 IC ──
    try:
        from daily.evaluation.advanced import compute_size_ic

        kw = {}
        if start and end:
            kw = {"start": start, "end": end}
        db = load_parquet("daily_basic", **kw).collect()
        if db.is_empty() and start and end:
            db = load_parquet("daily_basic").collect()
        cap_df = (
            clean_df.join(
                db.select(["trade_date", "ts_code", "total_mv"]),
                on=["trade_date", "ts_code"],
                how="left",
            )
            .join(
                ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]),
                on=["trade_date", "ts_code"],
                how="inner",
            )
            .filter(pl.col("total_mv").is_not_null())
        )
        if not cap_df.is_empty():
            advanced["size"] = compute_size_ic(
                cap_df,
                factor_col="factor_clean",
                ret_col="fwd_ret_1d",
                cap_col="total_mv",
                n_buckets=3,
                return_object=True,
            )
            logger.info(f"市值分层 IC: {advanced['size'].buckets}")
    except Exception as e:
        logger.warning(f"市值分层 IC 失败: {e}")

    return advanced if advanced else None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="因子 Tear Sheet 报告生成")
    parser.add_argument("--factor", required=True, help="因子名称")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument("--universe", default="csi300", help="股票池")
    parser.add_argument(
        "--frequency", default="daily", choices=["daily", "weekly", "monthly"], help="因子频率"
    )
    parser.add_argument(
        "--reuse", action="store_true", help="复用已有 parquet 结果，跳过重新计算（需先跑过一次）"
    )
    args = parser.parse_args()

    logger.info(f"──── 因子报告生成: {args.factor} | {args.start} ~ {args.end} ────")

    # ── 1. 获取因子类 ──
    try:
        factor_cls = get_factor(args.factor)
    except KeyError as e:
        logger.error(str(e))
        sys.exit(1)
    factor = factor_cls()
    logger.info(f"因子: {factor.name} | {factor.description}")

    # ── --reuse 路径 ──
    reused = None
    if args.reuse:
        reused = _load_results(args.factor, args.start, args.end)

    if reused is not None:
        clean_df, ic_result, bt_result, to_result = reused
        # 高级评价仍需 ret_df，重新从存储加载（快速路径：只读收盘价）
        try:
            fetch_daily(args.start, args.end)
        except Exception as e:
            logger.warning(f"数据拉取失败（高级评价可能跳过）: {e}")
        daily = load_parquet("daily", start=args.start, end=args.end).collect()
        if not daily.is_empty():
            ret_df = daily.select(["trade_date", "ts_code", "close"]).sort(
                ["ts_code", "trade_date"]
            )
            ret_df = ret_df.with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1).alias("ret")
            )
            ret_df = compute_fwd_returns(ret_df, ret_col="ret")
            advanced_results = _run_advanced_evaluation(
                clean_df, ret_df, args.frequency, args.start, args.end
            )
        else:
            logger.warning("日线数据为空，跳过高级评价")
            advanced_results = None
    else:
        if args.reuse:
            logger.info("--reuse: 未找到缓存，退回完整计算")

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

        # ── 6. 前向收益 ──
        daily = load_parquet("daily", start=args.start, end=args.end).collect()
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

        # ── 10. 高级评价 ──
        advanced_results = _run_advanced_evaluation(clean_df, ret_df, args.frequency)

        # ── 持久化中间结果 ──
        _save_results(args.factor, args.start, args.end, clean_df, ic_result, bt_result, to_result)

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

    # ── 12. 落盘 HTML ──
    OUTPUT_DAILY_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DAILY_REPORTS / f"{factor.name}_{args.start}_{args.end}.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"报告已生成: {report_path}")
    logger.info("完成!")


if __name__ == "__main__":
    main()
