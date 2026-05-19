#!/usr/bin/env python
"""Optuna 超参搜索（基于 walk-forward OOS Sharpe）。

用法:
  python scripts/run_hyperparameter_search.py --factor momentum_20d --strategy quantile_long_short --n_trials 30
"""

from __future__ import annotations

import argparse
import sys

from common.logger import get_logger, setup_logging
from config.settings import OUTPUT_DAILY_FACTORS
from daily.evaluation.backtest import BacktestConfig
from daily.evaluation.hyperparameter import ParamSpec, TuningSpace, run_optuna_search
from daily.evaluation.walk_forward import WalkForwardSplitter, run_walk_forward

setup_logging()
logger = get_logger(__name__)


def _build_strategy_factory(strategy_name: str):  # type: ignore[return]
    """根据策略名称构建 strategy_factory。"""
    from daily.evaluation.backtest import (
        FactorWeightedStrategy,
        QuantileLongShortStrategy,
        TopNLongOnlyStrategy,
    )

    factories = {
        "quantile_long_short": lambda p: QuantileLongShortStrategy(n_groups=p.get("n_groups", 10)),
        "topn_long_only": lambda p: TopNLongOnlyStrategy(n=p.get("n", 50)),
        "factor_weighted": lambda p: FactorWeightedStrategy(long_only=True),
    }
    if strategy_name not in factories:
        logger.error(f"未知策略: {strategy_name}，可用: {list(factories.keys())}")
        sys.exit(1)
    return factories[strategy_name]


def _build_tuning_space(strategy_name: str) -> TuningSpace:
    """根据策略名称构建 TuningSpace。"""
    if strategy_name == "quantile_long_short":
        return TuningSpace([ParamSpec("n_groups", "int", 5, 20)])
    elif strategy_name == "topn_long_only":
        return TuningSpace([ParamSpec("n", "int", 20, 100)])
    else:
        # factor_weighted 无可调超参
        return TuningSpace([])


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna 超参搜索（基于 walk-forward OOS Sharpe）")
    parser.add_argument("--factor", required=True, help="因子名称")
    parser.add_argument("--start", required=True, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="截止日期 YYYYMMDD")
    parser.add_argument(
        "--strategy",
        default="quantile_long_short",
        choices=["quantile_long_short", "topn_long_only", "factor_weighted"],
        help="策略类型",
    )
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna 搜索次数")
    parser.add_argument("--train_days", type=int, default=252, help="训练集长度（交易日）")
    parser.add_argument("--test_days", type=int, default=63, help="测试集长度（交易日）")
    parser.add_argument("--embargo_days", type=int, default=5, help="训练集末到测试集首的间隔")
    parser.add_argument("--universe", default="csi300", help="股票池")
    args = parser.parse_args()

    # ── 1. 加载因子数据 ──
    factor_path = OUTPUT_DAILY_FACTORS / f"{args.factor}_{args.start}_{args.end}.parquet"
    if not factor_path.exists():
        logger.error(f"因子文件不存在: {factor_path}")
        sys.exit(1)

    import polars as pl

    logger.info(f"加载因子数据: {factor_path}")
    factor_df = pl.read_parquet(str(factor_path))

    # ── 2. 加载价格数据 ──
    try:
        from common.storage import load_parquet

        price_df = load_parquet("daily", start=args.start, end=args.end).collect()
    except Exception as e:
        logger.error(f"价格数据加载失败: {e}")
        sys.exit(1)

    if price_df.is_empty():
        logger.error("价格数据为空")
        sys.exit(1)

    # ── 3. 构建策略工厂和搜索空间 ──
    strategy_factory = _build_strategy_factory(args.strategy)
    space = _build_tuning_space(args.strategy)

    if not space.specs:
        logger.warning(f"策略 {args.strategy} 无可调超参，直接运行默认参数")
        print(f"策略 {args.strategy} 无可调超参数，best_params = {{}}")
        return

    splitter = WalkForwardSplitter(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.test_days,
        embargo_days=args.embargo_days,
    )
    config = BacktestConfig()

    # ── 4. 定义 objective_fn ──
    def objective_fn(params: dict) -> float:
        result = run_walk_forward(
            strategy_factory=strategy_factory,
            factor_df=factor_df,
            price_df=price_df,
            splitter=splitter,
            config=config,
            factor_name=args.factor,
            params=params,
        )
        return result.oos_sharpe_mean

    # ── 5. 运行 Optuna 搜索 ──
    logger.info(f"开始 Optuna 搜索: strategy={args.strategy}, n_trials={args.n_trials}")
    best_params, study = run_optuna_search(
        objective_fn=objective_fn,
        space=space,
        n_trials=args.n_trials,
        direction="maximize",
        study_name=f"wf_{args.factor}_{args.strategy}",
    )

    # ── 6. 打印结果 ──
    print("\n" + "=" * 60)
    print(f"最优超参数 (策略: {args.strategy}):")
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print(f"最优 OOS Sharpe 均值: {study.best_value:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
