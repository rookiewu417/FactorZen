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
    parser.add_argument("--train_days", type=int, default=252, help="IS 历史观察期长度（交易日）")
    parser.add_argument("--test_days", type=int, default=63, help="OOS 未来验证期长度（交易日）")
    parser.add_argument("--embargo_days", type=int, default=5, help="IS 期末到 OOS 期首的间隔")
    parser.add_argument("--universe", default="csi300", help="股票池")
    parser.add_argument("--config", type=str, default=None, help="YAML 运行配置文件路径")
    parser.add_argument("--seed", type=int, default=None, help="全局随机种子")
    args = parser.parse_args()

    # ── 0. 加载 YAML 配置（可选），CLI 参数优先级更高 ──
    run_config = None
    if args.config:
        from common.config_loader import load_run_config

        run_config = load_run_config(args.config)
        if args.universe == "csi300" and run_config.universe:
            args.universe = run_config.universe
        if args.seed is None and run_config.seed is not None:
            args.seed = run_config.seed
        if args.n_trials == 30 and run_config.walk_forward.n_trials != 50:
            args.n_trials = run_config.walk_forward.n_trials
        if args.train_days == 252 and run_config.walk_forward.train_days != 504:
            args.train_days = run_config.walk_forward.train_days
        if args.test_days == 63 and run_config.walk_forward.test_days != 63:
            args.test_days = run_config.walk_forward.test_days
        if args.embargo_days == 5 and run_config.walk_forward.embargo_days != 5:
            args.embargo_days = run_config.walk_forward.embargo_days

    # ── 0b. 设置全局随机种子（可选）──
    seed: int | None = args.seed
    if seed is not None:
        from common.seed import set_global_seed

        set_global_seed(seed)
        logger.info(f"全局随机种子已设置: {seed}")

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
        seed=seed,
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
