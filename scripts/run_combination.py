"""多因子合成 CLI。

用法:
    pixi run python scripts/run_combination.py \\
        --factors momentum_20d reversal_5d volatility_20d \\
        --method ic_weighted \\
        --start 20240101 --end 20250101
"""

import argparse
import sys

import polars as pl

from common.storage import load_parquet
from daily.combination.methods import equal_weight, ic_weighted, max_ir
from daily.data.context import FactorDataContext
from daily.evaluation.backtest import CostModel
from daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from daily.factors.registry import get_factor
from daily.preprocessing.pipeline import quick_preprocess


def _instantiate_factor(fname: str, registry_getter=get_factor):
    factor_cls = registry_getter(fname)
    return factor_cls()


def _prepare_return_frame(
    price_df: pl.DataFrame, horizons: list[int] | None = None
) -> pl.DataFrame:
    ret_df = (
        price_df.select(["trade_date", "ts_code", "close"])
        .sort(["ts_code", "trade_date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0).alias("ret")
        )
    )
    return compute_fwd_returns(ret_df, horizons=horizons, ret_col="ret")


def parse_args():
    p = argparse.ArgumentParser(description="多因子合成评估")
    p.add_argument("--factors", nargs="+", required=True, help="参与合成的因子名列表")
    p.add_argument(
        "--method", default="equal_weight", choices=["equal_weight", "ic_weighted", "max_ir"]
    )
    p.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    p.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    p.add_argument("--n_groups", type=int, default=5, help="分层回测组数")
    p.add_argument("--no_cost", action="store_true", help="不扣交易成本")
    return p.parse_args()


def main():
    args = parse_args()

    factor_dfs: dict[str, pl.DataFrame] = {}
    for fname in args.factors:
        try:
            factor = _instantiate_factor(fname)
        except KeyError:
            print(f"[错误] 未知因子: {fname}")
            sys.exit(1)

        ctx = FactorDataContext(
            start=args.start,
            end=args.end,
            required_data=factor.required_data,
            lookback_days=factor.lookback_days,
        )
        raw = factor.compute(ctx)
        processed = quick_preprocess(raw)
        factor_dfs[fname] = processed.rename({"factor_clean": "factor_value"}).select(
            ["trade_date", "ts_code", "factor_value"]
        )
        print(f"  {fname}: {len(factor_dfs[fname])} 行")

    # 加载价格数据
    price_df = load_parquet("daily", start=args.start, end=args.end).collect()
    ret_df = _prepare_return_frame(price_df, horizons=[1, 5])

    # 合成
    fwd1 = ret_df.select(["trade_date", "ts_code", "fwd_ret_1d"]).rename({"fwd_ret_1d": "ret"})
    if args.method == "equal_weight":
        combined = equal_weight(factor_dfs)
    elif args.method == "ic_weighted":
        combined = ic_weighted(factor_dfs, fwd1)
    else:
        combined = max_ir(factor_dfs, fwd1)

    combined = combined.rename({"factor_value": "factor_clean"})
    ic_result = compute_rank_ic(combined, ret_df, factor_col="factor_clean", horizons=[1, 5])
    print("\n=== 合成因子 IC 分析 ===")
    print(ic_result.summary())

    cost_model = None if args.no_cost else CostModel()
    from daily.evaluation.backtest import run_stratified_backtest

    bt_result = run_stratified_backtest(
        combined,
        price_df,
        factor_col="factor_clean",
        n_groups=args.n_groups,
        cost_model=cost_model,
    )
    ls = bt_result.summary_stats.get("long_short", {})
    print(f"\n=== 多空回测 ({args.method}) ===")
    print(f"  年化收益: {ls.get('ann_ret', 0):.2%}")
    print(f"  Sharpe:   {ls.get('sharpe', 0):.2f}")
    print(f"  最大回撤: {ls.get('max_dd', 0):.2%}")


if __name__ == "__main__":
    main()
