"""策略化回测引擎。

核心口径：
- t 日因子生成目标权重。
- t+1 开盘按约束调仓。
- 旧持仓承担隔夜收益，成交后持仓承担开盘到收盘收益。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from factorzen.daily.optimization.base import OptimizerConstraints, PortfolioOptimizer

import numpy as np
import polars as pl

from factorzen.config.constants import (
    BORROW_RATE_ANNUAL,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
    STAMP_TAX_RATE,
    TRADING_DAYS_PER_YEAR,
)
from factorzen.core.universe import _get_board_limit
from factorzen.core.validation import require_columns
from factorzen.daily.evaluation.cost_models import CostModelBase


@dataclass
class CostModel:
    """A 股交易成本模型。

    所有费率均为小数（非百分比），例如 commission=0.00025 表示万 2.5。
    """

    commission: float = COMMISSION_RATE
    stamp_tax: float = STAMP_TAX_RATE
    slippage: float = SLIPPAGE_RATE
    borrow_annual: float = BORROW_RATE_ANNUAL

    def one_way_cost(self) -> float:
        """单边买入成本率。"""
        return self.commission + self.slippage

    def sell_cost(self) -> float:
        """单边卖出成本率。"""
        return self.commission + self.slippage + self.stamp_tax

    def round_trip_cost(self) -> float:
        """一次完整换手（卖旧 + 买新）的成本率。"""
        return self.sell_cost() + self.one_way_cost()

    def borrow_rate_per_period(self, frequency: str = "daily") -> float:
        """融券日/周/月费率（按年化利率折算）。"""
        days = {"daily": 1, "weekly": 5, "monthly": 21}.get(frequency, 1)
        return self.borrow_annual * days / TRADING_DAYS_PER_YEAR

    def trade_cost(self, delta_weight: float) -> float:
        """按权重变动方向计算单笔成交成本。"""
        if delta_weight > 0:
            return abs(delta_weight) * self.one_way_cost()
        if delta_weight < 0:
            return abs(delta_weight) * self.sell_cost()
        return 0.0


@dataclass
class BacktestConfig:
    """策略回测配置。"""

    factor_col: str = "factor_clean"
    frequency: str = "daily"
    initial_capital: float = 100_000_000.0
    max_participation_rate: float = 0.05
    max_gross_exposure: float = 2.0
    max_abs_weight: float = 1.0
    limit_up_pct: float = 9.8
    limit_down_pct: float = -9.8
    execution_price: str = "next_open"
    ret_definition: str = "open_to_close_with_overnight_carry"
    rebalance_threshold: float | None = None  # 换手率低于此阈值时跳过调仓（None=每期调仓）
    strategy_type: str | None = None
    strategy_params: dict[str, Any] = field(default_factory=dict)
    cost_model: str | None = None
    alpha: float | None = None
    fallback_adv: float | None = None


@dataclass
class BacktestContext:
    """策略生成目标权重时可见的上下文。"""

    signal_date: date
    execution_date: date
    factor_slice: pl.DataFrame
    price_slice: pl.DataFrame
    current_positions: pl.DataFrame
    factor_col: str = "factor_clean"
    price_history: pl.DataFrame = field(default_factory=pl.DataFrame)
    adv_20d: dict[str, float] = field(
        default_factory=dict
    )  # ts_code → 20日均成交额（元），由 _compute_adv_20d 填充


class Strategy(ABC):
    """策略抽象类。用户实现 generate_weights 返回目标权重。"""

    name: str = "strategy"

    @abstractmethod
    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        """返回列 ts_code, target_weight。"""


@dataclass
class StrategyBacktestResult:
    """策略回测结果。"""

    factor_name: str
    strategy_name: str
    n_groups: int
    returns: pl.DataFrame
    nav: pl.DataFrame
    positions: pl.DataFrame
    trades: pl.DataFrame
    summary_stats: dict[int | str, dict[str, float]]
    config: dict[str, Any]
    frequency: str = "daily"
    ret_definition: str = "open_to_close_with_overnight_carry"

    @property
    def daily_returns(self) -> pl.DataFrame:
        """兼容旧报告落盘字段。"""
        return self.returns

    @property
    def long_short_nav(self) -> pl.DataFrame:
        """兼容旧报告字段：策略净值即组合净值。"""
        return self.nav.select(["trade_date", pl.col("net_return").alias("ret"), "nav"])

    def summary(self) -> str:
        freq_label = {"daily": "日频", "weekly": "周频", "monthly": "月频"}.get(
            self.frequency, self.frequency
        )
        stats = self.summary_stats.get("long_short") or self.summary_stats.get("portfolio", {})
        return (
            f"Strategy Backtest ({self.strategy_name}, {freq_label}):\n"
            f"  Portfolio: ret={stats.get('ann_ret', 0):.2%} "
            f"Sharpe={stats.get('sharpe', 0):.2f} "
            f"MaxDD={stats.get('max_dd', 0):.1%}"
        )


BacktestResult = StrategyBacktestResult


def trim_backtest_to_first_trade(result: StrategyBacktestResult) -> StrategyBacktestResult:
    """Drop leading cash-only rows from cached backtest results."""
    if result.returns.is_empty():
        return result

    first_trade_date = None
    if not result.trades.is_empty() and "trade_date" in result.trades.columns:
        first_trade_date = result.trades.select(pl.col("trade_date").min()).item()
    if first_trade_date is None:
        candidate = result.returns.filter(
            (pl.col("turnover").abs() > 1e-12)
            | (pl.col("net_return").abs() > 1e-12)
            | ((pl.col("cash_weight") - 1.0).abs() > 1e-12)
        )
        if candidate.is_empty():
            return result
        first_trade_date = candidate.sort("trade_date")["trade_date"][0]

    returns = result.returns.filter(pl.col("trade_date") >= first_trade_date)
    nav = result.nav.filter(pl.col("trade_date") >= first_trade_date)
    positions = (
        result.positions.filter(pl.col("trade_date") >= first_trade_date)
        if not result.positions.is_empty()
        else result.positions
    )
    trades = (
        result.trades.filter(pl.col("trade_date") >= first_trade_date)
        if not result.trades.is_empty()
        else result.trades
    )
    if returns.height == result.returns.height:
        return result
    return StrategyBacktestResult(
        factor_name=result.factor_name,
        strategy_name=result.strategy_name,
        n_groups=result.n_groups,
        returns=returns,
        nav=nav,
        positions=positions,
        trades=trades,
        summary_stats=_summary_stats(returns, trades),
        config=result.config,
        frequency=result.frequency,
        ret_definition=result.ret_definition,
    )


class QuantileLongShortStrategy(Strategy):
    """做多最高分组、做空最低分组。"""

    name = "quantile_long_short"

    def __init__(self, n_groups: int = 10, factor_col: str = "factor_clean") -> None:
        self.n_groups = n_groups
        self.factor_col = factor_col

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        factor_col = context.factor_col or self.factor_col
        df = _valid_factor_slice(context.factor_slice, factor_col)
        if df.is_empty():
            return _empty_weights()

        grouped = (
            df.with_columns(pl.col(factor_col).rank("ordinal", descending=False).alias("_rank"))
            .with_columns(
                ((pl.col("_rank") - 1) * self.n_groups // pl.col("_rank").max())
                .cast(pl.Int32)
                .alias("_group")
            )
            .select(["ts_code", "_group"])
        )
        long_df = grouped.filter(pl.col("_group") == self.n_groups - 1)
        short_df = grouped.filter(pl.col("_group") == 0)
        parts: list[pl.DataFrame] = []
        if not long_df.is_empty():
            parts.append(
                long_df.select(["ts_code"]).with_columns(
                    pl.lit(1.0 / long_df.height).alias("target_weight")
                )
            )
        if not short_df.is_empty():
            parts.append(
                short_df.select(["ts_code"]).with_columns(
                    pl.lit(-1.0 / short_df.height).alias("target_weight")
                )
            )
        return pl.concat(parts) if parts else _empty_weights()


class TopNLongOnlyStrategy(Strategy):
    """做多因子值最高的 N 只股票。"""

    name = "topn_long_only"

    def __init__(self, n: int = 50, factor_col: str = "factor_clean") -> None:
        self.n = n
        self.factor_col = factor_col

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        factor_col = context.factor_col or self.factor_col
        top = (
            _valid_factor_slice(context.factor_slice, factor_col)
            .sort(factor_col, descending=True)
            .head(self.n)
        )
        if top.is_empty():
            return _empty_weights()
        return top.select(["ts_code"]).with_columns(pl.lit(1.0 / top.height).alias("target_weight"))


class PrecomputedWeightsStrategy(Strategy):
    """Use target weights that were precomputed by signal date."""

    name = "precomputed_weights"

    def __init__(self, weights_by_date: dict[date, pl.DataFrame]) -> None:
        self.weights_by_date = weights_by_date

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        weights = self.weights_by_date.get(context.signal_date)
        if weights is None:
            return _empty_weights()
        return weights


class FactorWeightedStrategy(Strategy):
    """按因子强度生成权重。"""

    name = "factor_weighted"

    def __init__(
        self,
        long_only: bool = False,
        gross_exposure: float = 2.0,
        long_exposure: float = 1.0,
        factor_col: str = "factor_clean",
    ) -> None:
        self.long_only = long_only
        self.gross_exposure = gross_exposure
        self.long_exposure = long_exposure
        self.factor_col = factor_col

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        factor_col = context.factor_col or self.factor_col
        df = _valid_factor_slice(context.factor_slice, factor_col)
        if df.is_empty():
            return _empty_weights()

        if self.long_only:
            weighted = df.with_columns(
                pl.when(pl.col(factor_col) > 0).then(pl.col(factor_col)).otherwise(0).alias("_w")
            )
            total = weighted["_w"].sum()
            if total is None or total <= 0:
                return _empty_weights()
            return weighted.select(
                ["ts_code", (pl.col("_w") / total * self.long_exposure).alias("target_weight")]
            )

        mean_val = df[factor_col].mean()
        demeaned = df.with_columns((pl.col(factor_col) - mean_val).alias("_score"))
        _long_raw = demeaned.filter(pl.col("_score") > 0)["_score"].sum()
        _short_raw = demeaned.filter(pl.col("_score") < 0)["_score"].sum()
        long_sum: float = float(_long_raw) if _long_raw is not None else 0.0  # type: ignore[arg-type]
        short_sum: float = abs(float(_short_raw) if _short_raw is not None else 0.0)  # type: ignore[arg-type]
        if long_sum <= 0 or short_sum <= 0:
            return _empty_weights()
        side_exposure = self.gross_exposure / 2
        return demeaned.select(
            [
                "ts_code",
                pl.when(pl.col("_score") > 0)
                .then(pl.col("_score") / long_sum * side_exposure)
                .when(pl.col("_score") < 0)
                .then(pl.col("_score") / short_sum * side_exposure)
                .otherwise(0.0)
                .alias("target_weight"),
            ]
        )


class OptimizerStrategy(Strategy):
    """凸优化策略：用因子值作为预期收益，历史价格估计协方差，调用 optimizer 求最优权重。

    Args:
        optimizer: PortfolioOptimizer 实例（MeanVarianceOptimizer 等）。
        lookback_days: 估计协方差的历史窗口长度（交易日）。
        factor_col: 因子列名。
        cov_estimator: 协方差估计方法，"sample" / "ledoit_wolf" / "ewma"。
        constraints: OptimizerConstraints，若 None 则使用默认。
        long_only: 是否仅允许多头（min_weight=0）。
        top_n: 仅保留因子值最高的 N 只股票进入优化（None=全部）。
    """

    name = "optimizer_strategy"

    def __init__(
        self,
        optimizer: PortfolioOptimizer,
        lookback_days: int = 60,
        factor_col: str = "factor_clean",
        cov_estimator: str = "sample",
        constraints: OptimizerConstraints | None = None,
        long_only: bool = True,
        top_n: int | None = 100,
    ) -> None:
        from factorzen.daily.optimization.base import OptimizerConstraints

        self.optimizer = optimizer
        self.lookback_days = lookback_days
        self.factor_col = factor_col
        self.cov_estimator = cov_estimator
        self.constraints = constraints or OptimizerConstraints()
        self.long_only = long_only
        self.top_n = top_n
        if long_only:
            self.constraints.min_weight = 0.0

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        from factorzen.daily.optimization.base import OptimizerConstraints
        from factorzen.daily.optimization.covariance import (
            ewma_covariance,
            ledoit_wolf_shrinkage,
            sample_covariance,
        )

        factor_col = context.factor_col or self.factor_col
        df = _valid_factor_slice(context.factor_slice, factor_col)
        if df.is_empty():
            return _empty_weights()

        # Select top_n by factor score if requested
        if self.top_n is not None and df.height > self.top_n:
            df = df.sort(factor_col, descending=True).head(self.top_n)

        codes = df["ts_code"].to_list()
        n = len(codes)

        # Build expected_returns from factor values (z-score normalization)
        factor_vals = df[factor_col].to_numpy().astype(float)
        std_f = float(np.std(factor_vals))
        if std_f > 1e-8:
            mu = factor_vals / std_f
        else:
            mu = np.ones(n)

        # Build covariance matrix from price_history
        cov = np.eye(n) * 0.0001  # default: tiny diagonal if no history
        if not context.price_history.is_empty():
            hist = context.price_history.filter(pl.col("ts_code").is_in(codes))
            # Pivot to wide format: rows=dates, cols=ts_code
            try:
                pivoted = hist.select(["trade_date", "ts_code", "close"]).sort(
                    ["trade_date", "ts_code"]
                )
                wide = pivoted.pivot(index="trade_date", on="ts_code", values="close")
                # Compute returns and extract in codes order
                code_cols = [c for c in wide.columns if c != "trade_date" and c in codes]
                present_codes = code_cols
                if len(present_codes) >= 2:
                    price_mat = wide.select(present_codes).to_numpy().astype(float)
                    ret_mat = np.diff(np.log(np.clip(price_mat, 1e-6, None)), axis=0)
                    # Remove rows with any NaN
                    valid_rows = np.isfinite(ret_mat).all(axis=1)
                    ret_mat = ret_mat[valid_rows]
                    if ret_mat.shape[0] >= 5:
                        if self.cov_estimator == "ledoit_wolf":
                            cov_present = ledoit_wolf_shrinkage(ret_mat)
                        elif self.cov_estimator == "ewma":
                            cov_present = ewma_covariance(ret_mat)
                        else:
                            cov_present = sample_covariance(ret_mat)
                        # Map back to codes order
                        code_to_idx = {c: i for i, c in enumerate(present_codes)}
                        cov = np.eye(n) * np.diag(cov_present).mean()
                        for i, ci in enumerate(codes):
                            for j, cj in enumerate(codes):
                                if ci in code_to_idx and cj in code_to_idx:
                                    cov[i, j] = cov_present[code_to_idx[ci], code_to_idx[cj]]
            except Exception:
                pass  # fallback to diagonal

        # Regularize covariance
        cov = cov + 1e-6 * np.eye(n)

        # Build constraints
        cons = OptimizerConstraints(
            max_weight=self.constraints.max_weight,
            min_weight=self.constraints.min_weight,
            gross_exposure=self.constraints.gross_exposure,
            net_exposure=self.constraints.net_exposure,
            turnover_limit=self.constraints.turnover_limit,
        )

        # Prev weights for turnover constraint
        if not context.current_positions.is_empty() and self.constraints.turnover_limit is not None:
            pos_map = dict(
                zip(
                    context.current_positions["ts_code"].to_list(),
                    context.current_positions["weight"].to_list(),
                    strict=False,
                )
            )
            cons.prev_weights = np.array([pos_map.get(c, 0.0) for c in codes])

        weights = self.optimizer.solve(mu, cov, cons)
        weights = np.clip(weights, cons.min_weight, cons.max_weight)
        total = np.sum(np.abs(weights))
        if total < 1e-8:
            weights = np.full(n, 1.0 / n)
        elif total > cons.gross_exposure:
            weights = weights / total * cons.gross_exposure

        return pl.DataFrame({"ts_code": codes, "target_weight": weights.tolist()})


def run_strategy_backtest(
    strategy: Strategy,
    factor_df: pl.DataFrame,
    price_df: pl.DataFrame,
    config: BacktestConfig | None = None,
    cost_model: CostModel | CostModelBase | None = None,
    factor_name: str = "",
    *,
    collect_positions: bool = True,
    collect_trades: bool = True,
    include_context_positions: bool = True,
) -> StrategyBacktestResult:
    """运行策略回测。"""
    cfg = config or BacktestConfig()
    factor = _prepare_factor_df(factor_df, cfg.factor_col)
    price = _prepare_price_df(price_df)
    trade_dates = price.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    if _can_use_precomputed_fast_path(
        strategy,
        cost_model,
        collect_positions=collect_positions,
        collect_trades=collect_trades,
        include_context_positions=include_context_positions,
    ):
        return _run_precomputed_weights_backtest_fast(
            strategy=cast(PrecomputedWeightsStrategy, strategy),
            price=price,
            trade_dates=trade_dates,
            config=cfg,
            cost_model=cast(CostModel | None, cost_model),
            factor_name=factor_name,
        )

    current_weights: dict[str, float] = {}
    nav_value = 1.0
    has_started = False
    nav_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    price_by_date = _group_frames_by_date(price)
    factor_by_date = _group_frames_by_date(factor)
    adv_20d_by_date = _precompute_adv_20d_by_date(price, trade_dates)

    # Determine lookback for OptimizerStrategy
    _lookback = getattr(strategy, "lookback_days", 0)

    for i, execution_date in enumerate(trade_dates):
        price_slice = price_by_date[execution_date]
        price_map = _price_records(price_slice)
        signal_date = trade_dates[i - 1] if i > 0 else None

        overnight_return = _weighted_return(current_weights, price_map, "overnight_ret")
        open_weights = _drift_weights(current_weights, price_map, overnight_return)
        open_nav_value = nav_value * (1.0 + overnight_return)

        target_weights: dict[str, float] = {}
        adv_20d: dict[str, float] = {}
        has_signal = signal_date is not None and signal_date in factor_by_date
        if has_signal:
            assert signal_date is not None  # has_signal 已蕴含
            has_started = True
            adv_20d = adv_20d_by_date.get(execution_date, {})
            context = BacktestContext(
                signal_date=signal_date,
                execution_date=execution_date,
                factor_slice=factor_by_date[signal_date],
                price_slice=price_slice,
                current_positions=_positions_frame(
                    open_weights, open_nav_value, cfg.initial_capital
                )
                if include_context_positions
                else pl.DataFrame(schema=_positions_schema()),
                factor_col=cfg.factor_col,
                price_history=_get_price_history(price, trade_dates, i, _lookback),
                adv_20d=adv_20d,
            )
            target_df = _validate_target_weights(strategy.generate_weights(context), cfg)
            target_weights = dict(
                zip(target_df["ts_code"], target_df["target_weight"], strict=True)
            )

            # 换手率低于阈值时跳过本次调仓
            if cfg.rebalance_threshold is not None:
                all_proposed = set(open_weights) | set(target_weights)
                proposed_turnover = sum(
                    abs(target_weights.get(c, 0.0) - open_weights.get(c, 0.0)) for c in all_proposed
                )
                if proposed_turnover <= cfg.rebalance_threshold:
                    target_weights = dict(open_weights)
        else:
            target_weights = dict(open_weights)

        all_codes = sorted(set(open_weights) | set(target_weights))
        next_weights = dict(open_weights)
        trade_cost = 0.0
        turnover = 0.0
        for code in all_codes:
            prev_weight = open_weights.get(code, 0.0)
            target_weight = target_weights.get(code, 0.0)
            filled_delta, reason = _apply_trade_constraints(
                code=code,
                delta=target_weight - prev_weight,
                price_map=price_map,
                portfolio_value=open_nav_value * cfg.initial_capital,
                config=cfg,
                adv=adv_20d.get(code),
            )
            next_weight = prev_weight + filled_delta
            if abs(next_weight) < 1e-12:
                next_weights.pop(code, None)
            else:
                next_weights[code] = next_weight
            cost = (
                _trade_cost(cost_model, filled_delta, adv_20d.get(code))
                if cost_model is not None
                else 0.0
            )
            trade_cost += cost
            turnover += abs(filled_delta)
            if collect_trades and (
                abs(filled_delta) > 0 or abs(target_weight - prev_weight) > 1e-12
            ):
                trade_rows.append(
                    {
                        "trade_date": execution_date,
                        "ts_code": code,
                        "prev_weight": prev_weight,
                        "target_weight": target_weight,
                        "filled_delta_weight": filled_delta,
                        "turnover": abs(filled_delta),
                        "cost": cost,
                        "block_reason": reason,
                    }
                )

        intraday_return = _weighted_return(next_weights, price_map, "intraday_ret")
        borrow_cost = 0.0
        if cost_model is not None:
            short_exposure = sum(abs(w) for w in next_weights.values() if w < 0)
            borrow_cost = short_exposure * cost_model.borrow_rate_per_period(cfg.frequency)
        gross_return = (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0
        period_cost_scale = 1.0 + overnight_return
        period_trade_cost = trade_cost * period_cost_scale
        period_borrow_cost = borrow_cost * period_cost_scale
        net_return = gross_return - period_trade_cost - period_borrow_cost
        nav_value *= 1.0 + net_return
        close_weights = _drift_weights(
            next_weights, price_map, intraday_return, return_col="intraday_ret"
        )
        if 1.0 + net_return > 1e-12:
            cost_scale = (1.0 + gross_return) / (1.0 + net_return)
            close_weights = {
                code: weight * cost_scale
                for code, weight in close_weights.items()
                if abs(weight * cost_scale) >= 1e-12
            }
        cash_weight = 1.0 - sum(close_weights.values())
        if has_started:
            nav_rows.append(
                {
                    "trade_date": execution_date,
                    "gross_return": gross_return,
                    "cost": period_trade_cost,
                    "borrow_cost": period_borrow_cost,
                    "net_return": net_return,
                    "nav": nav_value,
                    "cash_weight": cash_weight,
                    "turnover": turnover,
                }
            )
            if collect_positions:
                for code, weight in sorted(close_weights.items()):
                    position_rows.append(
                        {
                            "trade_date": execution_date,
                            "ts_code": code,
                            "weight": weight,
                            "market_value": weight * nav_value * cfg.initial_capital,
                        }
                    )
        current_weights = close_weights

    returns = pl.DataFrame(nav_rows, schema=_returns_schema())
    nav = _build_nav_frame(returns, trade_dates)
    positions = (
        pl.DataFrame(position_rows, schema=_positions_schema())
        if collect_positions and position_rows
        else pl.DataFrame(schema=_positions_schema())
    )
    trades = (
        pl.DataFrame(trade_rows, schema=_trades_schema())
        if collect_trades and trade_rows
        else pl.DataFrame(schema=_trades_schema())
    )
    summary = _summary_stats(returns, trades)
    n_groups = getattr(strategy, "n_groups", 1)

    return StrategyBacktestResult(
        factor_name=factor_name,
        strategy_name=strategy.name,
        n_groups=n_groups,
        returns=returns,
        nav=nav,
        positions=positions,
        trades=trades,
        summary_stats=summary,
        config=asdict(cfg),
        frequency=cfg.frequency,
        ret_definition=cfg.ret_definition,
    )


def run_stratified_backtest(
    factor_df: pl.DataFrame,
    price_df: pl.DataFrame,
    factor_col: str = "factor_clean",
    n_groups: int = 10,
    frequency: str = "daily",
    factor_name: str = "",
    cost_model: CostModel | CostModelBase | None = None,
    config: BacktestConfig | None = None,
) -> StrategyBacktestResult:
    """分层多空策略回测入口。"""
    cfg = config or BacktestConfig(factor_col=factor_col, frequency=frequency)
    strategy = QuantileLongShortStrategy(n_groups=n_groups, factor_col=factor_col)
    result = run_strategy_backtest(strategy, factor_df, price_df, cfg, cost_model, factor_name)
    return result


def _valid_factor_slice(df: pl.DataFrame, factor_col: str) -> pl.DataFrame:
    return df.filter(pl.col(factor_col).is_not_null() & pl.col(factor_col).is_finite()).select(
        ["ts_code", factor_col]
    )


def _empty_weights() -> pl.DataFrame:
    return pl.DataFrame(schema={"ts_code": pl.Utf8, "target_weight": pl.Float64})


def _prepare_factor_df(df: pl.DataFrame, factor_col: str) -> pl.DataFrame:
    require_columns(df, ["trade_date", "ts_code", factor_col], context="factor_df")
    return _ensure_date(df, "trade_date").select(["trade_date", "ts_code", factor_col])


def precompute_top_n_weights(
    factor_df: pl.DataFrame,
    *,
    top_n: int,
    factor_col: str = "factor_clean",
) -> dict[date, pl.DataFrame]:
    """Precompute TopN target weights for each signal date."""
    factor = _prepare_factor_df(factor_df, factor_col)
    weights_by_date: dict[date, pl.DataFrame] = {}
    n = max(1, int(top_n))
    for key, frame in factor.group_by("trade_date"):
        signal_date = key[0] if isinstance(key, tuple) else key
        top = _valid_factor_slice(frame, factor_col).sort(factor_col, descending=True).head(n)
        if top.is_empty():
            continue
        weights_by_date[signal_date] = top.select(["ts_code"]).with_columns(
            pl.lit(1.0 / top.height).alias("target_weight")
        )
    return weights_by_date


def _prepare_price_df(df: pl.DataFrame) -> pl.DataFrame:
    require_columns(df, ["trade_date", "ts_code", "close"], context="price_df")
    out = _ensure_date(df, "trade_date")
    if "open" not in out.columns:
        out = out.with_columns(pl.col("close").alias("open"))
    out = out.sort(["ts_code", "trade_date"])
    if "pre_close" not in out.columns:
        out = out.with_columns(pl.col("close").shift(1).over("ts_code").alias("pre_close"))
    out = out.with_columns(
        [
            pl.col("pre_close").fill_null(pl.col("open")).alias("pre_close"),
            pl.col("open").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        ]
    )
    if "pct_chg" not in out.columns:
        out = out.with_columns(
            ((pl.col("close") / pl.col("pre_close") - 1.0) * 100).alias("pct_chg")
        )
    if "vol" not in out.columns:
        out = out.with_columns(pl.lit(1.0).alias("vol"))
    if "amount" not in out.columns:
        out = out.with_columns(pl.lit(1.0e30).alias("amount"))
    out = out.with_columns(
        [
            (pl.col("open") / pl.col("pre_close") - 1.0).fill_null(0.0).alias("overnight_ret"),
            (pl.col("close") / pl.col("open") - 1.0).fill_null(0.0).alias("intraday_ret"),
            pl.col("pct_chg").cast(pl.Float64),
            pl.col("vol").cast(pl.Float64),
            pl.col("amount").cast(pl.Float64),
        ]
    )
    return out.select(
        [
            "trade_date",
            "ts_code",
            "open",
            "close",
            "pre_close",
            "pct_chg",
            "vol",
            "amount",
            "overnight_ret",
            "intraday_ret",
        ]
    )


def _ensure_date(df: pl.DataFrame, col: str) -> pl.DataFrame:
    dtype = df.schema[col]
    if dtype == pl.Date:
        return df
    if dtype == pl.Datetime:
        return df.with_columns(pl.col(col).dt.date().alias(col))
    if dtype == pl.Utf8:
        parsed_dash = pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        parsed_plain = pl.col(col).str.strptime(pl.Date, "%Y%m%d", strict=False)
        return df.with_columns(parsed_dash.fill_null(parsed_plain).alias(col))
    return df


def _validate_target_weights(weights: pl.DataFrame, config: BacktestConfig) -> pl.DataFrame:
    if "ts_code" not in weights.columns or "target_weight" not in weights.columns:
        raise ValueError("strategy output must contain ts_code and target_weight")
    try:
        out = weights.select(
            [pl.col("ts_code").cast(pl.Utf8), pl.col("target_weight").cast(pl.Float64)]
        )
    except Exception as exc:
        raise ValueError("target_weight must be numeric") from exc
    if out["ts_code"].n_unique() != out.height:
        raise ValueError("strategy output contains duplicate ts_code")
    invalid = out.filter(pl.col("target_weight").is_null() | ~pl.col("target_weight").is_finite())
    if not invalid.is_empty():
        raise ValueError("target_weight must be finite")
    if not out.is_empty():
        max_abs = out["target_weight"].abs().max()
        gross = out["target_weight"].abs().sum()
        if max_abs is not None and float(max_abs) > config.max_abs_weight + 1e-12:  # type: ignore[arg-type]
            raise ValueError("target_weight exceeds max_abs_weight")
        if gross is not None and float(gross) > config.max_gross_exposure + 1e-12:  # type: ignore[arg-type]
            raise ValueError("gross exposure exceeds max_gross_exposure")
    return out


def _get_price_history(
    price: pl.DataFrame, trade_dates: list, current_idx: int, lookback: int
) -> pl.DataFrame:
    """获取截至 current_idx-1 的 lookback 期价格历史。"""
    if lookback <= 0 or current_idx <= 0:
        return pl.DataFrame()
    start_idx = max(0, current_idx - lookback)
    hist_dates = set(trade_dates[start_idx:current_idx])
    return price.filter(pl.col("trade_date").is_in(hist_dates))


def _compute_adv_20d(price: pl.DataFrame, trade_dates: list, current_idx: int) -> dict[str, float]:
    """Compute trailing 20-period average amount before the execution date."""
    if current_idx <= 0:
        return {}
    start_idx = max(0, current_idx - 20)
    hist_dates = set(trade_dates[start_idx:current_idx])
    hist = price.filter(pl.col("trade_date").is_in(hist_dates))
    if hist.is_empty() or "amount" not in hist.columns:
        return {}
    hist = hist.filter(pl.col("amount").is_finite() & (pl.col("amount") > 0))
    if hist.is_empty():
        return {}
    adv = hist.group_by("ts_code").agg(pl.col("amount").mean().alias("adv_20d"))
    return dict(zip(adv["ts_code"].to_list(), adv["adv_20d"].to_list(), strict=False))


def _precompute_adv_20d_by_date(
    price: pl.DataFrame,
    trade_dates: list[date],
) -> dict[date, dict[str, float]]:
    """Precompute trailing 20-period ADV for each execution date."""
    if not trade_dates or "amount" not in price.columns:
        return {}

    adv_frame = (
        price.select(["trade_date", "ts_code", "amount"])
        .sort(["ts_code", "trade_date"])
        .with_columns(
            pl.when(
                pl.col("amount").cast(pl.Float64).is_finite()
                & (pl.col("amount").cast(pl.Float64) > 0)
            )
            .then(pl.col("amount").cast(pl.Float64))
            .otherwise(None)
            .alias("_amount_for_adv")
        )
        .with_columns(
            pl.col("_amount_for_adv")
            .rolling_mean(20, min_samples=1)
            .shift(1)
            .over("ts_code")
            .alias("adv_20d")
        )
        .filter(pl.col("trade_date").is_in(trade_dates))
        .filter(pl.col("adv_20d").is_not_null() & pl.col("adv_20d").is_finite())
        .select(["trade_date", "ts_code", "adv_20d"])
    )

    result: dict[date, dict[str, float]] = {}
    for row in adv_frame.iter_rows(named=True):
        result.setdefault(row["trade_date"], {})[row["ts_code"]] = row["adv_20d"]
    return result


def _can_use_precomputed_fast_path(
    strategy: Strategy,
    cost_model: CostModel | CostModelBase | None,
    *,
    collect_positions: bool,
    collect_trades: bool,
    include_context_positions: bool,
) -> bool:
    return (
        isinstance(strategy, PrecomputedWeightsStrategy)
        and not collect_positions
        and not collect_trades
        and not include_context_positions
        and (cost_model is None or isinstance(cost_model, CostModel))
    )


def _build_nav_frame(returns: pl.DataFrame, trade_dates: list) -> pl.DataFrame:
    nav_cols = [
        "trade_date",
        "gross_return",
        "cost",
        "borrow_cost",
        "net_return",
        "nav",
        "cash_weight",
    ]
    if returns.is_empty():
        return returns.select(nav_cols)
    sorted_returns = returns.sort("trade_date")
    first_return_date = sorted_returns["trade_date"][0]
    first_return_idx = trade_dates.index(first_return_date)
    first_signal_date = trade_dates[first_return_idx - 1]
    base_nav = pl.DataFrame(
        {
            "trade_date": [first_signal_date],
            "gross_return": [0.0],
            "cost": [0.0],
            "borrow_cost": [0.0],
            "net_return": [0.0],
            "nav": [1.0],
            "cash_weight": [1.0],
        },
        schema={col: _returns_schema()[col] for col in nav_cols},
    )
    return pl.concat([base_nav, sorted_returns.select(nav_cols)])


def _run_precomputed_weights_backtest_fast(
    *,
    strategy: PrecomputedWeightsStrategy,
    price: pl.DataFrame,
    trade_dates: list[date],
    config: BacktestConfig,
    cost_model: CostModel | None,
    factor_name: str,
) -> StrategyBacktestResult:
    codes = price.select("ts_code").unique().sort("ts_code")["ts_code"].to_list()
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    date_to_idx = {trade_date: idx for idx, trade_date in enumerate(trade_dates)}
    shape = (len(trade_dates), len(codes))
    open_px = np.full(shape, np.nan, dtype=float)
    pre_close = np.full(shape, np.nan, dtype=float)
    vol_data = np.full(shape, np.nan, dtype=float)
    overnight_ret = np.zeros(shape, dtype=float)
    intraday_ret = np.zeros(shape, dtype=float)

    for row in price.iter_rows(named=True):
        row_date = row["trade_date"]
        code = row["ts_code"]
        date_idx = date_to_idx.get(row_date)
        code_idx = code_to_idx.get(code)
        if date_idx is None or code_idx is None:
            continue
        open_px[date_idx, code_idx] = float(row["open"])
        pre_close[date_idx, code_idx] = float(row["pre_close"])
        vol_data[date_idx, code_idx] = float(row["vol"] or 0.0)
        overnight_ret[date_idx, code_idx] = float(row["overnight_ret"] or 0.0)
        intraday_ret[date_idx, code_idx] = float(row["intraday_ret"] or 0.0)

    adv_by_date = _precompute_adv_20d_by_date(price, trade_dates)
    adv = np.full(shape, np.nan, dtype=float)
    for row_date, adv_values in adv_by_date.items():
        date_idx = date_to_idx.get(row_date)
        if date_idx is None:
            continue
        for code, value in adv_values.items():
            code_idx = code_to_idx.get(code)
            if code_idx is not None:
                adv[date_idx, code_idx] = float(value)

    board_limits = np.array([_get_board_limit(code) * 100.0 for code in codes], dtype=float)
    target_by_signal_date: dict[date, tuple[np.ndarray, np.ndarray]] = {}
    for sig_date, weight_df in strategy.weights_by_date.items():
        indices: list[int] = []
        values: list[float] = []
        for row in weight_df.iter_rows(named=True):
            code_idx = code_to_idx.get(row["ts_code"])
            if code_idx is not None:
                indices.append(code_idx)
                values.append(float(row["target_weight"]))
        if indices:
            target_by_signal_date[sig_date] = (
                np.array(indices, dtype=int),
                np.array(values, dtype=float),
            )

    weights = np.zeros(len(codes), dtype=float)
    nav_value = 1.0
    has_started = False
    nav_rows: list[dict[str, Any]] = []

    for i, execution_date in enumerate(trade_dates):
        overnight = overnight_ret[i]
        overnight_return = float(np.dot(weights, overnight))
        denom = 1.0 + overnight_return
        if abs(denom) < 1e-12:
            open_weights = weights.copy()
        else:
            open_weights = weights * (1.0 + overnight) / denom
            open_weights[np.abs(open_weights) < 1e-12] = 0.0
        open_nav_value = nav_value * (1.0 + overnight_return)

        signal_date = trade_dates[i - 1] if i > 0 else None
        target_weights = open_weights.copy()
        if signal_date is not None and signal_date in target_by_signal_date:
            has_started = True
            target_weights = np.zeros(len(codes), dtype=float)
            idx, vals = target_by_signal_date[signal_date]
            target_weights[idx] = vals
            if config.rebalance_threshold is not None:
                proposed_turnover = float(np.sum(np.abs(target_weights - open_weights)))
                if proposed_turnover <= config.rebalance_threshold:
                    target_weights = open_weights.copy()

        delta = target_weights - open_weights
        active = np.abs(delta) > 1e-12
        filled = np.zeros(len(codes), dtype=float)
        if np.any(active):
            open_today = open_px[i]
            pre_close_today = pre_close[i]
            vol_today = vol_data[i]
            valid_price = (
                np.isfinite(open_today)
                & np.isfinite(pre_close_today)
                & (open_today > 0)
                & (pre_close_today > 0)
            )
            # Suspended stocks have vol == 0; block both buy and sell
            not_suspended = np.isfinite(vol_today) & (vol_today > 0)
            opening_pct = np.zeros(len(codes), dtype=float)
            opening_pct[valid_price] = (
                open_today[valid_price] / pre_close_today[valid_price] - 1.0
            ) * 100.0
            tradable = (
                active
                & valid_price
                & not_suspended
                # 浮点容差与慢路径 _apply_trade_constraints 保持一致：
                # 创业板 open=11.98/pre_close=10.0 → opening_pct=19.7999...，
                # 若不减 1e-9 则 19.7999... >= 19.8 为 False，涨停买单被漏判。
                & ~((delta > 0) & (opening_pct >= board_limits - 1e-9))
                & ~((delta < 0) & (opening_pct <= -board_limits + 1e-9))
            )
            filled[tradable] = delta[tradable]

            portfolio_value = open_nav_value * config.initial_capital
            if portfolio_value <= 0:
                filled[:] = 0.0
            else:
                adv_eff = adv[i].copy()
                valid_adv = np.isfinite(adv_eff) & (adv_eff > 0)
                fallback_adv = config.fallback_adv
                if (
                    fallback_adv is not None
                    and np.isfinite(float(fallback_adv))
                    and float(fallback_adv) > 0
                ):
                    adv_eff[~valid_adv] = float(fallback_adv)
                    valid_adv = np.isfinite(adv_eff) & (adv_eff > 0)
                capped = tradable & valid_adv
                if np.any(capped):
                    max_delta = adv_eff[capped] * config.max_participation_rate / portfolio_value
                    filled[capped] = np.sign(filled[capped]) * np.minimum(
                        np.abs(filled[capped]), max_delta
                    )

        next_weights = open_weights + filled
        next_weights[np.abs(next_weights) < 1e-12] = 0.0
        if cost_model is None:
            trade_cost = 0.0
        else:
            buy_cost = np.where(filled > 0, np.abs(filled) * cost_model.one_way_cost(), 0.0)
            sell_cost = np.where(filled < 0, np.abs(filled) * cost_model.sell_cost(), 0.0)
            trade_cost = float(np.sum(buy_cost + sell_cost))
        turnover = float(np.sum(np.abs(filled)))

        intraday = intraday_ret[i]
        intraday_return = float(np.dot(next_weights, intraday))
        gross_return = (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0
        period_cost_scale = 1.0 + overnight_return
        period_trade_cost = trade_cost * period_cost_scale
        net_return = gross_return - period_trade_cost
        nav_value *= 1.0 + net_return

        close_denom = 1.0 + intraday_return
        if abs(close_denom) < 1e-12:
            close_weights = next_weights.copy()
        else:
            close_weights = next_weights * (1.0 + intraday) / close_denom
            close_weights[np.abs(close_weights) < 1e-12] = 0.0
        if 1.0 + net_return > 1e-12:
            cost_scale = (1.0 + gross_return) / (1.0 + net_return)
            close_weights *= cost_scale
            close_weights[np.abs(close_weights) < 1e-12] = 0.0
        cash_weight = float(1.0 - np.sum(close_weights))

        if has_started:
            nav_rows.append(
                {
                    "trade_date": execution_date,
                    "gross_return": gross_return,
                    "cost": period_trade_cost,
                    "borrow_cost": 0.0,
                    "net_return": net_return,
                    "nav": nav_value,
                    "cash_weight": cash_weight,
                    "turnover": turnover,
                }
            )
        weights = close_weights

    returns = pl.DataFrame(nav_rows, schema=_returns_schema())
    nav = _build_nav_frame(returns, trade_dates)
    positions = pl.DataFrame(schema=_positions_schema())
    trades = pl.DataFrame(schema=_trades_schema())
    summary = _summary_stats(returns, trades)
    return StrategyBacktestResult(
        factor_name=factor_name,
        strategy_name=strategy.name,
        n_groups=1,
        returns=returns,
        nav=nav,
        positions=positions,
        trades=trades,
        summary_stats=summary,
        config=asdict(config),
        frequency=config.frequency,
        ret_definition=config.ret_definition,
    )


def _trade_cost(
    cost_model: CostModel | CostModelBase,
    delta_weight: float,
    adv: float | None,
) -> float:
    """Call old and new cost model interfaces without changing their public API."""
    if isinstance(cost_model, CostModelBase):
        return cost_model.trade_cost(delta_weight, adv=adv)
    return cost_model.trade_cost(delta_weight)


def _price_records(price_slice: pl.DataFrame) -> dict[str, dict[str, Any]]:
    return {row["ts_code"]: row for row in price_slice.to_dicts()}


def _group_frames_by_date(df: pl.DataFrame) -> dict[date, pl.DataFrame]:
    grouped: dict[date, pl.DataFrame] = {}
    for key, frame in df.group_by("trade_date"):
        group_date = key[0] if isinstance(key, tuple) else key
        grouped[group_date] = frame
    return grouped


def _positions_frame(
    weights: dict[str, float], nav_value: float, initial_capital: float
) -> pl.DataFrame:
    if not weights:
        return pl.DataFrame(schema=_positions_schema())
    return pl.DataFrame(
        [
            {
                "ts_code": code,
                "weight": weight,
                "market_value": weight * nav_value * initial_capital,
            }
            for code, weight in sorted(weights.items())
        ]
    )


def _weighted_return(
    weights: dict[str, float], price_map: dict[str, dict[str, Any]], col: str
) -> float:
    total = 0.0
    for code, weight in weights.items():
        rec = price_map.get(code)
        if rec is not None and rec.get(col) is not None:
            total += weight * float(rec[col])
    return total


def _drift_weights(
    weights: dict[str, float],
    price_map: dict[str, dict[str, Any]],
    portfolio_return: float,
    return_col: str = "overnight_ret",
) -> dict[str, float]:
    denom = 1.0 + portfolio_return
    if abs(denom) < 1e-12:
        return dict(weights)
    drifted = {}
    for code, weight in weights.items():
        rec = price_map.get(code)
        asset_ret = (
            float(rec[return_col]) if rec is not None and rec.get(return_col) is not None else 0.0
        )
        drifted_weight = weight * (1.0 + asset_ret) / denom
        if abs(drifted_weight) >= 1e-12:
            drifted[code] = drifted_weight
    return drifted


def _apply_trade_constraints(
    *,
    code: str,
    delta: float,
    price_map: dict[str, dict[str, Any]],
    portfolio_value: float,
    config: BacktestConfig,
    adv: float | None = None,
) -> tuple[float, str]:
    if abs(delta) < 1e-12:
        return 0.0, ""
    rec = price_map.get(code)
    if rec is None or rec.get("open") is None or rec.get("pre_close") is None:
        return 0.0, "missing_price"
    open_price = float(rec["open"])
    pre_close = float(rec["pre_close"])
    if (
        not np.isfinite(open_price)
        or not np.isfinite(pre_close)
        or open_price <= 0
        or pre_close <= 0
    ):
        return 0.0, "missing_price"

    # Check if stock is suspended (vol == 0)
    vol = rec.get("vol")
    if vol is not None and float(vol) == 0.0:
        return 0.0, "suspended"

    opening_pct = (open_price / pre_close - 1.0) * 100.0
    board_limit_pct = _get_board_limit(code) * 100 if code else config.limit_up_pct
    effective_limit_up = board_limit_pct
    effective_limit_down = -board_limit_pct
    # 浮点容差：防止 (11.98/10-1)*100=19.7999... >= 19.8 漏判
    if delta > 0 and opening_pct >= effective_limit_up - 1e-9:
        return 0.0, "limit_up"
    if delta < 0 and opening_pct <= effective_limit_down + 1e-9:
        return 0.0, "limit_down"

    adv_value = float(adv) if adv is not None else 0.0
    if not np.isfinite(adv_value) or adv_value <= 0:
        fallback_adv = config.fallback_adv
        adv_value = float(fallback_adv) if fallback_adv is not None else 0.0
    if not np.isfinite(adv_value) or adv_value <= 0:
        return delta, ""

    max_trade_value = adv_value * config.max_participation_rate
    if portfolio_value <= 0:
        return 0.0, "invalid_portfolio_value"
    max_delta = max_trade_value / portfolio_value
    if abs(delta) > max_delta + 1e-12:
        return float(np.sign(delta) * max_delta), "capacity"
    return delta, ""


def _summary_stats(
    returns: pl.DataFrame, trades: pl.DataFrame
) -> dict[int | str, dict[str, float]]:
    if returns.is_empty():
        stats = {
            "ann_ret": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "avg_turnover": 0.0,
            "total_cost": 0.0,
            "ann_turnover": 0.0,
        }
        return {"portfolio": stats, "long_short": stats}
    rets = returns["net_return"].to_numpy()
    valid = rets[np.isfinite(rets)]
    if len(valid) == 0:
        ann_ret = ann_vol = sharpe = max_dd = 0.0
    else:
        ann_ret = float(np.mean(valid) * TRADING_DAYS_PER_YEAR)
        ann_vol = float(np.std(valid) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cum = np.concatenate([[1.0], np.cumprod(1 + valid)])
        max_dd = float(np.min(cum / np.maximum.accumulate(cum) - 1))
    avg_turnover = float(returns["turnover"].mean() or 0.0)  # type: ignore[arg-type]
    total_cost = float(returns["cost"].sum() or 0.0)  # type: ignore[arg-type]
    stats = {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_turnover": avg_turnover,
        "total_cost": total_cost,
        "ann_turnover": avg_turnover * TRADING_DAYS_PER_YEAR,
    }
    return {"portfolio": stats, "long_short": stats}


def _returns_schema() -> dict[str, Any]:
    return {
        "trade_date": pl.Date,
        "gross_return": pl.Float64,
        "cost": pl.Float64,
        "borrow_cost": pl.Float64,
        "net_return": pl.Float64,
        "nav": pl.Float64,
        "cash_weight": pl.Float64,
        "turnover": pl.Float64,
    }


def _positions_schema() -> dict[str, Any]:
    return {
        "trade_date": pl.Date,
        "ts_code": pl.Utf8,
        "weight": pl.Float64,
        "market_value": pl.Float64,
    }


def _trades_schema() -> dict[str, Any]:
    return {
        "trade_date": pl.Date,
        "ts_code": pl.Utf8,
        "prev_weight": pl.Float64,
        "target_weight": pl.Float64,
        "filled_delta_weight": pl.Float64,
        "turnover": pl.Float64,
        "cost": pl.Float64,
        "block_reason": pl.Utf8,
    }
