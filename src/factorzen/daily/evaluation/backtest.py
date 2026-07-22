"""模拟交易回测引擎（日环撮合：约束/成本/仓位）。

因子研究的信号层评估（分层/多空/IC，毛收益向量化）在 ``signal_backtest.py``。

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
from factorzen.core.validation import require_columns
from factorzen.daily.evaluation.cost_models import CostModelBase
from factorzen.daily.evaluation.trade_constraints import (
    BLOCK_REASON_STR,
    apply_trade_constraints_batch,
    board_limit_pct_for_codes,
)
from factorzen.daily.evaluation.trade_constraints import (
    apply_trade_constraints as _apply_trade_constraints,  # noqa: F401 — 测试/外部兼容别名
)


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
    # 子类可声明不消费 context 字段，慢路径跳过对应物化（默认 True 保兼容）
    uses_context_positions: bool = True
    uses_context_adv: bool = True
    uses_context_price_slice: bool = True

    @abstractmethod
    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        """返回列 ts_code, target_weight。"""

    def has_target_for(self, signal_date: date) -> bool:
        """该策略在 ``signal_date`` 是否有明确调仓目标。默认 True（每个信号日都调仓）。

        PrecomputedWeightsStrategy 覆盖为『仅在有预计算权重的日期调仓，其余日持有』。
        统一日环引擎：缺权日 carry，显式空权表 flat（与历史快路径语义一致）。
        """
        return True


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
    # generate_weights 只读 factor_slice；跳过昂贵 context 物化
    uses_context_positions: bool = False
    uses_context_adv: bool = False
    uses_context_price_slice: bool = False

    def __init__(self, n_groups: int = 10, factor_col: str = "factor_clean") -> None:
        self.n_groups = n_groups
        self.factor_col = factor_col

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        factor_col = context.factor_col or self.factor_col
        df = _valid_factor_slice(context.factor_slice, factor_col)
        # 薄截面：股数不足分组数时无法同时填满 top/bottom → flat，禁止单腿裸头寸
        if df.is_empty() or df.height < self.n_groups:
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
        # 两腿必须成对：任一为空（近常数/退化分桶）→ flat，不建裸多/裸空
        if long_df.is_empty() or short_df.is_empty():
            return _empty_weights()
        parts = [
            long_df.select(["ts_code"]).with_columns(
                pl.lit(1.0 / long_df.height).alias("target_weight")
            ),
            short_df.select(["ts_code"]).with_columns(
                pl.lit(-1.0 / short_df.height).alias("target_weight")
            ),
        ]
        return pl.concat(parts)


class TopNLongOnlyStrategy(Strategy):
    """做多因子值最高的 N 只股票。"""

    name = "topn_long_only"
    uses_context_positions: bool = False
    uses_context_adv: bool = False
    uses_context_price_slice: bool = False

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
    uses_context_positions: bool = False
    uses_context_adv: bool = False
    uses_context_price_slice: bool = False

    def __init__(self, weights_by_date: dict[date, pl.DataFrame]) -> None:
        self.weights_by_date = weights_by_date

    def generate_weights(self, context: BacktestContext) -> pl.DataFrame:
        weights = self.weights_by_date.get(context.signal_date)
        if weights is None:
            return _empty_weights()
        return weights

    def has_target_for(self, signal_date: date) -> bool:
        # 只有该 signal 日有预计算权重才算调仓日；否则持有（与快路径一致）。
        return signal_date in self.weights_by_date


class FactorWeightedStrategy(Strategy):
    """按因子强度生成权重。"""

    name = "factor_weighted"
    uses_context_positions: bool = False
    uses_context_adv: bool = False
    uses_context_price_slice: bool = False

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
    factor_df: pl.DataFrame | None,
    price_df: pl.DataFrame,
    config: BacktestConfig | None = None,
    cost_model: CostModel | CostModelBase | None = None,
    factor_name: str = "",
    *,
    collect_positions: bool = True,
    collect_trades: bool = True,
    include_context_positions: bool = True,
    is_st_by_date: dict[date, set[str]] | None = None,
) -> StrategyBacktestResult:
    """运行策略回测（统一日环引擎）。

    单一实现：numpy 日状态机 + 向量约束核；``collect_*`` 仅控制是否写出明细，
    不再切换慢/快两套实现。``PrecomputedWeightsStrategy`` 启动时预填 target；
    其它策略在调仓日调用 ``generate_weights``（Optimizer 的 ``price_history`` 等
    最小上下文仍在日环内供给）。

    ``factor_df=None`` 仅适用于不消费因子内容的策略（如 PrecomputedWeightsStrategy）。

    Parameters
    ----------
    is_st_by_date : dict[date, set[str]] | None, optional
        按 ``execution_date`` 给出当日处于 ST/\\*ST 状态的 ``ts_code`` 集合，
        用于 PIT 正确地收窄主板 ST 股票的涨跌停阈值（4.8% 而非 9.8%，参见
        ``factorzen.core.universe._get_board_limit``）。为 ``None``（默认）
        时行为与未引入此参数前完全一致：一律按非 ST 板块阈值判断涨跌停。
    """
    cfg = config or BacktestConfig()
    if factor_df is None:
        factor_df = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                cfg.factor_col: pl.Float64,
            }
        )
    factor = _prepare_factor_df(factor_df, cfg.factor_col)
    price = _prepare_price_df(price_df)
    trade_dates = price.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    return _run_day_loop_engine(
        strategy=strategy,
        factor=factor,
        price=price,
        trade_dates=trade_dates,
        config=cfg,
        cost_model=cost_model,
        factor_name=factor_name,
        collect_positions=collect_positions,
        collect_trades=collect_trades,
        include_context_positions=include_context_positions,
        is_st_by_date=is_st_by_date,
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
    """交易口径的分层策略回测（完整日环：约束+成本+撮合）。

    研究用途的分层/多空信号评估请用 ``signal_backtest.run_signal_backtest``
    （向量化毛收益口径）。
    """
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

    # 机械向量化装载：避免 named iter_rows；语义仍是 date → {ts_code: adv}
    if adv_frame.is_empty():
        return {}
    result: dict[date, dict[str, float]] = {}
    dates = adv_frame["trade_date"].to_list()
    codes = adv_frame["ts_code"].to_list()
    vals = adv_frame["adv_20d"].to_list()
    for d, code, v in zip(dates, codes, vals, strict=True):
        result.setdefault(d, {})[code] = v
    return result


def _build_price_adv_matrices(
    price: pl.DataFrame,
    trade_dates: list[date],
    *,
    build_adv_dict: bool = True,
) -> tuple[
    list[str],
    dict[str, int],
    dict[date, int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[date, dict[str, float]],
]:
    """一次装载 (T×N) 价格/收益/ADV 矩阵 + 可选 ADV 按日字典，快慢路径共用。

    缺 open/pre_close/vol → NaN（对齐 missing_price）；
    overnight/intraday null → 0.0（对齐旧 float(x or 0.0)）。
    ``build_adv_dict=False`` 时跳过按日 dict（约束只用矩阵；Quantile/TopN 等不读 context.adv）。
    """
    codes = price.select("ts_code").unique().sort("ts_code")["ts_code"].to_list()
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    date_to_idx = {trade_date: idx for idx, trade_date in enumerate(trade_dates)}
    shape = (len(trade_dates), len(codes))
    open_px = np.full(shape, np.nan, dtype=float)
    pre_close = np.full(shape, np.nan, dtype=float)
    vol_data = np.full(shape, np.nan, dtype=float)
    overnight_ret = np.zeros(shape, dtype=float)
    intraday_ret = np.zeros(shape, dtype=float)

    if not price.is_empty():
        r_idx = np.fromiter(
            (date_to_idx.get(d, -1) for d in price["trade_date"].to_list()),
            dtype=np.int64,
            count=price.height,
        )
        c_idx = np.fromiter(
            (code_to_idx.get(s, -1) for s in price["ts_code"].to_list()),
            dtype=np.int64,
            count=price.height,
        )
        keep = (r_idx >= 0) & (c_idx >= 0)
        r_k, c_k = r_idx[keep], c_idx[keep]
        open_px[r_k, c_k] = price["open"].to_numpy().astype(np.float64, copy=False)[keep]
        pre_close[r_k, c_k] = (
            price["pre_close"].to_numpy().astype(np.float64, copy=False)[keep]
        )
        vol_data[r_k, c_k] = price["vol"].to_numpy().astype(np.float64, copy=False)[keep]
        overnight_ret[r_k, c_k] = (
            price["overnight_ret"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)[keep]
        )
        intraday_ret[r_k, c_k] = (
            price["intraday_ret"].fill_null(0.0).to_numpy().astype(np.float64, copy=False)[keep]
        )

    # ADV：rolling 一次 → 向量 scatter 到矩阵，并构建按日 dict（context 复用）
    adv = np.full(shape, np.nan, dtype=float)
    adv_by_date: dict[date, dict[str, float]] = {}
    if trade_dates and "amount" in price.columns:
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
        if not adv_frame.is_empty():
            dates_l = adv_frame["trade_date"].to_list()
            codes_l = adv_frame["ts_code"].to_list()
            vals_l = adv_frame["adv_20d"].to_numpy().astype(np.float64, copy=False)
            n_adv = len(dates_l)
            r_idx = np.fromiter(
                (date_to_idx.get(d, -1) for d in dates_l), dtype=np.int64, count=n_adv
            )
            c_idx = np.fromiter(
                (code_to_idx.get(c, -1) for c in codes_l), dtype=np.int64, count=n_adv
            )
            keep = (r_idx >= 0) & (c_idx >= 0)
            adv[r_idx[keep], c_idx[keep]] = vals_l[keep]
            if build_adv_dict:
                # context 字典：与 _precompute_adv_20d_by_date 同结构
                for d, code, v in zip(dates_l, codes_l, vals_l, strict=True):
                    adv_by_date.setdefault(d, {})[code] = float(v)

    return (
        codes,
        code_to_idx,
        date_to_idx,
        open_px,
        pre_close,
        vol_data,
        overnight_ret,
        intraday_ret,
        adv,
        adv_by_date,
    )


def _positions_frame_from_dense(
    code_arr: np.ndarray,
    weights: np.ndarray,
    nav_value: float,
    initial_capital: float,
) -> pl.DataFrame:
    """Dense 权重 → context 持仓帧（列式，避免 list[dict]+from_dicts）。

    与 ``_positions_frame`` 一致：非空帧无 trade_date 列（context 日持仓快照）。
    """
    nz = np.flatnonzero(np.abs(weights) >= 1e-12)
    if nz.size == 0:
        return pl.DataFrame(schema=_positions_schema())
    w = weights[nz]
    return pl.DataFrame(
        {
            "ts_code": code_arr[nz].tolist(),
            "weight": w.tolist(),
            "market_value": (w * nav_value * initial_capital).tolist(),
        }
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


def _prefill_precomputed_targets(
    strategy: PrecomputedWeightsStrategy,
    code_to_idx: dict[str, int],
    config: BacktestConfig,
) -> dict[date, tuple[np.ndarray, np.ndarray]]:
    """启动时校验并稀疏化 PrecomputedWeightsStrategy 目标权重。

    凡 ``weights_by_date`` 中存在的 sig_date 一律记录（含空权重表 = 显式空仓）：
      - 日期在表中且权重空 → flat
      - 日期不在表中 → carry（消费处 signal_date not in dict）
    """
    target_by_signal_date: dict[date, tuple[np.ndarray, np.ndarray]] = {}
    for sig_date, weight_df in strategy.weights_by_date.items():
        weight_df = _validate_target_weights(weight_df, config)
        if weight_df.is_empty():
            target_by_signal_date[sig_date] = (
                np.array([], dtype=int),
                np.array([], dtype=float),
            )
            continue
        w_codes = weight_df["ts_code"].to_list()
        w_vals = weight_df["target_weight"].to_numpy().astype(np.float64, copy=False)
        idx_buf: list[int] = []
        val_buf: list[float] = []
        for code, w in zip(w_codes, w_vals, strict=True):
            code_idx = code_to_idx.get(code)
            if code_idx is not None:
                idx_buf.append(code_idx)
                val_buf.append(float(w))
        target_by_signal_date[sig_date] = (
            np.asarray(idx_buf, dtype=int),
            np.asarray(val_buf, dtype=float),
        )
    return target_by_signal_date


def _run_day_loop_engine(
    *,
    strategy: Strategy,
    factor: pl.DataFrame,
    price: pl.DataFrame,
    trade_dates: list[date],
    config: BacktestConfig,
    cost_model: CostModel | CostModelBase | None,
    factor_name: str,
    collect_positions: bool,
    collect_trades: bool,
    include_context_positions: bool,
    is_st_by_date: dict[date, set[str]] | None,
) -> StrategyBacktestResult:
    """唯一日环实现（方案 B）：numpy 状态机 + 向量约束 + 可选列式明细。

    - PrecomputedWeightsStrategy：预填稀疏 target，调仓日判定 = 权重表键
    - 其它 Strategy：调仓日调用 generate_weights（factor 日历 ∩ has_target_for）
    - CostModel 向量化；CostModelBase 保留 per-name 语义
    - collect_* 仅控制明细缓冲，不切换实现
    """
    is_precomputed = isinstance(strategy, PrecomputedWeightsStrategy)
    need_adv_dict = (not is_precomputed) and getattr(strategy, "uses_context_adv", True)
    (
        codes,
        code_to_idx,
        _date_to_idx,
        open_px,
        pre_close_m,
        vol_data,
        overnight_ret,
        intraday_ret,
        adv_m,
        adv_20d_by_date,
    ) = _build_price_adv_matrices(
        price,
        trade_dates,
        build_adv_dict=need_adv_dict,
    )
    n_codes = len(codes)
    code_arr = np.asarray(codes, dtype=object)
    board_limits = board_limit_pct_for_codes(codes)
    board_limits_st: np.ndarray | None = (
        board_limit_pct_for_codes(codes, is_st=True) if is_st_by_date else None
    )
    reason_lut = np.asarray(BLOCK_REASON_STR, dtype=object)

    target_by_signal_date: dict[date, tuple[np.ndarray, np.ndarray]] | None = None
    factor_by_date: dict[date, pl.DataFrame] = {}
    price_by_date: dict[date, pl.DataFrame] = {}
    _lookback = 0
    _empty_price = pl.DataFrame(schema=price.schema)
    want_price_slice = False

    if is_precomputed:
        target_by_signal_date = _prefill_precomputed_targets(
            cast(PrecomputedWeightsStrategy, strategy), code_to_idx, config
        )
    else:
        want_price_slice = getattr(strategy, "uses_context_price_slice", True)
        price_by_date = _group_frames_by_date(price) if want_price_slice else {}
        factor_by_date = _group_frames_by_date(factor)
        _lookback = int(getattr(strategy, "lookback_days", 0) or 0)

    weights = np.zeros(n_codes, dtype=np.float64)
    nav_value = 1.0
    has_started = False
    nav_rows: list[dict[str, Any]] = []

    pos_dates: list[date] = []
    pos_codes: list[str] = []
    pos_weights: list[float] = []
    pos_mvs: list[float] = []
    tr_dates: list[date] = []
    tr_codes: list[str] = []
    tr_prev: list[float] = []
    tr_target: list[float] = []
    tr_filled: list[float] = []
    tr_turnover: list[float] = []
    tr_cost: list[float] = []
    tr_reason: list[str] = []

    for i, execution_date in enumerate(trade_dates):
        signal_date = trade_dates[i - 1] if i > 0 else None

        overnight = overnight_ret[i]
        overnight_return = float(np.dot(weights, overnight))
        denom = 1.0 + overnight_return
        if abs(denom) < 1e-12:
            open_w = weights.copy()
        else:
            open_w = weights * (1.0 + overnight) / denom
            open_w[np.abs(open_w) < 1e-12] = 0.0
        open_nav_value = nav_value * (1.0 + overnight_return)

        target_w = open_w.copy()
        if is_precomputed:
            assert target_by_signal_date is not None
            has_signal = signal_date is not None and signal_date in target_by_signal_date
        else:
            has_signal = (
                signal_date is not None
                and signal_date in factor_by_date
                and strategy.has_target_for(signal_date)
            )

        if has_signal:
            assert signal_date is not None
            has_started = True
            if is_precomputed:
                assert target_by_signal_date is not None
                target_w = np.zeros(n_codes, dtype=np.float64)
                idx, vals = target_by_signal_date[signal_date]
                if idx.size:
                    target_w[idx] = vals
            else:
                want_pos = include_context_positions and getattr(
                    strategy, "uses_context_positions", True
                )
                want_adv = getattr(strategy, "uses_context_adv", True)
                adv_20d = adv_20d_by_date.get(execution_date, {}) if want_adv else {}
                if want_pos:
                    pos_ctx = _positions_frame_from_dense(
                        code_arr, open_w, open_nav_value, config.initial_capital
                    )
                else:
                    pos_ctx = pl.DataFrame(schema=_positions_schema())
                price_slice = (
                    price_by_date.get(execution_date, _empty_price)
                    if want_price_slice
                    else _empty_price
                )
                context = BacktestContext(
                    signal_date=signal_date,
                    execution_date=execution_date,
                    factor_slice=factor_by_date[signal_date],
                    price_slice=price_slice,
                    current_positions=pos_ctx,
                    factor_col=config.factor_col,
                    price_history=_get_price_history(price, trade_dates, i, _lookback),
                    adv_20d=adv_20d,
                )
                target_df = _validate_target_weights(strategy.generate_weights(context), config)
                target_w = np.zeros(n_codes, dtype=np.float64)
                if not target_df.is_empty():
                    t_codes = target_df["ts_code"].to_list()
                    t_vals = target_df["target_weight"].to_numpy().astype(np.float64, copy=False)
                    for code, tw in zip(t_codes, t_vals, strict=True):
                        c_idx = code_to_idx.get(code)
                        if c_idx is not None:
                            target_w[c_idx] = float(tw)

            if config.rebalance_threshold is not None:
                proposed_turnover = float(np.sum(np.abs(target_w - open_w)))
                if proposed_turnover <= config.rebalance_threshold:
                    target_w = open_w.copy()

        delta = target_w - open_w
        active = np.abs(delta) > 1e-12
        filled = np.zeros(n_codes, dtype=np.float64)
        reasons = np.zeros(n_codes, dtype=np.int8)
        if np.any(active):
            effective_limits = board_limits
            if is_st_by_date and board_limits_st is not None:
                st_today = is_st_by_date.get(execution_date)
                if st_today:
                    st_mask = np.fromiter(
                        (c in st_today for c in codes), dtype=bool, count=n_codes
                    )
                    effective_limits = np.where(st_mask, board_limits_st, board_limits)

            # 与历史快路径一致：全截面 batch（inactive Δ≈0 被核内短路为 0）
            filled, reasons = apply_trade_constraints_batch(
                delta=delta,
                open_px=open_px[i],
                pre_close=pre_close_m[i],
                vol=vol_data[i],
                adv=adv_m[i],
                board_limits=effective_limits,
                portfolio_value=open_nav_value * config.initial_capital,
                max_participation_rate=config.max_participation_rate,
                fallback_adv=config.fallback_adv,
            )

        next_w = open_w + filled
        next_w[np.abs(next_w) < 1e-12] = 0.0

        # 成本：旧 CostModel 向量化；CostModelBase 保留 per-name 语义
        if cost_model is None:
            trade_costs = np.zeros(n_codes, dtype=np.float64)
            trade_cost = 0.0
            borrow_cost = 0.0
        elif isinstance(cost_model, CostModel):
            buy_c = np.where(filled > 0, np.abs(filled) * cost_model.one_way_cost(), 0.0)
            sell_c = np.where(filled < 0, np.abs(filled) * cost_model.sell_cost(), 0.0)
            trade_costs = buy_c + sell_c
            trade_cost = float(np.sum(trade_costs))
            short_exposure = float(np.sum(np.abs(next_w[next_w < 0])))
            borrow_cost = short_exposure * cost_model.borrow_rate_per_period("daily")
        else:
            trade_costs = np.zeros(n_codes, dtype=np.float64)
            trade_cost = 0.0
            if np.any(active):
                act_idx = np.flatnonzero(active)
                for j in act_idx:
                    adv_j = adv_m[i, j]
                    adv_arg = float(adv_j) if np.isfinite(adv_j) and adv_j > 0 else None
                    c = _trade_cost(cost_model, float(filled[j]), adv_arg)
                    trade_costs[j] = c
                    trade_cost += c
            short_exposure = float(np.sum(np.abs(next_w[next_w < 0])))
            borrow_cost = short_exposure * cost_model.borrow_rate_per_period("daily")

        turnover = float(np.sum(np.abs(filled)))

        if collect_trades and np.any(active):
            act_idx = np.flatnonzero(active)
            n_act = int(act_idx.size)
            tr_dates.extend([execution_date] * n_act)
            tr_codes.extend(code_arr[act_idx].tolist())
            tr_prev.extend(open_w[act_idx].tolist())
            tr_target.extend(target_w[act_idx].tolist())
            f_act = filled[act_idx]
            tr_filled.extend(f_act.tolist())
            tr_turnover.extend(np.abs(f_act).tolist())
            tr_cost.extend(trade_costs[act_idx].tolist())
            tr_reason.extend(reason_lut[reasons[act_idx]].tolist())

        intraday = intraday_ret[i]
        intraday_return = float(np.dot(next_w, intraday))
        # 融券是每日持有成本：回测循环恒按日迭代，按【日】费率计提。
        gross_return = (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0
        period_cost_scale = 1.0 + overnight_return
        period_trade_cost = trade_cost * period_cost_scale
        period_borrow_cost = borrow_cost * period_cost_scale
        net_return = gross_return - period_trade_cost - period_borrow_cost
        nav_value *= 1.0 + net_return

        close_denom = 1.0 + intraday_return
        if abs(close_denom) < 1e-12:
            close_w = next_w.copy()
        else:
            close_w = next_w * (1.0 + intraday) / close_denom
            close_w[np.abs(close_w) < 1e-12] = 0.0
        if 1.0 + net_return > 1e-12:
            cost_scale = (1.0 + gross_return) / (1.0 + net_return)
            close_w = close_w * cost_scale
            close_w[np.abs(close_w) < 1e-12] = 0.0
        cash_weight = float(1.0 - np.sum(close_w))

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
                nz = np.flatnonzero(np.abs(close_w) >= 1e-12)
                if nz.size:
                    mv_scale = nav_value * config.initial_capital
                    n_pos = int(nz.size)
                    pos_dates.extend([execution_date] * n_pos)
                    pos_codes.extend(code_arr[nz].tolist())
                    w_act = close_w[nz]
                    pos_weights.extend(w_act.tolist())
                    pos_mvs.extend((w_act * mv_scale).tolist())
        weights = close_w

    returns = pl.DataFrame(nav_rows, schema=_returns_schema())
    nav = _build_nav_frame(returns, trade_dates)
    if collect_positions and pos_dates:
        positions = pl.DataFrame(
            {
                "trade_date": pos_dates,
                "ts_code": pos_codes,
                "weight": pos_weights,
                "market_value": pos_mvs,
            },
            schema=_positions_schema(),
        )
    else:
        positions = pl.DataFrame(schema=_positions_schema())
    if collect_trades and tr_dates:
        trades = pl.DataFrame(
            {
                "trade_date": tr_dates,
                "ts_code": tr_codes,
                "prev_weight": tr_prev,
                "target_weight": tr_target,
                "filled_delta_weight": tr_filled,
                "turnover": tr_turnover,
                "cost": tr_cost,
                "block_reason": tr_reason,
            },
            schema=_trades_schema(),
        )
    else:
        trades = pl.DataFrame(schema=_trades_schema())
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
