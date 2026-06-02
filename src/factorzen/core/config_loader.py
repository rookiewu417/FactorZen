"""YAML 运行配置加载与 Pydantic v2 验证。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_BENCHMARK_BY_UNIVERSE = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
    "csi800": "000906.SH",
}


def default_benchmark_for_universe(universe: str | None) -> str:
    return DEFAULT_BENCHMARK_BY_UNIVERSE.get(universe or "csi300", "000300.SH")


class PreprocessingConfig(BaseModel):
    outlier: Literal["mad", "winsorize", "sigma"] = "mad"
    normalizer: Literal["zscore", "rank_uniform", "rank_normal", "quantile_normal"] = "zscore"
    neutralize: bool = False
    neutralize_by: Literal["industry", "size", "industry+size"] = "industry+size"


class StrategySpec(BaseModel):
    name: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    max_abs_weight: float | None = None
    rebalance_threshold: float | None = None
    cost_model: Literal["linear", "square_root_impact"] | None = None
    alpha: float | None = None
    fallback_adv: float | None = None


class BacktestConfig(BaseModel):
    top_n: int = 50
    quantiles: int = 5
    max_abs_weight: float = 0.1
    cost_model: Literal["linear", "square_root_impact"] = "linear"
    rebalance_threshold: float | None = None
    alpha: float = 0.1  # SquareRootImpactCostModel 冲击系数
    fallback_adv: float = 1e7  # ADV 缺失时的参考值（元）
    primary: str | None = None
    strategies: list[StrategySpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _populate_legacy_strategy(self) -> BacktestConfig:
        if not self.strategies:
            name = f"topn_{self.top_n}"
            self.strategies = [
                StrategySpec(
                    name=name,
                    type="topn_long_only",
                    params={"top_n": self.top_n},
                )
            ]
        if self.primary is None:
            self.primary = self.strategies[0].name
        return self

    @property
    def strategy_specs(self) -> list[StrategySpec]:
        return self.strategies


def default_all_strategy_specs() -> list[StrategySpec]:
    """Default built-in strategy suite used by --all when no YAML suite is provided."""
    return [
        StrategySpec(
            name="topn_50",
            type="topn_long_only",
            params={"top_n": 50},
            max_abs_weight=0.1,
            cost_model="linear",
        ),
        StrategySpec(
            name="quantile_ls_5",
            type="quantile_long_short",
            params={"quantiles": 5},
            max_abs_weight=0.1,
            cost_model="linear",
        ),
        StrategySpec(
            name="factor_weighted_ls",
            type="factor_weighted",
            params={"long_only": False, "gross_exposure": 2.0},
            max_abs_weight=0.05,
            cost_model="linear",
        ),
        StrategySpec(
            name="optimizer_mv_long_only",
            type="optimizer_strategy",
            params={
                "optimizer": "mean_variance",
                "risk_aversion": 1.0,
                "lookback_days": 60,
                "cov_estimator": "ledoit_wolf",
                "long_only": True,
                "top_n": 100,
                "max_weight": 0.08,
                "gross_exposure": 1.0,
                "net_exposure": 1.0,
            },
            max_abs_weight=0.08,
            cost_model="linear",
        ),
    ]


def with_default_all_strategies(config: RunConfig) -> RunConfig:
    """Return a copy configured to run the default built-in strategy suite."""
    return config.model_copy(
        update={
            "backtest": config.backtest.model_copy(
                update={
                    "primary": "topn_50",
                    "strategies": default_all_strategy_specs(),
                }
            )
        }
    )


class WalkForwardConfig(BaseModel):
    train_days: int = 504  # IS 历史观察期长度；字段名保留用于配置兼容
    test_days: int = 63  # OOS 未来验证期长度；字段名保留用于配置兼容
    step_days: int = 63
    embargo_days: int = 5
    n_trials: int = 50


class RunConfig(BaseModel):
    factor: str
    universe: str = "csi500"
    start: str  # YYYYMMDD
    end: str  # YYYYMMDD
    benchmark: str | None = None
    seed: int | None = None
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    ic_method: Literal["rank", "pearson", "both"] = "rank"
    event_study: bool = False
    neutralized_ic: bool = False


def load_run_config(path: Path | str) -> RunConfig:
    """从 YAML 文件加载并验证 RunConfig。

    Args:
        path: YAML 配置文件路径。

    Returns:
        验证后的 RunConfig 实例。

    Raises:
        ImportError: 若 PyYAML 未安装。
        pydantic.ValidationError: 若配置不符合 schema。
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML 未安装。请运行 `pixi add pyyaml` 或 `pip install pyyaml`。"
        ) from exc

    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(data)


def build_preprocessing_pipeline(config: RunConfig):
    """Build the runtime preprocessing pipeline from a validated run config."""
    from factorzen.daily.preprocessing.pipeline import PreprocessingPipeline

    return PreprocessingPipeline(
        steps=["outlier", "missing", "normalize"],
        outlier_method=config.preprocessing.outlier,
        normalizer_method=config.preprocessing.normalizer,
        neutralize=config.preprocessing.neutralize,
    )


def build_runtime_backtest_config(
    config: RunConfig,
    factor_col: str = "factor_clean",
    frequency: str = "daily",
    strategy_spec: StrategySpec | None = None,
):
    """Build daily.evaluation.backtest.BacktestConfig from RunConfig."""
    from factorzen.daily.evaluation.backtest import BacktestConfig as RuntimeBacktestConfig

    return RuntimeBacktestConfig(
        factor_col=factor_col,
        frequency=frequency,
        max_abs_weight=(
            strategy_spec.max_abs_weight
            if strategy_spec is not None and strategy_spec.max_abs_weight is not None
            else config.backtest.max_abs_weight
        ),
        rebalance_threshold=(
            strategy_spec.rebalance_threshold
            if strategy_spec is not None and strategy_spec.rebalance_threshold is not None
            else config.backtest.rebalance_threshold
        ),
        strategy_type=strategy_spec.type if strategy_spec is not None else None,
        strategy_params=dict(strategy_spec.params) if strategy_spec is not None else {},
        cost_model=(
            strategy_spec.cost_model
            if strategy_spec is not None and strategy_spec.cost_model is not None
            else config.backtest.cost_model
        ),
        alpha=(
            strategy_spec.alpha
            if strategy_spec is not None and strategy_spec.alpha is not None
            else config.backtest.alpha
        ),
        fallback_adv=(
            strategy_spec.fallback_adv
            if strategy_spec is not None and strategy_spec.fallback_adv is not None
            else config.backtest.fallback_adv
        ),
    )


def build_cost_model(config: RunConfig, strategy_spec: StrategySpec | None = None):
    """Build the configured transaction cost model."""
    from factorzen.daily.evaluation.cost_models import LinearCostModel, SquareRootImpactCostModel

    cost_model = strategy_spec.cost_model if strategy_spec and strategy_spec.cost_model else config.backtest.cost_model
    if cost_model == "square_root_impact":
        return SquareRootImpactCostModel(
            alpha=strategy_spec.alpha if strategy_spec and strategy_spec.alpha is not None else config.backtest.alpha,
            fallback_adv=(
                strategy_spec.fallback_adv
                if strategy_spec and strategy_spec.fallback_adv is not None
                else config.backtest.fallback_adv
            ),
        )
    return LinearCostModel()


def build_backtest_strategy(config: RunConfig):
    """Build the configured strategy for the default single-factor evaluation."""
    strategies = build_backtest_strategies(config)
    return strategies[config.backtest.primary or next(iter(strategies))]


def build_backtest_strategies(config: RunConfig):
    """Build named runtime strategies from RunConfig."""
    from factorzen.daily.evaluation.strategy_registry import build_strategy

    strategies = {}
    for spec in config.backtest.strategy_specs:
        strategy = build_strategy(spec.type, spec.params)
        strategy.name = spec.name
        strategies[spec.name] = strategy
    return strategies


def build_top_n_candidate_params(config: RunConfig) -> list[dict[str, int]]:
    """Build deterministic top_n candidates limited by walk_forward.n_trials."""
    top_n = max(1, int(config.backtest.top_n))
    n_trials = max(1, int(config.walk_forward.n_trials))
    count = min(top_n, n_trials)
    if count == 1:
        return [{"top_n": top_n}]

    values = {
        round(1 + i * (top_n - 1) / (count - 1))
        for i in range(count)
    }
    return [{"top_n": int(value)} for value in sorted(values)]
