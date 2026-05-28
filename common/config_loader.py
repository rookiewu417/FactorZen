"""YAML 运行配置加载与 Pydantic v2 验证。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    outlier: Literal["mad", "winsorize", "sigma"] = "mad"
    normalizer: Literal["zscore", "rank_uniform", "rank_normal", "quantile_normal"] = "zscore"
    neutralize: bool = False
    neutralize_by: Literal["industry", "size", "industry+size"] = "industry+size"


class BacktestConfig(BaseModel):
    top_n: int = 50
    quantiles: int = 5
    max_abs_weight: float = 0.1
    cost_model: Literal["linear", "square_root_impact"] = "linear"
    rebalance_threshold: float | None = None
    alpha: float = 0.1  # SquareRootImpactCostModel 冲击系数
    fallback_adv: float = 1e7  # ADV 缺失时的参考值（元）


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
    benchmark: str = "000300.SH"
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
    from daily.preprocessing.pipeline import PreprocessingPipeline

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
):
    """Build daily.evaluation.backtest.BacktestConfig from RunConfig."""
    from daily.evaluation.backtest import BacktestConfig as RuntimeBacktestConfig

    return RuntimeBacktestConfig(
        factor_col=factor_col,
        frequency=frequency,
        max_abs_weight=config.backtest.max_abs_weight,
        rebalance_threshold=config.backtest.rebalance_threshold,
    )


def build_cost_model(config: RunConfig):
    """Build the configured transaction cost model."""
    from daily.evaluation.cost_models import LinearCostModel, SquareRootImpactCostModel

    if config.backtest.cost_model == "square_root_impact":
        return SquareRootImpactCostModel(
            alpha=config.backtest.alpha,
            fallback_adv=config.backtest.fallback_adv,
        )
    return LinearCostModel()
