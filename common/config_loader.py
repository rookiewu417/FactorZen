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


class WalkForwardConfig(BaseModel):
    train_days: int = 504
    test_days: int = 63
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
