"""声明式研究配置 schema、YAML 加载与纯配置变换。"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
    enabled: bool = False
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


def build_default_daily_research_config(
    *,
    factor: str,
    start: str,
    end: str,
    universe: str | None = None,
    benchmark: str | None = None,
    seed: int | None = None,
) -> RunConfig:
    """Build the no-YAML daily research preset used for new factors."""
    resolved_universe = universe or "csi500"
    return RunConfig(
        factor=factor,
        start=start,
        end=end,
        universe=resolved_universe,
        benchmark=benchmark or default_benchmark_for_universe(resolved_universe),
        seed=42 if seed is None else seed,
        preprocessing=PreprocessingConfig(
            outlier="mad",
            normalizer="zscore",
            neutralize=True,
            neutralize_by="industry+size",
        ),
        backtest=BacktestConfig(
            primary="topn_50",
            strategies=default_all_strategy_specs(),
        ),
        walk_forward=WalkForwardConfig(
            enabled=False,
            train_days=504,
            test_days=63,
            step_days=63,
            embargo_days=5,
            n_trials=50,
        ),
        ic_method="both",
        neutralized_ic=True,
        event_study=True,
    )


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """把 ``key.path=value`` 形式的覆盖应用到原始配置 dict（pydantic 校验之前）。

    在校验前注入，可复用 pydantic 的取值/类型校验，并让 ``backtest.top_n`` 这类
    依赖 ``model_validator`` 自动填充策略的 legacy 字段用新值正确生成（无需特判）。

    - 值类型用 ``yaml.safe_load`` 推断，与 YAML 同源：``30→int``、``true→bool``、
      ``0.1→float``、``rank_normal→str``、``null→None``。
    - dotted key 走嵌套 dict（``backtest.top_n``、``preprocessing.normalizer``）；
      中间缺失的键按需创建空 dict。

    就地修改并返回 ``data``。
    """
    if not overrides:
        return data

    import yaml

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set 需要 key=value 形式，收到: {item!r}")
        key_path, _, raw_value = item.partition("=")
        keys = [part.strip() for part in key_path.strip().split(".")]
        if not all(keys):
            raise ValueError(f"--set 键名非法: {item!r}")
        value = yaml.safe_load(raw_value)
        node = data
        for key in keys[:-1]:
            existing = node.get(key)
            if existing is None:
                existing = {}
                node[key] = existing
            elif not isinstance(existing, dict):
                raise ValueError(f"--set 路径冲突：{key!r} 不是映射（在 {item!r} 中）")
            node = existing
        node[keys[-1]] = value
    return data


def _has_override(overrides: Sequence[str] | None, key: str) -> bool:
    return any(item.partition("=")[0].strip() == key for item in overrides or [])


def _sync_default_top_n_strategy(config: RunConfig) -> RunConfig:
    """Keep the built-in no-YAML primary top-N strategy aligned with backtest.top_n."""
    top_n = int(config.backtest.top_n)
    old_name = "topn_50"
    new_name = f"topn_{top_n}"
    changed = False
    strategies: list[StrategySpec] = []
    for spec in config.backtest.strategy_specs:
        if spec.name == old_name and spec.type == "topn_long_only":
            strategies.append(
                spec.model_copy(
                    update={
                        "name": new_name,
                        "params": {**spec.params, "top_n": top_n},
                    }
                )
            )
            changed = True
        else:
            strategies.append(spec)

    if not changed:
        return config

    primary = new_name if config.backtest.primary == old_name else config.backtest.primary
    return config.model_copy(
        update={
            "backtest": config.backtest.model_copy(
                update={"primary": primary, "strategies": strategies}
            )
        }
    )


def build_run_config_from_dict(
    data: dict[str, Any] | None, overrides: Sequence[str] | None = None
) -> RunConfig:
    """从原始 dict（叠加可选 overrides）构造并校验 RunConfig。无 YAML 文件时复用。"""
    merged: dict[str, Any] = dict(data or {})
    if overrides:
        apply_overrides(merged, overrides)
    config = RunConfig.model_validate(merged)
    if _has_override(overrides, "backtest.top_n"):
        config = _sync_default_top_n_strategy(config)
    return config


def load_run_config(path: Path | str, overrides: Sequence[str] | None = None) -> RunConfig:
    """从 YAML 文件加载并验证 RunConfig。

    Args:
        path: YAML 配置文件路径。
        overrides: 可选 ``key.path=value`` 覆盖列表，在校验前叠加到 YAML dict 上。

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
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if overrides:
        apply_overrides(data, overrides)
    return RunConfig.model_validate(data)


def build_top_n_candidate_params(config: RunConfig) -> list[dict[str, int]]:
    """Build deterministic top_n candidates limited by walk_forward.n_trials."""
    top_n = max(1, int(config.backtest.top_n))
    n_trials = max(1, int(config.walk_forward.n_trials))
    min_top_n = max(1, math.ceil(1.0 / float(config.backtest.max_abs_weight)))
    if min_top_n > top_n:
        return [{"top_n": top_n}]

    span = top_n - min_top_n + 1
    count = min(span, n_trials)
    if count == 1:
        return [{"top_n": top_n}]

    values = {round(min_top_n + i * (top_n - min_top_n) / (count - 1)) for i in range(count)}
    return [{"top_n": int(value)} for value in sorted(values)]
