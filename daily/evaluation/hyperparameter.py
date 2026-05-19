"""Optuna 超参搜索：基于 walk-forward OOS Sharpe 做策略参数优化。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ParamSpec:
    """单个超参数的搜索空间定义。"""

    name: str
    type: str  # "int", "float", "categorical"
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False  # log 空间搜索（适用于 int/float）
    choices: list[Any] | None = None  # type="categorical" 时使用


@dataclass
class TuningSpace:
    """超参搜索空间，封装 optuna trial.suggest_* 调用。

    Example:
        space = TuningSpace([
            ParamSpec("n_groups", "int", 5, 20),
            ParamSpec("risk_aversion", "float", 0.1, 10.0, log=True),
        ])
        params = space.suggest(trial)
    """

    specs: list[ParamSpec] = field(default_factory=list)

    def suggest(self, trial: Any) -> dict[str, Any]:
        """调用 trial.suggest_* 为每个参数采样并返回 params 字典。"""
        params: dict[str, Any] = {}
        for spec in self.specs:
            if spec.type == "int":
                params[spec.name] = trial.suggest_int(
                    spec.name,
                    int(spec.low),  # type: ignore[arg-type]
                    int(spec.high),  # type: ignore[arg-type]
                    step=int(spec.step) if spec.step else 1,
                    log=spec.log,
                )
            elif spec.type == "float":
                params[spec.name] = trial.suggest_float(
                    spec.name,
                    float(spec.low),  # type: ignore[arg-type]
                    float(spec.high),  # type: ignore[arg-type]
                    step=spec.step,
                    log=spec.log,
                )
            elif spec.type == "categorical":
                params[spec.name] = trial.suggest_categorical(
                    spec.name,
                    spec.choices,
                )
        return params


def run_optuna_search(
    objective_fn: Callable[[dict[str, Any]], float],
    space: TuningSpace,
    n_trials: int = 50,
    direction: str = "maximize",
    study_name: str = "walk_forward_tuning",
    timeout: float | None = None,
) -> tuple[dict[str, Any], Any]:
    """运行 Optuna 超参搜索。

    Args:
        objective_fn: 接受 params 字典，返回标量目标值（如 OOS Sharpe）。
        space: TuningSpace 搜索空间。
        n_trials: 搜索次数。
        direction: "maximize" 或 "minimize"。
        study_name: Optuna study 名称。
        timeout: 总超时秒数（None=不限）。

    Returns:
        (best_params, study) — best_params 为最优参数字典，study 为 optuna.Study 对象。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _wrapped_objective(trial: optuna.Trial) -> float:
        params = space.suggest(trial)
        try:
            return objective_fn(params)
        except Exception:
            return float("-inf") if direction == "maximize" else float("inf")

    study = optuna.create_study(direction=direction, study_name=study_name)
    study.optimize(_wrapped_objective, n_trials=n_trials, timeout=timeout)

    return study.best_params, study


# Suppress numpy import warning in type checkers
__all__ = ["ParamSpec", "TuningSpace", "run_optuna_search"]

# Avoid unused import warning
_np = np
