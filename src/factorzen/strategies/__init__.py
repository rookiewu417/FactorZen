"""模拟交易权重产物层：非因子研究回测插件，生成 weights/manifest 供 sim/execution 消费。

与 ``daily.evaluation.backtest.Strategy`` ABC（因子研究回测插件）语义分离：
本包产出 ``weights.parquet[ts_code, target_weight]`` + ``manifest.json{signal_date, status}``，
契约对齐 ``sim.engine._load_weights_by_date``。
"""
from __future__ import annotations

from factorzen.strategies.momentum_rotation import generate_momentum_rotation_products
from factorzen.strategies.quantile_group import (
    build_group_weights,
    generate_quantile_group_products,
)
from factorzen.strategies.runner import (
    run_momentum_rotation_experiment,
    run_strategy_simulation,
    run_trend_timing_experiment,
)
from factorzen.strategies.sleeve import build_sleeve_weights, generate_sleeve_products
from factorzen.strategies.trend_timing import generate_trend_timing_products

__all__ = [
    "build_group_weights",
    "build_sleeve_weights",
    "generate_momentum_rotation_products",
    "generate_quantile_group_products",
    "generate_sleeve_products",
    "generate_trend_timing_products",
    "run_momentum_rotation_experiment",
    "run_strategy_simulation",
    "run_trend_timing_experiment",
]
