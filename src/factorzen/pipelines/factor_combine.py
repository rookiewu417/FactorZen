"""fz combine run 流水线:加载因子/收益 parquet → 四方法 OOS 对比实验。

因子 parquet 需含 [trade_date, ts_code, factor_value](来源:因子评估产物或
`fz mine export-alpha` 导出的 α 截面);收益 parquet 需含 [trade_date, ts_code, ret]
(对齐到因子日的前向收益)。因子名取文件名 stem。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.experiment import run_combination_experiment


def run_factor_combination(
    *,
    factor_files: list[str],
    ret_file: str,
    train_days: int = 120,
    test_days: int = 20,
    purge_days: int = 5,
    embargo_days: int = 0,
    methods: list[str] | None = None,
    seed: int = 0,
    out_dir: str = "workspace/combinations",
    run_id: str | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """从 parquet 加载因子/收益,跑 OOS 对比实验。"""
    factor_dfs: dict[str, pl.DataFrame] = {}
    for f in factor_files:
        name = Path(f).stem
        factor_dfs[name] = pl.read_parquet(f).select(
            ["trade_date", "ts_code", "factor_value"]
        )
    ret_df = pl.read_parquet(ret_file).select(["trade_date", "ts_code", "ret"])
    cv = PurgedWalkForwardCV(
        train_days=train_days,
        test_days=test_days,
        purge_days=purge_days,
        embargo_days=embargo_days,
    )
    return run_combination_experiment(
        factor_dfs,
        ret_df,
        cv=cv,
        methods=methods,
        seed=seed,
        out_dir=out_dir,
        run_id=run_id,
        command=command,
    )
