"""分位组 long-only：指定分位组、组内等权，每日调仓。

语义对齐草稿 ``workspace/configs/quantile_group_longonly.py::build_group_weights``，
分组口径与 ``QuantileLongShortStrategy`` 一致：

    rank = 分数.rank("ordinal", descending=False)          # 升序，1=最低分
    group = (rank - 1) * n_groups // rank.max()            # 0..n_groups-1
    # group 0   = 分数最低组
    # group n-1 = 分数最高组

``group`` 用人类直觉编号：``1`` = 分数最高组，``n_groups`` = 分数最低组。
内部映射 ``code_group = n_groups - group``。

截面股数 ``< n_groups`` 时该日不建仓（与草稿一致的退化护卫）。
PIT：只用当日截面分数；manifest 写 ``signal_date`` + ``status:"optimal"``。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl


def build_group_weights(
    scores: pl.DataFrame,
    *,
    score_col: str,
    n_groups: int,
    group: int,
) -> dict[date, pl.DataFrame]:
    """按分层回测的分组公式取指定组，组内等权。group=1 为分数最高组。"""
    if not 1 <= group <= n_groups:
        raise ValueError(f"group 必须在 1..{n_groups}，收到 {group}")
    code_group = n_groups - group  # 1 → n_groups-1（最高组）；n_groups → 0（最低组）
    valid = scores.filter(
        pl.col(score_col).is_not_null() & pl.col(score_col).is_finite()
    )
    grouped = (
        valid.with_columns(
            pl.col(score_col).rank("ordinal", descending=False).over("trade_date").alias("_rk")
        )
        .with_columns(pl.col("_rk").max().over("trade_date").alias("_rkmax"))
        # 与引擎一致：整数除法分组；截面股数 < n_groups 时该日不建仓
        .filter(pl.col("_rkmax") >= n_groups)
        .with_columns(
            (((pl.col("_rk") - 1) * n_groups) // pl.col("_rkmax")).cast(pl.Int32).alias("_g")
        )
        .filter(pl.col("_g") == code_group)
        .select(["trade_date", "ts_code"])
    )
    sizes = grouped.group_by("trade_date").agg(pl.len().alias("_n"))
    weighted = grouped.join(sizes, on="trade_date", how="left").with_columns(
        (1.0 / pl.col("_n")).alias("target_weight")
    )
    return {
        (k[0] if isinstance(k, tuple) else k): g.select(["ts_code", "target_weight"])
        for k, g in weighted.partition_by("trade_date", as_dict=True).items()
    }


def generate_quantile_group_products(
    out_dir: str,
    scores: pl.DataFrame,
    *,
    score_col: str,
    n_groups: int = 5,
    group: int = 1,
    trade_dates: list[date] | None = None,
) -> list[str]:
    """分位组 long-only：PIT 生成多期 weights 产物（形态对齐 trend_timing）。

    Parameters
    ----------
    out_dir : str
        产物根目录，每个 signal_date 落 ``out_dir/<T.isoformat()>/``。
    scores : pl.DataFrame
        因子/分数面板，须含 ``trade_date``、``ts_code`` 与 ``score_col``。
    n_groups, group
        分组数与人类编号组（1=最高分）。
    trade_dates : list[date], optional
        若给定则按此列表落盘（无权重日写空仓）；默认仅落有权重的交易日。

    Returns
    -------
    list[str]
        各 signal_date 对应 run_dir。
    """
    weights = build_group_weights(
        scores, score_col=score_col, n_groups=n_groups, group=group
    )
    if trade_dates is None:
        # 仅有权重的日；排序保证确定性
        dates = sorted(weights.keys())
    else:
        dates = list(trade_dates)

    run_dirs: list[str] = []
    for T in dates:
        wdf = weights.get(T)
        rd = Path(out_dir) / (T.isoformat() if isinstance(T, date) else str(T))
        rd.mkdir(parents=True, exist_ok=True)
        if wdf is None or wdf.is_empty():
            pl.DataFrame(
                {"ts_code": [], "target_weight": []},
                schema={"ts_code": pl.Utf8, "target_weight": pl.Float64},
            ).write_parquet(rd / "weights.parquet")
        else:
            wdf.select(["ts_code", "target_weight"]).write_parquet(rd / "weights.parquet")
        sig = T.isoformat() if isinstance(T, date) else str(T)
        (rd / "manifest.json").write_text(
            json.dumps(
                {"signal_date": sig, "status": "optimal"},
                ensure_ascii=False,
                indent=2,
            )
        )
        run_dirs.append(str(rd))
    return run_dirs
