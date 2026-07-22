"""滚动分层建仓：因子前 top_n 名每日建一层、持有 holding_days 日，层内等权。

语义对齐草稿 ``workspace/configs/sleeve_top200_h10.py::build_sleeve_weights``：

- 每个交易日 t：按分数在截面内取前 ``top_n`` 名（``direction="bottom"`` 则取最低）。
- 每天投入 ``1/holding_days`` 资金等权买入该层；每层持有 ``holding_days`` 个交易日。
- 稳态下同时持有 H 层、总仓位约 100%；同一票跨层权重累加。

目标权重（signal_date = t）::

    w_t(i) = Σ_{j=0}^{H-1}  1{i ∈ Top_N(t-j)} / (H · N_{t-j})

PIT：权重只用 ``trade_date <= signal_date`` 的分数截面；manifest 写
``signal_date`` + ``status:"optimal"``，契约对齐 sim。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl


def build_sleeve_weights(
    scores: pl.DataFrame,
    *,
    score_col: str,
    top_n: int,
    holding_days: int,
    trade_dates: list[date],
    direction: str = "top",
) -> dict[date, pl.DataFrame]:
    """构造滚动分层目标权重 ``dict[signal_date] -> DataFrame[ts_code, target_weight]``。

    ``direction="bottom"`` 取分数**最低**的 top_n 只（反向对照）。
    分组边界与层权重公式与草稿 ``build_sleeve_weights`` 逐行对齐。
    """
    date_idx = {d: i for i, d in enumerate(trade_dates)}
    n_dates = len(trade_dates)

    # 每日 top_n：分数降序（bottom 则升序），ordinal 打散并列（避免并列被整组纳入）
    top = (
        scores.filter(pl.col(score_col).is_not_null() & pl.col(score_col).is_finite())
        .with_columns(
            pl.col(score_col)
            .rank("ordinal", descending=(direction == "top"))
            .over("trade_date")
            .alias("_rk")
        )
        .filter(pl.col("_rk") <= top_n)
        .select(["trade_date", "ts_code"])
    )
    # 每层内部等权满仓 1/H：层权重 = 1 / (H · 该层实际股票数)
    layer_size = top.group_by("trade_date").agg(pl.len().alias("_n"))
    top = top.join(layer_size, on="trade_date", how="left").with_columns(
        (1.0 / (holding_days * pl.col("_n"))).alias("_w")
    )
    top = top.with_columns(
        pl.col("trade_date").replace_strict(date_idx, return_dtype=pl.Int32).alias("_i")
    )

    # 层 d 贡献到 t ∈ [d, d+H-1]：cross join 偏移量后按目标日聚合
    offsets = pl.DataFrame({"_off": list(range(holding_days))}, schema={"_off": pl.Int32})
    spread = (
        top.join(offsets, how="cross")
        .with_columns((pl.col("_i") + pl.col("_off")).alias("_t"))
        .filter(pl.col("_t") < n_dates)
        .group_by(["_t", "ts_code"])
        .agg(pl.col("_w").sum().alias("target_weight"))
    )

    weights: dict[date, pl.DataFrame] = {}
    for (t_idx,), grp in spread.partition_by("_t", as_dict=True).items():
        weights[trade_dates[int(t_idx)]] = grp.select(["ts_code", "target_weight"])
    return weights


def generate_sleeve_products(
    out_dir: str,
    scores: pl.DataFrame,
    *,
    score_col: str,
    top_n: int = 200,
    holding_days: int = 10,
    trade_dates: list[date] | None = None,
    direction: str = "top",
) -> list[str]:
    """滚动分层建仓：PIT 生成多期 weights 产物（形态对齐 trend_timing）。

    Parameters
    ----------
    out_dir : str
        产物根目录，每个 signal_date 落 ``out_dir/<T.isoformat()>/``。
    scores : pl.DataFrame
        因子/分数面板，须含 ``trade_date``、``ts_code`` 与 ``score_col``。
    score_col : str
        分数列名。
    top_n, holding_days, direction
        透传 ``build_sleeve_weights``。
    trade_dates : list[date], optional
        交易日日历（层偏移用）；默认取 scores 中唯一 trade_date 升序。

    Returns
    -------
    list[str]
        各 signal_date 对应 run_dir（按 trade_dates 中出现顺序）。
    """
    if trade_dates is None:
        trade_dates = sorted(scores.select("trade_date").unique()["trade_date"].to_list())
    weights = build_sleeve_weights(
        scores,
        score_col=score_col,
        top_n=top_n,
        holding_days=holding_days,
        trade_dates=trade_dates,
        direction=direction,
    )

    run_dirs: list[str] = []
    for T in trade_dates:
        wdf = weights.get(T)
        rd = Path(out_dir) / T.isoformat()
        rd.mkdir(parents=True, exist_ok=True)
        if wdf is None or wdf.is_empty():
            pl.DataFrame(
                {"ts_code": [], "target_weight": []},
                schema={"ts_code": pl.Utf8, "target_weight": pl.Float64},
            ).write_parquet(rd / "weights.parquet")
        else:
            wdf.select(["ts_code", "target_weight"]).write_parquet(rd / "weights.parquet")
        (rd / "manifest.json").write_text(
            json.dumps(
                {"signal_date": T.isoformat(), "status": "optimal"},
                ensure_ascii=False,
                indent=2,
            )
        )
        run_dirs.append(str(rd))
    return run_dirs
