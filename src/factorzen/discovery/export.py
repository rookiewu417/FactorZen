"""把挖出的表达式渲染成独立 .py，落入 workspace/factors/daily/ 供 registry 发现。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from factorzen.discovery.factor import ExpressionFactor


def _class_name(name: str) -> str:
    return "".join(p.capitalize() for p in name.replace("-", "_").split("_"))


def render_factor_file(expression: str, name: str) -> str:
    cls = _class_name(name)
    expr_literal = repr(expression)
    return f'''"""Mined factor: {name}. 由 fz mine 自动生成。表达式: {expression}"""

from factorzen.discovery.factor import ExpressionFactor


class {cls}(ExpressionFactor):
    name = "{name}"
    frequency = "daily"
    expression = {expr_literal}
    mined_name = "{name}"
    lookback_days = 60


{cls}()  # 模块级实例化供 registry 自动发现
'''


def export_candidate(expression: str, name: str, dest_dir: str) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.py"
    path.write_text(render_factor_file(expression, name), encoding="utf-8")
    return path


def read_candidate_expression(session_dir: str, rank: int = 1) -> str:
    """从挖掘 session 的 candidates.csv 读取第 ``rank`` 名（1-based）候选表达式。"""
    csv = Path(session_dir) / "candidates.csv"
    if not csv.exists():
        raise FileNotFoundError(f"找不到候选文件: {csv}")
    df = pl.read_csv(csv)
    if "expression" not in df.columns or "rank" not in df.columns:
        raise ValueError(f"{csv} 缺少 rank/expression 列")
    row = df.filter(pl.col("rank") == rank)
    if row.height == 0:
        raise ValueError(f"rank={rank} 不在 {csv}（共 {df.height} 个候选）")
    return str(row["expression"][0])


def alpha_cross_section(expression: str, ctx: object, date: str) -> pl.DataFrame:
    """计算 ``expression`` 在 ``date`` 当日的截面 α，返回 ``[ts_code, alpha]`` 两列长表。

    复用 :class:`ExpressionFactor.compute`（含停牌掩码/派生列/有限性过滤），
    再取 ``date`` 当日截面并把因子值列重命名为 ``alpha``，
    直接喂给 ``fz portfolio build --alpha-file``。
    """
    target = datetime.strptime(date, "%Y%m%d").date()
    fdf = ExpressionFactor(expression=expression).compute(ctx)
    return (
        fdf.filter(pl.col("trade_date") == target)
        .select([pl.col("ts_code"), pl.col("factor_value").alias("alpha")])
        .filter(pl.col("alpha").is_finite())
    )


def export_alpha_cross_section(
    expression: str, ctx: object, date: str, out_path: str
) -> Path:
    """计算 ``date`` 当日截面 α 并落 ``[ts_code, alpha]`` 两列 parquet，返回输出路径。"""
    cross = alpha_cross_section(expression, ctx, date)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cross.write_parquet(out)
    return out
