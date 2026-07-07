"""把挖出的表达式渲染成独立 .py，落入 workspace/factors/daily/ 供 registry 发现。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from factorzen.discovery.expression import (
    compile_expr,
    parse_expr,
    required_lookback,
)
from factorzen.discovery.factor import ExpressionFactor

# 导出因子 lookback 下限（与内置默认一致）；表达式实际需求更大时按 AST 上取。
_MIN_LOOKBACK_DAYS = 60


def _lookback_for_expression(expression: str) -> int:
    """按表达式 AST 推导 lookback_days，至少 _MIN_LOOKBACK_DAYS。畸形表达式回退下限。"""
    try:
        return max(_MIN_LOOKBACK_DAYS, required_lookback(parse_expr(expression)))
    except ValueError:
        return _MIN_LOOKBACK_DAYS


def agent_candidates_csv_df(candidates: list[dict]) -> pl.DataFrame:
    """把 Agent/Team 候选 dict 列组装成 candidates.csv 帧，补 rank + passed 列。

    否则 read_candidate_expression 因缺 rank 列报 'ValueError: 缺少 rank/expression 列'，
    fz mine export-alpha 无法消费 M5/M6 session；且缺 passed 列使护栏过滤对这类 session
    静默不生效。Agent 候选本就全过护栏，passed=True。
    """
    if not candidates:
        return pl.DataFrame({"rank": [], "expression": [], "passed": [],
                             "holdout_ic": [], "dsr": []})
    return pl.DataFrame([{"rank": i + 1, "passed": True, **c}
                         for i, c in enumerate(candidates)])


def _class_name(name: str) -> str:
    return "".join(p.capitalize() for p in name.replace("-", "_").split("_"))


def render_factor_file(expression: str, name: str) -> str:
    cls = _class_name(name)
    expr_literal = repr(expression)
    lookback = _lookback_for_expression(expression)
    return f'''"""Mined factor: {name}. 由 fz mine 自动生成。表达式: {expression}"""

from factorzen.discovery.factor import ExpressionFactor


class {cls}(ExpressionFactor):
    name = "{name}"
    frequency = "daily"
    expression = {expr_literal}
    mined_name = "{name}"
    lookback_days = {lookback}


{cls}()  # 模块级实例化供 registry 自动发现
'''


def export_candidate(expression: str, name: str, dest_dir: str) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.py"
    path.write_text(render_factor_file(expression, name), encoding="utf-8")
    return path


def read_candidate_expression(session_dir: str, rank: int = 1, require_passed: bool = False) -> str:
    """从挖掘 session 的 candidates.csv 读取第 ``rank`` 名（1-based）候选表达式。

    ``require_passed=True`` 时，若该候选未通过防过拟合护栏（``passed`` 列为 false）则报错，
    提示用 ``--all`` 强制导出——这是 export-alpha 默认只放行 passed 候选的实现。老 session
    无 ``passed`` 列时该参数不生效（向后兼容）。
    """
    csv = Path(session_dir) / "candidates.csv"
    if not csv.exists():
        raise FileNotFoundError(f"找不到候选文件: {csv}")
    df = pl.read_csv(csv)
    if "expression" not in df.columns or "rank" not in df.columns:
        raise ValueError(f"{csv} 缺少 rank/expression 列")
    row = df.filter(pl.col("rank") == rank)
    if row.height == 0:
        raise ValueError(f"rank={rank} 不在 {csv}（共 {df.height} 个候选）")
    if require_passed and "passed" in df.columns:
        pv = row["passed"][0]
        if not (pv is True or str(pv).strip().lower() == "true"):
            raise ValueError(
                f"rank={rank} 未通过防过拟合护栏（passed=false）；用 --all 强制导出，"
                f"或换一个 passed=true 的候选。session: {csv}"
            )
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


def alpha_cross_section_from_daily(
    expression: str,
    daily: pl.DataFrame,
    date: str,
    leaf_map: dict[str, str] | None = None,
) -> pl.DataFrame:
    """市场无关版 α 截面：在**已含派生列**的 daily 帧上直接编译表达式。

    与 :func:`alpha_cross_section` 不同，不依赖 A 股 ``FactorDataContext``；
    ``leaf_map`` 为该市场叶子名→列名映射（默认 A 股 LEAF_FEATURES）。crypto 传
    ``profile.factors.leaf_features()``，``daily`` 需先经 ``derived_columns`` 加派生列。
    返回 ``[ts_code, alpha]``。
    """
    target = datetime.strptime(date, "%Y%m%d").date()
    node = parse_expr(expression, leaf_map)
    fdf = daily.sort(["ts_code", "trade_date"]).with_columns(
        compile_expr(node, leaf_map).alias("factor_value")
    )
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
