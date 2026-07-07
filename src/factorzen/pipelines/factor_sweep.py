"""factor sweep：在 ``--set`` 之上做参数网格扫描，串行跑每个组合并汇总对比表。

设计为"纯逻辑 + 可注入 runner"：``expand_grid`` / ``format_sweep_table`` 是纯函数，
``run_sweep`` 接受注入的 ``runner``（便于离线单测）；``pipeline_runner`` 是默认真实实现，
复用完整单因子评估管线并读回 ``*_ic.parquet`` 计算 IC 指标（需本地数据，不进 CI）。
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any

# runner: 接收一组 ``key=value`` 覆盖，返回该组合的指标 dict（ic_mean/ir/t/...）
Runner = Callable[[list[str]], dict[str, Any]]

# IC 维度看 ic_mean/ir；回测维度（如 top_n）看 sharpe/ann_ret/turnover
DEFAULT_METRIC_COLS = ("ic_mean", "ir", "sharpe", "ann_ret", "avg_turnover", "n")


def expand_grid(grid_tokens: Sequence[str]) -> list[list[str]]:
    """把 ``key=v1,v2,...`` 维度列表展开为笛卡尔积，每个组合是一组 ``key=value`` 覆盖串。

    例：``["backtest.top_n=30,50", "preprocessing.normalizer=zscore,rank_normal"]``
    → 4 个组合，每个形如 ``["backtest.top_n=30", "preprocessing.normalizer=zscore"]``。
    """
    dims: list[tuple[str, list[str]]] = []
    for token in grid_tokens:
        if "=" not in token:
            raise ValueError(f"--grid 需要 key=v1,v2,... 形式，收到: {token!r}")
        key, _, values_str = token.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--grid 键名非法: {token!r}")
        values = [v.strip() for v in values_str.split(",") if v.strip() != ""]
        if not values:
            raise ValueError(f"--grid 维度无取值: {token!r}")
        dims.append((key, values))

    if not dims:
        return []

    keys = [k for k, _ in dims]
    value_lists = [vs for _, vs in dims]
    return [
        [f"{k}={v}" for k, v in zip(keys, combo, strict=True)]
        for combo in itertools.product(*value_lists)
    ]


def run_sweep(
    grid_tokens: Sequence[str],
    runner: Runner,
    *,
    sort_by: str = "ir",
    extra_overrides: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """对网格每个组合调用 ``runner`` 收集指标，按 ``sort_by`` 降序排序后返回。

    单个组合评估失败不会中断整个 sweep：记录 ``error`` 字段，指标置 NaN 并排到末尾。
    """
    combos = expand_grid(grid_tokens)
    extra = list(extra_overrides or [])
    rows: list[dict[str, Any]] = []
    for overrides in combos:
        row: dict[str, Any] = {"overrides": overrides}
        try:
            row.update(runner(extra + overrides))
        except (Exception, SystemExit) as exc:
            # daily_single.main() 用 sys.exit(1/2) 报告内部错误 = SystemExit(BaseException，
            # 非 Exception)，会逃逸 except Exception 而中止整批网格、丢掉已跑组合。本函数
            # 契约是"单组合失败不中断整个 sweep"，故一并捕获、记 error 并继续。
            row["error"] = str(exc) or f"SystemExit(code={getattr(exc, 'code', None)})"
        rows.append(row)

    def sort_key(row: dict[str, Any]) -> float:
        value = row.get(sort_by)
        if not isinstance(value, (int, float)) or math.isnan(float(value)):
            return float("-inf")
        return float(value)

    rows.sort(key=sort_key, reverse=True)
    return rows


def _fmt_cell(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def format_sweep_table(
    rows: Sequence[dict[str, Any]],
    *,
    metric_cols: Sequence[str] = DEFAULT_METRIC_COLS,
) -> str:
    """把 sweep 结果渲染成等宽对齐的对比表（参数列 + 指标列）。"""
    if not rows:
        return "(空 sweep)"

    param_keys = [ov.split("=", 1)[0] for ov in rows[0]["overrides"]]
    headers = [k.split(".")[-1] for k in param_keys] + list(metric_cols)

    table: list[list[str]] = []
    for row in rows:
        param_vals = [ov.split("=", 1)[1] for ov in row["overrides"]]
        if "error" in row:
            metric_vals = [row["error"]] + [""] * (len(metric_cols) - 1)
        else:
            metric_vals = [_fmt_cell(row.get(c, "")) for c in metric_cols]
        table.append(param_vals + metric_vals)

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in table)) for i in range(len(headers))
    ]

    def render(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [render(headers), render(["-" * w for w in widths])]
    lines += [render(r) for r in table]
    return "\n".join(lines)


def format_sweep_csv(
    rows: Sequence[dict[str, Any]],
    *,
    metric_cols: Sequence[str] = DEFAULT_METRIC_COLS,
) -> str:
    """把 sweep 结果渲染成 CSV 文本（参数列 + 指标列 + error 列）。"""
    import csv
    import io

    if not rows:
        return ""

    param_keys = [ov.split("=", 1)[0] for ov in rows[0]["overrides"]]
    headers = param_keys + list(metric_cols) + ["error"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row in rows:
        param_vals = [ov.split("=", 1)[1] for ov in row["overrides"]]
        metric_vals = [row.get(c, "") for c in metric_cols]
        writer.writerow(param_vals + metric_vals + [row.get("error", "")])
    return buf.getvalue()


def pipeline_runner(
    *,
    factor: str,
    start: str,
    end: str,
    config_path: str | None = None,
    universe: str | None = None,
) -> Runner:
    """默认真实 runner：跑完整单因子评估，经 ``--metrics-out`` 读回 IC + 主策略回测指标。

    需要本地数据，故不在 CI 中运行——CI 通过注入假 runner 覆盖 ``run_sweep`` 逻辑。
    """
    import json
    import os
    import sys
    import tempfile

    from factorzen.pipelines import daily_single

    def runner(overrides: list[str]) -> dict[str, Any]:
        fd, metrics_path = tempfile.mkstemp(prefix="fz_sweep_", suffix=".json")
        os.close(fd)
        argv = ["fz-sweep"]
        if config_path:
            argv += ["--config", config_path]
        # 即便有 --config 也追加 --factor：daily_single 显式 --factor 优先于 config merge，
        # 否则同时给位置参数 factor 与 --config 时，位置参数被静默忽略、评估的是 config 里的因子。
        if factor:
            argv += ["--factor", factor]
        argv += ["--start", start, "--end", end]
        if universe:
            argv += ["--universe", universe]
        for override in overrides:
            argv += ["--set", override]
        argv += ["--metrics-out", metrics_path]

        old_argv = sys.argv
        try:
            sys.argv = argv
            daily_single.main()
            with open(metrics_path, encoding="utf-8") as fh:
                return json.load(fh)
        finally:
            sys.argv = old_argv
            if os.path.exists(metrics_path):
                os.unlink(metrics_path)

    return runner
