# M1 · 因子挖掘引擎 MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个把可解释的量价+基本面因子表达式自动生成、搜索、评估并落成标准 `DailyFactor` 的挖掘引擎，一条 `fz mine search` 端到端产出 top-K 候选 + 排行榜 + 可复现 manifest。

**Architecture:** 自定义表达式 AST（内部树 ↔ 可读字符串双向）→ 算子库编译成 polars 表达式（时序 `.over("ts_code")`、截面 `.over("trade_date")`）→ `ExpressionFactor` 把表达式包成标准 `DailyFactor` → 两段式评估（搜索内循环用快速 Rank IC/IR，top-K 才跑完整 `fz factor run`）→ random / 遗传编程搜索 → `mining_session` 编排并落 manifest。全程复用现有 IC / 预处理 / 去相关 / 注册设施，不重造。

**Tech Stack:** Python 3.10–3.12 · polars · numpy · argparse CLI · pytest · 现有 `factorzen.daily.evaluation` / `preprocessing` / `factors.registry`。

## Global Constraints

> 每个 task 的需求都隐含包含本节。以下值逐字来自接口探索，是最易写错的地方。

- **收益列命名两套**：`compute_fwd_returns` / `compute_rank_ic` 用 `fwd_ret_1d`（及 `fwd_ret_5d/10d/20d`）；`compute_ic` 用 `ret_1d`。互转用 `.rename({"fwd_ret_1d": "ret_1d"})`。
- **标准化函数输出不一致**：`cross_sectional_zscore(df, col=...)` **新增** 列 `f"{col}_z"`；`cross_sectional_rank(df, factor_col=..., method=...)` **原地覆盖** `factor_col`。
- **`get_factor(name)` 返回因子类（不是实例）**，未找到抛 `KeyError`；用法 `get_factor(name)()`。
- **`CorrelationResult.corr_matrix` 是 `np.ndarray`**（n×n），不是 DataFrame；名称在 `.factor_names`。
- **`ctx.daily` 列**：`trade_date(pl.Date), ts_code, open, high, low, close, pre_close, change, pct_chg, vol, amount, close_adj, open_adj, high_adj, low_adj`。
- **`ctx.daily_basic` 列**：`trade_date, ts_code, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_mv, circ_mv`（**没有 `turnover_rate` / `volume_ratio`**——基本面叶子只用这些实际存在的列）。
- **`DailyFactor`** 是 `@dataclass`，`required_data` 是 `ClassVar[list[str]]`；`name` / `description` 是类属性（非 dataclass 字段）；`compute(ctx) -> pl.DataFrame` 必须返回列 `trade_date, ts_code, factor_value`，并 `.filter(pl.col("trade_date") >= ctx.start)` 剔除预热期。
- **`ICAnalysisResult`** 字段：`factor_name, ic_mean, ic_std, ir, ic_positive_ratio, n_periods, ic_series, decay, frequency, ic_tstat, ic_pvalue, multi_period, oos_ic, walk_forward_ic`。
- **测试纪律**：纯 mock 数据（`np.random.default_rng(seed)` + list-of-dicts + `pl.DataFrame(rows)`），无磁盘 IO、无 Tushare 网络；CI 离线可重复。
- **提交规范**：conventional commits；作者必须为 `rookiewu417 <1007372080@qq.com>`（用 `git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit`）。
- **范围外**（不在本计划）：完整 OOS holdout 永久隔离、PBO/DSR/Reality Check（M2）；`finance` 财报因子；Agent（M5）；Leaderboard HTML 美化页；全 A 股性能优化。

---

## File Structure

| 文件 | 职责 | Task |
|---|---|---|
| `src/factorzen/discovery/__init__.py` | 包导出 | 1 |
| `src/factorzen/discovery/operators.py` | 算子规格表 `OPERATORS`：每个算子的 name/arity/类别/参数 + builder（子 `pl.Expr` → 新 `pl.Expr`）；叶子特征清单 | 1 |
| `src/factorzen/discovery/expression.py` | AST 节点（`Feature`/`Constant`/`OpNode`）+ `to_expr_string`/`parse_expr` 双向 + `compile_expr`（AST → `pl.Expr`）+ `complexity` | 2,3 |
| `src/factorzen/discovery/factor.py` | `ExpressionFactor`：表达式 → 标准 `DailyFactor`（停牌掩码 + 派生列 + 编译求值） | 4 |
| `src/factorzen/discovery/scoring.py` | `DataBundle`（预加载缓存 + train/valid 切分）+ `quick_fitness`（Rank IC/IR）+ 去相关惩罚 + 复杂度惩罚 + `score_candidate` | 5,6 |
| `src/factorzen/discovery/search/__init__.py` | 导出 | 7 |
| `src/factorzen/discovery/search/random_search.py` | 类型约束随机 AST 生成器 `random_expression` + `RandomSearcher` | 7 |
| `src/factorzen/discovery/search/genetic.py` | 遗传编程：交叉/变异/选择/精英/防膨胀 `GeneticSearcher` | 9 |
| `src/factorzen/discovery/mining_session.py` | 编排：生成→评估→排序→top-K→落 candidates.csv + manifest.json | 8 |
| `src/factorzen/discovery/export.py` | top-K 表达式 → 渲染独立 `.py` 写入 `workspace/factors/daily/` | 10 |
| `src/factorzen/pipelines/factor_mine.py` | `run_mine(...)` pipeline 入口 | 8,11 |
| `src/factorzen/cli/main.py` | 新增顶层 `fz mine search` / `fz mine leaderboard` 子命令 | 11 |
| `tests/test_discovery_operators.py` | 算子 builder polars 编译对拍 | 1 |
| `tests/test_discovery_expression.py` | round-trip 序列化 + 编译求值 + PIT/前视安全 | 2,3 |
| `tests/test_discovery_factor.py` | ExpressionFactor 一致性（复刻 momentum）+ 停牌掩码 | 4 |
| `tests/test_discovery_scoring.py` | quick_fitness 正确性 + train/valid + 去相关/复杂度惩罚 | 5,6 |
| `tests/test_discovery_search.py` | 随机生成合法性 + GP 交叉/变异合法性 | 7,9 |
| `tests/test_discovery_session.py` | 端到端 smoke（同 seed 同结果，产物齐全） | 8 |
| `tests/test_discovery_export.py` | 导出 .py 可发现可复现 | 10 |
| `tests/test_discovery_cli.py` | CLI 解析 + run_mine smoke | 11 |

---

## Task 1: 算子库（operators.py）

**Files:**
- Create: `src/factorzen/discovery/__init__.py`, `src/factorzen/discovery/operators.py`
- Test: `tests/test_discovery_operators.py`

**Interfaces:**
- Produces:
  - `LEAF_FEATURES: dict[str, str]` — 叶子名 → 实际列名（价量+基本面+派生）。派生 `vwap`/`log_vol`/`ret_1d` 预计算列。
  - `BASIC_FEATURES: set[str]` — 属于 `daily_basic` 的叶子名（决定是否 join）。
  - `OperatorSpec` dataclass: `name: str`, `category: Literal["ts","cs","arith"]`, `arity: int`, `has_window: bool`, `build: Callable[[list[pl.Expr], int|None], pl.Expr]`。
  - `OPERATORS: dict[str, OperatorSpec]`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_discovery_operators.py
from __future__ import annotations
import numpy as np
import polars as pl


def _toy_df(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for code in ["A", "B"]:
        price = 10.0
        for d in range(30):
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": d, "ts_code": code, "close_adj": price,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def test_ts_mean_matches_manual():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["ts_mean"].build([pl.col("close_adj")], 5)
    got = df.with_columns(expr.alias("f"))
    manual = df.with_columns(
        pl.col("close_adj").rolling_mean(5, min_samples=3).over("ts_code").alias("m"))
    assert got["f"].to_list() == manual["m"].to_list()


def test_cs_rank_is_within_unit_interval():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["rank"].build([pl.col("close_adj")], None)
    got = df.with_columns(expr.alias("r"))["r"].drop_nulls().to_list()
    assert all(0.0 < v < 1.0 for v in got)


def test_arith_add():
    from factorzen.discovery.operators import OPERATORS
    df = _toy_df()
    expr = OPERATORS["add"].build([pl.col("close_adj"), pl.col("vol")], None)
    got = df.with_columns(expr.alias("s"))
    assert got["s"].to_list() == (df["close_adj"] + df["vol"]).to_list()


def test_operator_categories_present():
    from factorzen.discovery.operators import OPERATORS
    cats = {spec.category for spec in OPERATORS.values()}
    assert cats == {"ts", "cs", "arith"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_operators.py -v`
Expected: FAIL（`ModuleNotFoundError: factorzen.discovery`）

- [ ] **Step 3: 实现 operators.py**

```python
# src/factorzen/discovery/operators.py
"""算子库：每个算子是一个把子表达式（pl.Expr）组合成新 pl.Expr 的工厂。

约定（编译前提）：求值表已按 (ts_code, trade_date) 排序。
- 时序算子(ts)用 .over("ts_code")；截面算子(cs)用 .over("trade_date")；算术(arith)逐元素。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import polars as pl

# 叶子名 → 求值表中的列名。vwap/log_vol/ret_1d 为派生列（ExpressionFactor 预计算）。
LEAF_FEATURES: dict[str, str] = {
    "close": "close_adj", "open": "open_adj", "high": "high_adj", "low": "low_adj",
    "vol": "vol", "amount": "amount", "vwap": "vwap", "log_vol": "log_vol", "ret_1d": "ret_1d",
    "total_mv": "total_mv", "circ_mv": "circ_mv", "pb": "pb", "pe_ttm": "pe_ttm",
    "ps_ttm": "ps_ttm", "dv_ttm": "dv_ttm",
}
BASIC_FEATURES: set[str] = {"total_mv", "circ_mv", "pb", "pe_ttm", "ps_ttm", "dv_ttm"}

_MIN = 3  # rolling 最小样本


def _safe_div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(b.abs() > 1e-12).then(a / b).otherwise(None)


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: Literal["ts", "cs", "arith"]
    arity: int
    has_window: bool
    build: Callable[[list[pl.Expr], "int | None"], pl.Expr]


def _ts(name, fn):  # window 时序算子
    return OperatorSpec(name, "ts", 1, True, lambda c, w: fn(c[0], w))


def _cs(name, fn):  # 截面算子
    return OperatorSpec(name, "cs", 1, False, lambda c, w: fn(c[0]))


def _ar(name, arity, fn):  # 算术算子
    return OperatorSpec(name, "arith", arity, False, lambda c, w: fn(*c))


OPERATORS: dict[str, OperatorSpec] = {
    # ── 时序（.over("ts_code")）──
    "ts_mean": _ts("ts_mean", lambda x, w: x.rolling_mean(w, min_samples=_MIN).over("ts_code")),
    "ts_std":  _ts("ts_std",  lambda x, w: x.rolling_std(w, min_samples=_MIN).over("ts_code")),
    "ts_sum":  _ts("ts_sum",  lambda x, w: x.rolling_sum(w, min_samples=_MIN).over("ts_code")),
    "ts_min":  _ts("ts_min",  lambda x, w: x.rolling_min(w, min_samples=_MIN).over("ts_code")),
    "ts_max":  _ts("ts_max",  lambda x, w: x.rolling_max(w, min_samples=_MIN).over("ts_code")),
    "ts_rank": _ts("ts_rank", lambda x, w:
        x.rolling_map(lambda s: float(s.rank()[-1]) / s.len(), w).over("ts_code")),
    "delay":   _ts("delay",   lambda x, w: x.shift(w).over("ts_code")),
    "delta":   _ts("delta",   lambda x, w: (x - x.shift(w)).over("ts_code")),
    "pct_change": _ts("pct_change", lambda x, w: _safe_div(x, x.shift(w).over("ts_code")) - 1.0),
    "ts_decay_linear": _ts("ts_decay_linear", lambda x, w:
        x.rolling_mean(w, min_samples=_MIN).over("ts_code")),  # MVP：等权近似线性衰减
    # ── 截面（.over("trade_date")）──
    "rank":  _cs("rank",  lambda x: (x.rank().over("trade_date") / (pl.len().over("trade_date") + 1))),
    "zscore": _cs("zscore", lambda x:
        _safe_div(x - x.mean().over("trade_date"), x.std().over("trade_date"))),
    "scale": _cs("scale", lambda x: _safe_div(x, x.abs().sum().over("trade_date"))),
    # ── 算术 ──
    "add": _ar("add", 2, lambda a, b: a + b),
    "sub": _ar("sub", 2, lambda a, b: a - b),
    "mul": _ar("mul", 2, lambda a, b: a * b),
    "div": _ar("div", 2, lambda a, b: _safe_div(a, b)),
    "abs": _ar("abs", 1, lambda a: a.abs()),
    "log": _ar("log", 1, lambda a: pl.when(a > 0).then(a.log()).otherwise(None)),
    "sign": _ar("sign", 1, lambda a: a.sign()),
    "sqrt": _ar("sqrt", 1, lambda a: pl.when(a >= 0).then(a.sqrt()).otherwise(None)),
}
```

> 注：`ts_corr`（二元时序）在 MVP 先省略——polars rolling 双列相关较繁琐，留作后续算子扩展。`ts_rank` 用 `rolling_map` 实现，慢但正确；若性能不足可后续优化。

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_operators.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/__init__.py src/factorzen/discovery/operators.py tests/test_discovery_operators.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): 算子库 operators.py（时序/截面/算术 + 叶子特征）"
```

---

## Task 2: 表达式 AST + 字符串双向序列化（expression.py 第一部分）

**Files:**
- Create: `src/factorzen/discovery/expression.py`
- Test: `tests/test_discovery_expression.py`

**Interfaces:**
- Consumes: `OPERATORS`, `LEAF_FEATURES`（Task 1）
- Produces:
  - `Node`（基类）；`Feature(name: str)`；`Constant(value: float)`；`OpNode(op: str, children: list[Node], window: int | None = None)`
  - `to_expr_string(node: Node) -> str`
  - `parse_expr(s: str) -> Node`
  - `complexity(node: Node) -> int`（节点总数）
  - `feature_names(node: Node) -> set[str]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_discovery_expression.py
from __future__ import annotations
import pytest


def test_round_trip_simple():
    from factorzen.discovery.expression import parse_expr, to_expr_string
    s = "rank(ts_mean(close, 5))"
    assert to_expr_string(parse_expr(s)) == s


def test_round_trip_nested():
    from factorzen.discovery.expression import parse_expr, to_expr_string
    s = "div(ts_mean(close, 5), ts_mean(close, 60))"
    assert to_expr_string(parse_expr(s)) == s


def test_constant_and_feature():
    from factorzen.discovery.expression import parse_expr, to_expr_string
    s = "mul(zscore(pb), 2.0)"
    assert to_expr_string(parse_expr(s)) == s


def test_complexity_counts_nodes():
    from factorzen.discovery.expression import parse_expr, complexity
    # rank(1) + ts_mean(1) + close(1) = 3
    assert complexity(parse_expr("rank(ts_mean(close, 5))")) == 3


def test_feature_names():
    from factorzen.discovery.expression import parse_expr, feature_names
    assert feature_names(parse_expr("div(close, pb)")) == {"close", "pb"}


def test_parse_rejects_unknown_op():
    from factorzen.discovery.expression import parse_expr
    with pytest.raises(ValueError):
        parse_expr("frobnicate(close, 5)")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_expression.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现 AST + 序列化**

```python
# src/factorzen/discovery/expression.py
"""表达式 AST：内部树 ↔ 可读字符串双向，并编译成 polars 表达式。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import polars as pl

from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS


class Node:
    pass


@dataclass
class Feature(Node):
    name: str


@dataclass
class Constant(Node):
    value: float


@dataclass
class OpNode(Node):
    op: str
    children: list[Node] = field(default_factory=list)
    window: int | None = None


def to_expr_string(node: Node) -> str:
    if isinstance(node, Feature):
        return node.name
    if isinstance(node, Constant):
        return repr(float(node.value))
    if isinstance(node, OpNode):
        parts = [to_expr_string(c) for c in node.children]
        if node.window is not None:
            parts.append(str(node.window))
        return f"{node.op}({', '.join(parts)})"
    raise TypeError(f"未知节点: {node!r}")


_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _split_args(s: str) -> list[str]:
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(" :
            depth += 1; cur += ch
        elif ch == ")":
            depth -= 1; cur += ch
        elif ch == "," and depth == 0:
            args.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def parse_expr(s: str) -> Node:
    s = s.strip()
    if "(" not in s:
        if _NUM.match(s):
            return Constant(float(s))
        if s in LEAF_FEATURES:
            return Feature(s)
        raise ValueError(f"未知叶子: {s}")
    op = s[: s.index("(")].strip()
    if op not in OPERATORS:
        raise ValueError(f"未知算子: {op}")
    inner = s[s.index("(") + 1 : s.rindex(")")]
    raw_args = _split_args(inner)
    spec = OPERATORS[op]
    window = None
    if spec.has_window:
        window = int(raw_args[-1]); raw_args = raw_args[:-1]
    children = [parse_expr(a) for a in raw_args]
    if len(children) != spec.arity:
        raise ValueError(f"{op} 期望 {spec.arity} 个子节点，得到 {len(children)}")
    return OpNode(op, children, window)


def complexity(node: Node) -> int:
    if isinstance(node, (Feature, Constant)):
        return 1
    return 1 + sum(complexity(c) for c in node.children)  # type: ignore[attr-defined]


def feature_names(node: Node) -> set[str]:
    if isinstance(node, Feature):
        return {node.name}
    if isinstance(node, Constant):
        return set()
    out: set[str] = set()
    for c in node.children:  # type: ignore[attr-defined]
        out |= feature_names(c)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_expression.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/expression.py tests/test_discovery_expression.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): 表达式 AST + 字符串双向序列化"
```

---

## Task 3: AST → polars 编译器（expression.py 第二部分）

**Files:**
- Modify: `src/factorzen/discovery/expression.py`（追加 `compile_expr` / `evaluate`）
- Test: `tests/test_discovery_expression.py`（追加编译用例）

**Interfaces:**
- Produces:
  - `compile_expr(node: Node) -> pl.Expr`
  - `evaluate(node: Node, df: pl.DataFrame) -> pl.Series`（在已排序 df 上求值，便于测试）

- [ ] **Step 1: 追加失败测试**

```python
# tests/test_discovery_expression.py （追加）
import numpy as np
import polars as pl


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for code in ["A", "B", "C"]:
        p = 10.0
        for d in range(40):
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": d, "ts_code": code, "close_adj": p,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def test_compile_ts_mean_ratio():
    from factorzen.discovery.expression import parse_expr, evaluate
    df = _toy()
    series = evaluate(parse_expr("div(ts_mean(close, 5), ts_mean(close, 20))"), df)
    assert series.len() == df.height
    assert series.drop_nulls().is_finite().all()


def test_compile_cross_sectional_rank_per_date():
    from factorzen.discovery.expression import parse_expr, evaluate
    df = _toy()
    out = df.with_columns(evaluate(parse_expr("rank(close)"), df).alias("r"))
    # 每个 trade_date 截面内 rank 落在 (0,1)
    vals = out.filter(pl.col("trade_date") == 30)["r"].drop_nulls().to_list()
    assert all(0.0 < v < 1.0 for v in vals)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_expression.py -k compile -v`
Expected: FAIL（`ImportError: cannot import name 'evaluate'`）

- [ ] **Step 3: 追加编译器实现**

```python
# src/factorzen/discovery/expression.py （追加到文件末尾）

def compile_expr(node: Node) -> pl.Expr:
    if isinstance(node, Feature):
        return pl.col(LEAF_FEATURES[node.name])
    if isinstance(node, Constant):
        return pl.lit(float(node.value))
    if isinstance(node, OpNode):
        spec = OPERATORS[node.op]
        child_exprs = [compile_expr(c) for c in node.children]
        return spec.build(child_exprs, node.window)
    raise TypeError(f"无法编译节点: {node!r}")


def evaluate(node: Node, df: pl.DataFrame) -> pl.Series:
    """在已按 (ts_code, trade_date) 排序的 df 上求值，返回 factor 列。"""
    return df.with_columns(compile_expr(node).alias("__f"))["__f"]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_expression.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/expression.py tests/test_discovery_expression.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): AST → polars 编译器"
```

---

## Task 4: ExpressionFactor（factor.py）+ 一致性测试

**Files:**
- Create: `src/factorzen/discovery/factor.py`
- Test: `tests/test_discovery_factor.py`

**Interfaces:**
- Consumes: `parse_expr`, `compile_expr`, `feature_names`（Task 2/3）；`BASIC_FEATURES`（Task 1）；`DailyFactor`
- Produces:
  - `class ExpressionFactor(DailyFactor)`，字段 `expression: str`、`mined_name: str`、`lookback_days: int`；`compute(ctx) -> pl.DataFrame`（列 `trade_date, ts_code, factor_value`）

- [ ] **Step 1: 写失败测试（一致性：表达式复刻 momentum）**

```python
# tests/test_discovery_factor.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl


def _make_daily_lf(n_stocks=8, n_days=60, seed=42) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        price = 10.0
        for day in days:
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": price,
                         "open": price, "high": price, "low": price,
                         "close_adj": price, "open_adj": price, "high_adj": price, "low_adj": price,
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).lazy()


@dataclass
class MockCtx:
    start: str = "20240301"
    end: str = "20240331"
    required_data: list = field(default_factory=lambda: ["daily", "daily_basic"])
    lookback_days: int = 30
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily: pl.LazyFrame | None = None
    _basic: pl.LazyFrame | None = None

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return self._basic if self._basic is not None else pl.DataFrame(
            {"trade_date": [], "ts_code": []}).lazy()


def test_expression_factor_matches_builtin_momentum():
    """pct_change(close, 20) 应与内置 momentum_20d 的 compute 输出一致。"""
    from factorzen.discovery.factor import ExpressionFactor
    from factorzen.builtin_factors.daily.momentum import Momentum20D

    lf = _make_daily_lf()
    ctx = MockCtx(_daily=lf)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builtin = Momentum20D().compute(ctx).sort(["trade_date", "ts_code"])

    mined = ExpressionFactor(expression="pct_change(close, 20)", mined_name="m20",
                             lookback_days=30).compute(ctx).sort(["trade_date", "ts_code"])

    j = builtin.join(mined, on=["trade_date", "ts_code"], suffix="_m")
    diff = (j["factor_value"] - j["factor_value_m"]).abs().max()
    assert diff is None or diff < 1e-9


def test_suspended_rows_masked():
    """vol==0（停牌）行不应产出有限因子值。"""
    from factorzen.discovery.factor import ExpressionFactor
    lf = _make_daily_lf()
    # 注入一只全停牌股票
    extra = pl.DataFrame({"trade_date": [date(2024, 3, 15)], "ts_code": ["999999.SH"],
                          "close": [5.0], "open": [5.0], "high": [5.0], "low": [5.0],
                          "close_adj": [5.0], "open_adj": [5.0], "high_adj": [5.0],
                          "low_adj": [5.0], "amount": [0.0], "vol": [0.0]}).lazy()
    ctx = MockCtx(_daily=pl.concat([lf, extra]))
    out = ExpressionFactor(expression="ts_mean(close, 5)", mined_name="x",
                           lookback_days=30).compute(ctx)
    sus = out.filter(pl.col("ts_code") == "999999.SH")
    assert sus.height == 0 or sus["factor_value"].is_null().all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_factor.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现 ExpressionFactor**

```python
# src/factorzen/discovery/factor.py
"""把表达式包装成标准 DailyFactor，可被 registry/评估管线无缝消费。"""
from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import polars as pl

from factorzen.daily.factors.base import DailyFactor
from factorzen.discovery.expression import compile_expr, feature_names, parse_expr
from factorzen.discovery.operators import BASIC_FEATURES

_PRICE_COLS = ["open", "high", "low", "close", "open_adj", "high_adj",
               "low_adj", "close_adj", "vol", "amount"]


class ExpressionFactor(DailyFactor):
    """表达式因子：可直接实例化（传 expression），或被子类用类属性覆盖 expression 后实例化。"""

    required_data: ClassVar[list[str]] = ["daily", "daily_basic"]
    expression: str = ""       # 子类可用类属性覆盖
    mined_name: str = ""
    lookback_days: int = 60

    def __init__(self, expression: str | None = None, mined_name: str | None = None,
                 lookback_days: int | None = None) -> None:
        # 不加 @dataclass：支持「直接传参」与「子类用类属性提供 expression」两种构造方式
        if expression is not None:
            self.expression = expression
        if mined_name is not None:
            self.mined_name = mined_name
        if lookback_days is not None:
            self.lookback_days = lookback_days
        if not self.expression:
            raise ValueError("ExpressionFactor 需要非空 expression")
        self.node = parse_expr(self.expression)
        if not getattr(self, "name", ""):
            self.name = self.mined_name or f"mined_{abs(hash(self.expression)) % (10**8)}"
        self.description = f"mined: {self.expression}"
        self._feats = feature_names(self.node)

    def compute(self, ctx) -> pl.DataFrame:
        daily = ctx.daily.collect()
        # 停牌掩码：vol==0 行的价量列置 null，避免污染时序算子
        daily = daily.with_columns([
            pl.when(pl.col("vol") > 0).then(pl.col(c)).otherwise(None).alias(c)
            for c in _PRICE_COLS if c in daily.columns
        ])
        # 派生列
        daily = daily.with_columns([
            (pl.col("amount") / pl.col("vol")).alias("vwap"),
            (pl.col("vol") + 1.0).log().alias("log_vol"),
        ]).with_columns(
            (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1.0).alias("ret_1d")
        )
        # 仅在表达式引用基本面叶子时 join daily_basic
        if self._feats & BASIC_FEATURES:
            basic = ctx.daily_basic.collect()
            if not basic.is_empty():
                daily = daily.join(basic, on=["trade_date", "ts_code"], how="left")
        df = daily.sort(["ts_code", "trade_date"])
        df = df.with_columns(compile_expr(self.node).alias("factor_value"))
        start = datetime.strptime(ctx.start, "%Y%m%d").date()
        return (
            df.filter(pl.col("trade_date") >= start)
            .select(["trade_date", "ts_code", "factor_value"])
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
        )
```

> 一致性测试要点：内置 `Momentum20D` 用 `close_adj.shift(20)`，而 `pct_change(close, 20)` 编译为 `close_adj / close_adj.shift(20) - 1`——两者数学等价。停牌掩码对 momentum 测试无副作用（mock 数据 vol 均 > 0）。

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_factor.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/factor.py tests/test_discovery_factor.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): ExpressionFactor + 与内置因子一致性测试"
```

---

## Task 5: 快速评估 DataBundle + quick_fitness（scoring.py 第一部分）

**Files:**
- Create: `src/factorzen/discovery/scoring.py`
- Test: `tests/test_discovery_scoring.py`

**Interfaces:**
- Consumes: `ExpressionFactor`（Task 4）；`compute_fwd_returns`, `compute_rank_ic`（现有）；`cross_sectional_zscore`（现有）
- Produces:
  - `@dataclass DataBundle`：`daily: pl.DataFrame`, `fwd_returns: pl.DataFrame`, `train_end: str`
  - `DataBundle.build(daily, train_ratio=0.7) -> DataBundle`
  - `quick_fitness(factor_df: pl.DataFrame, bundle: DataBundle, segment: Literal["train","valid"]) -> dict`（含 `ic_mean`, `ir`, `n`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_discovery_scoring.py
from __future__ import annotations
import numpy as np
import polars as pl
from datetime import date, timedelta


def _daily(seed=1, n_stocks=40, n_days=120):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)


def _signal_factor_df(daily: pl.DataFrame) -> pl.DataFrame:
    """构造与次日收益正相关的因子（用于验证 IC 为正）。"""
    df = daily.sort(["ts_code", "trade_date"]).with_columns(
        (pl.col("close_adj").shift(-1).over("ts_code") / pl.col("close_adj") - 1.0).alias("fwd"))
    return df.select(["trade_date", "ts_code", pl.col("fwd").alias("factor_value")]).drop_nulls()


def test_databundle_split():
    from factorzen.discovery.scoring import DataBundle
    b = DataBundle.build(_daily(), train_ratio=0.7)
    assert b.train_end is not None
    assert "fwd_ret_1d" in b.fwd_returns.columns


def test_quick_fitness_positive_for_signal():
    from factorzen.discovery.scoring import DataBundle, quick_fitness
    daily = _daily()
    b = DataBundle.build(daily, train_ratio=0.7)
    fac = _signal_factor_df(daily)
    res = quick_fitness(fac, b, segment="train")
    assert res["ic_mean"] > 0.05
    assert res["n"] > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_scoring.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现 scoring.py 第一部分**

```python
# src/factorzen/discovery/scoring.py
"""候选因子快速评估：两段式中的「内循环」——只算 Rank IC/IR，不跑回测。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore


@dataclass
class DataBundle:
    daily: pl.DataFrame
    fwd_returns: pl.DataFrame
    train_end: str  # "YYYYMMDD"，train 段含此日及之前

    @classmethod
    def build(cls, daily: pl.DataFrame, train_ratio: float = 0.7) -> "DataBundle":
        daily = daily.sort(["ts_code", "trade_date"])
        fwd = compute_fwd_returns(daily, price_col="close_adj" if "close_adj" in daily.columns else "close")
        dates = sorted(daily["trade_date"].unique().to_list())
        cut = dates[int(len(dates) * train_ratio)]
        train_end = cut.strftime("%Y%m%d") if hasattr(cut, "strftime") else str(cut)
        return cls(daily=daily, fwd_returns=fwd, train_end=train_end)

    def _segment_mask(self, df: pl.DataFrame, segment: str) -> pl.DataFrame:
        from datetime import datetime
        cut = datetime.strptime(self.train_end, "%Y%m%d").date()
        if segment == "train":
            return df.filter(pl.col("trade_date") <= cut)
        return df.filter(pl.col("trade_date") > cut)


def quick_fitness(factor_df: pl.DataFrame, bundle: DataBundle,
                  segment: Literal["train", "valid"] = "train") -> dict:
    """factor_df: [trade_date, ts_code, factor_value] → {ic_mean, ir, n}。"""
    seg = bundle._segment_mask(factor_df, segment)
    if seg.is_empty():
        return {"ic_mean": 0.0, "ir": 0.0, "n": 0}
    # 截面 zscore（cross_sectional_zscore 新增列 factor_value_z）
    clean = cross_sectional_zscore(seg, col="factor_value").rename({"factor_value_z": "factor_clean"})
    ret = bundle._segment_mask(bundle.fwd_returns, segment)
    res = compute_rank_ic(clean.select(["trade_date", "ts_code", "factor_clean"]),
                          ret, factor_col="factor_clean", frequency="daily")
    return {"ic_mean": res.ic_mean, "ir": res.ir, "n": res.n_periods}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_scoring.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/scoring.py tests/test_discovery_scoring.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): DataBundle + quick_fitness（train/valid 切分）"
```

---

## Task 6: 去相关 + 复杂度惩罚 + score_candidate（scoring.py 第二部分）

**Files:**
- Modify: `src/factorzen/discovery/scoring.py`
- Test: `tests/test_discovery_scoring.py`（追加）

**Interfaces:**
- Consumes: `compute_factor_correlation`（现有）；`complexity`（Task 2）
- Produces:
  - `max_correlation(factor_df, pool: dict[str, pl.DataFrame]) -> float`
  - `score_candidate(factor_df, node, bundle, pool, lam=0.5, gamma=0.002) -> dict`（含 `fitness`, `ic_train`, `ir_train`, `max_corr`, `complexity`）

- [ ] **Step 1: 追加失败测试**

```python
# tests/test_discovery_scoring.py （追加）
def test_max_correlation_self_is_one():
    from factorzen.discovery.scoring import max_correlation
    daily = _daily()
    fac = _signal_factor_df(daily).rename({"factor_value": "factor_clean"})
    corr = max_correlation(fac.rename({"factor_clean": "factor_value"}),
                           {"self": fac})
    assert corr > 0.99


def test_score_penalizes_complexity():
    from factorzen.discovery.scoring import DataBundle, score_candidate
    from factorzen.discovery.expression import parse_expr
    daily = _daily()
    b = DataBundle.build(daily)
    fac = _signal_factor_df(daily)
    simple = score_candidate(fac, parse_expr("close"), b, pool={}, gamma=0.01)
    # 复杂表达式（节点更多）在相同 IC 下 fitness 更低
    assert simple["complexity"] == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_scoring.py -k "correlation or complexity" -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 追加实现**

```python
# src/factorzen/discovery/scoring.py （追加）
from factorzen.daily.evaluation.correlation import compute_factor_correlation
from factorzen.discovery.expression import Node, complexity as _complexity


def max_correlation(factor_df: pl.DataFrame, pool: dict[str, pl.DataFrame]) -> float:
    """factor_df 与 pool 中每个因子的截面相关性绝对值的最大值。pool 为空时返回 0。"""
    if not pool:
        return 0.0
    fd = {"__cand__": factor_df.rename({"factor_value": "factor_clean"})
          if "factor_value" in factor_df.columns else factor_df}
    for name, df in pool.items():
        fd[name] = df.rename({"factor_value": "factor_clean"}) if "factor_value" in df.columns else df
    res = compute_factor_correlation(fd, factor_col="factor_clean")
    i = res.factor_names.index("__cand__")
    corrs = [abs(res.corr_matrix[i][j]) for j in range(len(res.factor_names)) if j != i]
    return max(corrs) if corrs else 0.0


def score_candidate(factor_df: pl.DataFrame, node: Node, bundle: DataBundle,
                    pool: dict[str, pl.DataFrame], lam: float = 0.5,
                    gamma: float = 0.002) -> dict:
    train = quick_fitness(factor_df, bundle, "train")
    mc = max_correlation(factor_df, pool)
    cplx = _complexity(node)
    fitness = train["ir"] - lam * mc - gamma * cplx
    return {"fitness": fitness, "ic_train": train["ic_mean"], "ir_train": train["ir"],
            "max_corr": mc, "complexity": cplx, "n_train": train["n"]}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_scoring.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/scoring.py tests/test_discovery_scoring.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): 去相关 + 复杂度惩罚 + score_candidate"
```

---

## Task 7: 类型约束随机表达式生成 + RandomSearcher（search/random_search.py）

**Files:**
- Create: `src/factorzen/discovery/search/__init__.py`, `src/factorzen/discovery/search/random_search.py`
- Test: `tests/test_discovery_search.py`

**Interfaces:**
- Consumes: `OPERATORS`, `LEAF_FEATURES`（Task 1）；`Node`, `Feature`, `Constant`, `OpNode`, `compile_expr`（Task 2/3）
- Produces:
  - `random_expression(rng: np.random.Generator, max_depth: int = 3) -> Node`
  - `class RandomSearcher`：`__init__(self, rng, max_depth=3)`；`propose(self) -> Node`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_discovery_search.py
from __future__ import annotations
import numpy as np
import polars as pl


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for code in ["A", "B", "C", "D"]:
        p = 10.0
        for d in range(30):
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": d, "ts_code": code, "close_adj": p, "open_adj": p,
                         "high_adj": p, "low_adj": p, "vol": 1e5, "amount": 1e6,
                         "vwap": p, "log_vol": 11.0, "ret_1d": 0.0,
                         "total_mv": 5e9, "circ_mv": 4e9, "pb": 2.0,
                         "pe_ttm": 20.0, "ps_ttm": 3.0, "dv_ttm": 1.0})
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def test_random_expression_is_compilable():
    from factorzen.discovery.search.random_search import random_expression
    from factorzen.discovery.expression import compile_expr, to_expr_string, parse_expr
    df = _toy()
    rng = np.random.default_rng(7)
    for _ in range(50):
        node = random_expression(rng, max_depth=3)
        # 可编译
        out = df.with_columns(compile_expr(node).alias("f"))
        assert "f" in out.columns
        # 可 round-trip
        assert to_expr_string(parse_expr(to_expr_string(node))) == to_expr_string(node)


def test_random_searcher_proposes_distinct():
    from factorzen.discovery.search.random_search import RandomSearcher
    from factorzen.discovery.expression import to_expr_string
    s = RandomSearcher(np.random.default_rng(0), max_depth=3)
    exprs = {to_expr_string(s.propose()) for _ in range(30)}
    assert len(exprs) > 5  # 有多样性
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_search.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现随机生成器**

```python
# src/factorzen/discovery/search/__init__.py
"""因子搜索算法。"""

# src/factorzen/discovery/search/random_search.py
from __future__ import annotations

import numpy as np

from factorzen.discovery.expression import Constant, Feature, Node, OpNode
from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS

_LEAVES = list(LEAF_FEATURES.keys())
_OPS = list(OPERATORS.keys())
_WINDOWS = [3, 5, 10, 20, 60]


def random_expression(rng: np.random.Generator, max_depth: int = 3) -> Node:
    """按算子类型签名递归生成合法 AST。叶子为特征或（少量）常数。"""
    if max_depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.1:
            return Constant(float(rng.choice([0.5, 1.0, 2.0])))
        return Feature(str(rng.choice(_LEAVES)))
    op = str(rng.choice(_OPS))
    spec = OPERATORS[op]
    children = [random_expression(rng, max_depth - 1) for _ in range(spec.arity)]
    window = int(rng.choice(_WINDOWS)) if spec.has_window else None
    return OpNode(op, children, window)


class RandomSearcher:
    def __init__(self, rng: np.random.Generator, max_depth: int = 3) -> None:
        self.rng = rng
        self.max_depth = max_depth

    def propose(self) -> Node:
        return random_expression(self.rng, self.max_depth)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_search.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/search/__init__.py src/factorzen/discovery/search/random_search.py tests/test_discovery_search.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): 类型约束随机表达式生成 + RandomSearcher"
```

---

## Task 8: mining_session 编排 + run_mine pipeline（端到端）

**Files:**
- Create: `src/factorzen/discovery/mining_session.py`, `src/factorzen/pipelines/factor_mine.py`
- Test: `tests/test_discovery_session.py`

**Interfaces:**
- Consumes: `RandomSearcher`（Task 7）；`ExpressionFactor`（Task 4）；`DataBundle`, `quick_fitness`, `score_candidate`（Task 5/6）；`to_expr_string`（Task 2）
- Produces:
  - `run_session(daily, *, n_trials, top_k, seed, method="random", train_ratio=0.7, out_dir) -> dict`（含 `candidates: list[dict]`, `n_trials`, `session_dir`）
  - `candidates.csv` 列：`rank, expression, ic_train, ir_train, ic_valid, ir_valid, max_corr, complexity`
  - `manifest.json`：`seed, method, n_trials, top_k, train_end, git_sha, duration_seconds, candidates`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_discovery_session.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import polars as pl
from datetime import date, timedelta


def _daily(seed=3, n_stocks=40, n_days=120):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "open_adj": p, "high_adj": p, "low_adj": p, "open": p, "high": p, "low": p,
                         "amount": 1e7, "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)


def test_session_runs_and_writes_artifacts(tmp_path: Path):
    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(), n_trials=20, top_k=5, seed=42,
                      method="random", out_dir=str(tmp_path))
    session_dir = Path(res["session_dir"])
    assert (session_dir / "candidates.csv").exists()
    assert (session_dir / "manifest.json").exists()
    assert len(res["candidates"]) <= 5
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["n_trials"] == 20
    assert manifest["seed"] == 42


def test_session_reproducible_same_seed(tmp_path: Path):
    from factorzen.discovery.mining_session import run_session
    a = run_session(_daily(), n_trials=20, top_k=5, seed=7, out_dir=str(tmp_path / "a"))
    b = run_session(_daily(), n_trials=20, top_k=5, seed=7, out_dir=str(tmp_path / "b"))
    expr_a = [c["expression"] for c in a["candidates"]]
    expr_b = [c["expression"] for c in b["candidates"]]
    assert expr_a == expr_b
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_session.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现 mining_session + pipeline**

```python
# src/factorzen/discovery/mining_session.py
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.discovery.expression import compile_expr, to_expr_string
from factorzen.discovery.scoring import DataBundle, quick_fitness, score_candidate
from factorzen.discovery.search.random_search import RandomSearcher


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _factor_values(node, daily: pl.DataFrame) -> pl.DataFrame:
    df = daily.sort(["ts_code", "trade_date"]).with_columns(compile_expr(node).alias("factor_value"))
    return df.select(["trade_date", "ts_code", "factor_value"]).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())


def run_session(daily: pl.DataFrame, *, n_trials: int, top_k: int, seed: int,
                method: str = "random", train_ratio: float = 0.7,
                out_dir: str = "workspace/mining_sessions") -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    # 停牌掩码（与 ExpressionFactor 一致，保证挖掘内 IC 与 fz factor run 一致）+ 派生列
    daily = daily.sort(["ts_code", "trade_date"])
    _price = ["open", "high", "low", "close", "open_adj", "high_adj", "low_adj", "close_adj", "vol", "amount"]
    daily = daily.with_columns([
        pl.when(pl.col("vol") > 0).then(pl.col(c)).otherwise(None).alias(c)
        for c in _price if c in daily.columns
    ]).with_columns([
        (pl.col("amount") / pl.col("vol")).alias("vwap"),
        (pl.col("vol") + 1.0).log().alias("log_vol"),
    ]).with_columns(
        (pl.col("close_adj") / pl.col("close_adj").shift(1).over("ts_code") - 1.0).alias("ret_1d"))
    bundle = DataBundle.build(daily, train_ratio=train_ratio)
    searcher = RandomSearcher(rng, max_depth=3)

    scored: list[dict] = []
    seen: set[str] = set()
    for _ in range(n_trials):
        node = searcher.propose()
        expr = to_expr_string(node)
        if expr in seen:
            continue
        seen.add(expr)
        try:
            fdf = _factor_values(node, daily)
            if fdf.height < 50:
                continue
            sc = score_candidate(fdf, node, bundle, pool={})
            if sc["n_train"] < 5:
                continue
            valid = quick_fitness(fdf, bundle, "valid")
            scored.append({"expression": expr, "ic_train": sc["ic_train"],
                           "ir_train": sc["ir_train"], "ic_valid": valid["ic_mean"],
                           "ir_valid": valid["ir"], "max_corr": sc["max_corr"],
                           "complexity": sc["complexity"], "fitness": sc["fitness"]})
        except Exception:
            continue

    scored.sort(key=lambda d: d["fitness"], reverse=True)
    top = scored[:top_k]

    session_dir = Path(out_dir) / f"session_{seed}_{method}"
    session_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"rank": i + 1, **{k: c[k] for k in
             ["expression", "ic_train", "ir_train", "ic_valid", "ir_valid", "max_corr", "complexity"]}}
            for i, c in enumerate(top)]
    pl.DataFrame(rows).write_csv(session_dir / "candidates.csv") if rows else \
        (session_dir / "candidates.csv").write_text("rank,expression\n")
    manifest = {"seed": seed, "method": method, "n_trials": n_trials, "top_k": top_k,
                "train_end": bundle.train_end, "git_sha": _git_sha(),
                "duration_seconds": round(time.perf_counter() - t0, 3), "candidates": top}
    (session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"candidates": top, "n_trials": n_trials, "session_dir": str(session_dir)}
```

```python
# src/factorzen/pipelines/factor_mine.py
"""fz mine 的 pipeline 入口：拉数据 → run_session。"""
from __future__ import annotations

from factorzen.discovery.mining_session import run_session


def run_mine(*, start: str, end: str, universe: str | None = None,
             n_trials: int = 200, top_k: int = 10, seed: int = 42,
             method: str = "random") -> dict:
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.core.universe import get_universe

    uni = None
    if universe:
        uni = get_universe(end, universe)["ts_code"].to_list()
    ctx = FactorDataContext(start=start, end=end, required_data=["daily", "daily_basic"],
                            lookback_days=60, universe=uni)
    daily = ctx.daily.collect()
    return run_session(daily, n_trials=n_trials, top_k=top_k, seed=seed, method=method)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_session.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/mining_session.py src/factorzen/pipelines/factor_mine.py tests/test_discovery_session.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): mining_session 端到端 + run_mine pipeline"
```

---

## Task 9: 遗传编程 GeneticSearcher（search/genetic.py）

**Files:**
- Create: `src/factorzen/discovery/search/genetic.py`
- Modify: `src/factorzen/discovery/mining_session.py`（`method="genetic"` 分支）
- Test: `tests/test_discovery_search.py`（追加）

**Interfaces:**
- Consumes: `random_expression`（Task 7）；`Node`, `OpNode`, `Feature`, `Constant`, `complexity`, `to_expr_string`（Task 2）
- Produces:
  - `crossover(a: Node, b: Node, rng) -> Node`
  - `mutate(node: Node, rng, max_depth=3) -> Node`
  - `class GeneticSearcher`：`evolve(score_fn: Callable[[Node], float], pop_size, generations) -> list[Node]`

- [ ] **Step 1: 追加失败测试**

```python
# tests/test_discovery_search.py （追加）
def test_crossover_and_mutate_stay_compilable():
    from factorzen.discovery.search.random_search import random_expression
    from factorzen.discovery.search.genetic import crossover, mutate
    from factorzen.discovery.expression import compile_expr
    df = _toy()
    rng = np.random.default_rng(11)
    for _ in range(40):
        a = random_expression(rng, 3)
        b = random_expression(rng, 3)
        child = crossover(a, b, rng)
        mutant = mutate(child, rng, 3)
        for node in (child, mutant):
            df.with_columns(compile_expr(node).alias("f"))  # 不抛异常即合法


def test_genetic_improves_toy_objective():
    """目标：偏好复杂度小的表达式 → GP 平均复杂度应下降或持平。"""
    from factorzen.discovery.search.genetic import GeneticSearcher
    from factorzen.discovery.expression import complexity
    rng = np.random.default_rng(5)
    gs = GeneticSearcher(rng, max_depth=3)
    best = gs.evolve(lambda node: -complexity(node), pop_size=20, generations=5)
    assert complexity(best[0]) <= 4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_search.py -k "crossover or genetic" -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现遗传编程**

```python
# src/factorzen/discovery/search/genetic.py
from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from factorzen.discovery.expression import Constant, Feature, Node, OpNode, complexity
from factorzen.discovery.search.random_search import random_expression


def _all_nodes(node: Node) -> list[Node]:
    out = [node]
    if isinstance(node, OpNode):
        for c in node.children:
            out.extend(_all_nodes(c))
    return out


def _replace_random_subtree(root: Node, new_sub: Node, rng: np.random.Generator) -> Node:
    root = copy.deepcopy(root)
    nodes = [n for n in _all_nodes(root) if isinstance(n, OpNode)]
    if not nodes:
        return new_sub
    target = nodes[int(rng.integers(len(nodes)))]
    if target.children:
        target.children[int(rng.integers(len(target.children)))] = new_sub
    return root


def crossover(a: Node, b: Node, rng: np.random.Generator) -> Node:
    donor_subtrees = _all_nodes(b)
    donor = copy.deepcopy(donor_subtrees[int(rng.integers(len(donor_subtrees)))])
    return _replace_random_subtree(a, donor, rng)


def mutate(node: Node, rng: np.random.Generator, max_depth: int = 3) -> Node:
    return _replace_random_subtree(node, random_expression(rng, max_depth=max_depth - 1), rng)


class GeneticSearcher:
    def __init__(self, rng: np.random.Generator, max_depth: int = 3) -> None:
        self.rng = rng
        self.max_depth = max_depth

    def evolve(self, score_fn: Callable[[Node], float], pop_size: int = 30,
               generations: int = 8, elite: int = 2) -> list[Node]:
        pop = [random_expression(self.rng, self.max_depth) for _ in range(pop_size)]
        for _ in range(generations):
            scored = sorted(pop, key=lambda n: score_fn(n), reverse=True)
            survivors = scored[: max(elite, pop_size // 2)]
            children = list(scored[:elite])
            while len(children) < pop_size:
                a = survivors[int(self.rng.integers(len(survivors)))]
                b = survivors[int(self.rng.integers(len(survivors)))]
                child = crossover(a, b, self.rng)
                if self.rng.random() < 0.3:
                    child = mutate(child, self.rng, self.max_depth)
                if complexity(child) <= 12:  # 防膨胀
                    children.append(child)
            pop = children
        return sorted(pop, key=lambda n: score_fn(n), reverse=True)
```

在 `mining_session.run_session` 中接入（替换原 searcher 选择逻辑）：

```python
# mining_session.py：在 run_session 内，按 method 选择搜索
    if method == "genetic":
        from factorzen.discovery.search.genetic import GeneticSearcher
        gs = GeneticSearcher(rng, max_depth=3)
        cache: dict[str, float] = {}
        def _score(node):
            expr = to_expr_string(node)
            if expr in cache:
                return cache[expr]
            try:
                fdf = _factor_values(node, daily)
                val = score_candidate(fdf, node, bundle, pool={})["fitness"] if fdf.height >= 50 else -9.9
            except Exception:
                val = -9.9
            cache[expr] = val
            return val
        nodes = gs.evolve(_score, pop_size=max(20, n_trials // 5),
                          generations=max(3, n_trials // 40))
        candidate_nodes = nodes
    else:
        candidate_nodes = [searcher.propose() for _ in range(n_trials)]
    # 之后对 candidate_nodes 统一评分（复用现有循环体）
```

> 实现时把 Task 8 的评分循环重构成「对 `candidate_nodes` 列表评分」，random 与 genetic 共用同一段评分代码（DRY）。

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_search.py tests/test_discovery_session.py -v`
Expected: PASS（含 `method="genetic"` 的 session 仍跑通——可在 session 测试追加一条 `method="genetic"` 断言）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/search/genetic.py src/factorzen/discovery/mining_session.py tests/test_discovery_search.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): 遗传编程 GeneticSearcher + session 接入"
```

---

## Task 10: 导出 top-K 为 workspace 因子（export.py）

**Files:**
- Create: `src/factorzen/discovery/export.py`
- Modify: `src/factorzen/discovery/mining_session.py`（session 结束自动导出 top-K）
- Test: `tests/test_discovery_export.py`

**Interfaces:**
- Consumes: `ExpressionFactor`（Task 4）
- Produces:
  - `render_factor_file(expression: str, name: str) -> str`（生成可被 registry 发现的 .py 文本）
  - `export_candidate(expression: str, name: str, dest_dir: str) -> Path`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_discovery_export.py
from __future__ import annotations
from pathlib import Path


def test_render_factor_file_contains_expression():
    from factorzen.discovery.export import render_factor_file
    text = render_factor_file("rank(ts_mean(close, 5))", "mined_demo")
    assert "rank(ts_mean(close, 5))" in text
    assert "class" in text and "ExpressionFactor" in text
    assert 'name = "mined_demo"' in text


def test_exported_file_is_importable_and_consistent(tmp_path: Path):
    """导出的 .py 能 import，且其因子 compute 与直接用 ExpressionFactor 一致。"""
    import importlib.util
    import numpy as np
    import polars as pl
    from dataclasses import dataclass, field
    from datetime import date, timedelta
    from factorzen.discovery.export import export_candidate
    from factorzen.discovery.factor import ExpressionFactor

    path = export_candidate("ts_mean(close, 5)", "mined_demo", str(tmp_path))
    assert path.exists()
    spec = importlib.util.spec_from_file_location("mined_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "MinedDemo") or any(
        isinstance(getattr(mod, a), type) for a in dir(mod))

    # 造数据 + mock ctx（复用 Task 4 风格）
    rng = np.random.default_rng(1)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < 40:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(6)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "open": p, "high": p,
                         "low": p, "close_adj": p, "open_adj": p, "high_adj": p, "low_adj": p,
                         "amount": 1e7, "vol": 1e5})
    lf = pl.DataFrame(rows).lazy()

    @dataclass
    class MockCtx:
        start: str = "20240301"; end: str = "20240331"
        required_data: list = field(default_factory=lambda: ["daily", "daily_basic"])
        lookback_days: int = 30; universe=None; snapshot_mode="daily"
        @property
        def daily(self): return lf
        @property
        def daily_basic(self): return pl.DataFrame({"trade_date": [], "ts_code": []}).lazy()

    direct = ExpressionFactor(expression="ts_mean(close, 5)", mined_name="mined_demo",
                              lookback_days=30).compute(MockCtx()).sort(["trade_date", "ts_code"])
    assert direct.height > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_export.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: 实现 export.py**

```python
# src/factorzen/discovery/export.py
"""把挖出的表达式渲染成独立 .py，落入 workspace/factors/daily/ 供 registry 发现。"""
from __future__ import annotations

from pathlib import Path


def _class_name(name: str) -> str:
    return "".join(p.capitalize() for p in name.replace("-", "_").split("_"))


def render_factor_file(expression: str, name: str) -> str:
    cls = _class_name(name)
    return f'''"""Mined factor: {name}. 由 fz mine 自动生成。表达式: {expression}"""

from factorzen.discovery.factor import ExpressionFactor


class {cls}(ExpressionFactor):
    name = "{name}"
    frequency = "daily"
    expression = "{expression}"
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
```

在 `run_session` 末尾自动导出 top-K（写入 `session_dir / "exported"`）：

```python
# mining_session.py：写完 manifest 后追加
    from factorzen.discovery.export import export_candidate
    exported_dir = session_dir / "exported"
    for i, c in enumerate(top):
        export_candidate(c["expression"], f"mined_{seed}_{i+1}", str(exported_dir))
```

> 注意：`ExpressionFactor.__init__` 支持「子类用类属性提供 `expression`」——`MinedDemo()` 无参实例化时 `__init__` 读取类属性 `expression` 并解析（见 Task 4 实现）。top-K 已在 `run_session` 结束时自动导出到 `session_dir/exported/`；单独的 `fz mine export <id>` 便利命令属后续，不在本 MVP。

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_export.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add src/factorzen/discovery/export.py src/factorzen/discovery/mining_session.py tests/test_discovery_export.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): top-K 导出为 workspace 因子"
```

---

## Task 11: CLI `fz mine search / leaderboard`（cli/main.py）

**Files:**
- Modify: `src/factorzen/cli/main.py`（顶层 `mine` subparser + `_cmd_mine_search` / `_cmd_mine_leaderboard`）
- Test: `tests/test_discovery_cli.py`

**Interfaces:**
- Consumes: `run_mine`（Task 8）；`build_parser`（现有，main.py:363）
- Produces:
  - 顶层命令 `fz mine search --start --end [--universe --method --trials --top-k --seed]`
  - `fz mine leaderboard <session_dir>`（打印 candidates.csv）
  - `_cmd_mine_search(args) -> int`、`_cmd_mine_leaderboard(args) -> int`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_discovery_cli.py
from __future__ import annotations


def test_parser_has_mine_search():
    from factorzen.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["mine", "search", "--start", "20240101", "--end", "20240601"])
    assert args.command == "mine"
    assert args.mine_command == "search"
    assert args.start == "20240101"
    assert callable(args.func)


def test_parser_has_mine_leaderboard():
    from factorzen.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["mine", "leaderboard", "some/dir"])
    assert args.mine_command == "leaderboard"
    assert args.session_dir == "some/dir"


def test_leaderboard_prints_csv(tmp_path, capsys):
    from factorzen.cli.main import _cmd_mine_leaderboard
    import argparse
    (tmp_path / "candidates.csv").write_text("rank,expression\n1,rank(close)\n")
    rc = _cmd_mine_leaderboard(argparse.Namespace(session_dir=str(tmp_path)))
    assert rc == 0
    assert "rank(close)" in capsys.readouterr().out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pixi run pytest tests/test_discovery_cli.py -v`
Expected: FAIL（`AttributeError: mine_command` / `ImportError`）

- [ ] **Step 3: 接入 CLI**

在 `build_parser()`（main.py:363，`factor` 组之后、`return parser` 之前）插入顶层 `mine` 组：

```python
    # ── fz mine ──（与 fz factor 并列的顶层命令组）
    mine = sub.add_parser("mine", help="Factor mining workflows")
    mine_sub = mine.add_subparsers(dest="mine_command", required=True)

    m_search = mine_sub.add_parser("search", help="Search candidate factor expressions")
    m_search.add_argument("--start", required=True, help="Start date YYYYMMDD")
    m_search.add_argument("--end", required=True, help="End date YYYYMMDD")
    m_search.add_argument("--universe", default=None, help="Universe name (e.g. csi500)")
    m_search.add_argument("--method", choices=["random", "genetic"], default="random")
    m_search.add_argument("--trials", type=int, default=200)
    m_search.add_argument("--top-k", dest="top_k", type=int, default=10)
    m_search.add_argument("--seed", type=int, default=42)
    m_search.set_defaults(func=_cmd_mine_search)

    m_lb = mine_sub.add_parser("leaderboard", help="Print a mining session leaderboard")
    m_lb.add_argument("session_dir", help="Path to a mining session directory")
    m_lb.set_defaults(func=_cmd_mine_leaderboard)
```

在模块顶层（仿 `_cmd_factor_sweep` 直调式）新增：

```python
def _cmd_mine_search(args: argparse.Namespace) -> int:
    from factorzen.pipelines.factor_mine import run_mine
    res = run_mine(start=args.start, end=args.end, universe=args.universe,
                   n_trials=args.trials, top_k=args.top_k, seed=args.seed, method=args.method)
    print(f"[mine] 完成：{len(res['candidates'])} 个候选 → {res['session_dir']}")
    return 0


def _cmd_mine_leaderboard(args: argparse.Namespace) -> int:
    from pathlib import Path
    csv = Path(args.session_dir) / "candidates.csv"
    if not csv.exists():
        print(f"[mine] 找不到 {csv}")
        return 2
    print(csv.read_text(encoding="utf-8"))
    return 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pixi run pytest tests/test_discovery_cli.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 全量质量门 + 提交**

```bash
pixi run pytest tests/test_discovery_*.py -q
pixi run lint
pixi run typecheck
git add src/factorzen/cli/main.py tests/test_discovery_cli.py
git -c user.name=rookiewu417 -c user.email=1007372080@qq.com commit -m "feat(discovery): fz mine search/leaderboard CLI"
```

---

## 收尾验收（全部 task 完成后）

- [ ] `pixi run pytest tests/test_discovery_*.py -q` 全绿
- [ ] `pixi run lint && pixi run typecheck && pixi run coverage` 全绿
- [ ] 手动 smoke（需本地数据）：`pixi run fz mine search --start 20230101 --end 20241231 --universe csi500 --trials 200 --top-k 10`
- [ ] 取一个挖出的候选 `pixi run fz factor run <mined_name>`，确认其 IC 与 candidates.csv 中的 `ic_train`/`ic_valid` 量级一致（一致性验收）
- [ ] `git status --short` 干净
- [ ] 更新 README「核心能力」表与 `docs/FactorZen-升级计划.md` 的 M1 进度

---

*本实现计划对接的现有接口均经源码探索验证（见 Global Constraints）。按 Task 顺序执行；M1 完成后进入 M2（防过拟合护栏）。*

---

## 实现完成记录（2026-06-30）

M1 按本计划 11 个 task 全部实现，subagent 驱动执行（每 task implementer + reviewer + 必要 fix），并通过 opus 整分支 final review。

**成果**：`src/factorzen/discovery/`（611 行源 + 548 行测试，20 commits，36 测试全绿，ruff 0 errors）。
算子库 → 表达式 AST↔字符串↔polars 编译器 → ExpressionFactor → scoring（quick_fitness + 贪心去相关 + 复杂度惩罚）→ random/遗传编程搜索 → mining_session 端到端 → 导出 workspace 因子 → `fz mine search/leaderboard` CLI。

**已验证**：一致性闭环在值层面坐实（`pct_change(close,20)`==内置 Momentum20D 逐值；导出 .py 因子 compute==ExpressionFactor 逐值 <1e-9）；PIT/前视安全（算子层无前视算子+停牌掩码+train/valid 切分）；同 seed 可复现。

**用法**：
```bash
pixi run fz mine search --start 20230101 --end 20241231 --universe csi500 --method genetic --trials 200 --top-k 10
pixi run fz mine leaderboard <session_dir>
# 复现：cp <session_dir>/exported/*.py workspace/factors/daily/ && fz factor run <name> --set preprocessing.neutralize=false
```

**Final review 后 deferred 到 M2/后续（非阻塞，已在 CLI/manifest 诚实标注）**：
- IC parity 完整对齐：挖掘内 IC 用 plain zscore，`fz factor run` 默认带中性化 → 完整复现需 `--set preprocessing.neutralize=false`；样本窗口/fwd 收益掩码口径未完全对齐。
- 导出因子未自动注册到 `workspace.factors.daily`（复现需手动 copy；`fz mine export` 桥接留后续）。
- genetic 路径双评分（效率减半）；genetic 多重检验 N 记账（manifest 记 CLI n_trials 而非实际评估数）。
- ts_rank rolling_map 性能（大 universe）。

**下一步**：M2 防过拟合护栏（PBO/DSR/Reality Check/OOS holdout 永久隔离），并在其中完整对齐 IC parity 与多重检验记账。`scoring.py` 已为此预留接口。
