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


_NUM = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _split_args(s: str) -> list[str]:
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def parse_expr(s: str, leaves: dict[str, str] | set[str] | None = None) -> Node:
    """解析表达式字符串。``leaves`` 为合法叶子集(默认 A 股 LEAF_FEATURES)，
    传入其他市场的叶子集(dict 键或 set)即可解析该市场表达式。"""
    valid_leaves = LEAF_FEATURES if leaves is None else leaves
    s = s.strip()
    if "(" not in s:
        if _NUM.match(s):
            return Constant(float(s))
        if s in valid_leaves:
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
        # 统一异常契约：空参数/窗口非整数都抛 ValueError（而非 IndexError），
        # 否则畸形 LLM 输出（如 'ts_mean()'）越界的 IndexError 逃过只捕 ValueError 的
        # 解析点，崩掉整个挖掘 session。
        if not raw_args:
            raise ValueError(f"{op} 需要窗口参数，但表达式无参数")
        try:
            window = int(raw_args[-1])
        except ValueError as e:
            raise ValueError(f"{op} 的窗口参数非整数: {raw_args[-1]!r}") from e
        raw_args = raw_args[:-1]
    children = [parse_expr(a, valid_leaves) for a in raw_args]
    if len(children) != spec.arity:
        raise ValueError(f"{op} 期望 {spec.arity} 个子节点，得到 {len(children)}")
    return OpNode(op, children, window)


def complexity(node: Node) -> int:
    if isinstance(node, (Feature, Constant)):
        return 1
    return 1 + sum(complexity(c) for c in node.children)  # type: ignore[attr-defined, misc]


def required_lookback(node: Node) -> int:
    """表达式求值所需的最小历史 bar 数：沿最深时序路径累加各 ts 算子的 window。

    如 ``ts_mean(delta(close, 5), 20)``：delta 需回看 5、其上 ts_mean 需再回看 20，
    故首个有效值要 5+20=25 根历史。截面/算术算子不加窗口（window=None→0）。
    导出因子据此设 lookback_days，避免硬编码 60 对大窗口/嵌套表达式欠预热（首段 NaN）。
    """
    if isinstance(node, (Feature, Constant)):
        return 0
    child_max = max((required_lookback(c) for c in node.children), default=0)  # type: ignore[attr-defined]
    own = getattr(node, "window", None) or 0
    return own + child_max


def feature_names(node: Node) -> set[str]:
    if isinstance(node, Feature):
        return {node.name}
    if isinstance(node, Constant):
        return set()
    out: set[str] = set()
    for c in node.children:  # type: ignore[attr-defined]
        out |= feature_names(c)
    return out


def compile_expr(node: Node, leaf_map: dict[str, str] | None = None) -> pl.Expr:
    """把 AST 编译成 polars 表达式。``leaf_map`` 为叶子名→列名映射
    (默认 A 股 LEAF_FEATURES)，传入其他市场映射即可编译该市场表达式。"""
    lm = LEAF_FEATURES if leaf_map is None else leaf_map
    if isinstance(node, Feature):
        return pl.col(lm[node.name])
    if isinstance(node, Constant):
        return pl.lit(float(node.value))
    if isinstance(node, OpNode):
        spec = OPERATORS[node.op]
        child_exprs = [compile_expr(c, lm) for c in node.children]
        return spec.build(child_exprs, node.window)
    raise TypeError(f"无法编译节点: {node!r}")


def evaluate(node: Node, df: pl.DataFrame) -> pl.Series:
    """在已按 (ts_code, trade_date) 排序的 df 上求值，返回 factor 列。"""
    return df.with_columns(compile_expr(node).alias("__f"))["__f"]
