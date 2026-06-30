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
        window = int(raw_args[-1])
        raw_args = raw_args[:-1]
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
