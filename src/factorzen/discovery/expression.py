"""表达式 AST：内部树 ↔ 可读字符串双向，并编译成 polars 表达式。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import polars as pl

from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS


class LookaheadWindowError(ValueError):
    """时序算子窗口 <1（负=前视/未来函数、零=无意义）。

    ``ValueError`` 子类：所有 ``except ValueError`` 的外部输入解析点（LLM/配置/历史产物）
    仍原样接住（异常契约统一，CLAUDE.md 陷阱#7），同时可被特判以区分「前视」与「未知叶子」等
    其它解析失败（见 `is_lookahead_expr`）。铁律#1「PIT 无未来函数」的 parse 层根治。
    """


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
        # 铁律#1 根治：所有 has_window 算子的窗口都是**回看期**，须 ≥1。负窗口 = 前视
        # （delay(x,-1)=明日值、delta/pct_change(x,-w)=用未来）；零窗口无意义（rolling(0)/
        # shift(0)）。随机/遗传搜索 _WINDOWS 全正，负/零窗口只可能来自 LLM/历史产物——一律拒。
        if window < 1:
            raise LookaheadWindowError(
                f"{op} 的窗口必须 ≥1（回看期），得到 {window}：负窗口=前视/未来函数"
                f"（违反 PIT 无未来函数铁律），零窗口无意义")
        raw_args = raw_args[:-1]
    children = [parse_expr(a, valid_leaves) for a in raw_args]
    if len(children) != spec.arity:
        raise ValueError(f"{op} 期望 {spec.arity} 个子节点，得到 {len(children)}")
    return OpNode(op, children, window)


def is_lookahead_expr(s: str, leaves: dict[str, str] | set[str] | None = None) -> bool:
    """表达式是否含前视窗口（时序算子窗口 <1）。**只认前视**，不把其它解析失败误判成前视。

    用于把历史产物里的前视因子从「喂回 LLM 的正/负例」中剔除（否则引导 LLM 继续生成前视）。
    ``LookaheadWindowError`` → True；其它 ``ValueError``（未知叶子=别市场表达式、语法错）→ False
    （保持既有 raw 语义，不误伤干净的跨市场表达式）；正常解析 → False。
    """
    try:
        parse_expr(s, leaves)
    except LookaheadWindowError:
        return True
    except ValueError:
        return False
    return False


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


def leaf_lookbacks(node: Node) -> dict[str, int]:
    """每个叶子沿『根→该叶』路径累加的窗口数（同叶多次出现取最大）。

    与 `required_lookback`（只给最深路径的全局最大）互补：叶子只需填满它**上方**的
    窗口，不受别处更深路径拖累。据此逐叶判预热，避免『浅派生叶（如 `ts_sum(ret_1d, 20)`
    的 ret_1d，need=20）拖垮深 raw 叶路（如 `ts_zscore(delta(close,60),120)` 的 close，
    need=180）』的假拒绝。非常数表达式满足 ``max(leaf_lookbacks(node).values()) ==
    required_lookback(node)``（同为最深路径）。
    """
    out: dict[str, int] = {}

    def walk(n: Node, acc: int) -> None:
        if isinstance(n, Feature):
            if acc > out.get(n.name, -1):
                out[n.name] = acc
            return
        if isinstance(n, Constant):
            return
        own = getattr(n, "window", None) or 0
        for c in n.children:  # type: ignore[attr-defined]
            walk(c, acc + own)

    walk(node, 0)
    return out


def feature_names(node: Node) -> set[str]:
    if isinstance(node, Feature):
        return {node.name}
    if isinstance(node, Constant):
        return set()
    out: set[str] = set()
    for c in node.children:  # type: ignore[attr-defined]
        out |= feature_names(c)
    return out


def clamp_window_literals(
    expr_str: str,
    budgets: dict[str, int],
    leaf_map: dict[str, str] | set[str] | None = None,
) -> tuple[str, bool]:
    """把时序算子窗口字面量钳到「表达式所涉叶子」的最小可用预算。

    预算语义 = 可用上限（与 ``leaf_budgets`` / ``leaf_warmup_budgets`` 一致）。
    窗口参数识别口径同 ``required_lookback`` / ``leaf_lookbacks``：``OpNode.window``。
    仅对 ``budgets`` 中有条目的叶子取 min 作为 cap；无约束叶子视作无限。
    parse 失败 → ``(expr_str, False)`` 原样返回；预算充足或无窗口 → 不改写。

    返回 ``(钳后表达式串, 是否发生钳制)``。
    """
    if not budgets:
        return expr_str, False
    try:
        node = parse_expr(expr_str, leaf_map)
    except ValueError:
        return expr_str, False
    leaves = feature_names(node)
    constrained = [budgets[L] for L in leaves if L in budgets]
    if not constrained:
        return expr_str, False
    cap = min(constrained)
    if cap < 1:
        return expr_str, False

    changed = False

    def walk(n: Node) -> Node:
        nonlocal changed
        if isinstance(n, (Feature, Constant)):
            return n
        if isinstance(n, OpNode):
            new_children = [walk(c) for c in n.children]
            new_window = n.window
            if n.window is not None and n.window > cap:
                new_window = cap
                changed = True
            if new_window == n.window and new_children == n.children:
                return n
            return OpNode(n.op, new_children, new_window)
        return n

    new_node = walk(node)
    if not changed:
        return expr_str, False
    return to_expr_string(new_node), True


def warmup_bars(node, prepped: pl.DataFrame, eval_start,
                leaf_map: dict[str, str] | None = None) -> int:
    """表达式各叶子在 `eval_start` 之前的**非空且非 NaN 交易日数**的最小值 = 真实可用预热 bar 数。

    M1 搜索路径（`mining_session`）与 agent 路径（`discovery/evaluation`）共用本判定，
    两侧的预热门（`required_lookback` 对照）据此对齐，消除双路径漂移。

    不能按预热段交易日数算：daily_basic 缺 2019 时 dv_ttm 在预热段全 null，
    帧里有 57 个交易日，该叶子的可用预热却是 0。取各叶子最小值——
    任一叶子欠预热，整个表达式的首段就是噪声。

    non-null 不够：polars 里 NaN 不是 null，`is_not_null()` 对 NaN 单元格返回 True。
    NaN 预热单元格不是可用历史（如 `ret_1d = close_adj / close_adj.shift(1) - 1.0`
    在分母为 0 时产出 NaN 而非 null），必须一并剔除，否则会把噪声段误报为已预热。
    `is_not_nan()`/`is_nan()` 只对浮点列合法，整数/字符串列会报错，故按 schema 分流。

    ``leaf_map``：叶子名→列名映射（默认 A 股 `LEAF_FEATURES`）；crypto 等市场传各自映射，
    否则会用错列名判预热。``prepped`` 须是派生列（ret_1d/amplitude 等）已物化的帧。
    ``eval_start`` 须是与 ``prepped`` 的 trade_date dtype 匹配的字面量（调用方用
    `_cut_literal` 转换 "YYYYMMDD"，或直接传 date）。
    """
    warm = prepped.filter(pl.col("trade_date") < eval_start)
    if warm.is_empty():
        return 0
    leaves = feature_names(node)
    if not leaves:  # 纯常数表达式，无需预热
        return warm["trade_date"].n_unique()
    return min(warmup_bars_by_leaf(node, prepped, eval_start, leaf_map).values())


def leaf_warmup_budgets(prepped: pl.DataFrame, eval_start, leaves,
                        leaf_map: dict[str, str] | None = None) -> dict[str, int]:
    """每个叶子在 `eval_start` 之前的**可用预热 bar 数**（非空且非 NaN 交易日数）。

    这是 `warmup_bars_by_leaf` 的按叶名版本，也是它的**唯一实现来源**——生成侧要给 LLM
    报「叶子历史预算」，报的数必须与预热门 `warmup_shortfall` 判 have 的数逐值相等，否则
    prompt 承诺的预算与预热判定漂移就是继续骗 LLM。二者共用本函数即保证一致（B4.1）。

    ``leaves``：要查预算的叶子名可迭代对象（列表/集合）。语义同 `warmup_bars_by_leaf`：
    列缺失（如未拉 daily_basic）记 0；non-null 不够——polars 里 NaN 不是 null，NaN 预热
    单元格不是可用历史（`ret_1d` 分母为 0 时产 NaN 而非 null），须一并剔除；
    `is_not_nan()` 只对浮点列合法，按 schema 分流。
    """
    lm = LEAF_FEATURES if leaf_map is None else leaf_map
    warm = prepped.filter(pl.col("trade_date") < eval_start)
    out: dict[str, int] = {}
    for leaf in leaves:
        col = lm.get(leaf, leaf)
        if warm.is_empty() or col not in warm.columns:
            out[leaf] = 0
            continue
        valid = pl.col(col).is_not_null()
        if warm.schema[col].is_float():
            valid = valid & pl.col(col).is_not_nan()
        out[leaf] = warm.filter(valid)["trade_date"].n_unique()
    return out


def warmup_bars_by_leaf(node, prepped: pl.DataFrame, eval_start,
                        leaf_map: dict[str, str] | None = None) -> dict[str, int]:
    """各叶子在 `eval_start` 之前的**非空且非 NaN 交易日数**（逐叶，不取最小）。

    `warmup_bars` 取本函数各值的最小；预热门 `warmup_shortfall` 逐叶对照
    `leaf_lookbacks`——避免浅叶把深叶路拖成假拒绝。列缺失（如未拉 daily_basic）记 0。
    实现委托 `leaf_warmup_budgets`（唯一实现来源），保证生成侧报的预算与预热判定同源。
    """
    return leaf_warmup_budgets(prepped, eval_start, feature_names(node), leaf_map)


def warmup_shortfall(node, prepped: pl.DataFrame, eval_start,
                     leaf_map: dict[str, str] | None = None) -> tuple[str, int, int] | None:
    """逐叶预热门：返回缺口最大的 ``(leaf, need, have)``；预热充分返回 None。

    对每个叶子 L：``need_L = leaf_lookbacks(node)[L]``（根→L 路径窗口累加），
    ``have_L = warmup_bars_by_leaf(...)[L]``（L 在 eval_start 前的可用预热 bar 数）。
    仅当某叶 ``have_L < need_L`` 才算欠预热，取缺口（need-have）最大的一条用于报错。

    这样只需 20 根的浅派生叶（`ret_1d`, have=19）不会拖垮需要且满足 180 的深 raw 叶路
    （`close`），消除『全叶最小 warmup vs 最深路径 max lookback』跨叶错配的假拒绝。
    M1（`mining_session._underwarmed`）与 agent（`evaluate_expressions`）共用本门，
    两条路判定天然一致，消除双路径漂移。``eval_start`` 契约同 `warmup_bars`。
    """
    needs = {leaf: n for leaf, n in leaf_lookbacks(node).items() if n > 0}
    if not needs:                        # 纯常数/纯截面表达式，无需预热
        return None
    have_by_leaf = warmup_bars_by_leaf(node, prepped, eval_start, leaf_map)
    worst: tuple[str, int, int] | None = None
    worst_gap = 0
    for leaf, need in needs.items():
        have = have_by_leaf.get(leaf, 0)
        gap = need - have
        if gap > worst_gap:
            worst_gap = gap
            worst = (leaf, need, have)
    return worst


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


def evaluate_materialized(
    node: Node, df: pl.DataFrame, leaf_map: dict[str, str] | None = None
) -> pl.Series:
    """逐节点求值：时序/截面（带 .over()）算子的结果先物化成列，算术算子保持内联。

    修复 compile_expr 的嵌套 bug——单个嵌套 pl.Expr 里「截面 .over("trade_date") 套时序
    .over("ts_code")」在 polars 下（分组键冲突）求值为全 null。这里保证任何 .over() 的输入都是
    **已物化的列**（无 .over()），永不嵌套冲突。语义等价于「内层算子算完 → 外层算子作用于其结果」，
    因此在旧嵌套求值本就正确的形状上逐值相等（见 test_parity_on_previously_working_shapes）。

    只物化 ts/cs 类算子（.over() 所在处）、算术内联，是因为算术算子不含 .over()
    （由 test_arith_operators_carry_no_over_invariant 守卫）。df 须已按 (ts_code, trade_date) 排序。
    """
    lm = LEAF_FEATURES if leaf_map is None else leaf_map
    work = df
    counter = [0]

    def rec(n: Node) -> pl.Expr:
        nonlocal work
        if isinstance(n, Feature):
            return pl.col(lm[n.name])
        if isinstance(n, Constant):
            return pl.lit(float(n.value))
        if isinstance(n, OpNode):
            spec = OPERATORS[n.op]
            child_exprs = [rec(c) for c in n.children]
            expr = spec.build(child_exprs, n.window)
            if spec.category in ("ts", "cs"):
                name = f"__mz{counter[0]}"
                counter[0] += 1
                work = work.with_columns(expr.alias(name))
                return pl.col(name)
            return expr
        raise TypeError(f"无法求值节点: {n!r}")

    final_expr = rec(node)
    return work.with_columns(final_expr.alias("__f"))["__f"]


def evaluate(node: Node, df: pl.DataFrame) -> pl.Series:
    """在已按 (ts_code, trade_date) 排序的 df 上求值，返回 factor 列。"""
    return evaluate_materialized(node, df)
