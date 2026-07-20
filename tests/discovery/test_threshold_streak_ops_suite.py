"""P2-B 阈值/游程算子: ts_count_gt / ts_streak_gt / ts_count_cross_up。

TDD 铁律:
- golden 期望值为手算硬编码,禁止用被测函数自产期望。
- 覆盖 null / NaN / y=Constant / y=表达式。
- round-trip 幂等;既有表达式零回归。
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from factorzen.discovery.expression import evaluate, parse_expr, to_expr_string
from factorzen.discovery.operators import OPERATORS


def _panel(ts_code: str, x: list[float | None], y: list[float | None] | float) -> pl.DataFrame:
    n = len(x)
    if isinstance(y, (int, float)):
        y_col: list[float | None] = [float(y)] * n
    else:
        y_col = list(y)
    return pl.DataFrame(
        {
            "ts_code": [ts_code] * n,
            "trade_date": list(range(n)),
            "x": x,
            "y": y_col,
            # 叶子别名,供 parse_expr 路径
            "close_adj": x,
            "open_adj": y_col,
            "ret_1d": x,
            "pb": y_col,
        }
    ).sort(["ts_code", "trade_date"])


def _eval_op(name: str, df: pl.DataFrame, w: int, x_col: str = "x", y_col: str = "y") -> list:
    expr = OPERATORS[name].build([pl.col(x_col), pl.col(y_col)], w)
    return df.with_columns(expr.alias("r"))["r"].to_list()


def _approx_eq(got: list, exp: list, tol: float = 1e-12) -> None:
    assert len(got) == len(exp), f"len {len(got)} != {len(exp)}"
    for i, (g, e) in enumerate(zip(got, exp, strict=True)):
        if e is None:
            assert g is None or (isinstance(g, float) and math.isnan(g)), (
                f"idx{i}: expected null, got {g!r}"
            )
        else:
            assert g is not None and not (isinstance(g, float) and math.isnan(g)), (
                f"idx{i}: expected {e}, got {g!r}"
            )
            assert abs(float(g) - float(e)) < tol, f"idx{i}: {g} != {e}"


# ── 元信息 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["ts_count_gt", "ts_streak_gt", "ts_count_cross_up"])
def test_meta_arity_window_category(name: str):
    spec = OPERATORS[name]
    assert spec.category == "ts"
    assert spec.arity == 2
    assert spec.has_window is True


# ── ts_count_gt golden ───────────────────────────────────────────────────────

def test_ts_count_gt_constant_y_hand_calc():
    """y=Constant(1.5), w=4。gt 序列手算: F T F T F T T T F T。"""
    # x:        1  2  0  3  1  4  2  5 -1  6
    # >1.5?:    F  T  F  T  F  T  T  T  F  T
    # den thr=2; 期望见下
    x = [1.0, 2.0, 0.0, 3.0, 1.0, 4.0, 2.0, 5.0, -1.0, 6.0]
    df = _panel("A", x, 1.5)
    got = _eval_op("ts_count_gt", df, 4)
    exp = [
        None,   # den=1 < 2
        0.5,    # den=2 num=1
        1 / 3,  # den=3 num=1
        0.5,    # den=4 num=2
        0.5,    # win F T F T → 2/4
        0.5,    # T F T F → 2/4
        0.75,   # F T F T → wait positions 3-6: T F T T → 3/4
        0.75,   # 4-7: F T T T → 3/4
        0.75,   # 5-8: T T T F → 3/4
        0.75,   # 6-9: T T F T → 3/4
    ]
    _approx_eq(got, exp)


def test_ts_count_gt_null_and_nan_excluded():
    """null/NaN 对不计分子分母; NaN 不得当 True 泄漏。"""
    # x: 1, None, 3, 0, 5, 2, NaN, 4 ; y=1.5; w=4 thr=2
    # valid gt: F, inv, T, F, T, T, inv, T
    x = [1.0, None, 3.0, 0.0, 5.0, 2.0, float("nan"), 4.0]
    df = _panel("A", x, 1.5)
    got = _eval_op("ts_count_gt", df, 4)
    exp = [
        None,       # den=1
        None,       # den=1 (None 不计)
        0.5,        # den=2 num=1 (1.0, 3.0)
        1 / 3,      # den=3 num=1
        2 / 3,      # inv,T,F,T → den3 num2
        0.75,       # T,F,T,T → den4 num3
        2 / 3,      # F,T,T,inv → den3 num2
        1.0,        # T,T,inv,T → den3 num3
    ]
    _approx_eq(got, exp)


def test_ts_count_gt_y_as_expression_column():
    """y 为另一列(模拟子表达式物化列),非常数。"""
    # x: 2,3,1,4,0  y: 1,1,2,2,1  w=3 thr=1.5
    # gt: T T F T F
    x = [2.0, 3.0, 1.0, 4.0, 0.0]
    y = [1.0, 1.0, 2.0, 2.0, 1.0]
    df = _panel("A", x, y)
    got = _eval_op("ts_count_gt", df, 3)
    exp = [
        None,   # den=1 < 1.5
        1.0,    # den=2 num=2
        2 / 3,  # T T F
        2 / 3,  # T F T
        1 / 3,  # F T F
    ]
    _approx_eq(got, exp)


def test_ts_count_gt_via_parse_constant():
    """parse + evaluate 路径, y=Constant 0.0。"""
    # ret 手算: 1,-1,2,-2,3  vs 0 → T F T F T; w=4 thr=2
    rows = []
    rets = [1.0, -1.0, 2.0, -2.0, 3.0, 0.5, -0.5, 1.5]
    for i, r in enumerate(rets):
        rows.append({
            "ts_code": "A", "trade_date": i,
            "close_adj": 10.0, "ret_1d": r,
            "open_adj": 10.0, "vol": 1.0, "pb": 1.0,
        })
    df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
    node = parse_expr("ts_count_gt(ret_1d, 0.0, 4)")
    got = evaluate(node, df).to_list()
    # gt: T F T F T T F T
    exp = [
        None,   # den1
        0.5,    # T F
        2 / 3,  # T F T
        0.5,    # T F T F
        0.5,    # F T F T
        0.75,   # T F T T
        0.75,   # F T T F → 2/4? F T T F → num2=0.5  WAIT
        # positions 3-6: F T T F → num=2 → 0.5
        # positions 4-7: T T F T → num=3 → 0.75
    ]
    # recount:
    # i0 T den1 null
    # i1 TF den2 0.5
    # i2 TFT den3 2/3
    # i3 TFTF den4 0.5
    # i4 FTFT den4 0.5
    # i5 TFTT den4 0.75
    # i6 FTTF den4 0.5
    # i7 TTFT den4 0.75
    exp = [None, 0.5, 2 / 3, 0.5, 0.5, 0.75, 0.5, 0.75]
    _approx_eq(got, exp)


# ── ts_streak_gt golden ──────────────────────────────────────────────────────

def test_ts_streak_gt_constant_y_hand_calc():
    """连续上涨截断 w=3。手算: [1,0,1,2,3,0,1,2,3,0]。"""
    x = [1.0, -1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, -5.0]
    df = _panel("A", x, 0.0)
    got = _eval_op("ts_streak_gt", df, 3)
    exp = [1, 0, 1, 2, 3, 0, 1, 2, 3, 0]
    _approx_eq(got, exp)


def test_ts_streak_gt_null_breaks_and_outputs_null():
    """x null → 当日 null; 后续 True 从 1 重启。"""
    x = [1.0, 2.0, None, 3.0, 4.0]
    df = _panel("A", x, 0.0)
    got = _eval_op("ts_streak_gt", df, 5)
    exp = [1, 2, None, 1, 2]
    _approx_eq(got, exp)


def test_ts_streak_gt_nan_treated_as_null():
    """NaN 不得泄漏为 True 拉长 streak。"""
    x = [1.0, float("nan"), 2.0, 3.0]
    df = _panel("A", x, 0.0)
    got = _eval_op("ts_streak_gt", df, 5)
    exp = [1, None, 1, 2]
    _approx_eq(got, exp)


def test_ts_streak_gt_false_today_is_zero():
    """当日 x>y 为假 → 0(非 null)。"""
    x = [5.0, 4.0, 3.0]
    df = _panel("A", x, 10.0)  # 全假
    got = _eval_op("ts_streak_gt", df, 10)
    _approx_eq(got, [0, 0, 0])


def test_ts_streak_gt_y_expression_via_parse():
    """y = ts_mean(ret_1d, 3) 子表达式; 手算小窗。"""
    # ret: 1,1,1, -1,-1, 2,2,2
    rets = [1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0, 2.0]
    rows = [
        {"ts_code": "A", "trade_date": i, "ret_1d": r, "close_adj": 10.0,
         "open_adj": 10.0, "vol": 1.0, "pb": 1.0}
        for i, r in enumerate(rets)
    ]
    df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
    # 先物化 y=ts_mean(ret,3) 手算(min_samples=3):
    # i0-1 None; i2 mean(1,1,1)=1; i3 mean(1,1,-1)=1/3; i4 mean(1,-1,-1)=-1/3
    # i5 mean(-1,-1,2)=0; i6 mean(-1,2,2)=1; i7 mean(2,2,2)=2
    # streak ret > y, w=4:
    # i0: ret1 vs null → null (y null → x or y null → null)
    # i1: null
    # i2: 1>1? F → 0
    # i3: -1 > 1/3? F → 0
    # i4: -1 > -1/3? F → 0
    # i5: 2 > 0? T → 1
    # i6: 2 > 1? T → 2
    # i7: 2 > 2? F → 0
    node = parse_expr("ts_streak_gt(ret_1d, ts_mean(ret_1d, 3), 4)")
    got = evaluate(node, df).to_list()
    exp = [None, None, 0, 0, 0, 1, 2, 0]
    _approx_eq(got, exp)


# ── ts_count_cross_up golden ─────────────────────────────────────────────────

def test_ts_count_cross_up_constant_y_hand_calc():
    """上穿 y=0.5; 事件在 i=1 与 i=5; w=4 滚动计数。"""
    # x: 0,1,2,1,0,1,2,3
    # >0.5: F T T T F T T T
    # cross:  - T F F F T F F
    x = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0]
    df = _panel("A", x, 0.5)
    got = _eval_op("ts_count_cross_up", df, 4)
    exp = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    _approx_eq(got, exp)


def test_ts_count_cross_up_null_nan_no_false_cross():
    """null/NaN 不得制造假上穿。"""
    # x: 0, 1, None, 0, NaN, 1
    # y: 0.5
    # cross candidates: i1 (0→1) yes; i2 null no; i3 no; i4 nan no; i5 (nan→1) no prev valid
    x = [0.0, 1.0, None, 0.0, float("nan"), 1.0]
    df = _panel("A", x, 0.5)
    got = _eval_op("ts_count_cross_up", df, 6)
    # cumulative cross count in w=6 window = total crosses so far
    exp = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    _approx_eq(got, exp)


def test_ts_count_cross_up_y_expression_via_parse():
    """y=open 叶子; x=close。手算: close 上穿 open。"""
    # day:  0     1     2     3     4
    # open: 10    10    12    11    10
    # close:9     11    11    12    9
    # close>open: F T F T F
    # cross: prev close<=prev open and today close>open
    # i0: no prev → F
    # i1: prev 9<=10, 11>10 → T
    # i2: prev 11>10, 11<=12 → F (not today >)
    # i3: prev 11<=12, 12>11 → T
    # i4: prev 12>11, 9<=10 → F
    rows = []
    opens = [10.0, 10.0, 12.0, 11.0, 10.0]
    closes = [9.0, 11.0, 11.0, 12.0, 9.0]
    for i in range(5):
        rows.append({
            "ts_code": "A", "trade_date": i,
            "open_adj": opens[i], "close_adj": closes[i],
            "ret_1d": 0.0, "vol": 1.0, "pb": 1.0,
        })
    df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
    node = parse_expr("ts_count_cross_up(close, open, 3)")
    got = evaluate(node, df).to_list()
    # cross flags: 0,1,0,1,0
    # rolling w=3:
    # i0:0 i1:1 i2:1 i3:2 (1,0,1) i4:1 (0,1,0)
    exp = [0.0, 1.0, 1.0, 2.0, 1.0]
    _approx_eq(got, exp)


def test_ts_count_cross_up_no_cross_when_touch_from_above():
    """从上方触及再上不算上穿: 需昨 ≤ 且今 >。昨已 > 则不算。"""
    x = [2.0, 3.0, 4.0]  # 一直 > 0
    df = _panel("A", x, 0.0)
    got = _eval_op("ts_count_cross_up", df, 3)
    _approx_eq(got, [0.0, 0.0, 0.0])


# ── multi-stock isolation ────────────────────────────────────────────────────

def test_ops_respect_ts_code_boundary():
    """A 的窗口不得吃进 B。"""
    # A: 全 1 vs 0 → streak 递增; B: 全 -1 → streak 0
    rows = []
    for code, val in [("A", 1.0), ("B", -1.0)]:
        for d in range(5):
            rows.append({
                "ts_code": code, "trade_date": d,
                "x": val, "y": 0.0,
            })
    df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
    streak = _eval_op("ts_streak_gt", df, 5)
    a = streak[:5]
    b = streak[5:]
    _approx_eq(a, [1, 2, 3, 4, 5])
    _approx_eq(b, [0, 0, 0, 0, 0])


# ── round-trip ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("s", [
    "ts_count_gt(ret_1d, 0.0, 10)",
    "ts_streak_gt(ret_1d, 0.0, 10)",
    "ts_count_cross_up(close, open, 20)",
    "ts_streak_gt(ret_1d, ts_mean(ret_1d, 5), 10)",
    "ts_count_gt(close, ts_mean(close, 20), 60)",
    "ts_count_cross_up(ret_1d, 0.0, 5)",
])
def test_roundtrip_parse_to_string(s: str):
    node = parse_expr(s)
    s2 = to_expr_string(node)
    assert to_expr_string(parse_expr(s2)) == s2
    # 再解一次与原 AST 结构一致(Constant 用 float repr)
    node2 = parse_expr(s2)
    assert to_expr_string(node2) == s2


# ── 零回归:既有表达式 ────────────────────────────────────────────────────────

def _zero_reg_panel() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for code in ["A", "B"]:
        price = 10.0
        for d in range(20):
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({
                "trade_date": d,
                "ts_code": code,
                "close_adj": price,
                "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4),
                "pb": float(1 + abs(rng.standard_normal())),
                "ret_1d": float(rng.standard_normal() * 0.02),
                "open_adj": price * 0.99,
            })
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


# 实现前用 seed=42 面板捕获的硬编码期望(不经新算子)
_ZERO_REG_EXPECTED = {
    "rank(close)": [
        0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
        0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
        0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
        0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.3333333333333333,
        0.3333333333333333, 0.3333333333333333, 0.6666666666666666, 0.3333333333333333,
        0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666,
        0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
        0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
        0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.6666666666666666,
        0.6666666666666666, 0.6666666666666666, 0.3333333333333333, 0.6666666666666666,
    ],
    "ts_mean(ret_1d, 5)": [
        None, None, 0.0093474270631772, 0.0027141079829667092, 0.0019715827424283557,
        -0.0024087940514117084, 0.000317952575418934, -0.004418875231835322,
        0.0035341837905811966, 0.0063362585858250135, 0.0078830218063671,
        0.007315627744009517, 0.01146644071290214, 0.004395040149045206,
        0.005665782416112325, 0.007082026449282558, 0.004338237083525561,
        -0.002720064675842524, 0.002597388133401916, -0.0025131114438449605,
        None, None, -0.0031181950652004356, -0.009574208661015764, -0.011281283150252853,
        -0.00680426930039119, -0.00821037807890419, -0.006865493731537385,
        -0.0031950146781002456, -0.000530520966552099, 0.0019561670605413007,
        -0.0014427528531478523, -0.0079004398527022, -0.009844785341277158,
        0.002768087032644259, -0.001597817418942145, 0.0025940712526837176,
        0.003958200831605707, 0.015092236810755406, -0.0013029881189420118,
    ],
    "div(ts_std(close, 5), ts_mean(close, 5))": [
        None, None, 0.02322918623309764, 0.01999609406570839, 0.017345240712667916,
        0.0037084764117120473, 0.004723142858000164, 0.004611943438149084,
        0.006140024809266154, 0.005989267073483313, 0.0065112408377251355,
        0.007687262375368154, 0.01239608417535435, 0.012682096792162768,
        0.013905003640537856, 0.026033405955151225, 0.026547895307030327,
        0.016343003565276085, 0.014170915410040438, 0.012069083396194006,
        None, None, 0.012206786795645302, 0.011221965015338153, 0.018135877101782664,
        0.018635192063955065, 0.025320551431706136, 0.01954753084130873,
        0.010780410756992122, 0.007921247457608563, 0.010478039433262606,
        0.010236989378868999, 0.008984796257244998, 0.009019962842496612,
        0.00952780452168072, 0.009925611961049751, 0.010053785720475236,
        0.009118475194977958, 0.013521397833878282, 0.015425208310182334,
    ],
}


@pytest.mark.parametrize("expr", list(_ZERO_REG_EXPECTED.keys()))
def test_zero_regression_existing_exprs(expr: str):
    """不用新算子的既有表达式求值必须与实现前硬编码期望逐位一致。"""
    df = _zero_reg_panel()
    got = evaluate(parse_expr(expr), df).to_list()
    _approx_eq(got, _ZERO_REG_EXPECTED[expr], tol=1e-12)


# ── 搜索可采样 ───────────────────────────────────────────────────────────────

def test_random_search_can_emit_new_ops_and_zero_constant():
    """注册后 random_expression 能抽出新算子; 常数池含 0.0。"""
    from factorzen.discovery.expression import Constant, OpNode
    from factorzen.discovery.search import random_search as rs

    assert "ts_count_gt" in rs._OPS
    assert "ts_streak_gt" in rs._OPS
    assert "ts_count_cross_up" in rs._OPS
    # 常数池含 0.0(阈值/游程常用)
    # 间接:多次采样叶子常数,或直接读源保证
    rng = np.random.default_rng(0)
    saw_new_op = False
    saw_zero = False
    for _ in range(500):
        n = rs.random_expression(rng, max_depth=3)
        if isinstance(n, OpNode) and n.op in {
            "ts_count_gt", "ts_streak_gt", "ts_count_cross_up",
        }:
            saw_new_op = True
            assert n.window is not None and n.window >= 1
            assert len(n.children) == 2
        # 深搜常数
        stack = [n]
        while stack:
            cur = stack.pop()
            if isinstance(cur, Constant) and cur.value == 0.0:
                saw_zero = True
            if isinstance(cur, OpNode):
                stack.extend(cur.children)
    assert saw_new_op, "500 次采样未见新算子"
    assert saw_zero, "500 次采样未见 Constant(0.0),常数池应含 0.0"
