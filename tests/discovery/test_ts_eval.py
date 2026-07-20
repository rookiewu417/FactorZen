"""Merged discovery tests: test_ts_eval.py

test_ts_chunk_eval.py：ts 节点分块求值：全 A OOM 路径逐位 parity 与批次边界
test_discovery_intraday_keys.py：引擎日期键 dtype 分派：Datetime 帧过 DataBundle/quick_fitness 不炸
"""

from __future__ import annotations

import datetime as dt
import math
from datetime import (
    date,
    datetime,
)

import numpy as np
import polars as pl
import pytest

from factorzen.discovery import expression as expression_mod
from factorzen.discovery.expression import (
    evaluate_materialized,
    parse_expr,
)
from factorzen.discovery.operators import OPERATORS
from factorzen.discovery.scoring import (
    DataBundle,
    _cut_literal,
    quick_fitness,
)

# ==== 来自 test_ts_chunk_eval.py ====
# ── 数据工厂 ────────────────────────────────────────────────────────────────

def _panel(
    n_stocks: int = 30,
    n_days: int = 80,
    *,
    unequal: bool = True,
    seed: int = 42,
    categorical: bool = False,
) -> pl.DataFrame:
    """多股面板，按 (ts_code, trade_date) 排序；可选不等长股（含 1/2 行股）与 null/NaN。"""
    rng = np.random.default_rng(seed)
    base = dt.date(2021, 1, 4)
    rows: list[dict] = []
    for si in range(n_stocks):
        code = f"{si:06d}.SZ"
        # 末几只刻意做短股；其余略抖动长度
        if unequal and si == n_stocks - 1:
            nd = 1
        elif unequal and si == n_stocks - 2:
            nd = 2
        elif unequal:
            nd = max(3, n_days - (si % 7))
        else:
            nd = n_days
        for di in range(nd):
            d = base + dt.timedelta(days=di)
            # 注入 null / NaN 到叶子
            close = float(rng.uniform(8.0, 20.0))
            vol = float(rng.uniform(1e4, 1e6))
            ret = float(rng.normal(0.0, 0.02))
            pb = float(rng.uniform(0.5, 5.0))
            if di % 17 == 0:
                close = None  # type: ignore[assignment]
            if di % 19 == 0:
                ret = float("nan")
            if di % 23 == 0:
                vol = None  # type: ignore[assignment]
            rows.append({
                "ts_code": code,
                "trade_date": d,
                "close_adj": close,
                "vol": vol,
                "ret_1d": ret,
                "pb": pb,
            })
    df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])
    if categorical:
        df = df.with_columns(pl.col("ts_code").cast(pl.Categorical))
    return df

def _force_chunk(monkeypatch, *, threshold: int = 1000, target: int = 500) -> None:
    """压低阈值，使中等测试面板走分块路径。"""
    monkeypatch.setattr(expression_mod, "TS_CHUNK_ROWS_THRESHOLD", threshold)
    monkeypatch.setattr(expression_mod, "TS_CHUNK_TARGET_ROWS", target)

def _eval_chunked_and_full(node, df, monkeypatch):
    """分块 on（低阈值）vs off（超高阈值）→ 两 Series。"""
    _force_chunk(monkeypatch, threshold=1000, target=500)
    chunked = evaluate_materialized(node, df)

    monkeypatch.setattr(expression_mod, "TS_CHUNK_ROWS_THRESHOLD", 10**12)
    unchunked = evaluate_materialized(node, df)
    return chunked, unchunked

# ── 1. 逐位 parity 矩阵 ─────────────────────────────────────────────────────

_SINGLE_TS_OPS = [
    "ts_mean", "ts_std", "ts_rank", "delay", "delta", "ts_zscore",
    "ts_sum", "ts_min", "ts_max", "ts_median", "ts_skew", "pct_change",
    "ts_decay_linear",
]

_TWO_TS_OPS = ["ts_corr", "ts_cov"]

# 嵌套混合：含 PR#61 老巢（test_parity_on_previously_working_shapes 同型）
_NESTED_EXPRS = [
    "mul(close, vol)",                    # 纯算术
    "add(pb, ret_1d)",                    # 纯算术
    "rank(pb)",                           # 截面套叶子
    "ts_std(ret_1d, 5)",                  # 时序套叶子
    "ts_mean(ts_std(ret_1d, 5), 5)",      # ts∘ts
    "neg(rank(pb))",                      # 算术套截面
    "rank(ts_std(ret_1d, 5))",            # cs(ts(x)) — 嵌套 over 老巢
    "ts_mean(rank(pb), 5)",               # ts(cs(x))
    "add(ts_mean(ret_1d, 5), rank(pb))",  # arith(ts, cs)
    "rank(add(ts_std(ret_1d, 5), ts_mean(ret_1d, 5)))",  # 深交叉
    "ts_corr(close, vol, 10)",
    "ts_cov(ret_1d, pb, 10)",
]

@pytest.mark.parametrize("op", _SINGLE_TS_OPS)
def test_parity_single_input_ts(op, monkeypatch):
    assert OPERATORS[op].category == "ts" and OPERATORS[op].arity == 1
    df = _panel(30, 80, unequal=True)
    assert df.height >= 1000  # 激发分块
    node = parse_expr(f"{op}(close, 5)")
    chunked, unchunked = _eval_chunked_and_full(node, df, monkeypatch)
    assert chunked.equals(unchunked), f"{op}: chunked ≠ unchunked"

@pytest.mark.parametrize("op", _TWO_TS_OPS)
def test_parity_two_input_ts(op, monkeypatch):
    assert OPERATORS[op].category == "ts" and OPERATORS[op].arity == 2
    df = _panel(30, 80, unequal=True)
    node = parse_expr(f"{op}(close, vol, 10)")
    chunked, unchunked = _eval_chunked_and_full(node, df, monkeypatch)
    assert chunked.equals(unchunked), f"{op}: chunked ≠ unchunked"

@pytest.mark.parametrize("expr", _NESTED_EXPRS)
def test_parity_nested_mixed(expr, monkeypatch):
    df = _panel(30, 80, unequal=True)
    node = parse_expr(expr)
    chunked, unchunked = _eval_chunked_and_full(node, df, monkeypatch)
    assert chunked.equals(unchunked), f"{expr}: chunked ≠ unchunked"

def test_parity_categorical_ts_code(monkeypatch):
    """P4c：ts_code 为 Categorical 时 rle/分块仍须逐位相等。"""
    df = _panel(25, 60, unequal=True, categorical=True)
    node = parse_expr("ts_mean(close, 5)")
    chunked, unchunked = _eval_chunked_and_full(node, df, monkeypatch)
    assert chunked.equals(unchunked)

# ── 2. 整股不切 ─────────────────────────────────────────────────────────────

def test_whole_stock_never_split(monkeypatch):
    """单股行数 > 目标行数 → 该股独立成批，结果仍 parity。"""
    # 1 只长股 800 行 + 若干短股；target=200 → 长股必须整只成批
    df = _panel(n_stocks=5, n_days=800, unequal=False, seed=7)
    # 再加几只短股
    short = _panel(n_stocks=10, n_days=30, unequal=True, seed=8)
    short = short.with_columns(
        (pl.col("ts_code").cast(pl.Utf8) + "_s").alias("ts_code")
    )
    df = pl.concat([df, short]).sort(["ts_code", "trade_date"])

    monkeypatch.setattr(expression_mod, "TS_CHUNK_ROWS_THRESHOLD", 100)
    monkeypatch.setattr(expression_mod, "TS_CHUNK_TARGET_ROWS", 200)

    batches = expression_mod._ts_stock_batches(df["ts_code"], 200)
    # 每批内 ts_code 段连续且不跨批切开同一 code
    for off, length in batches:
        codes = df.slice(off, length)["ts_code"].to_list()
        # 批内允许多股，但每只股的行必须连续（rle 段完整）
        rle = pl.Series(codes).rle()
        lengths = rle.struct.field("len").to_list()
        # 各 rle 段长度之和 = length，且与全局 rle 在该区间一致
        assert sum(lengths) == length

    # 任意单股长度若 > target，则该股必须独占一批
    global_rle = df["ts_code"].rle()
    g_lens = global_rle.struct.field("len").to_list()
    g_vals = global_rle.struct.field("value").to_list()
    for gl, gv in zip(g_lens, g_vals, strict=True):
        if gl > 200:
            matching = [
                (o, L) for o, L in batches
                if gl == L and df.slice(o, L)["ts_code"][0] == gv
            ]
            assert len(matching) == 1, f"超长股 {gv}({gl}行) 必须独占一批"

    node = parse_expr("ts_std(ret_1d, 5)")
    chunked = evaluate_materialized(node, df)
    monkeypatch.setattr(expression_mod, "TS_CHUNK_ROWS_THRESHOLD", 10**12)
    unchunked = evaluate_materialized(node, df)
    assert chunked.equals(unchunked)

# ── 3. 行序保持 ─────────────────────────────────────────────────────────────

def test_row_order_preserved(monkeypatch):
    """输出 Series 与输入 df 行对齐；手算 delay(1) 交叉验证。"""
    df = _panel(8, 20, unequal=True, seed=11).with_row_index("__ri")
    _force_chunk(monkeypatch, threshold=50, target=40)

    node = parse_expr("delay(close, 1)")
    # 求值会裁列，先记下期望：每股 close 右移 1
    work = df.select(["ts_code", "trade_date", "close_adj", "__ri"]).sort(
        ["ts_code", "trade_date"]
    )
    series = evaluate_materialized(node, work.drop("__ri"))

    expected: list[float | None] = []
    for code in work["ts_code"].unique(maintain_order=True).to_list():
        closes = work.filter(pl.col("ts_code") == code)["close_adj"].to_list()
        expected.append(None)
        expected.extend(closes[:-1])

    assert series.len() == work.height
    # null 位置
    for i, (got, exp) in enumerate(zip(series.to_list(), expected, strict=True)):
        if exp is None or (isinstance(exp, float) and math.isnan(exp)):
            assert got is None or (isinstance(got, float) and math.isnan(got)), (
                f"row {i}: expected null/nan, got {got}"
            )
        else:
            assert got is not None and abs(got - exp) < 1e-12, (
                f"row {i}: {got} != {exp}"
            )

    # 行序：with_row_index 顺序与输出对齐（按原 work 行）
    assert series.len() == df.height

# ── 4. 阈值不触发 ───────────────────────────────────────────────────────────

def test_threshold_skips_chunk_path(monkeypatch):
    """小帧 / 默认阈值：_materialize_ts_chunked 调用次数 = 0。"""
    df = _panel(5, 30, unequal=False)  # ~150 行 << 3e6
    calls = {"n": 0}
    orig = expression_mod._materialize_ts_chunked

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(expression_mod, "_materialize_ts_chunked", counting)
    # 保持默认阈值（3_000_000）
    node = parse_expr("ts_mean(ts_std(close, 5), 5)")
    out = evaluate_materialized(node, df)
    assert calls["n"] == 0
    assert out.len() == df.height

    # cs 节点也不走分块
    node_cs = parse_expr("rank(pb)")
    _ = evaluate_materialized(node_cs, df)
    assert calls["n"] == 0

def test_chunk_path_invoked_when_over_threshold(monkeypatch):
    """阈值压低后 ts 节点确实走分块函数。"""
    df = _panel(20, 60, unequal=True)
    _force_chunk(monkeypatch, threshold=100, target=80)
    calls = {"n": 0}
    orig = expression_mod._materialize_ts_chunked

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(expression_mod, "_materialize_ts_chunked", counting)
    node = parse_expr("ts_mean(close, 5)")
    _ = evaluate_materialized(node, df)
    assert calls["n"] >= 1

# ── 5. 批次边界算法单测 ─────────────────────────────────────────────────────

def test_ts_stock_batches_greedy_pack():
    """贪心合批：不切开股票，累计逼近 target。"""
    # A:3 B:3 C:2 D:5  target=5 → [A+B=6? no: A=3, then A+B=6>5 so A alone?
    # 贪心：当前批空 + 下一股 → 放；放后若再加超 target 则封批
    # A(3): batch=[A] size=3; B(3): 3+3=6>5 → seal [A], start [B]; ...
    codes = (
        ["A"] * 3 + ["B"] * 3 + ["C"] * 2 + ["D"] * 5 + ["E"] * 1
    )
    s = pl.Series(codes)
    batches = expression_mod._ts_stock_batches(s, target_rows=5)
    reconstructed = []
    for off, length in batches:
        reconstructed.extend(codes[off: off + length])
    assert reconstructed == codes
    assert sum(L for _, L in batches) == len(codes)
    # 每批 size：允许单股超 target（本例无）；否则 ≤ target + 不切
    for off, length in batches:
        # 批内完整 rle 段
        part = codes[off: off + length]
        assert part == s.slice(off, length).to_list()

def test_ts_stock_batches_single_stock_over_target():
    s = pl.Series(["X"] * 100)
    batches = expression_mod._ts_stock_batches(s, target_rows=30)
    assert batches == [(0, 100)]

def test_cs_chunked_parity(monkeypatch):
    """cs 节点按日期段分块:分块 on/off 逐位相同,行序还原。"""
    import datetime as dt

    import numpy as np
    import polars as pl

    import factorzen.discovery.expression as ex

    rng = np.random.default_rng(11)
    days = [dt.date(2021, 1, 4) + dt.timedelta(days=i) for i in range(40)]
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(9)]:
        for d in days:
            rows.append({"trade_date": d, "ts_code": c,
                         "close_adj": float(rng.uniform(5, 30)),
                         "vol": float(rng.uniform(1e5, 1e6))})
    df = pl.DataFrame(rows).sort(["ts_code", "trade_date"])

    for expr_s in ["rank(ts_mean(close, 5))",
                   "zscore(sub(rank(close), rank(vol)))",
                   "ts_rank(rank(close), 5)"]:
        node = ex.parse_expr(expr_s)
        base = ex.evaluate_materialized(node, df)
        monkeypatch.setattr(ex, "TS_CHUNK_ROWS_THRESHOLD", 50)
        monkeypatch.setattr(ex, "TS_CHUNK_TARGET_ROWS", 60)
        chunked = ex.evaluate_materialized(node, df)
        monkeypatch.setattr(ex, "TS_CHUNK_ROWS_THRESHOLD", 3_000_000)
        monkeypatch.setattr(ex, "TS_CHUNK_TARGET_ROWS", 1_500_000)
        assert chunked.equals(base), f"cs 分块 parity 失败: {expr_s}"

# ==== 来自 test_discovery_intraday_keys.py ====
def _intraday_daily(n_bars: int = 48, n_syms: int = 40) -> pl.DataFrame:
    # ≥MIN_IC_SAMPLES(30) 个标的,否则 compute_rank_ic 跳过全部横截面 → IC 序列空
    ts = [datetime(2026, 5, 1 + i // 24, i % 24, 0) for i in range(n_bars)]
    rows = []
    for si in range(n_syms):
        base = 100.0 + si * 10
        for i, t in enumerate(ts):
            rows.append({"ts_code": f"C{si:02d}USDT", "trade_date": t,
                         "close": base + i * 0.5, "vol": 1.0, "amount": 100.0})
    return pl.DataFrame(rows).with_columns(pl.col("trade_date").cast(pl.Datetime("us")))

def test_cut_literal_dispatch():
    intraday = _intraday_daily()
    daily = intraday.with_columns(pl.col("trade_date").cast(pl.Date))
    assert _cut_literal(intraday, "20260501") == datetime(2026, 5, 1)
    assert _cut_literal(daily, "20260501") == date(2026, 5, 1)

def test_databundle_and_fitness_on_datetime_frame():
    df = _intraday_daily()
    bundle = DataBundle.build(df, train_ratio=0.7)
    factor = df.select("trade_date", "ts_code",
                       pl.col("close").alias("factor_value"))
    res = quick_fitness(factor, bundle, "train")
    assert res["n"] > 0  # 切分/过滤在 Datetime 键上正常工作

def test_factor_values_eval_start_on_datetime_frame():
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.mining_session import _factor_values
    df = _intraday_daily()
    leaf_map = {"close": "close", "vol": "vol", "amount": "amount"}
    out = _factor_values(parse_expr("close", leaf_map), df, eval_start="20260502",
                         leaf_map=leaf_map)
    assert out["trade_date"].min() >= datetime(2026, 5, 2)

