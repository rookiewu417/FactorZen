"""
test_search_genetic.py：Merged discovery tests: test_search_genetic.py
test_ts_eval.py：Merged discovery tests: test_ts_eval.py
"""

from __future__ import annotations

import datetime as dt
import math
import tempfile
from datetime import (
    date,
    datetime,
    timedelta,
)

import numpy as np
import polars as pl
import pytest

import factorzen.discovery.mining_session as ms
from factorzen.discovery import expression as expression_mod
from factorzen.discovery.evaluation import evaluate_expressions
from factorzen.discovery.expression import (
    evaluate_materialized,
    parse_expr,
)
from factorzen.discovery.mining_session import run_session
from factorzen.discovery.operators import BASIC_FEATURES, OPERATORS
from factorzen.discovery.scoring import (
    DataBundle,
    _cut_literal,
    quick_fitness,
    score_candidate,
)


# ==== 来自 test_search_genetic.py ====
# ==== 来自 test_discovery_search.py ====
def _toy(seed=0):
    # 动态覆盖 LEAF_FEATURES 的所有列名，避免每次新增叶子都要手动同步 fixture
    # （历史上 amplitude、turnover_rate 等新叶子都曾因此漏加，导致随机表达式 compile 时崩）。
    # 本文件的测试只验证「可编译 / 可求值不抛异常」，故所有叶子列填同一正值即可。
    from factorzen.discovery.operators import LEAF_FEATURES

    leaf_cols = sorted(set(LEAF_FEATURES.values()))
    rng = np.random.default_rng(seed)
    rows = []
    for code in ["A", "B", "C", "D"]:
        p = 10.0
        for d in range(30):
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            row: dict = {"trade_date": d, "ts_code": code}
            for col in leaf_cols:
                row[col] = p
            rows.append(row)
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])

def test_search_compilability_suite():
    """test_random_expression_is_compilable；test_random_searcher_proposes_distinct；test_crossover_and_mutate_stay_compilable"""
    # -- 原 test_random_expression_is_compilable --
    def _section_0_test_random_expression_is_compilable():
        from factorzen.discovery.expression import compile_expr, parse_expr, to_expr_string
        from factorzen.discovery.search.random_search import random_expression
        df = _toy()
        rng = np.random.default_rng(7)
        for _ in range(50):
            node = random_expression(rng, max_depth=3)
            # 可编译
            out = df.with_columns(compile_expr(node).alias("f"))
            assert "f" in out.columns
            # 可 round-trip
            assert to_expr_string(parse_expr(to_expr_string(node))) == to_expr_string(node)

    _section_0_test_random_expression_is_compilable()

    # -- 原 test_random_searcher_proposes_distinct --
    def _section_1_test_random_searcher_proposes_distinct():
        from factorzen.discovery.expression import to_expr_string
        from factorzen.discovery.search.random_search import RandomSearcher
        s = RandomSearcher(np.random.default_rng(0), max_depth=3)
        exprs = {to_expr_string(s.propose()) for _ in range(30)}
        assert len(exprs) > 5  # 有多样性

    _section_1_test_random_searcher_proposes_distinct()

    # -- 原 test_crossover_and_mutate_stay_compilable --
    def _section_2_test_crossover_and_mutate_stay_compilable():
        from factorzen.discovery.expression import compile_expr
        from factorzen.discovery.search.genetic import crossover, mutate
        from factorzen.discovery.search.random_search import random_expression
        df = _toy()
        rng = np.random.default_rng(11)
        for _ in range(40):
            a = random_expression(rng, 3)
            b = random_expression(rng, 3)
            child = crossover(a, b, rng)
            mutant = mutate(child, rng, 3)
            for node in (child, mutant):
                df.with_columns(compile_expr(node).alias("f"))  # 不抛异常即合法

    _section_2_test_crossover_and_mutate_stay_compilable()


def test_genetic_toy_objective_suite():
    """目标：偏好复杂度小的表达式 → GP 平均复杂度应下降或持平。；即使目标偏好高复杂度（防膨胀过滤压力最大），evolve 也必须在有限时间内终止。"""
    # -- 原 test_genetic_improves_toy_objective --
    def _section_0_test_genetic_improves_toy_objective():
        from factorzen.discovery.expression import complexity
        from factorzen.discovery.search.genetic import GeneticSearcher
        rng = np.random.default_rng(5)
        gs = GeneticSearcher(rng, max_depth=3)
        best = gs.evolve(lambda node: -complexity(node), pop_size=20, generations=5)
        assert complexity(best[0]) <= 4

    _section_0_test_genetic_improves_toy_objective()

    # -- 原 test_genetic_terminates_under_complexity_pressure --
    def _section_1_test_genetic_terminates_under_complexity_pressure():
        from factorzen.discovery.expression import complexity
        from factorzen.discovery.search.genetic import GeneticSearcher
        rng = np.random.default_rng(13)
        gs = GeneticSearcher(rng, max_depth=3)
        best = gs.evolve(lambda node: float(complexity(node)), pop_size=15, generations=6)
        assert len(best) == 15  # 种群规模维持，未因死循环卡住

    _section_1_test_genetic_terminates_under_complexity_pressure()


def test_search_space_max_lookback_tracks_constants():
    """预热前缀按搜索空间派生：= max(_WINDOWS) × _DEFAULT_MAX_DEPTH，随常量联动而非硬编码。

    最深路径全取最大窗口 → required_lookback 上界。prepare_mining_daily 据此设预热，
    保证搜索空间内任意表达式都不会因预热门被误拒。
    """
    from factorzen.discovery.search.random_search import (
        _DEFAULT_MAX_DEPTH,
        _WINDOWS,
        search_space_max_lookback,
    )

    assert search_space_max_lookback() == max(_WINDOWS) * _DEFAULT_MAX_DEPTH

# ==== 来自 test_discovery_genetic_parallel.py ====
def _synthetic_daily(n_stocks=40, n_days=160, seed=3):
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 1)
    prices = {s: 10.0 + s for s in range(n_stocks)}
    rows = []
    for d in range(n_days):
        dt = start + timedelta(days=d)
        for s in range(n_stocks):
            prev = prices[s]
            price = max(1.0, prev * (1 + rng.normal(0, 0.02)))
            prices[s] = price
            vol = float(rng.uniform(1e5, 1e6))
            rows.append(
                {
                    "ts_code": f"{s:04d}.SZ",
                    "trade_date": dt,
                    "pre_close": prev,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "open_adj": price,
                    "high_adj": price * 1.01,
                    "low_adj": price * 0.99,
                    "close_adj": price,
                    "vol": vol,
                    "amount": price * vol,
                }
            )
    daily = pl.DataFrame(rows)
    return daily.with_columns(
        [pl.lit(1.0).alias(c) for c in sorted(BASIC_FEATURES) if c not in daily.columns]
    )

def test_genetic_parallel_deterministic(tmp_path):
    daily = _synthetic_daily()
    r1 = run_session(
        daily, n_trials=40, top_k=3, seed=7, method="genetic",
        out_dir=str(tmp_path / "w1"), workers=1,
    )
    r4 = run_session(
        daily, n_trials=40, top_k=3, seed=7, method="genetic",
        out_dir=str(tmp_path / "w4"), workers=4,
    )
    e1 = [c["expression"] for c in r1["candidates"]]
    e4 = [c["expression"] for c in r4["candidates"]]
    assert e1 == e4, "并行与串行的 leaderboard 表达式序列必须一致"
    f1 = [round(float(c["ir_train"]), 6) for c in r1["candidates"]]
    f4 = [round(float(c["ir_train"]), 6) for c in r4["candidates"]]
    assert f1 == f4, "并行与串行的候选分数必须一致"

# ==== 来自 test_genetic_pool_hygiene.py ====
# 做薄的 basic 叶子。只薄一个（如 pb）不够——random 的 40 次试验很可能一次都没抽到它，
# 于是 `test_random_ir_pool_also_excludes_them` 变成空跑（变异实证：删掉 random 那道门，
# 它照样绿）。薄掉一组，才让两条路径都真的接触到死表达式。
_THIN_LEAVES = ("pb", "pe_ttm", "ps_ttm", "dv_ttm")

def _daily_with_thin_leaf(n_stocks: int = 40, n_days: int = 200, seed: int = 5) -> pl.DataFrame:
    """`_THIN_LEAVES` 只对前 20 只非 null → 每个截面 20 只 < `_MIN_CROSS_SAMPLES`(=30) → n_train=0。

    但因子帧行数 = 20×200 = 4000 ≫ 50，绕过 `_score_one` 的 `fdf.height < 50` 那道门。
    这正是真实 A 股的形态：新股/停牌让 basic 叶子大面积缺失。
    """
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        px = 10.0 + i
        for dd in days:
            prev = px
            px = max(1.0, px * (1 + rng.normal(0, 0.02)))
            vol = float(rng.uniform(1e5, 1e6))
            rows.append({
                "ts_code": f"{600000 + i:06d}.SH", "trade_date": dd, "pre_close": prev,
                "open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
                "open_adj": px, "high_adj": px * 1.01, "low_adj": px * 0.99, "close_adj": px,
                "vol": vol, "amount": px * vol,
                "total_mv": px * 1e6, "circ_mv": px * 8e5,
                **{leaf: (1.5 + j if i < 20 else None)      # ← 薄叶子
                   for j, leaf in enumerate(_THIN_LEAVES)},
            })
    daily = pl.DataFrame(rows)
    return daily.with_columns([
        pl.lit(1.0).alias(c) for c in sorted(BASIC_FEATURES) if c not in daily.columns
    ])

def test_thin_leaf_yields_sentinel_zero_not_nan():
    """前提坐实：死表达式的 `ir_train` 是 `0.0`（有限值），`from_ir_pool` 剔不掉它。

    没有这条，下面两个测试可能因为「其实是 nan、早就被剔了」而失去意义。
    """
    from factorzen.discovery.derived import add_derived_columns
    from factorzen.discovery.guardrails import DeflationBasis

    daily = add_derived_columns(_daily_with_thin_leaf())
    bundle = DataBundle.build(daily)
    node = parse_expr("rank(pb)")
    fdf = ms._factor_values(node, daily, None, None)

    assert fdf.height >= 50, "前提：因子帧要够长，否则 _score_one 的 height 门就挡住了"
    sc = score_candidate(fdf, node, bundle, pool={})
    assert sc["n_train"] == 0
    assert sc["ir_train"] == 0.0, "是 sentinel 0.0，不是 nan"

    basis = DeflationBasis.from_ir_pool([sc["ir_train"], 0.2, 0.3])
    assert basis.n_trials == 3, "0.0 是有限值，from_ir_pool 剔不掉——必须在上游拦"

def _capture_ir_pool(monkeypatch, daily, *, method: str) -> list:
    """跑一遍 run_session，抓它真正喂给 `DeflationBasis.from_ir_pool` 的池子。

    不 spy `score_candidate`：genetic 会在 `_score_one` 与后面的 `scored` 循环里各调一次，
    数出来是双份。直接抓 deflation 池，才是「N 与 sharpe_variance 的真实入参」。
    """
    from factorzen.discovery.guardrails import DeflationBasis as _Real

    pools: list[list] = []

    class _Spy:
        @classmethod
        def from_ir_pool(cls, pool, **kw):
            pools.append(list(pool))
            return _Real.from_ir_pool(pool, **kw)

    monkeypatch.setattr(ms, "DeflationBasis", _Spy)
    with tempfile.TemporaryDirectory() as td:
        ms.run_session(daily, n_trials=60, top_k=3, seed=11, method=method, out_dir=td)
    assert pools, "run_session 应构造过 DeflationBasis"
    return pools[-1]

def test_ir_pool_hygiene_dual_path_suite():
    """`_score_one` 必须与 random 路径同一道门：n_train 不足者不进 `eval_ir`。；random 路径本就有这道门。两条一起断言，才守得住「双路径登记簿」。"""
    # -- 原 test_genetic_ir_pool_excludes_expressions_below_min_n_train --
    def _section_0_test_genetic_ir_pool_excludes_expressions_below_min_n_train(mp):
        from factorzen.discovery.derived import add_derived_columns

        ir_pool = _capture_ir_pool(mp, add_derived_columns(_daily_with_thin_leaf()),
                                   method="genetic")

        assert ir_pool, "genetic 的 ir_pool 不该为空（否则测试失去判别力）"
        assert 0.0 not in ir_pool, (
            f"eval_ir 里混进了 sentinel 0.0（死表达式）：{sorted(set(ir_pool))[:5]}。"
            "genetic 的 _score_one 缺少 random 路径那道 min_n_train 门。"
        )

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_genetic_ir_pool_excludes_expressions_below_min_n_train(mp)

    # -- 原 test_random_ir_pool_also_excludes_them --
    def _section_1_test_random_ir_pool_also_excludes_them(mp):
        from factorzen.discovery.derived import add_derived_columns

        ir_pool = _capture_ir_pool(mp, add_derived_columns(_daily_with_thin_leaf()),
                                   method="random")

        assert ir_pool
        assert 0.0 not in ir_pool

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_random_ir_pool_also_excludes_them(mp)


# ==== 来自 test_w4_rank_fingerprint_eval.py ====
# tests/test_w4_rank_fingerprint_eval.py

def _mock_daily(n_stocks=40, n_days=80, seed=7):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c = f"{i:06d}.SZ"
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            amt = float(abs(rng.standard_normal()) * 1e7 + 1e6) + i * 1e3
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": amt,
            })
    return pl.DataFrame(rows)


def test_evaluate_fingerprint_suite():
    """同截面秩序：rank(amount) 与 rank(mul(amount,2)) 第二记 duplicate_fingerprint、不计 N。；seen_fingerprints=None（默认）→ 不算指纹，两等价表达式都出 IC。；调用方持有跨批 set：第二批同源表达式被去重。"""
    # -- 原 test_evaluate_fingerprint_dup_monotone_equivalent --
    def _section_0_test_evaluate_fingerprint_dup_monotone_equivalent():
        daily = _mock_daily()
        bundle = DataBundle.build(daily)
        seen: set[str] = set()
        out = evaluate_expressions(
            ["rank(amount)", "rank(mul(amount, 2.0))"],
            daily, bundle, seen_fingerprints=seen,
        )
        assert len(out) == 2
        assert out[0]["error"] is None and out[0]["ic_train"] is not None
        assert out[0]["n_train"] > 0
        assert out[1]["error"] == "duplicate_fingerprint"
        assert out[1]["ic_train"] is None
        assert out[1]["n_train"] == 0
        assert out[1]["compile_ok"] is True
        assert len(seen) == 1  # 只登记首个指纹

    _section_0_test_evaluate_fingerprint_dup_monotone_equivalent()

    # -- 原 test_evaluate_fingerprint_none_gating_zero_regression --
    def _section_1_test_evaluate_fingerprint_none_gating_zero_regression():
        daily = _mock_daily()
        bundle = DataBundle.build(daily)
        out = evaluate_expressions(
            ["rank(amount)", "rank(mul(amount, 2.0))"],
            daily, bundle,
        )
        assert all(r["error"] != "duplicate_fingerprint" for r in out)
        assert out[0]["ic_train"] is not None
        assert out[1]["ic_train"] is not None

    _section_1_test_evaluate_fingerprint_none_gating_zero_regression()

    # -- 原 test_evaluate_fingerprint_persists_across_batches --
    def _section_2_test_evaluate_fingerprint_persists_across_batches():
        daily = _mock_daily()
        bundle = DataBundle.build(daily)
        seen: set[str] = set()
        out1 = evaluate_expressions(["rank(amount)"], daily, bundle, seen_fingerprints=seen)
        assert out1[0]["error"] is None
        out2 = evaluate_expressions(
            ["rank(mul(amount, 2.0))"], daily, bundle, seen_fingerprints=seen,
        )
        assert out2[0]["error"] == "duplicate_fingerprint"
        assert out2[0]["n_train"] == 0

    _section_2_test_evaluate_fingerprint_persists_across_batches()


# ==== 来自 test_ts_eval.py ====
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
    "ts_mean", "delay", "ts_std", "ts_skew", "ts_decay_linear",
]

_TWO_TS_OPS = [
    "ts_corr", "ts_cov",
    "ts_count_gt", "ts_streak_gt", "ts_count_cross_up",
]

# 嵌套混合：含 PR#61 老巢（test_parity_on_previously_working_shapes 同型）
_NESTED_EXPRS = [
    "mul(close, vol)",                    # 纯算术
    "ts_mean(ts_std(ret_1d, 5), 5)",      # ts∘ts
    "rank(ts_std(ret_1d, 5))",            # cs(ts(x)) — 嵌套 over 老巢
    "ts_mean(rank(pb), 5)",               # ts(cs(x))
    "add(ts_mean(ret_1d, 5), rank(pb))",  # arith(ts, cs)
    "rank(add(ts_std(ret_1d, 5), ts_mean(ret_1d, 5)))",  # 深交叉
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

def test_ts_chunk_threshold_dispatch_suite():
    """小帧 / 默认阈值：_materialize_ts_chunked 调用次数 = 0。；阈值压低后 ts 节点确实走分块函数。"""
    # -- 原 test_threshold_skips_chunk_path --
    def _section_0_test_threshold_skips_chunk_path(mp):
        df = _panel(5, 30, unequal=False)  # ~150 行 << 3e6
        calls = {"n": 0}
        orig = expression_mod._materialize_ts_chunked

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        mp.setattr(expression_mod, "_materialize_ts_chunked", counting)
        # 保持默认阈值（3_000_000）
        node = parse_expr("ts_mean(ts_std(close, 5), 5)")
        out = evaluate_materialized(node, df)
        assert calls["n"] == 0
        assert out.len() == df.height

        # cs 节点也不走分块
        node_cs = parse_expr("rank(pb)")
        _ = evaluate_materialized(node_cs, df)
        assert calls["n"] == 0

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_threshold_skips_chunk_path(mp)

    # -- 原 test_chunk_path_invoked_when_over_threshold --
    def _section_1_test_chunk_path_invoked_when_over_threshold(mp):
        df = _panel(20, 60, unequal=True)
        _force_chunk(mp, threshold=100, target=80)
        calls = {"n": 0}
        orig = expression_mod._materialize_ts_chunked

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        mp.setattr(expression_mod, "_materialize_ts_chunked", counting)
        node = parse_expr("ts_mean(close, 5)")
        _ = evaluate_materialized(node, df)
        assert calls["n"] >= 1

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_chunk_path_invoked_when_over_threshold(mp)


# ── 5. 批次边界算法单测 ─────────────────────────────────────────────────────

def test_ts_stock_batches_suite():
    """贪心合批：不切开股票，累计逼近 target。；test_ts_stock_batches_single_stock_over_target"""
    # -- 原 test_ts_stock_batches_greedy_pack --
    def _section_0_test_ts_stock_batches_greedy_pack():
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

    _section_0_test_ts_stock_batches_greedy_pack()

    # -- 原 test_ts_stock_batches_single_stock_over_target --
    def _section_1_test_ts_stock_batches_single_stock_over_target():
        s = pl.Series(["X"] * 100)
        batches = expression_mod._ts_stock_batches(s, target_rows=30)
        assert batches == [(0, 100)]

    _section_1_test_ts_stock_batches_single_stock_over_target()


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

def test_datetime_key_scoring_suite():
    """test_cut_literal_dispatch；test_databundle_and_fitness_on_datetime_frame；test_factor_values_eval_start_on_datetime_frame"""
    # -- 原 test_cut_literal_dispatch --
    def _section_0_test_cut_literal_dispatch():
        intraday = _intraday_daily()
        daily = intraday.with_columns(pl.col("trade_date").cast(pl.Date))
        assert _cut_literal(intraday, "20260501") == datetime(2026, 5, 1)
        assert _cut_literal(daily, "20260501") == date(2026, 5, 1)

    _section_0_test_cut_literal_dispatch()

    # -- 原 test_databundle_and_fitness_on_datetime_frame --
    def _section_1_test_databundle_and_fitness_on_datetime_frame():
        df = _intraday_daily()
        bundle = DataBundle.build(df, train_ratio=0.7)
        factor = df.select("trade_date", "ts_code",
                           pl.col("close").alias("factor_value"))
        res = quick_fitness(factor, bundle, "train")
        assert res["n"] > 0  # 切分/过滤在 Datetime 键上正常工作

    _section_1_test_databundle_and_fitness_on_datetime_frame()

    # -- 原 test_factor_values_eval_start_on_datetime_frame --
    def _section_2_test_factor_values_eval_start_on_datetime_frame():
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.mining_session import _factor_values
        df = _intraday_daily()
        leaf_map = {"close": "close", "vol": "vol", "amount": "amount"}
        out = _factor_values(parse_expr("close", leaf_map), df, eval_start="20260502",
                             leaf_map=leaf_map)
        assert out["trade_date"].min() >= datetime(2026, 5, 2)

    _section_2_test_factor_values_eval_start_on_datetime_frame()


