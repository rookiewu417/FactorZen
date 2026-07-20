"""Merged discovery tests: test_search_genetic.py

test_discovery_search.py：随机/遗传搜索：可编译、多样性、crossover/mutate 合法与目标改进
test_discovery_genetic_parallel.py：遗传搜索并行评分：同 seed 下 workers=1 与 N 结果逐项等价
test_genetic_pool_hygiene.py：genetic eval_ir 与 random scored 共用 min_n_train 门禁
test_w4_rank_fingerprint_eval.py：W4 evaluate 指纹去重：duplicate_fingerprint 门控与跨批 set
"""

from __future__ import annotations

import datetime as dt
import tempfile
from datetime import (
    date,
    timedelta,
)

import numpy as np
import polars as pl

import factorzen.discovery.mining_session as ms
from factorzen.discovery.evaluation import evaluate_expressions
from factorzen.discovery.expression import parse_expr
from factorzen.discovery.mining_session import run_session
from factorzen.discovery.operators import BASIC_FEATURES
from factorzen.discovery.scoring import (
    DataBundle,
    score_candidate,
)


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

def test_random_expression_is_compilable():
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

def test_random_searcher_proposes_distinct():
    from factorzen.discovery.expression import to_expr_string
    from factorzen.discovery.search.random_search import RandomSearcher
    s = RandomSearcher(np.random.default_rng(0), max_depth=3)
    exprs = {to_expr_string(s.propose()) for _ in range(30)}
    assert len(exprs) > 5  # 有多样性

def test_crossover_and_mutate_stay_compilable():
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

def test_genetic_improves_toy_objective():
    """目标：偏好复杂度小的表达式 → GP 平均复杂度应下降或持平。"""
    from factorzen.discovery.expression import complexity
    from factorzen.discovery.search.genetic import GeneticSearcher
    rng = np.random.default_rng(5)
    gs = GeneticSearcher(rng, max_depth=3)
    best = gs.evolve(lambda node: -complexity(node), pop_size=20, generations=5)
    assert complexity(best[0]) <= 4

def test_genetic_terminates_under_complexity_pressure():
    """即使目标偏好高复杂度（防膨胀过滤压力最大），evolve 也必须在有限时间内终止。"""
    from factorzen.discovery.expression import complexity
    from factorzen.discovery.search.genetic import GeneticSearcher
    rng = np.random.default_rng(13)
    gs = GeneticSearcher(rng, max_depth=3)
    best = gs.evolve(lambda node: float(complexity(node)), pop_size=15, generations=6)
    assert len(best) == 15  # 种群规模维持，未因死循环卡住

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

def test_genetic_ir_pool_excludes_expressions_below_min_n_train(monkeypatch):
    """`_score_one` 必须与 random 路径同一道门：n_train 不足者不进 `eval_ir`。"""
    from factorzen.discovery.derived import add_derived_columns

    ir_pool = _capture_ir_pool(monkeypatch, add_derived_columns(_daily_with_thin_leaf()),
                               method="genetic")

    assert ir_pool, "genetic 的 ir_pool 不该为空（否则测试失去判别力）"
    assert 0.0 not in ir_pool, (
        f"eval_ir 里混进了 sentinel 0.0（死表达式）：{sorted(set(ir_pool))[:5]}。"
        "genetic 的 _score_one 缺少 random 路径那道 min_n_train 门。"
    )

def test_random_ir_pool_also_excludes_them(monkeypatch):
    """random 路径本就有这道门。两条一起断言，才守得住「双路径登记簿」。"""
    from factorzen.discovery.derived import add_derived_columns

    ir_pool = _capture_ir_pool(monkeypatch, add_derived_columns(_daily_with_thin_leaf()),
                               method="random")

    assert ir_pool
    assert 0.0 not in ir_pool

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


def test_evaluate_fingerprint_dup_monotone_equivalent():
    """同截面秩序：rank(amount) 与 rank(mul(amount,2)) 第二记 duplicate_fingerprint、不计 N。"""
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

def test_evaluate_fingerprint_none_gating_zero_regression():
    """seen_fingerprints=None（默认）→ 不算指纹，两等价表达式都出 IC。"""
    daily = _mock_daily()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(
        ["rank(amount)", "rank(mul(amount, 2.0))"],
        daily, bundle,
    )
    assert all(r["error"] != "duplicate_fingerprint" for r in out)
    assert out[0]["ic_train"] is not None
    assert out[1]["ic_train"] is not None

def test_evaluate_fingerprint_persists_across_batches():
    """调用方持有跨批 set：第二批同源表达式被去重。"""
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


