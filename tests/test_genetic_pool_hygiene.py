"""genetic 路径的 `eval_ir` 必须与 random 路径的 `scored` 过同一道 `min_n_train` 门。

`quick_fitness` 对「求值后没有任何有效截面」的表达式返回 **sentinel `ic=0.0, ir=0.0, n=0`**
（不是 nan）。random 路径在 `mining_session.py:251` 用 `if sc["n_train"] < min_n_train: continue`
把它们挡在 `scored` 之外；genetic 路径的 `_score_one` **没有这道门**，`eval_ir` 照单全收。

后果：`ir_pool = list(eval_ir.values())` 里混入一批 `0.0`——
- N 被膨胀（偏严）
- 经验方差被压低（偏松，且这一侧占优，因 `expected_max_sharpe ∝ sqrt(var)`）

净效应**偏松**，与已修的 DSR 三条缺陷同向。

实测（真实 csi300 数据，`method="genetic"`, n_trials=200）：214 次评分里 8 次 `n_train==0`
（3.7%），`ir` 全为 `0.0`；`sr0` 因此从 0.3679 掉到 0.3625（**-1.4%**）。
`workspace/mining_sessions/session_42_genetic` 的池子有 9031 个 trial —— 这条路径是 live 的。

`n_train=0` 的表达式没有可比较的 IR，永远不可能是 `max`，故不该计入多重检验的 N
（Bailey & López de Prado 的 N = 「最大值是从多少个统计量里选出来的」）。
random 路径与 Agent 路径都已如此，genetic 是最后一处。
"""

from __future__ import annotations

import datetime as dt
import tempfile

import numpy as np
import polars as pl

import factorzen.discovery.mining_session as ms
from factorzen.discovery.expression import parse_expr
from factorzen.discovery.operators import BASIC_FEATURES
from factorzen.discovery.scoring import DataBundle, score_candidate

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
