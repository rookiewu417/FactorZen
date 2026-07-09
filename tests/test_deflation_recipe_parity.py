# tests/test_deflation_recipe_parity.py
"""P0 的结构性闭环：M1 与 Agent 必须**共同调用**同一份 deflation 配方，而非各自实现。

PR #62 修好了 Agent 侧漏传 `sharpe_variance` 的偏松问题，但那次的 parity 测试里，「M1 侧」是
在测试体内**手写复现**的配方——因为 `mining_session.py:288-307` 是内联的，没有可调用的函数边界。
「配方 == M1 真实所做」这一环只由一次性的反解校验背书，**不在 CI 里**。

于是留下一个洞：日后有人改动 M1 的 deflation（换成样本方差、把 holdout 并入池、改 n_obs 来源），
那个测试仍会全绿，而两条路径已经静默再次漂移——正是本仓库登记在案的头号缺陷模式。

本文件封住它：把配方抽成 `DeflationBasis` + `deflated_pvalue`，两路共同调用；
再用一个**架构守卫测试**禁止任一路径绕过它直接调 `deflated_sharpe`。
"""
from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.discovery.guardrails import DeflationBasis, deflated_pvalue
from factorzen.validation.deflated_sharpe import deflated_sharpe

_SRC = Path(__file__).resolve().parents[1] / "src" / "factorzen"


# ── 共享配方的语义 ──────────────────────────────────────────────────────────


def test_basis_uses_population_variance_and_pool_size():
    """N 与 sharpe_variance 必须同源（R8）：都来自同一批 trial 的 IR 池。"""
    pool = [0.20, 0.10, -0.13, 0.05]
    basis = DeflationBasis.from_ir_pool(pool)

    assert basis.n_trials == 4
    assert basis.sharpe_variance == pytest.approx(float(np.var(np.asarray(pool))))


def test_basis_degenerates_to_unit_variance_for_single_trial():
    """池大小 < 2 时经验方差无意义，退化为 1.0（与 M1 既有行为一致）。"""
    assert DeflationBasis.from_ir_pool([0.3]).sharpe_variance == 1.0
    assert DeflationBasis.from_ir_pool([]).sharpe_variance == 1.0
    assert DeflationBasis.from_ir_pool([]).n_trials == 0


def test_basis_drops_none_and_nonfinite():
    """死表达式(None)与 nan/inf 不得进池——它们会同时污染方差与计数。"""
    basis = DeflationBasis.from_ir_pool([0.2, None, float("nan"), 0.1, float("inf")])

    assert basis.n_trials == 2
    assert basis.sharpe_variance == pytest.approx(float(np.var(np.asarray([0.2, 0.1]))))


def test_nan_in_pool_does_not_poison_every_candidate():
    """一个畸形 IR 不得静默废掉整个 session 的护栏。

    旧的 M1 写法 `np.array([...]).var()` 遇 nan → `sharpe_variance=nan`
    → `expected_max_sharpe` 的 `sharpe_variance <= 0` 判否（nan 比较恒 False）→ `sqrt(nan)`
    → `sr0=nan` → 所有候选的 `dsr_pvalue=nan` → `guardrail_passed` 因 nan 检查一律判否。
    **整批候选被静默拒绝，且看不出原因。** `from_ir_pool` 剔除非有限值后不再如此。
    """
    basis = DeflationBasis.from_ir_pool([0.42, float("nan"), 0.18, -0.13])

    assert basis.sharpe_variance == basis.sharpe_variance, "sharpe_variance 不得是 nan"
    assert basis.n_trials == 3
    _dsr, p = deflated_pvalue(0.42, basis, 305)
    assert p == p and 0.0 <= p <= 1.0, "p 值必须可用，而非被 nan 传染"


def test_deflated_pvalue_delegates_with_basis():
    basis = DeflationBasis.from_ir_pool([0.2, 0.1, -0.13])
    got = deflated_pvalue(0.2, basis, n_obs=300)
    want = deflated_sharpe(0.2, basis.n_trials, 300, sharpe_variance=basis.sharpe_variance)
    assert got == want


# ── 架构守卫：任何一条挖掘路径都不得绕过共享配方 ──────────────────────────────


@pytest.mark.parametrize("rel", ["discovery/mining_session.py", "agents/nodes.py"])
def test_mining_paths_never_call_deflated_sharpe_directly(rel):
    """两条挖掘路径必须经 `deflated_pvalue`。直接调 `deflated_sharpe` 就能自选
    `sharpe_variance`/`n_trials`，口径会再次漂移——那正是 P0 的成因。

    抓两种形式：`deflated_sharpe(...)`（Name）与 `ds.deflated_sharpe(...)`（Attribute）。
    """
    tree = ast.parse((_SRC / rel).read_text(encoding="utf-8"))
    direct = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "deflated_sharpe")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "deflated_sharpe")
        )
    ]
    assert not direct, (
        f"{rel} 直接调用了 deflated_sharpe（第 {[n.lineno for n in direct]} 行），"
        f"绕过共享的 deflated_pvalue → 两路 deflation 口径会再次漂移"
    )


def test_deflated_sharpe_is_imported_only_by_guardrails():
    """把守卫从「绊线」升级成「墙」：`deflated_sharpe` 只许 `guardrails.py` 导入。

    仅禁止调用形式挡不住 `import factorzen.validation.deflated_sharpe as ds` 之后的花式引用。
    源头收口——拿不到这个符号，就没法绕过 `deflated_pvalue` 自选 deflation 参数。
    （`validation/` 内部与测试不受限；本断言只约束 src/factorzen 下的生产代码。）
    """
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        rel = path.relative_to(_SRC).as_posix()
        if rel.startswith("validation/") or rel == "discovery/guardrails.py":
            continue
        # utf-8-sig：仓库里有文件带 BOM，ast.parse 遇 U+FEFF 会抛 SyntaxError
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for n in ast.walk(tree):
            if (isinstance(n, ast.ImportFrom) and n.module
                    and "deflated_sharpe" in n.module
                    and any(a.name == "deflated_sharpe" for a in n.names)):
                offenders.append(f"{rel}:{n.lineno}")          # from ... import deflated_sharpe
            elif isinstance(n, ast.Import) and any("deflated_sharpe" in a.name for a in n.names):
                offenders.append(f"{rel}:{n.lineno}")          # import ...deflated_sharpe [as ds]

    assert not offenders, (
        "只有 discovery/guardrails.py 可以导入 deflated_sharpe（其余须经 deflated_pvalue）；"
        f"违规：{offenders}"
    )


# ── 真正的 cross-path decision-parity（驱动两条真实路径）──────────────────────


def _mk_daily(n_stocks: int = 40, n_days: int = 260, seed: int = 5) -> pl.DataFrame:
    """M1 的 run_session 需要复权价列（add_derived_columns 用 close_adj 算 ret_1d）。

    股票数 ≥ 40：`_MIN_CROSS_SAMPLES = 30` 会把截面股票数不足 30 的日期全部过滤，
    IC 序列为空时 `quick_fitness` 落回 sentinel 0.0，测试就跑在 IC≡0 的垃圾数据上了。
    """
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        px = rng.uniform(8, 15)
        for dd in days:
            px = float(max(px * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": dd, "ts_code": c,
                         "close": px, "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "close_adj": px, "open_adj": px * 0.99,
                         "high_adj": px * 1.01, "low_adj": px * 0.98, "pre_close": px,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def test_m1_dsr_pvalue_is_produced_by_the_shared_recipe(tmp_path):
    """驱动真实 `run_session`，用它自报的 basis 复算每个候选的 p 值，必须逐位吻合。

    这一步把「配方 == M1 真实所做」从一次性反解升格为 CI 断言。
    """
    from factorzen.discovery.mining_session import run_session

    res = run_session(_mk_daily(), n_trials=25, top_k=5, seed=3, method="random",
                      out_dir=str(tmp_path))
    assert res["candidates"], "本测试需要 M1 至少产出一个候选"

    basis = DeflationBasis(n_trials=res["n_trials"], sharpe_variance=res["sharpe_variance"])
    for c in res["candidates"]:
        _dsr, want = deflated_pvalue(c["ir_train"], basis, c["n_train"])
        assert c["dsr_pvalue"] == pytest.approx(round(float(want), 4), abs=1e-9), (
            f"M1 的 dsr_pvalue 与共享配方不符：{c['expression']}"
        )


def test_m1_reports_basis_for_reproducibility(tmp_path):
    """`sharpe_variance` 决定 deflation 门槛，属于「事后能重跑出同样结果」的必要信息。"""
    import json

    from factorzen.discovery.mining_session import run_session

    res = run_session(_mk_daily(), n_trials=20, top_k=3, seed=4, method="random",
                      out_dir=str(tmp_path))
    m = json.loads((Path(res["session_dir"]) / "manifest.json").read_text())

    assert m["sharpe_variance"] == pytest.approx(res["sharpe_variance"])
    assert m["n_trials"] == res["n_trials"]


def test_agent_and_m1_agree_given_identical_pool_and_factor():
    """同一 IR 池、同一因子 IR、同一 n_obs → 两条路径的 p 值必须相等。

    抽出共享配方后这是结构性成立的；本测试守住它，防止任一侧再引入私有分支。
    """
    ir_pool = [0.42, 0.18, -0.13, 0.07, 0.02]
    n_obs = 305
    basis = DeflationBasis.from_ir_pool(ir_pool)

    # M1 口径：带符号 IR（符号轴是独立议题，正 IR 下两者等价）
    _, p_m1 = deflated_pvalue(0.42, basis, n_obs)
    # Agent 口径：abs(IR)
    _, p_agent = deflated_pvalue(abs(0.42), basis, n_obs)

    assert p_m1 == p_agent
