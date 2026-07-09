"""DSR 的符号轴：统计量取 |IR| 时，deflation 基准必须按 2N 算。

背景——两条挖掘路径的**选择规则**不同，所以 deflation 基准也必须不同：

  M1 (`discovery/mining_session`)
      `fitness = tstat - λ·max_corr - γ·complexity`，`scored.sort(reverse=True)`。
      tstat 带符号 ⇒ 搜索最大化 **有符号** IR。反转因子以 `neg(x)` 的正 IR 形式被搜到
      （`neg` 在算子集内，搜索空间对取负封闭）。故统计量 = `max_i IR_i` ⇒ 基准用 N。

  Agent (`agents/nodes`)
      `passed.sort(key=lambda a: abs(a.ic_train), reverse=True)`，且 `guardrail_passed`
      经 `ci_high < 0` 分支接纳负 IC 因子。统计量 = `max_i |IR_i|` ⇒ 基准必须用 2N。

为什么是 2N（对称零分布）::

    P(max_{i≤N}|Z_i| ≤ t) = [2Φ(t) − 1]^N ≈ [Φ(t)²]^N = Φ(t)^{2N} = P(max_{j≤2N} Z_j ≤ t)

即「取绝对值」渐近等价于「试验数翻倍」。本文件第一个测试用蒙特卡洛把这条断言钉成
CI 断言——它是本次修复的 ground truth，不依赖任何人的推导。

历史：Agent 曾传 `abs(ir_train)` 却用 N 做基准 ⇒ sr0 系统性少算 0.20σ–0.41σ（N 越小越糟），
门槛偏低、放行过拟合因子。方向与已修的 `sharpe_variance` 漂移（PR #62）**同向**。
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from factorzen.discovery.guardrails import DeflationBasis, deflated_pvalue
from factorzen.validation.deflated_sharpe import expected_max_sharpe

_SRC = Path(__file__).resolve().parents[1] / "src" / "factorzen"

# team_51_6r 真实 run 反解出的 basis（6/6 候选的 dsr_pvalue 逐位复算吻合）
_REAL_VAR = 0.00127844
_REAL_N = 18
_REAL_NOBS = 303


# ── ground truth：取绝对值 ≡ 试验数翻倍 ────────────────────────────────────


@pytest.mark.parametrize("n", [5, 20, 100])
def test_abs_max_null_baseline_matches_double_trials(n):
    """E[max_N |Z|] ≈ expected_max_sharpe(1, 2N)，而非 (1, N)。

    容差 0.03σ：`expected_max_sharpe` 对 E[max_N Z] 本身就有 ~0.02σ–0.04σ 的逼近误差，
    再苛刻就是在测这个公式的近似质量，而不是测 2N 这条结论。
    """
    rng = np.random.default_rng(20260709)
    z = rng.standard_normal((60_000, n))
    e_absmax = float(np.abs(z).max(axis=1).mean())

    assert e_absmax == pytest.approx(expected_max_sharpe(1.0, 2 * n), abs=0.03), (
        f"N={n}: E[max|Z|]={e_absmax:.4f} 应与 2N 基准 {expected_max_sharpe(1.0, 2 * n):.4f} 吻合"
    )


@pytest.mark.parametrize("n", [5, 20, 100])
def test_one_sided_baseline_understates_the_abs_max_statistic(n):
    """反向断言：用 N 做 |IR| 的基准会少算一大截——这正是漂移的量级。

    没有这条，上一个测试可以靠「N 与 2N 差不多」而假绿。
    """
    rng = np.random.default_rng(20260709)
    z = rng.standard_normal((60_000, n))
    e_absmax = float(np.abs(z).max(axis=1).mean())
    gap = e_absmax - expected_max_sharpe(1.0, n)

    assert gap > 0.15, f"N={n}: 单边基准少算 {gap:.4f}σ，应显著为正（实测 0.20σ–0.41σ）"


# ── DeflationBasis 的 two_sided 契约 ───────────────────────────────────────


def test_two_sided_basis_doubles_effective_trials_but_reports_honest_n():
    """`n_trials` 是诚实计数（写进 manifest 的那个），`effective_trials` 才是 deflation 用的。

    两者混用会让 manifest 声称「试了 36 个表达式」——那是谎报。
    """
    pool = [0.2, -0.1, 0.35, 0.05]
    one = DeflationBasis.from_ir_pool(pool)
    two = DeflationBasis.from_ir_pool(pool, two_sided=True)

    assert one.n_trials == two.n_trials == 4, "诚实计数不受 sidedness 影响"
    assert one.effective_trials == 4
    assert two.effective_trials == 8
    assert one.sharpe_variance == pytest.approx(two.sharpe_variance), "方差口径与 sidedness 无关"


def test_deflated_pvalue_applies_abs_only_when_two_sided():
    """调用方一律传**带符号** IR；取不取绝对值由 basis 决定。

    这样「统计量 abs 了没有」与「基准 N 还是 2N」不可能各说各话——两者由同一个字段决定。
    """
    var, n_obs = 0.01, 300
    two = DeflationBasis(n_trials=10, sharpe_variance=var, two_sided=True)
    one = DeflationBasis(n_trials=10, sharpe_variance=var)

    p_pos = deflated_pvalue(0.30, two, n_obs)[1]
    p_neg = deflated_pvalue(-0.30, two, n_obs)[1]
    assert p_neg == pytest.approx(p_pos), "双边下 ±0.30 必须同 p（反转因子只是符号相反）"

    p_neg_one = deflated_pvalue(-0.30, one, n_obs)[1]
    assert p_neg_one > 0.5, "单边下负 IR 不该显著——M1 的反转因子须以 neg(x) 形式出现"


def test_two_sided_basis_is_stricter_on_the_same_statistic():
    """同一个 |IR|，双边基准给出的 p 必须更大（门槛更高）。方向错了就是把漂移修反了。"""
    var, n_obs, ir = _REAL_VAR, _REAL_NOBS, 0.20
    p_one = deflated_pvalue(ir, DeflationBasis(_REAL_N, var), n_obs)[1]
    p_two = deflated_pvalue(ir, DeflationBasis(_REAL_N, var, two_sided=True), n_obs)[1]
    assert p_two > p_one


def test_sidedness_flips_a_real_run_candidate():
    """team_51_6r 的真实 basis 上，IR=0.172 的因子单边过关、双边被拒。

    数字来自真实 run 的 manifest 反解（复算 p 与记录 p 逐位吻合），不是合成的。
    """
    p_one = deflated_pvalue(0.172, DeflationBasis(_REAL_N, _REAL_VAR), _REAL_NOBS)[1]
    p_two = deflated_pvalue(0.172, DeflationBasis(_REAL_N, _REAL_VAR, two_sided=True),
                            _REAL_NOBS)[1]

    assert p_one < 0.05, f"单边口径下应过关（p={p_one:.4f}）"
    assert p_two >= 0.05, f"双边口径下应被拒（p={p_two:.4f}）"


# ── 架构守卫：禁止调用方自己 abs，否则 sidedness 又会与基准脱钩 ──────────────


def _abs_tainted_names(tree: ast.AST) -> set[str]:
    """收集所有「值来自 abs(...) 调用」的局部名字。

    历史 bug 的形态是 `sharpe = abs(a.ir_train) if ... else abs(...)` 再 `deflated_pvalue(sharpe, ...)`
    ——只查内联 `abs()` 的守卫会在有 bug 的代码上通过（我第一版就是这样，白写）。
    """
    def has_abs(node: ast.AST) -> bool:
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "abs"
                   for n in ast.walk(node))

    tainted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and has_abs(node.value):
            tainted.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and has_abs(node.value) and isinstance(node.target, ast.Name)):
            tainted.add(node.target.id)
    return tainted


def test_no_caller_feeds_an_abs_value_into_deflated_pvalue():
    """`abs()` 只许发生在 `deflated_pvalue` 内部。

    调用方一旦自己取绝对值，「统计量是 |IR|」这件事就对 deflation 基准不可见，
    基准 N/2N 的选择随即与统计量脱钩——这正是本次修复前的形态。
    """
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        tainted = _abs_tainted_names(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "deflated_pvalue" and node.args):
                continue
            first = node.args[0]
            inline_abs = (isinstance(first, ast.Call) and isinstance(first.func, ast.Name)
                          and first.func.id == "abs")
            via_name = isinstance(first, ast.Name) and first.id in tainted
            if inline_abs or via_name:
                offenders.append(f"{path.relative_to(_SRC)}:{node.lineno}")

    assert not offenders, (
        "以下调用点把 abs(...) 的值传给了 deflated_pvalue——统计量与 deflation 基准会脱钩。"
        f"改传带符号 IR，并在构造 DeflationBasis 时指定 two_sided=True：{offenders}"
    )


# ── 行为守卫：AST 挡不住的，由真实 node_guardrails 挡 ──────────────────────


def _seed(state, irs, n_train):
    from factorzen.agents.state import AttemptRecord
    for i, ir in enumerate(irs):
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis="h",
            expression=f"rank(neg(ts_min(low, {5 + i})))", compile_ok=True,
            ic_train=ir / 10.0, passed_guardrails=False, critic_verdict=None,
            error=None, ir_train=ir, turnover=0.3, n_train=n_train,
        ))


def test_agent_guardrails_deflate_two_sided(monkeypatch):
    """驱动真实 `node_guardrails`，其 dsr_pvalue 必须等于**双边**配方逐位算出的值。

    AST 守卫只看形状；这条看行为。有人换个写法绕过 abs 检查，这里照样红。
    """
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.multiple_testing import TrialLedger
    from tests.test_agent_dsr_parity import _mk_daily

    monkeypatch.setattr("factorzen.validation.holdout.holdout_ic",
                        lambda fdf, hdf: (0.05, 0.5, (0.01, 0.09)))
    monkeypatch.setattr("factorzen.discovery.scoring.max_correlation", lambda fdf, pool: 0.0)

    daily = _mk_daily()
    ir_pool, n_train = [0.45, 0.1048, -0.1285], 305
    state = AgentState(seed=1)
    _seed(state, ir_pool, n_train)

    node_guardrails(state, daily=daily, holdout_df=daily, bundle=DataBundle.build(daily),
                    ledger=TrialLedger(), top_k=5)
    assert state.candidates, "IR=0.45 应过关，否则本测试失去判别力"

    top = max(state.candidates, key=lambda c: abs(c["ir_train"]))
    want = deflated_pvalue(0.45, DeflationBasis.from_ir_pool(ir_pool, two_sided=True), n_train)[1]
    assert top["dsr_pvalue"] == pytest.approx(want, abs=1e-9), (
        f"Agent 的 dsr_pvalue={top['dsr_pvalue']} 不等于双边配方的 {want} —— "
        "sidedness 与 deflation 基准脱钩了"
    )
