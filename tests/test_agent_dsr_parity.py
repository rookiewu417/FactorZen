# tests/test_agent_dsr_parity.py
"""P0: DSR 双路径 decision-parity —— Agent 护栏必须与 M1 用同一 deflation 口径。

背景：M1(`mining_session.py`) 给 `deflated_sharpe` 传 `sharpe_variance=ir_pool.var()`
（trial 池 IR 的经验方差，与 N 同源）；Agent(`agents/nodes.py`) 从前不传，静默回落到
`deflated_sharpe` 的 H0 默认 `1/n_obs`。因 `expected_max_sharpe ∝ sqrt(sharpe_variance)`，
而多样化 trial 池的经验方差恒大于 `1/n_obs`，Agent 的 deflation 基准系统性偏小 → 放行
M1 会拒绝的因子。真实 run `agent_43_3r` 的 2 个 passed 候选按 M1 口径均不合格（2/2 翻转）。

本文件的 parity 断言只覆盖**正 IR** 因子——这正是 P0 被验证的区间，且正 IR 时
`abs(ir) == ir`，与「M1 传带符号 IR / Agent 传 abs(ir)」这条独立的符号分歧正交。
符号轴是另一个议题（M1 的带符号 IR 使 `guardrail_passed` 的反向因子分支成为死代码），
不在本文件范围。

⚠ 本文件的覆盖边界（勿高估）：下面的「M1 侧」是在测试体内**手写复现**的 deflation 配方，
不是去调用 `mining_session.py` 的真实代码——因为那段逻辑（:292-307）是内联的，没有可调用的
函数边界。「配方 == M1 真实所做」这一环，目前只由一次性的反解校验背书
（用 `session_42_random` 的真实 `dsr_pvalue` 反解参数，6/6 逐位吻合），**该校验不在 CI 里**。

因此：若日后有人改动 M1 的 deflation（换成样本方差、把 holdout 并入池、改 n_obs 来源），
**本文件仍会全绿，而两条路径已静默再次漂移**——正是本仓库登记在案的头号缺陷模式。
真正的结构性 parity 需要把 deflation 配方抽成 M1 与 Agent **共同调用**的函数，
届时本文件的 recipe 镜像应当替换为对该共享函数的断言。见 follow-up。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from factorzen.agents.evaluation import evaluate_expressions
from factorzen.agents.nodes import node_guardrails
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.discovery.scoring import DataBundle
from factorzen.validation.deflated_sharpe import deflated_sharpe
from factorzen.validation.multiple_testing import TrialLedger


def _mk_daily(n_days: int = 300, n_stocks: int = 30, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px,
                         "high": px * 1.01, "low": px * 0.99,
                         "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7})
    return pl.DataFrame(rows)


# ── F14（P0 的前置条件）：死表达式不得冒充有效 trial ──────────────────────────


def test_dead_expression_yields_none_ic_not_zero():
    """求值后无任何有效截面的表达式必须记 ic=None，而非 0.0。

    `quick_fitness` 空段返回 `ic_mean=0.0`（sentinel）。若照单全收，死表达式会以
    「IC 恰好为 0」的身份混进 `passed` → 膨胀 N，且把 0.0 灌进 IR 池拉低经验方差，
    使 P0 的 deflation 基准算在垃圾上。真实 run `agent_42_2r` 5/5 皆为此形态。
    """
    daily = _mk_daily()
    bundle = DataBundle.build(daily)
    # 分母恒零 → 全 inf/nan → 过滤后空帧
    res = evaluate_expressions(["div(close, sub(close, close))"], daily, bundle)[0]

    assert res["compile_ok"] is True, "语法合法，compile 应成功"
    assert res["ic_train"] is None, "死表达式的 ic_train 必须是 None，不是 0.0"
    assert res["ir_train"] is None
    assert res["n_train"] == 0, "有效 IC 天数为 0"


def test_live_expression_reports_n_train():
    """正常表达式必须回报 train 段有效 IC 天数（供 DSR 的 n_obs 用，对齐 M1 的 c['n_train']）。"""
    daily = _mk_daily()
    bundle = DataBundle.build(daily)
    res = evaluate_expressions(["rank(close)"], daily, bundle)[0]

    assert res["ic_train"] is not None
    assert res["n_train"] > 0
    # n_train 是「因子有效 IC 天数」，不是 train 段日历交易日数
    train_days = bundle._segment_mask(daily, "train")["trade_date"].n_unique()
    assert res["n_train"] <= train_days


# ── P0 主体：decision-parity ────────────────────────────────────────────────


def _seed_attempts(state: AgentState, irs: list[float], n_train: int) -> None:
    """按给定 signed IR 池填充本轮 attempts（ic 与 ir 同号，holdout 由 monkeypatch 控制）。"""
    for i, ir in enumerate(irs):
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis="h", expression=f"rank(neg(ts_min(low, {5 + i})))",
            compile_ok=True, ic_train=ir / 10.0, passed_guardrails=False,
            critic_verdict=None, error=None, ir_train=ir, turnover=0.3, n_train=n_train,
        ))


@pytest.fixture
def _stub_holdout(monkeypatch):
    """把 holdout 关卡固定为「同号且 CI 方向正确」，隔离出 DSR 这一关。

    只 stub holdout（外部关卡），DSR 走真实 `deflated_sharpe`——被测的就是它的入参。
    """
    monkeypatch.setattr("factorzen.validation.holdout.holdout_ic",
                        lambda fdf, hdf: (0.05, 0.5, (0.01, 0.09)))
    monkeypatch.setattr("factorzen.discovery.scoring.max_correlation", lambda fdf, pool: 0.0)


def test_node_guardrails_dsr_pvalue_matches_m1_recipe(_stub_holdout):
    """核心 decision-parity：Agent 算出的 dsr_pvalue 必须等于 M1 配方逐位算出的值。

    M1 配方（mining_session.py:292-307）：
        ir_pool   = 全体评估过的唯一表达式的 signed train IR
        N         = len(ir_pool)
        sharpe_var= ir_pool.var()
        pval      = deflated_sharpe(sharpe, N, n_train, sharpe_variance=sharpe_var)[1]
    """
    daily = _mk_daily()
    mining_df, holdout_df = daily, daily
    bundle = DataBundle.build(mining_df)

    ir_pool = [0.45, 0.1048, -0.1285]     # 正 IR 被测因子 + 散布，signed
    n_train = 305
    state = AgentState(seed=1)
    _seed_attempts(state, ir_pool, n_train)

    ledger = TrialLedger()
    node_guardrails(state, daily=mining_df, holdout_df=holdout_df, bundle=bundle,
                    ledger=ledger, top_k=5)

    assert ledger.n_trials == len(ir_pool), "N 必须等于 IR 池大小（与 sharpe_variance 同源）"
    assert state.candidates, "被测因子 IR=0.45 应过护栏（否则测试失去判别力）"

    top = max(state.candidates, key=lambda c: abs(c["ir_train"]))
    expected_var = float(np.var(np.array(ir_pool)))
    _, expected_p = deflated_sharpe(abs(0.45), len(ir_pool), n_train,
                                    sharpe_variance=expected_var)

    assert top["dsr_pvalue"] == pytest.approx(expected_p, abs=1e-9), (
        f"Agent 的 dsr_pvalue={top['dsr_pvalue']} 与 M1 配方 {expected_p} 不符 —— "
        f"sharpe_variance 漂移（Agent 若回落 1/n_obs 会得到显著更小的 p）"
    )


def test_agent_rejects_factor_that_m1_rejects_regression(_stub_holdout):
    """回归：真实 run `agent_43_3r` 里 Agent 放行、M1 拒绝的那个因子，修复后必须被拒。

    实测参数：IR=0.1698，轮内累积 N=3，n_train(n_obs)=305，IR 池 signed。
    默认 1/n_obs 口径 → p=0.0181（放行）；池经验方差口径 → p>0.05（拒绝）。
    """
    daily = _mk_daily()
    bundle = DataBundle.build(daily)

    ir_pool = [0.1698, 0.1048, -0.1285]
    n_train = 305
    state = AgentState(seed=1)
    _seed_attempts(state, ir_pool, n_train)

    # 前置：确认这组数字真的构成「默认口径放行 / 经验方差口径拒绝」的分歧，否则测试无判别力
    _, p_default = deflated_sharpe(0.1698, 3, n_train)
    _, p_empirical = deflated_sharpe(0.1698, 3, n_train,
                                     sharpe_variance=float(np.var(np.array(ir_pool))))
    assert p_default < 0.05 < p_empirical, "测试数据须落在两口径的分歧区"

    node_guardrails(state, daily=daily, holdout_df=daily, bundle=bundle,
                    ledger=TrialLedger(), top_k=5)

    top_attempt = max(state.attempts, key=lambda a: abs(a.ir_train or 0.0))
    assert top_attempt.passed_guardrails is False, (
        "IR=0.1698 在池经验方差口径下 DSR 不显著，必须被护栏拒绝"
    )
    assert not state.candidates


def test_node_guardrails_uses_factor_n_train_not_calendar_days(_stub_holdout):
    """n_obs 口径对齐：DSR 用因子自己的有效 IC 天数，不是 train 段日历交易日数。

    M1 注释（mining_session.py:304-306）明说不能用全段交易日数——后者更大，
    会系统性放大显著性（危险方向）。Agent 从前传的正是日历交易日数。

    构造一个只在「误用日历日数」时才会被放行的因子：n_train=40 时 DSR 不显著，
    但若拿 train 段的 ~210 个日历日去算，z 会被 sqrt(n_obs-1) 放大到显著。
    """
    daily = _mk_daily()
    bundle = DataBundle.build(daily)
    calendar_days = bundle._segment_mask(daily, "train")["trade_date"].n_unique()

    n_train = 40                       # 该因子真实有效 IC 天数，远小于日历日数
    assert n_train < calendar_days
    ir_pool = [0.45, 0.1048, -0.1285]
    sharpe_var = float(np.var(np.array(ir_pool)))

    # 判别性前置：两种 n_obs 口径必须给出相反判定，否则本测试无法区分实现
    _, p_correct = deflated_sharpe(0.45, 3, n_train, sharpe_variance=sharpe_var)
    _, p_calendar = deflated_sharpe(0.45, 3, calendar_days, sharpe_variance=sharpe_var)
    assert p_calendar < 0.05 < p_correct, (
        "测试数据须使「误用日历日数→放行 / 正确用 n_train→拒绝」成立"
    )

    state = AgentState(seed=1)
    _seed_attempts(state, ir_pool, n_train)
    node_guardrails(state, daily=daily, holdout_df=daily, bundle=bundle,
                    ledger=TrialLedger(), top_k=5)

    top_attempt = max(state.attempts, key=lambda a: abs(a.ir_train or 0.0))
    assert top_attempt.passed_guardrails is False, (
        f"n_train={n_train} 下 DSR 不显著(p={p_correct:.4f})，必须拒绝；"
        f"若放行说明仍在用日历日数(p={p_calendar:.4f})"
    )
    assert not state.candidates
