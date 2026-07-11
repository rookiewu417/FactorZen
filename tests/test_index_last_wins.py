"""长期记忆是**事件日志**：同一表达式的当前状态 = 它最新的那条记录。

背景：Librarian 在**每轮**把 attempts 写进 index，`passed` 取当轮护栏结论。而收尾复核
（`node_finalize_guardrails`）会用最终 N 把早轮候选降级。此时 index 里已经躺着一条
`passed=True`——若 `known_valid()` 无差别扫全部记录，被降级的因子仍会以「已验证有效」
喂给后续 session。补写一条 `passed=False` 也压不住它：两条记录会同时命中。

故 `_scoped()` 必须按归一化表达式做**后写覆盖**。这同时修好一个潜在问题：
同一表达式在不同 session 被重新评估时，旧结论不该与新结论并存。
"""

from __future__ import annotations

import json

from factorzen.agents.experiment_index import ExperimentIndex


def _rec(expr: str, *, passed: bool, ic: float, holdout: float | None = None,
         verdict: str | None = None, decorrelated: bool = False) -> dict:
    return {"expression": expr, "passed": passed, "ic_train": ic, "compile_ok": True,
            "holdout_ic": holdout, "verdict": verdict, "decorrelated": decorrelated,
            "data_window": {"start": "20220101", "end": "20231229",
                            "universe": "csi800", "market": "ashare"}}


_WINDOW = {"start": "20220101", "end": "20231229", "universe": "csi800", "market": "ashare"}


def _write(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def test_later_record_overrides_earlier_for_same_expression(tmp_path):
    """收尾降级：先写 passed=True，后写 passed=False → 不得再出现在 known_valid。"""
    p = tmp_path / "idx.jsonl"
    _write(p, [
        _rec("rank(neg(pb))", passed=True, ic=0.02, holdout=0.05),   # 早轮结论
        _rec("rank(neg(pb))", passed=False, ic=0.02, holdout=0.05),  # 收尾复核后的更正
    ])
    index = ExperimentIndex(str(p))

    assert index.known_valid(data_window=_WINDOW) == [], "被降级的因子不该还算「已验证有效」"
    assert "rank(neg(pb))" in index.known_invalid(data_window=_WINDOW)


def test_override_works_in_the_other_direction_too(tmp_path):
    """反向断言：后写 passed=True 也必须覆盖先写的 False，否则「后写覆盖」是假的。"""
    p = tmp_path / "idx.jsonl"
    _write(p, [
        _rec("rank(neg(pb))", passed=False, ic=0.02, holdout=0.05),
        _rec("rank(neg(pb))", passed=True, ic=0.02, holdout=0.05),
    ])
    index = ExperimentIndex(str(p))

    assert index.known_valid(data_window=_WINDOW) == ["rank(neg(pb))"]
    assert index.known_invalid(data_window=_WINDOW) == []


def test_override_matches_on_normalized_expression(tmp_path):
    """覆盖判定按**归一化**表达式，而非裸字符串——否则空格差异就能绕过。"""
    p = tmp_path / "idx.jsonl"
    _write(p, [
        _rec("rank(neg(pb))", passed=True, ic=0.02, holdout=0.05),
        _rec("rank( neg( pb ) )", passed=False, ic=0.02, holdout=0.05),
    ])
    index = ExperimentIndex(str(p))

    assert index.known_valid(data_window=_WINDOW) == []


def test_distinct_expressions_are_not_collapsed(tmp_path):
    """判别力：别把「后写覆盖」实现成「只留最后一条记录」。"""
    p = tmp_path / "idx.jsonl"
    _write(p, [
        _rec("rank(neg(pb))", passed=True, ic=0.02, holdout=0.09),
        _rec("rank(neg(pe_ttm))", passed=True, ic=0.03, holdout=0.05),
    ])
    index = ExperimentIndex(str(p))

    assert sorted(index.known_valid(data_window=_WINDOW)) == [
        "rank(neg(pb))", "rank(neg(pe_ttm))"
    ]


def test_override_is_scoped_per_data_window(tmp_path):
    """不同数据窗口的同名表达式互不覆盖——族边界优先于时间顺序。"""
    p = tmp_path / "idx.jsonl"
    other = dict(_WINDOW, end="20241231")
    r_old = _rec("rank(neg(pb))", passed=True, ic=0.02, holdout=0.05)
    r_new = _rec("rank(neg(pb))", passed=False, ic=0.02, holdout=0.05)
    r_new["data_window"] = other
    _write(p, [r_old, r_new])
    index = ExperimentIndex(str(p))

    assert index.known_valid(data_window=_WINDOW) == ["rank(neg(pb))"], \
        "另一个窗口的 False 不该影响本窗口"
    assert index.known_valid(data_window=other) == []


def test_seen_expressions_unaffected_by_dedup(tmp_path):
    """去重不该让「见过的表达式」漏掉任何一个。"""
    p = tmp_path / "idx.jsonl"
    _write(p, [
        _rec("rank(neg(pb))", passed=True, ic=0.02),
        _rec("rank(neg(pb))", passed=False, ic=0.02),
        _rec("rank(neg(pe_ttm))", passed=False, ic=0.01),
    ])
    index = ExperimentIndex(str(p))

    assert index.seen_expressions(data_window=_WINDOW) == {
        "rank(neg(pb))", "rank(neg(pe_ttm))"
    }


# ── 接线守卫：收尾降级后，team 必须把更正写回 index ──────────────────────
#
# 变异实证：把 team_orchestrator 里补写更正的分支关掉，上面 6 个测试全绿——
# 它们只测 ExperimentIndex 的读语义，没人验证「降级后真的补写了」。


def test_team_writes_correction_to_index_after_demotion(tmp_path, monkeypatch):
    """收尾把候选降级 → index 里那条 passed=True 必须被后写的 passed=False 覆盖。

    否则被最终 N 否掉的因子仍以「已验证有效」喂给后续 session——长期记忆被污染。
    """
    import datetime as dt
    import json

    import numpy as np
    import polars as pl

    import factorzen.agents.team_orchestrator as team
    from factorzen.agents.state import AttemptRecord
    from factorzen.discovery.guardrails import DeflationBasis

    def _daily(n_stocks=40, n_days=180, seed=1):
        rng = np.random.default_rng(seed)
        days, d = [], dt.date(2022, 1, 3)
        while len(days) < n_days:
            if d.weekday() < 5:
                days.append(d)
            d += dt.timedelta(days=1)
        rows = []
        for c in [f"{i:06d}.SZ" for i in range(n_stocks)]:
            px = 10.0
            for dd in days:
                px *= 1 + rng.standard_normal() * 0.02
                rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                             "high": px * 1.01, "low": px * 0.98,
                             "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                             "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
        return pl.DataFrame(rows)

    def fake_guardrails(state, *, daily, holdout_df, bundle, ledger, top_k=5,
                        dsr_alpha=0.05, warmup_daily=None, eval_start=None):
        """每轮产一个「当轮 N 下过关」的候选（passed=True 会被 Librarian 落盘）。"""
        ledger.record(1)
        state.attempts.append(AttemptRecord(
            iteration=state.iteration, hypothesis="h", expression="rank(neg(pb))",
            compile_ok=True, ic_train=0.05, passed_guardrails=True, critic_verdict=None,
            error=None, ir_train=0.4, turnover=0.3, n_train=300))
        if not state.candidates:
            state.candidates.append({
                "expression": "rank(neg(pb))", "hypothesis": "h", "ic_train": 0.05,
                "ir_train": 0.4, "turnover": 0.3, "holdout_ic": 0.04, "holdout_ir": 0.3,
                "dsr": 0.9, "dsr_pvalue": 0.01, "n_train": 300,
                "ic_ci_low": 0.01, "ic_ci_high": 0.07})
        return state

    def fake_finalize(state, *, dsr_alpha=0.05, daily=None, bundle=None):
        """模拟「最终 N 下不再显著」：清空候选并回落事实。"""
        for a in state.attempts:
            a.passed_guardrails = False
        state.candidates = []
        return DeflationBasis(n_trials=3, sharpe_variance=0.01, two_sided=True)

    monkeypatch.setattr(team, "node_guardrails", fake_guardrails)
    monkeypatch.setattr(team, "node_finalize_guardrails", fake_finalize)

    seq = [json.dumps({"hypotheses": ["h"]}),
           json.dumps({"expressions": ["rank(neg(pb))"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    idx_path = tmp_path / "e.jsonl"
    team.run_team_agent(_daily(), fn, n_rounds=2, seed=1, index_path=str(idx_path),
                        heal_rounds=0)

    index = ExperimentIndex(str(idx_path))
    assert index.known_valid() == [], (
        "收尾已把候选降级，index 却仍把它当「已验证有效」—— 更正记录没补写，"
        "或后写覆盖没生效"
    )
    assert "rank(neg(pb))" in index.known_invalid()
