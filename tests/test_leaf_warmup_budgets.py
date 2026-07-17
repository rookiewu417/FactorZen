"""任务 B：叶子预热预算注入生成侧 + 预热错误回灌 revise。

核心契约（B4.1）：`leaf_warmup_budgets` 报的 have 数，必须与预热门 `warmup_shortfall`
内部 `warmup_bars_by_leaf` 给出的 have 数**逐值相等**——prompt 里承诺的预算与预热判定
不一致就是继续骗 LLM。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.discovery.expression import (
    leaf_warmup_budgets,
    parse_expr,
    warmup_shortfall,
)


def _workdays(anchor: dt.date, count: int, *, forward: bool) -> list[dt.date]:
    out: list[dt.date] = []
    d = anchor
    step = dt.timedelta(days=1 if forward else -1)
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += step
    return out


def _frame_with_north_ratio(n_warm: int = 100, n_after: int = 40,
                            eval_start: dt.date = dt.date(2022, 1, 3)):
    """两只股票的帧：north_ratio 在 eval_start 前恰好 n_warm 个非空交易日。"""
    before = sorted(_workdays(eval_start - dt.timedelta(days=1), n_warm, forward=False))
    after = _workdays(eval_start, n_after, forward=True)
    days = before + after
    rows = []
    for c in ("000001.SZ", "000002.SZ"):
        for i, d in enumerate(days):
            rows.append({"trade_date": d, "ts_code": c,
                         "close": 10.0 + i * 0.1, "north_ratio": float(i)})
    return pl.DataFrame(rows), eval_start


# ── B4.1 一致性（关键）─────────────────────────────────────────────────────────
def test_budget_equals_warmup_shortfall_have():
    prepped, es = _frame_with_north_ratio(n_warm=100)
    budgets = leaf_warmup_budgets(prepped, es, ["north_ratio"])
    sf = warmup_shortfall(parse_expr("ts_mean(north_ratio, 200)"), prepped, es)
    assert sf is not None, "need=200 > have=100，应报预热不足"
    leaf, need, have = sf
    assert leaf == "north_ratio" and need == 200
    assert budgets["north_ratio"] == have == 100


def test_budget_missing_column_is_zero():
    prepped, es = _frame_with_north_ratio(n_warm=30)
    # or_yoy 列不在帧里 → 预算 0（与 warmup_bars_by_leaf 列缺失记 0 一致）
    budgets = leaf_warmup_budgets(prepped, es, ["north_ratio", "or_yoy"])
    assert budgets["or_yoy"] == 0
    assert budgets["north_ratio"] == 30


def test_budget_matches_warmup_shortfall_across_random_windows():
    """随机窗口下逐一对照：budget[leaf] 恒等于 shortfall 的 have（或充分时的 have）。"""
    prepped, es = _frame_with_north_ratio(n_warm=120)
    have_budget = leaf_warmup_budgets(prepped, es, ["north_ratio"])["north_ratio"]
    for w in (10, 50, 119, 120, 121, 300):
        sf = warmup_shortfall(parse_expr(f"ts_mean(north_ratio, {w})"), prepped, es)
        if w > have_budget:
            assert sf is not None and sf[2] == have_budget
        else:
            assert sf is None  # have 足够，无缺口


# ── B4.2 build_agent_messages（单 agent 路径）─────────────────────────────────
def test_build_agent_messages_lists_budget():
    from factorzen.llm.generation import build_agent_messages
    msgs = build_agent_messages(["ts_mean", "rank"], ["close", "north_ratio"],
                                leaf_budgets={"north_ratio": 238})
    sysmsg = msgs[0]["content"]
    assert "north_ratio" in sysmsg and "238" in sysmsg and "历史较短" in sysmsg


def test_build_agent_messages_none_is_byte_identical():
    from factorzen.llm.generation import build_agent_messages
    base = build_agent_messages(["ts_mean", "rank"], ["close", "north_ratio"])
    with_none = build_agent_messages(["ts_mean", "rank"], ["close", "north_ratio"],
                                     leaf_budgets=None)
    assert base == with_none
    assert "历史较短" not in base[0]["content"]  # 零回归：无预算时不加预算文案


def test_build_agent_messages_empty_budget_no_text():
    """budgets 为空 dict（无短历史叶子）→ 不加文案，避免 prompt 膨胀。"""
    from factorzen.llm.generation import build_agent_messages
    base = build_agent_messages(["ts_mean"], ["close"])
    empty = build_agent_messages(["ts_mean"], ["close"], leaf_budgets={})
    assert base == empty


# ── B4.3 coder._syntax_prompt（team 路径）+ 双路径 parity ──────────────────────
def test_syntax_prompt_lists_budget():
    from factorzen.agents.roles.coder import _syntax_prompt
    p = _syntax_prompt(leaf_budgets={"north_ratio": 238})
    assert "north_ratio" in p and "238" in p and "历史较短" in p


def test_syntax_prompt_none_is_byte_identical():
    from factorzen.agents.roles.coder import _syntax_prompt
    assert _syntax_prompt() == _syntax_prompt(leaf_budgets=None)
    assert "历史较短" not in _syntax_prompt()


def test_budget_hint_shared_between_two_paths():
    """双路径登记簿：两侧预算文案共用同一 fragment，杜绝漂移。"""
    from factorzen.agents.roles.coder import _syntax_prompt
    from factorzen.llm.generation import build_agent_messages, format_leaf_budget_hint
    b = {"north_ratio": 238, "or_yoy": 424}
    hint = format_leaf_budget_hint(b)
    assert hint  # 非空
    assert hint in build_agent_messages(["rank"], ["close"], leaf_budgets=b)[0]["content"]
    assert hint in _syntax_prompt(leaf_budgets=b)


# ── B4.4 预热错误回灌（只回灌一轮）────────────────────────────────────────────
def _mock_daily_with_north_ratio(n_days=250, n_stocks=20, eval_start_idx=60, seed=3):
    """含预热前缀的帧：north_ratio 全程有值但预热段仅 eval_start_idx 天。"""
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    eval_start = days[eval_start_idx]
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for i, dd in enumerate(days):
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                         "north_ratio": float(abs(rng.standard_normal()) + i * 0.01)})
    return pl.DataFrame(rows), eval_start.strftime("%Y%m%d")


def test_warmup_error_refeed_one_round(tmp_path: Path):
    """预热不足的表达式经 revise_from_error 回灌 → 修正版被评估；且**只回灌一轮**：
    修正版仍预热不足时不再二次回灌。

    W5b 后单个超预算窗口字面量会被 clamp_window_literals 钳掉、不再触发预热错误——
    refeed 的存量场景是**嵌套窗口叠加**超预算（单个字面量都 ≤ 预算,
    required_lookback 沿最深路径累加后仍超),钳制治不了,故用嵌套表达式构造。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    daily, eval_start = _mock_daily_with_north_ratio()
    calls: list[str] = []

    hyp = json.dumps({"hypotheses": ["北向持股占比高的股票未来收益更高"]})
    # 预热 ~60 根：50+50=100 > 60 必预热不足；单字面量 50 ≤ 预算不被钳
    bad = json.dumps({"expressions": ["ts_mean(ts_mean(north_ratio, 50), 50)"]})
    # 修正版 45+45=90 > 60 仍预热不足
    revised = json.dumps({"expressions": ["ts_mean(ts_mean(north_ratio, 45), 45)"]})
    keep = json.dumps({"verdict": "keep", "reason": "ok"})

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        calls.append(text)
        if "诊断信息" in text:            # revise_from_error（回灌）
            return revised
        if "翻译成" in text:              # write_expressions
            return bad
        if "风控审计员" in text:          # critic
            return keep
        return hyp                        # propose_hypotheses

    res = run_team_agent(daily, fn, n_rounds=1, seed=7, heal_rounds=0,
                         index_path=str(tmp_path / "e.jsonl"), eval_start=eval_start)

    refeed_calls = [t for t in calls if "诊断信息" in t]
    assert len(refeed_calls) == 1, f"应恰好回灌一轮，实得 {len(refeed_calls)} 次"
    exprs_seen = {a.expression for a in res.state.attempts}
    assert "ts_mean(ts_mean(north_ratio, 50), 50)" in exprs_seen, \
        "原始预热不足表达式应被评估并落 attempt"
    assert "ts_mean(ts_mean(north_ratio, 45), 45)" in exprs_seen, "修正版应被评估（回灌一轮）"
    # 只回灌一轮：修正版仍预热不足，但不再触发第二次 revise_from_error（上面已断言 ==1）


def test_oversized_window_literal_clamped_no_refeed(tmp_path: Path):
    """W5b：单个超预算窗口字面量被本地钳制 → 评估正常、不触发 refeed LLM。"""
    from factorzen.agents.team_orchestrator import run_team_agent

    daily, eval_start = _mock_daily_with_north_ratio()
    calls: list[str] = []

    hyp = json.dumps({"hypotheses": ["北向"]})
    bad = json.dumps({"expressions": ["ts_mean(north_ratio, 999)"]})  # 超预算 → 被钳
    keep = json.dumps({"verdict": "keep", "reason": "ok"})

    def fn(messages):
        text = "\n".join(m["content"] for m in messages)
        calls.append(text)
        if "诊断信息" in text:
            return json.dumps({"expressions": []})
        if "翻译成" in text:
            return bad
        if "风控审计员" in text:
            return keep
        return hyp

    res = run_team_agent(daily, fn, n_rounds=1, seed=7, heal_rounds=0,
                         index_path=str(tmp_path / "e.jsonl"), eval_start=eval_start)

    assert not [t for t in calls if "诊断信息" in t], "钳制后不该再触发 refeed"
    assert res.rounds_log[0].get("n_window_clamped", 0) >= 1, "rounds_log 应记钳制次数"
    exprs_seen = {a.expression for a in res.state.attempts}
    assert any(e.startswith("ts_mean(north_ratio, ") and "999" not in e for e in exprs_seen), \
        f"应评估钳后表达式(窗口<999): {exprs_seen}"
