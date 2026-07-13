"""P3：叶子级负面/正面经验回灌（leaf_stats + Librarian leaf_guidance + prompt 注入）。"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles import hypothesis as hyp_mod
from factorzen.agents.roles.librarian import (
    EXHAUSTED_MIN_TRIES,
    UNEXPLORED_MAX_TRIES,
    format_leaf_guidance,
    recall,
)
from factorzen.discovery.guardrails import REJECT_CATEGORY_HOLDOUT_COVERAGE
from factorzen.llm import generation as gen_mod


def _rec(
    expr: str,
    *,
    passed: bool = False,
    ic: float | None = 0.01,
    compile_ok: bool = True,
    reject_category: str | None = None,
    reject_reason: str | None = None,
    data_window: dict | None = None,
) -> dict:
    r: dict = {
        "expression": expr,
        "passed": passed,
        "ic_train": ic,
        "compile_ok": compile_ok,
    }
    if reject_category is not None:
        r["reject_category"] = reject_category
    if reject_reason is not None:
        r["reject_reason"] = reject_reason
    if data_window is not None:
        r["data_window"] = data_window
    return r


def _write(idx: ExperimentIndex, recs: list[dict]) -> None:
    idx.append(recs)


# ── A. leaf_stats 计数口径 ────────────────────────────────────────────────────


def test_leaf_stats_counts_and_word_boundary(tmp_path: Path):
    """词边界匹配 + passed/coverage/编译失败/|IC| 逐项计数。

    真实陷阱对：
    - ``roa`` vs ``roe``：互不命中
    - ``ret_1d`` 在 ``ts_mean(ret_1d,5)`` 中命中
    - ``pe_ttm`` 不被 ``ps_ttm`` 命中
    """
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    _write(idx, [
        _rec("ts_mean(roe, 5)", passed=True, ic=0.04),
        _rec("rank(roa)", passed=False, ic=-0.02),
        _rec("ts_mean(ret_1d,5)", passed=False, ic=0.01),
        _rec("rank(ps_ttm)", passed=False, ic=0.005),
        _rec("rank(pe_ttm)", passed=True, ic=0.03),
        # 编译失败：不计入 n_exprs
        _rec("broken(roe", compile_ok=False, ic=None),
        # coverage 失败：计入 n_exprs + n_coverage_fail，不算方向失败以外的特殊通道
        _rec(
            "ts_mean(north_ratio,20)",
            passed=False,
            ic=None,
            reject_category=REJECT_CATEGORY_HOLDOUT_COVERAGE,
            reject_reason="holdout覆盖不足(days=5/291)",
        ),
    ])
    stats = idx.leaf_stats(
        ["roe", "roa", "ret_1d", "pe_ttm", "ps_ttm", "north_ratio", "unused_leaf"]
    )

    assert stats["roe"]["n_exprs"] == 1
    assert stats["roe"]["n_passed"] == 1
    assert stats["roe"]["best_abs_ic"] == 0.04
    assert stats["roe"]["n_coverage_fail"] == 0

    assert stats["roa"]["n_exprs"] == 1
    assert stats["roa"]["n_passed"] == 0
    assert stats["roa"]["best_abs_ic"] == 0.02

    assert stats["ret_1d"]["n_exprs"] == 1
    assert stats["pe_ttm"]["n_exprs"] == 1 and stats["pe_ttm"]["n_passed"] == 1
    assert stats["ps_ttm"]["n_exprs"] == 1
    # pe 不命中 ps 表达式
    assert stats["pe_ttm"]["n_exprs"] == 1

    assert stats["north_ratio"]["n_exprs"] == 1
    assert stats["north_ratio"]["n_coverage_fail"] == 1
    assert stats["north_ratio"]["n_passed"] == 0
    assert stats["north_ratio"]["best_abs_ic"] == 0.0  # ic=None → 0

    assert stats["unused_leaf"] == {
        "n_exprs": 0, "n_passed": 0, "best_abs_ic": 0.0, "n_coverage_fail": 0,
    }


def test_leaf_stats_unique_expr_last_wins(tmp_path: Path):
    """同表达式后写覆盖：只计最新状态。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    _write(idx, [
        _rec("rank(roe)", passed=True, ic=0.05),
        _rec("rank(roe)", passed=False, ic=0.01),  # last wins
    ])
    s = idx.leaf_stats(["roe"])["roe"]
    assert s["n_exprs"] == 1
    assert s["n_passed"] == 0
    assert s["best_abs_ic"] == 0.01


def test_leaf_stats_scoped_by_data_window(tmp_path: Path):
    w1 = {"start": "20220101", "end": "20231231", "universe": "csi300", "market": "ashare"}
    w2 = {"start": "20150101", "end": "20211231", "universe": "csi800", "market": "ashare"}
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    _write(idx, [
        _rec("rank(roe)", passed=False, ic=0.01, data_window=w1),
        _rec("rank(roe)", passed=False, ic=0.02, data_window=w2),
        _rec("ts_mean(roe, 10)", passed=False, ic=0.03, data_window=w1),
    ])
    s = idx.leaf_stats(["roe"], data_window=w1)["roe"]
    assert s["n_exprs"] == 2
    assert abs(s["best_abs_ic"] - 0.03) < 1e-12


# ── B. 挖穿区 / 未探索区 ──────────────────────────────────────────────────────


def test_exhausted_excludes_coverage_only_and_passed(tmp_path: Path, monkeypatch):
    """coverage 失败不计入尝试数；n_passed>0 不入挖穿区。"""
    monkeypatch.setattr("factorzen.agents.roles.librarian.EXHAUSTED_MIN_TRIES", 5)
    monkeypatch.setattr("factorzen.agents.roles.librarian.UNEXPLORED_MAX_TRIES", 2)

    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    # north_ratio：20 次全 coverage → 方向尝试 0 → 不挖穿
    cov = [
        _rec(
            f"ts_mean(north_ratio,{i})",
            passed=False,
            ic=None,
            reject_category=REJECT_CATEGORY_HOLDOUT_COVERAGE,
        )
        for i in range(20)
    ]
    # grossprofit_margin：6 次方向失败 0 过关 → 挖穿
    gpm = [
        _rec(f"rank(ts_mean(grossprofit_margin,{i}))", passed=False, ic=0.01 + i * 0.001)
        for i in range(6)
    ]
    # roe：5 次失败但 1 次 passed → 不挖穿
    roe = [_rec("rank(roe)", passed=True, ic=0.05)] + [
        _rec(f"ts_mean(roe,{i})", passed=False, ic=0.01) for i in range(5)
    ]
    # assets_yoy：0 次 → 未探索
    _write(idx, cov + gpm + roe)

    # leaf_names 只传存活叶（模拟 leaf_health 已摘除 dead_leaf）——死叶不得出现在任一侧
    r = recall(idx, k=5, leaf_names=["north_ratio", "grossprofit_margin", "roe", "assets_yoy"])
    assert r.leaf_guidance is not None
    exhausted_blob = " ".join(r.leaf_guidance["exhausted"])
    unexplored = r.leaf_guidance["unexplored"]

    assert "grossprofit_margin" in exhausted_blob
    assert "north_ratio" not in exhausted_blob
    assert "roe" not in exhausted_blob
    assert "assets_yoy" in unexplored
    assert "dead_leaf" not in unexplored
    assert "dead_leaf" not in exhausted_blob
    # 文案含统计
    assert "试" in exhausted_blob and "0 过关" in exhausted_blob


def test_unexplored_excludes_dead_leaves_from_leaf_health(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("factorzen.agents.roles.librarian.UNEXPLORED_MAX_TRIES", 2)
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    # index 里有 north_ratio 历史，但 session 存活叶子已不含它
    _write(idx, [_rec("rank(close)", passed=False, ic=0.0)])
    r = recall(idx, k=5, leaf_names=["close", "vol", "roe"])
    assert r.leaf_guidance is not None
    u = r.leaf_guidance["unexplored"]
    assert "north_ratio" not in u
    assert set(u) <= {"close", "vol", "roe"}
    # close 有 1 条 ≤2 → 未探索；roe/vol 0 条 → 未探索
    assert "roe" in u and "vol" in u and "close" in u


def test_recall_without_leaf_names_has_no_guidance(tmp_path: Path):
    """不传 leaf_names → leaf_guidance=None，旧调用零回归。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    _write(idx, [_rec("rank(vol)", passed=False, ic=0.0)])
    r = recall(idx, k=5)
    assert r.leaf_guidance is None


# ── C. prompt 注入 + 静态偏置移除 ─────────────────────────────────────────────


def test_propose_structured_injects_leaf_guidance():
    cap: dict = {}

    def fake(msgs):
        cap["user"] = msgs[1]["content"]
        return json.dumps({"hypotheses": [{"direction": "d", "mechanism": "m",
                                           "expected_sign": 1, "falsification": "f"}]})

    guidance = {
        "exhausted": ["grossprofit_margin(试 86 次 0 过关, best|IC|=0.011)"],
        "unexplored": ["assets_yoy", "or_yoy"],
    }
    hyp_mod.propose_structured(
        fake, known_invalid=[], known_valid=[], leaf_guidance=guidance,
    )
    user = cap["user"]
    assert "已挖穿" in user
    assert "未探索" in user
    assert "grossprofit_margin" in user
    assert "assets_yoy" in user and "or_yoy" in user


def test_propose_without_leaf_guidance_zero_regression():
    """不传 leaf_guidance 时 user prompt 不含挖穿/未探索段。"""
    cap: dict = {}

    def fake(msgs):
        cap["user"] = msgs[1]["content"]
        return '{"hypotheses":["x"]}'

    hyp_mod.propose_hypotheses(fake, known_invalid=["a"], known_valid=["b"], n=1)
    user = cap["user"]
    assert "已挖穿" not in user
    assert "未探索" not in user
    assert "a" in user and "b" in user


def test_signal_families_no_north_encouragement():
    """静态偏置移除：不再点名鼓励「北向」。"""
    fam = hyp_mod.signal_families("ashare")
    # 允许中性列举资金流叶子，但不得出现「北向」鼓励文案
    assert "北向" not in fam
    assert "多族组合" in fam or "避开拥挤" in fam


def test_format_leaf_guidance_shared_by_both_paths():
    """双路径架构守卫：hypothesis 与 build_agent_messages 共用 format_leaf_guidance。"""
    # generation 与 hypothesis 均 import/调用同一注入函数
    assert gen_mod.format_leaf_guidance is format_leaf_guidance
    src_hyp = inspect.getsource(hyp_mod.propose_structured)
    src_bam = inspect.getsource(gen_mod.build_agent_messages)
    assert "format_leaf_guidance" in src_hyp
    assert "format_leaf_guidance" in src_bam

    guidance = {
        "exhausted": ["gpm(试 15 次 0 过关, best|IC|=0.01)"],
        "unexplored": ["assets_yoy"],
    }
    text = format_leaf_guidance(guidance)
    assert "已挖穿" in text and "未探索" in text

    cap: dict = {}

    def fake(msgs):
        cap["u"] = msgs[1]["content"]
        return '{"hypothesis":"h","expressions":["rank(close)"],"rationale":"r"}'

    msgs = gen_mod.build_agent_messages(
        ["ts_mean"], ["close"], leaf_guidance=guidance,
    )
    blob = " ".join(m["content"] for m in msgs)
    assert "已挖穿" in blob and "assets_yoy" in blob

    # 不传 → 零回归（无动态挖穿段；静态文案不得冒充注入标记）
    base = gen_mod.build_agent_messages(["ts_mean"], ["close"])
    base_blob = " ".join(m["content"] for m in base)
    assert "已挖穿(避开" not in base_blob
    assert "未探索(优先考虑)" not in base_blob


def test_team_round_logs_leaf_guidance_summary(tmp_path: Path, monkeypatch):
    """team 路径 rounds_log 落 leaf_guidance 摘要。"""
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.agents.team_orchestrator import run_team_agent

    # 压低挖穿阈值，保证本测 index 种子能触发
    monkeypatch.setattr("factorzen.agents.roles.librarian.EXHAUSTED_MIN_TRIES", 3)
    monkeypatch.setattr("factorzen.agents.roles.librarian.UNEXPLORED_MAX_TRIES", 2)

    rng = np.random.default_rng(0)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 120:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(15)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    daily = pl.DataFrame(rows)

    idx_path = tmp_path / "e.jsonl"
    # 预置 grossprofit_margin 挖穿种子（leaf 在默认 LEAF_FEATURES 中）
    seed_idx = ExperimentIndex(str(idx_path))
    seed_idx.append([
        _rec(f"rank(ts_mean(grossprofit_margin,{i}))", passed=False, ic=0.01)
        for i in range(4)
    ])

    seq = [
        json.dumps({"hypotheses": ["动量"]}),
        json.dumps({"expressions": ["ts_mean(close,5)"]}),
        json.dumps({"verdict": "keep", "reason": "ok"}),
    ]
    i = {"k": 0}
    cap_prompts: list[str] = []

    def fn(msgs):
        blob = " ".join(m["content"] for m in msgs)
        if "提出" in blob and "方向" in blob:
            cap_prompts.append(blob)
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    res = run_team_agent(
        daily, fn, n_rounds=1, seed=1, index_path=str(idx_path), heal_rounds=0,
    )
    assert res.rounds_log, "应有一轮 rounds_log"
    lg = res.rounds_log[0].get("leaf_guidance")
    assert lg is not None
    assert "exhausted" in lg and "unexplored" in lg
    # Hypothesis prompt 应收到 guidance
    assert any("已挖穿" in p or "未探索" in p for p in cap_prompts), (
        f"team 路径 hypothesis prompt 未注入 leaf_guidance; prompts={cap_prompts!r}"
    )


def test_constants_defaults():
    assert EXHAUSTED_MIN_TRIES == 15
    assert UNEXPLORED_MAX_TRIES == 2
