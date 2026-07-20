"""
test_leaf_feedback.py：P3：叶子级负面/正面经验回灌（leaf_stats + Librarian leaf_guidance + prompt 注入）。
test_scout.py：B-W2：Feature Scout 角色 + scout_support 编排 + team/agent 接线（全离线 mock）。
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.nodes import AgentContext
from factorzen.agents.roles import hypothesis as hyp_mod
from factorzen.agents.roles.feature_scout import propose_intraday_features
from factorzen.agents.roles.librarian import (
    EXHAUSTED_MIN_TRIES,
    UNEXPLORED_MAX_TRIES,
    recall,
)
from factorzen.agents.scout_support import (
    ScoutState,
    promote_admitted_exprs,
    run_scout_round,
)
from factorzen.discovery.guardrails import REJECT_CATEGORY_HOLDOUT_COVERAGE
from factorzen.discovery.intraday_expr import make_expr_spec
from factorzen.llm import generation as gen_mod
from factorzen.llm.prompt_fragments import format_leaf_guidance


# ==== 来自 test_leaf_feedback.py ====
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


def test_leaf_stats_suite(tmp_path):
    """词边界匹配 + passed/coverage/编译失败/|IC| 逐项计数。；同表达式后写覆盖：只计最新状态。；test_leaf_stats_scoped_by_data_window"""
    # -- 原 test_leaf_stats_counts_and_word_boundary --
    def _section_0_test_leaf_stats_counts_and_word_boundary(tmp_path):
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

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_leaf_stats_counts_and_word_boundary(_tp0)

    # -- 原 test_leaf_stats_unique_expr_last_wins --
    def _section_1_test_leaf_stats_unique_expr_last_wins(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        _write(idx, [
            _rec("rank(roe)", passed=True, ic=0.05),
            _rec("rank(roe)", passed=False, ic=0.01),  # last wins
        ])
        s = idx.leaf_stats(["roe"])["roe"]
        assert s["n_exprs"] == 1
        assert s["n_passed"] == 0
        assert s["best_abs_ic"] == 0.01

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_leaf_stats_unique_expr_last_wins(_tp1)

    # -- 原 test_leaf_stats_scoped_by_data_window --
    def _section_2_test_leaf_stats_scoped_by_data_window(tmp_path):
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

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_leaf_stats_scoped_by_data_window(_tp2)


# ── B. 挖穿区 / 未探索区 ──────────────────────────────────────────────────────


def test_exhausted_guidance_suite(tmp_path, monkeypatch):
    """**进了库**的叶子不判枯竭，哪怕单因子护栏零通过。；豁免只认**在任**记录 → 因子被降级后该叶重新可判枯竭。；库内表达式按词边界匹配叶名，`roe` 不得被 `grossprofit_margin` 之类子串误命中。；`library_exprs=None`（旧调用方）→ 行为与加该参数前完全一致。；coverage 失败不计入尝试数；n_passed>0 不入挖穿区。；test_unexplored_excludes_dead_leaves_from_leaf_health；不传 leaf_names → leaf_guidance=None，旧调用零回归。"""
    # -- 原 test_exhausted_excludes_leaf_present_in_library --
    def _section_0_test_exhausted_excludes_leaf_present_in_library(tmp_path, mp):
        mp.setattr("factorzen.agents.roles.librarian.EXHAUSTED_MIN_TRIES", 5)

        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        # 两个叶子都是「6 次方向失败、单因子护栏 0 过关」——按旧判据都该枯竭
        recs = [
            _rec(f"rank(ts_mean(grossprofit_margin,{i}))", passed=False, ic=0.01)
            for i in range(6)
        ] + [
            _rec(f"rank(ts_mean(roa,{i}))", passed=False, ic=0.01) for i in range(6)
        ]
        _write(idx, recs)

        # roa 有一条经 lift 轨进了库（护栏没过，但库是权威成功记录）
        r = recall(
            idx, k=5,
            leaf_names=["grossprofit_margin", "roa"],
            library_exprs=["ts_decay_linear(mul(rank(roa), 2.0), 20)"],
        )
        blob = " ".join(r.leaf_guidance["exhausted"])
        assert "grossprofit_margin" in blob, "无库内证据的叶子仍应判枯竭"
        assert "roa" not in blob, "库里有该叶子的因子，不该判枯竭"
        # 硬过滤用的裸名列表必须同口径（两个消费方不许漂移）
        assert r.exhausted_leaves == ["grossprofit_margin"], r.exhausted_leaves

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_exhausted_excludes_leaf_present_in_library(_tp0, mp)

    # -- 原 test_exhausted_library_exemption_expires_with_demotion --
    def _section_1_test_exhausted_library_exemption_expires_with_demotion(tmp_path, mp):
        mp.setattr("factorzen.agents.roles.librarian.EXHAUSTED_MIN_TRIES", 5)

        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        _write(idx, [_rec(f"rank(ts_mean(roa,{i}))", passed=False, ic=0.01) for i in range(6)])

        # 在任 → 豁免
        live = recall(idx, k=5, leaf_names=["roa"], library_exprs=["rank(mul(roa, 2.0))"])
        assert live.exhausted_leaves is None, live.exhausted_leaves

        # 该因子被降级出 active/probation → 调用方不再传它 → 豁免消失
        demoted = recall(idx, k=5, leaf_names=["roa"], library_exprs=[])
        assert demoted.exhausted_leaves == ["roa"], demoted.exhausted_leaves

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_exhausted_library_exemption_expires_with_demotion(_tp1, mp)

    # -- 原 test_exhausted_library_match_uses_word_boundary --
    def _section_2_test_exhausted_library_match_uses_word_boundary(tmp_path, mp):
        mp.setattr("factorzen.agents.roles.librarian.EXHAUSTED_MIN_TRIES", 5)

        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        _write(idx, [_rec(f"rank(ts_mean(roe,{i}))", passed=False, ic=0.01) for i in range(6)])

        # 库里只有含 `roe_ttm` 的表达式——不是 `roe`，不该救下 roe
        r = recall(idx, k=5, leaf_names=["roe"], library_exprs=["rank(roe_ttm)"])
        assert "roe" in " ".join(r.leaf_guidance["exhausted"]), "子串命中导致误救"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_exhausted_library_match_uses_word_boundary(_tp2, mp)

    # -- 原 test_exhausted_library_exprs_none_is_zero_regression --
    def _section_3_test_exhausted_library_exprs_none_is_zero_regression(tmp_path, mp):
        mp.setattr("factorzen.agents.roles.librarian.EXHAUSTED_MIN_TRIES", 5)

        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        _write(idx, [_rec(f"rank(ts_mean(roa,{i}))", passed=False, ic=0.01) for i in range(6)])

        r = recall(idx, k=5, leaf_names=["roa"])
        assert "roa" in " ".join(r.leaf_guidance["exhausted"])
        assert r.exhausted_leaves == ["roa"]

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_exhausted_library_exprs_none_is_zero_regression(_tp3, mp)

    # -- 原 test_exhausted_excludes_coverage_only_and_passed --
    def _section_4_test_exhausted_excludes_coverage_only_and_passed(tmp_path, mp):
        mp.setattr("factorzen.agents.roles.librarian.EXHAUSTED_MIN_TRIES", 5)
        mp.setattr("factorzen.agents.roles.librarian.UNEXPLORED_MAX_TRIES", 2)

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

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_exhausted_excludes_coverage_only_and_passed(_tp4, mp)

    # -- 原 test_unexplored_excludes_dead_leaves_from_leaf_health --
    def _section_5_test_unexplored_excludes_dead_leaves_from_leaf_health(tmp_path, mp):
        mp.setattr("factorzen.agents.roles.librarian.UNEXPLORED_MAX_TRIES", 2)
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

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_unexplored_excludes_dead_leaves_from_leaf_health(_tp5, mp)

    # -- 原 test_recall_without_leaf_names_has_no_guidance --
    def _section_6_test_recall_without_leaf_names_has_no_guidance(tmp_path):
        idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
        _write(idx, [_rec("rank(vol)", passed=False, ic=0.0)])
        r = recall(idx, k=5)
        assert r.leaf_guidance is None

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    _section_6_test_recall_without_leaf_names_has_no_guidance(_tp6)


# ── C. prompt 注入 + 静态偏置移除 ─────────────────────────────────────────────


def test_leaf_guidance_prompt_suite():
    """test_propose_structured_injects_leaf_guidance；不传 leaf_guidance 时 user prompt 不含挖穿/未探索段。；静态偏置移除：不再点名鼓励「北向」。；双路径架构守卫：hypothesis 与 build_agent_messages 共用 format_leaf_guidance。；test_constants_defaults"""
    # -- 原 test_propose_structured_injects_leaf_guidance --
    def _section_0_test_propose_structured_injects_leaf_guidance():
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

    _section_0_test_propose_structured_injects_leaf_guidance()

    # -- 原 test_propose_without_leaf_guidance_zero_regression --
    def _section_1_test_propose_without_leaf_guidance_zero_regression():
        cap: dict = {}

        def fake(msgs):
            cap["user"] = msgs[1]["content"]
            return '{"hypotheses":["x"]}'

        hyp_mod.propose_hypotheses(fake, known_invalid=["a"], known_valid=["b"], n=1)
        user = cap["user"]
        assert "已挖穿" not in user
        assert "未探索" not in user
        assert "a" in user and "b" in user

    _section_1_test_propose_without_leaf_guidance_zero_regression()

    # -- 原 test_signal_families_no_north_encouragement --
    def _section_2_test_signal_families_no_north_encouragement():
        fam = hyp_mod.signal_families("ashare")
        # 允许中性列举资金流叶子，但不得出现「北向」鼓励文案
        assert "北向" not in fam
        assert "多族组合" in fam or "避开拥挤" in fam

    _section_2_test_signal_families_no_north_encouragement()

    # -- 原 test_format_leaf_guidance_shared_by_both_paths --
    def _section_3_test_format_leaf_guidance_shared_by_both_paths():
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

    _section_3_test_format_leaf_guidance_shared_by_both_paths()

    # -- 原 test_constants_defaults --
    def _section_4_test_constants_defaults():
        assert EXHAUSTED_MIN_TRIES == 15
        assert UNEXPLORED_MAX_TRIES == 2

    _section_4_test_constants_defaults()


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


# ── E. 接线层：orchestrator 必须把「在任库表达式」透传给挖穿判定 ──────────────


def test_run_team_agent_passes_live_library_exprs_to_recall(tmp_path: Path, monkeypatch):
    """从最外层 `run_team_agent` 出发，验证 `library_exprs` 真到达 `recall`。

    能力层↔接线层漂移是本项目头号 bug 源：`build_leaf_guidance` 支持库兜底，但只要
    orchestrator 不传，挖穿判定照旧误杀（分钟叶 17 个全部 `n_passed==0`，正是被误杀
    的那批）。断言必须从 CLI/入口出发，不能手工拼「已经正确」的参数。

    同时锁住**只传在任记录**：库的生命周期即豁免的过期机制，传了 correlated/no_lift
    就等于给已降级的因子发终身豁免。
    """
    import datetime as dt

    import polars as pl

    from factorzen.agents import team_orchestrator as tor
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    lib_root = tmp_path / "factor_library"
    _save_library("ashare", [
        FactorRecord(expression="rank(roe)", market="ashare", status="active",
                     ic_train=0.05, added_at="2026-07-01", updated_at="2026-07-01"),
        FactorRecord(expression="rank(pb)", market="ashare", status="probation",
                     ic_train=0.03, added_at="2026-07-01", updated_at="2026-07-01"),
        # 以下两条已降级 → 不得进 library_exprs（否则豁免永不过期）
        FactorRecord(expression="rank(vol)", market="ashare", status="correlated",
                     ic_train=0.04, added_at="2026-07-01", updated_at="2026-07-01"),
        FactorRecord(expression="rank(amount)", market="ashare", status="no_lift",
                     ic_train=0.02, added_at="2026-07-01", updated_at="2026-07-01"),
    ], root=str(lib_root))

    seen: dict = {}
    orig_recall = tor.recall

    def _spy(index, **kw):
        seen["library_exprs"] = kw.get("library_exprs")
        return orig_recall(index, **kw)

    monkeypatch.setattr(tor, "recall", _spy)

    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    rows = [
        {"trade_date": d, "ts_code": f"{c:06d}.SZ",
         "close": 10.0 + i * 0.1 + c, "open": 10.0 + i * 0.1 + c,
         "high": 11.0 + i * 0.1 + c, "low": 9.0 + i * 0.1 + c,
         "close_adj": 10.0 + i * 0.1 + c, "open_adj": 10.0 + i * 0.1 + c,
         "high_adj": 11.0 + i * 0.1 + c, "low_adj": 9.0 + i * 0.1 + c,
         "vol": 1000.0 + c, "amount": 5000.0 + c}
        for i, d in enumerate(dates) for c in range(30)
    ]
    daily = pl.DataFrame(rows)

    seq = [
        json.dumps({"hypotheses": ["动量"]}),
        json.dumps({"expressions": ["ts_mean(close, 20)"]}),
        json.dumps({"verdict": "keep", "reason": "ok"}),
    ] * 10
    k = {"i": 0}

    def llm_fn(messages):
        v = seq[k["i"] % len(seq)]
        k["i"] += 1
        return v

    tor.run_team_agent(
        daily, llm_fn, n_rounds=1, seed=1,
        index_path=str(tmp_path / "idx.jsonl"), heal_rounds=0,
        library_root=str(lib_root),
    )

    got = seen.get("library_exprs")
    assert got is not None, "orchestrator 没把 library_exprs 传给 recall（接线漏斗）"
    assert set(got) == {"rank(roe)", "rank(pb)"}, got

# ==== 来自 test_scout.py ====
# ── fixtures ────────────────────────────────────────────────────────────


def _mock_daily(n_stocks: int = 8, n_days: int = 60, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days: list[dt.date] = []
    d = dt.date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for c in codes:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append(
                {
                    "trade_date": dd,
                    "ts_code": c,
                    "close": px,
                    "open": px * 0.99,
                    "high": px * 1.01,
                    "low": px * 0.98,
                    "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                    "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                    "i_rv": float(abs(rng.standard_normal()) * 0.01 + 0.001),
                    "i_amihud": float(abs(rng.standard_normal()) * 1e-6),
                }
            )
    return pl.DataFrame(rows)


def _fake_panel_for_specs(specs, mining: pl.DataFrame) -> pl.DataFrame:
    """按 mining 键构造与 specs 同名的 ix 列面板（可过 screen）。"""
    keys = mining.select(["trade_date", "ts_code"]).unique().sort(["trade_date", "ts_code"])
    out = keys
    rng = np.random.default_rng(0)
    for sp in specs:
        noise = rng.standard_normal(out.height)
        # 与 i_rv 低相关：独立噪声
        out = out.with_columns(pl.Series(sp.name, noise.astype(np.float64)))
    return out


# ── propose_intraday_features ────────────────────────────────────────────


def test_scout_role_core_suite(monkeypatch):
    """test_propose_valid_json_array；test_propose_malformed_returns_empty；test_propose_mixed_skips_non_dict_and_caps_k；test_propose_llm_raises_returns_empty；test_propose_wrapped_features_key；test_run_scout_round_injects_and_audits；test_run_scout_round_max_leaves_skips_llm"""
    # -- 原 test_propose_valid_json_array --
    def _section_0_test_propose_valid_json_array():
        payload = [
            {"bar_expr": "sub(div(close, open), 1)", "agg": "std", "hypothesis": "开盘跳空波动"},
            {"bar_expr": "sub(high, low)", "agg": "mean", "hypothesis": "日内振幅"},
        ]

        def llm_fn(messages):
            return json.dumps(payload)

        out = propose_intraday_features(llm_fn, k=2, avoid=[], known_features="")
        assert len(out) == 2
        assert out[0]["bar_expr"] == "sub(div(close, open), 1)"
        assert out[0]["agg"] == "std"
        assert out[1]["hypothesis"] == "日内振幅"

    _section_0_test_propose_valid_json_array()

    # -- 原 test_propose_malformed_returns_empty --
    def _section_1_test_propose_malformed_returns_empty():
        def llm_fn(messages):
            return "这不是 JSON 也不是数组"

        assert propose_intraday_features(llm_fn, k=3, avoid=[], known_features="") == []

    _section_1_test_propose_malformed_returns_empty()

    # -- 原 test_propose_mixed_skips_non_dict_and_caps_k --
    def _section_2_test_propose_mixed_skips_non_dict_and_caps_k():
        payload = [
            "junk",
            {"bar_expr": "vol", "agg": "sum", "hypothesis": "成交量"},
            {"bar_expr": "close", "agg": "last", "hypothesis": "收盘"},
            {"bar_expr": "amount", "agg": "mean", "hypothesis": "额"},
        ]

        def llm_fn(messages):
            return json.dumps(payload)

        out = propose_intraday_features(llm_fn, k=2, avoid=[], known_features="")
        assert len(out) == 2
        assert all("bar_expr" in x and "agg" in x for x in out)

    _section_2_test_propose_mixed_skips_non_dict_and_caps_k()

    # -- 原 test_propose_llm_raises_returns_empty --
    def _section_3_test_propose_llm_raises_returns_empty():
        def llm_fn(messages):
            raise RuntimeError("network")

        assert propose_intraday_features(llm_fn, k=1, avoid=[], known_features="") == []

    _section_3_test_propose_llm_raises_returns_empty()

    # -- 原 test_propose_wrapped_features_key --
    def _section_4_test_propose_wrapped_features_key():
        payload = {
            "features": [
                {"bar_expr": "bar_ret", "agg": "std", "hypothesis": "已实现波动"},
            ]
        }

        def llm_fn(messages):
            return json.dumps(payload)

        out = propose_intraday_features(llm_fn, k=1, avoid=[], known_features="")
        assert len(out) == 1
        assert out[0]["agg"] == "std"

    _section_4_test_propose_wrapped_features_key()

    # -- 原 test_run_scout_round_injects_and_audits --
    def _section_5_test_run_scout_round_injects_and_audits(mp):
        daily = _mock_daily()
        mid = daily["trade_date"].unique().sort()
        n = mid.len()
        mining = daily.filter(pl.col("trade_date") < mid[int(n * 0.8)])
        holdout = daily.filter(pl.col("trade_date") >= mid[int(n * 0.8)])
        ctx = AgentContext()
        state = ScoutState()

        proposals = [
            {"bar_expr": "sub(high, low)", "agg": "mean", "hypothesis": "振幅"},
            {"bar_expr": "bar_ret", "agg": "std", "hypothesis": "波动"},
        ]

        def llm_fn(messages):
            return json.dumps(proposals)

        def fake_mat(specs, start, end, *, freq="5min", **_kw):
            # mining∪holdout∪daily 全键
            keys = daily.select(["trade_date", "ts_code"]).unique()
            out = keys
            rng = np.random.default_rng(7)
            for sp in specs:
                out = out.with_columns(
                    pl.Series(sp.name, rng.standard_normal(out.height).astype(np.float64))
                )
            return out

        def fake_screen(panel, reference=None, **_kw):
            return {c: "keep" for c in panel.columns if c.startswith("ix_")}

        mp.setattr(
            "factorzen.agents.scout_support.materialize_expr_features", fake_mat,
        )
        mp.setattr(
            "factorzen.agents.scout_support.screen_expr_panel", fake_screen,
        )
        # 跳过 leaf_health 死叶（合成帧覆盖可能为 0）
        mp.setattr(
            "factorzen.discovery.leaf_health.leaf_holdout_coverage",
            lambda *a, **k: {n: 1.0 for n in (a[1] if len(a) > 1 else [])},
        )

        frames = run_scout_round(
            llm_fn=llm_fn,
            state=state,
            k=2,
            max_leaves=12,
            start="20220103",
            end="20220331",
            freq="5min",
            frames={"mining": mining, "holdout": holdout, "daily": daily},
            ctx=ctx,
        )

        assert len(state.injected) == 2
        for name in state.injected:
            assert name in frames["mining"].columns
            assert name in frames["holdout"].columns
            assert name in frames["daily"].columns
            assert name in ctx.leaf_names
            assert ctx.leaf_map is not None and name in ctx.leaf_map
        keeps = [a for a in state.audit if a["verdict"] == "keep"]
        assert len(keeps) == 2

    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_run_scout_round_injects_and_audits(mp)

    # -- 原 test_run_scout_round_max_leaves_skips_llm --
    def _section_6_test_run_scout_round_max_leaves_skips_llm():
        daily = _mock_daily(n_days=40)
        ctx = AgentContext()
        state = ScoutState()
        state.injected = [f"ix_fake{i:02d}" for i in range(3)]
        called = {"n": 0}

        def llm_fn(messages):
            called["n"] += 1
            return "[]"

        frames_in = {"mining": daily, "holdout": daily, "daily": daily}
        frames_out = run_scout_round(
            llm_fn=llm_fn,
            state=state,
            k=4,
            max_leaves=3,
            start="20220103",
            end="20220301",
            freq="5min",
            frames=frames_in,
            ctx=ctx,
        )
        assert called["n"] == 0
        assert frames_out is frames_in

    _section_6_test_run_scout_round_max_leaves_skips_llm()


# ── run_scout_round ──────────────────────────────────────────────────────


def test_run_scout_round_screen_reject_not_injected(monkeypatch):
    daily = _mock_daily(n_days=40)
    ctx = AgentContext()
    state = ScoutState()

    def llm_fn(messages):
        return json.dumps([
            {"bar_expr": "vol", "agg": "sum", "hypothesis": "量"},
        ])

    monkeypatch.setattr(
        "factorzen.agents.scout_support.materialize_expr_features",
        lambda specs, *a, **k: _fake_panel_for_specs(specs, daily),
    )
    monkeypatch.setattr(
        "factorzen.agents.scout_support.screen_expr_panel",
        lambda panel, reference=None, **kw: {
            c: "degenerate" for c in panel.columns if c.startswith("ix_")
        },
    )

    run_scout_round(
        llm_fn=llm_fn,
        state=state,
        k=1,
        max_leaves=12,
        start="20220103",
        end="20220301",
        freq="5min",
        frames={"mining": daily, "holdout": daily, "daily": daily},
        ctx=ctx,
    )
    assert state.injected == []
    assert any(a["verdict"] == "degenerate" for a in state.audit)


def test_run_scout_round_dedup_repeat_proposal(monkeypatch):
    daily = _mock_daily(n_days=40)
    ctx = AgentContext()
    state = ScoutState()
    prop = {"bar_expr": "sub(high, low)", "agg": "mean", "hypothesis": "振幅"}

    def llm_fn(messages):
        return json.dumps([prop])

    monkeypatch.setattr(
        "factorzen.agents.scout_support.materialize_expr_features",
        lambda specs, *a, **k: _fake_panel_for_specs(specs, daily),
    )
    monkeypatch.setattr(
        "factorzen.agents.scout_support.screen_expr_panel",
        lambda panel, reference=None, **kw: {
            c: "keep" for c in panel.columns if c.startswith("ix_")
        },
    )
    monkeypatch.setattr(
        "factorzen.discovery.leaf_health.leaf_holdout_coverage",
        lambda *a, **k: {n: 1.0 for n in (a[1] if len(a) > 1 else [])},
    )

    frames = {"mining": daily, "holdout": daily, "daily": daily}
    run_scout_round(
        llm_fn=llm_fn, state=state, k=1, max_leaves=12,
        start="20220103", end="20220301", freq="5min",
        frames=frames, ctx=ctx,
    )
    n1 = len(state.injected)
    assert n1 == 1
    # 第二轮同提案 → duplicate，不再注入
    run_scout_round(
        llm_fn=llm_fn, state=state, k=1, max_leaves=12,
        start="20220103", end="20220301", freq="5min",
        frames={"mining": frames["mining"] if False else daily,
                "holdout": daily, "daily": daily},
        ctx=ctx,
    )
    assert len(state.injected) == n1
    assert any(a["verdict"] == "duplicate" for a in state.audit)


# ── promote_admitted_exprs ───────────────────────────────────────────────


def test_scout_wiring_suite(tmp_path, monkeypatch, capsys):
    """test_promote_only_referenced；flag-off：不建 ScoutState，result.intraday_scout 为 None（行为与改前一致）。；test_cli_parser_intraday_scout_flags；test_cli_intraday_scout_non_ashare_returns_2；--intraday-scout 隐含 intraday_leaves=True 再进 prepare。"""
    # -- 原 test_promote_only_referenced --
    def _section_0_test_promote_only_referenced(tmp_path, mp):
        sp_keep = make_expr_spec("sub(high, low)", "mean", freq="5min", hypothesis="振幅")
        sp_skip = make_expr_spec("vol", "sum", freq="5min", hypothesis="量")
        state = ScoutState(
            injected=[sp_keep.name, sp_skip.name],
            specs={sp_keep.name: sp_keep, sp_skip.name: sp_skip},
        )
        registered: list[str] = []
        ensured: list[str] = []

        def fake_reg(specs, *, session, base_dir=None):
            for s in specs:
                registered.append(s.name)

        def fake_ensure(name, start, end, *, base_dir=None, source_dir=None):
            ensured.append(name)
            return pl.DataFrame(
                schema={"trade_date": pl.Date, "ts_code": pl.String, name: pl.Float64}
            )

        mp.setattr(
            "factorzen.agents.scout_support.register_expr_features", fake_reg,
        )
        mp.setattr(
            "factorzen.agents.scout_support.ensure_expr_panel", fake_ensure,
        )

        admitted = [f"ts_mean({sp_keep.name}, 5)"]
        promoted = promote_admitted_exprs(
            session_dir=tmp_path,
            admitted_exprs=admitted,
            state=state,
            session="test_sess",
            full_start="20220101",
            full_end="20221231",
            freq="5min",
            base_dir=tmp_path,
            leaf_map={sp_keep.name: sp_keep.name, sp_skip.name: sp_skip.name},
        )
        assert sp_keep.name in promoted
        assert sp_skip.name not in promoted
        assert registered == [sp_keep.name]
        assert ensured == [sp_keep.name]

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_promote_only_referenced(_tp0, mp)

    # -- 原 test_team_flag_off_no_scout_block --
    def _section_1_test_team_flag_off_no_scout_block(tmp_path):
        from factorzen.agents.team_orchestrator import run_team_agent

        hyp = json.dumps({"hypotheses": ["动量"]})
        code = json.dumps({"expressions": ["ts_mean(close,5)"]})
        crit = json.dumps({"verdict": "keep", "reason": "ok"})
        seq = [hyp, code, crit] * 20
        i = {"k": 0}

        def fn(messages):
            v = seq[i["k"] % len(seq)]
            i["k"] += 1
            return v

        daily = _mock_daily(n_days=60)
        res = run_team_agent(
            daily, fn, n_rounds=1, seed=1,
            index_path=str(tmp_path / "e.jsonl"),
            heal_rounds=0, auto_lift=False, update_library=False,
            library_orthogonal=False, campaign_prior_enabled=False,
            intraday_scout=False,
        )
        assert res.intraday_scout is None

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_team_flag_off_no_scout_block(_tp1)

    # -- 原 test_cli_parser_intraday_scout_flags --
    def _section_2_test_cli_parser_intraday_scout_flags():
        from factorzen.cli.main import build_parser

        p = build_parser()
        args = p.parse_args([
            "mine", "team", "--start", "20220101", "--end", "20231231",
            "--intraday-scout", "--scout-k", "3", "--scout-max-leaves", "8",
        ])
        assert args.intraday_scout is True
        assert args.scout_k == 3
        assert args.scout_max_leaves == 8

        args2 = p.parse_args([
            "mine", "agent", "--start", "20220101", "--end", "20231231",
        ])
        assert getattr(args2, "intraday_scout", False) is False
        assert args2.scout_k == 4
        assert args2.scout_max_leaves == 12

    _section_2_test_cli_parser_intraday_scout_flags()

    # -- 原 test_cli_intraday_scout_non_ashare_returns_2 --
    def _section_3_test_cli_intraday_scout_non_ashare_returns_2(capsys):
        from factorzen.cli import main as cli

        rc = cli.main([
            "mine", "team",
            "--start", "20220101", "--end", "20231231",
            "--market", "crypto",
            "--intraday-scout",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "intraday-scout" in err and "ashare" in err

    _section_3_test_cli_intraday_scout_non_ashare_returns_2(capsys)

    # -- 原 test_cli_intraday_scout_implies_leaves --
    def _section_4_test_cli_intraday_scout_implies_leaves(mp, capsys):
        import polars as pl

        from factorzen.cli import main as cli

        seen: dict = {}

        def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
            seen["intraday"] = kw.get("intraday")
            return pl.DataFrame({
                "ts_code": ["000001.SZ"],
                "trade_date": [dt.date(2022, 1, 4)],
                "close": [10.0],
                "open": [10.0],
                "high": [10.0],
                "low": [10.0],
                "vol": [1e5],
                "amount": [1e6],
            })

        def fake_run(daily, **kwargs):
            seen["scout"] = kwargs.get("intraday_scout")
            return {
                "n_candidates": 0,
                "n_trials": 0,
                "run_dir": "workspace/mine_team/x",
            }

        mp.setattr(
            "factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare,
        )
        mp.setattr(
            "factorzen.pipelines.factor_mine_team.run_team_mine", fake_run,
        )
        rc = cli.main([
            "mine", "team",
            "--start", "20220101", "--end", "20231231",
            "--intraday-scout",
        ])
        assert rc == 0
        assert seen.get("intraday") is True
        assert seen.get("scout") is True

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_cli_intraday_scout_implies_leaves(mp, capsys)


# ── e2e team ─────────────────────────────────────────────────────────────


def _team_llm_with_scout(scout_payload: list[dict]):
    """hypothesis/coder/critic 脚本 + scout 固定表达式。"""
    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": ["ts_mean(close,5)"]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    scout = json.dumps(scout_payload)
    i = {"k": 0}

    def fn(messages):
        text = "\n".join(m.get("content", "") for m in messages)
        if "日内特征 Scout" in text or "BAR_LEAVES" in text or "bar 级叶子" in text:
            return scout
        if "风控审计员" in text or ("verdict" in text and "审计" in text):
            return crit
        if "翻译成" in text or "修正" in text:
            return code
        if "提出" in text and "方向" in text:
            return hyp
        # critic 角色常见措辞
        if "审计" in text or "风控" in text:
            return crit
        # 默认按 team 序列
        seq = [hyp, code, crit]
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    return fn


def test_team_e2e_intraday_scout_manifest(tmp_path: Path, monkeypatch):
    daily = _mock_daily(n_days=90, n_stocks=12)
    scout_payload = [
        {"bar_expr": "sub(high, low)", "agg": "mean", "hypothesis": "振幅"},
    ]

    monkeypatch.setattr(
        "factorzen.agents.scout_support.materialize_expr_features",
        lambda specs, *a, **k: _fake_panel_for_specs(specs, daily),
    )
    monkeypatch.setattr(
        "factorzen.agents.scout_support.screen_expr_panel",
        lambda panel, reference=None, **kw: {
            c: "keep" for c in panel.columns if c.startswith("ix_")
        },
    )
    monkeypatch.setattr(
        "factorzen.discovery.leaf_health.leaf_holdout_coverage",
        lambda *a, **k: {n: 1.0 for n in (a[1] if len(a) > 1 else [])},
    )
    # promote 不碰真实盘
    monkeypatch.setattr(
        "factorzen.agents.scout_support.register_expr_features",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "factorzen.agents.scout_support.ensure_expr_panel",
        lambda name, *a, **k: pl.DataFrame(
            schema={"trade_date": pl.Date, "ts_code": pl.String, name: pl.Float64}
        ),
    )

    from factorzen.agents.team_orchestrator import run_team_agent, write_team_manifest

    res = run_team_agent(
        daily,
        _team_llm_with_scout(scout_payload),
        n_rounds=2,
        seed=7,
        index_path=str(tmp_path / "e.jsonl"),
        heal_rounds=0,
        auto_lift=False,
        update_library=False,
        library_orthogonal=False,
        campaign_prior_enabled=False,
        intraday_scout=True,
        scout_k=1,
        scout_max_leaves=4,
        scout_base_dir=tmp_path / "ix_base",
    )
    assert res.intraday_scout is not None
    assert "injected" in res.intraday_scout
    assert "audit" in res.intraday_scout
    assert "promoted" in res.intraday_scout
    assert res.intraday_scout["proposed"] >= 1
    # 至少一轮 keep 注入
    assert len(res.intraday_scout["injected"]) >= 1

    path = write_team_manifest(
        res, out_dir=str(tmp_path / "runs"), run_id="scout_e2e", params={},
    )
    man = json.loads(path.read_text(encoding="utf-8"))
    assert "intraday_scout" in man
    assert man["intraday_scout"]["injected"] == res.intraday_scout["injected"]


