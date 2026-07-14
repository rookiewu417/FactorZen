"""P4：挖掘目标升级为「对因子库的增量正交 alpha」。

覆盖：
1. build_library_pool 物化（合法/非法/status 过滤/缺文件）
2. team/node_guardrails 库级去相关
3. M1 与 team 共用相关函数（架构守卫）
4. 空库零回归
5. Hypothesis prompt 注入「库内已有」
6. known_invalid 排除 library_correlated
7. manifest 字段
"""
from __future__ import annotations

import ast
import datetime as dt
import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl

_SRC = Path(__file__).resolve().parents[1] / "src" / "factorzen"


# ── 合成数据 ────────────────────────────────────────────────────────────────


def _mk_daily(n_days: int = 120, n_stocks: int = 40, seed: int = 7) -> pl.DataFrame:
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
            rows.append({
                "trade_date": dd, "ts_code": c,
                "close": px, "open": px, "high": px * 1.01, "low": px * 0.99,
                # LEAF_FEATURES 把 close→close_adj；与真实日线/预处理帧同口径
                "close_adj": px, "open_adj": px, "high_adj": px * 1.01, "low_adj": px * 0.99,
                "pre_close": px / (1 + 0.001 * max(i, 1)),
                "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7 + rng.normal(0, 1e5),
            })
    return pl.DataFrame(rows)


def _write_lib(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


# ── 1. build_library_pool ────────────────────────────────────────────────────


def test_build_library_pool_materializes_active_skips_bad_and_correlated(tmp_path):
    """2 条可物化 active + 1 非法 + 1 correlated(默认不取) → pool 恰含 2 项。"""
    from factorzen.discovery.factor_library import build_library_pool

    daily = _mk_daily()
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
        {"expression": "rank(vol)", "market": "ashare", "status": "active",
         "ic_train": 0.03},
        {"expression": "not_a_real_op(close, 1)", "market": "ashare", "status": "active",
         "ic_train": 0.09},
        {"expression": "rank(amount)", "market": "ashare", "status": "correlated",
         "ic_train": 0.08},
    ])
    pool = build_library_pool("ashare", daily, root=str(tmp_path))
    assert set(pool.keys()) == {"rank(close)", "rank(vol)"}
    for fdf in pool.values():
        assert set(fdf.columns) >= {"trade_date", "ts_code", "factor_value"}
        assert fdf.height > 0
        assert fdf["factor_value"].null_count() < fdf.height


def test_build_library_pool_missing_file_returns_empty(tmp_path):
    from factorzen.discovery.factor_library import build_library_pool

    pool = build_library_pool("ashare", _mk_daily(), root=str(tmp_path / "no_such"))
    assert pool == {}


def test_build_library_pool_bad_record_does_not_crash(tmp_path):
    """一条坏记录不得崩整个 pool——异常契约。"""
    from factorzen.discovery.factor_library import build_library_pool

    daily = _mk_daily()
    _write_lib(tmp_path, "ashare", [
        {"expression": "((((broken", "market": "ashare", "status": "active",
         "ic_train": 0.1},
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.04},
    ])
    pool = build_library_pool("ashare", daily, root=str(tmp_path))
    assert "rank(close)" in pool
    assert len(pool) == 1


# ── 2. team / node_guardrails 库级去相关 ─────────────────────────────────────


def _seed_attempt(state, expr: str, *, ic: float = 0.05, ir: float = 0.4, n: int = 100):
    from factorzen.agents.state import AttemptRecord
    state.attempts.append(AttemptRecord(
        iteration=state.iteration, hypothesis="h", expression=expr,
        compile_ok=True, ic_train=ic, passed_guardrails=False,
        critic_verdict=None, error=None, ir_train=ir, turnover=0.3, n_train=n,
    ))


def test_node_guardrails_rejects_library_correlated(tmp_path, monkeypatch):
    """与库因子数学等价 → library_correlated，不占候选位。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState
    from factorzen.discovery.factor_library import build_library_pool
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIBRARY_CORRELATED
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import HoldoutICResult
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mk_daily()
    bundle = DataBundle.build(daily)
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.06},
    ])
    lib_pool = build_library_pool("ashare", daily, root=str(tmp_path))
    assert "rank(close)" in lib_pool

    # holdout 固定过关；session 池为空 → session 去相关不干扰
    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda fdf, hdf: HoldoutICResult(0.05, 0.5, (0.01, 0.09), n_days=100),
    )

    state = AgentState(seed=1)
    _seed_attempt(state, "rank(close)")          # 与库等价 → 应拒
    _seed_attempt(state, "rank(vol)", ic=0.04)   # 与库近似正交 → 应入池

    node_guardrails(
        state, daily=daily, holdout_df=daily, bundle=bundle,
        ledger=TrialLedger(), top_k=5, lib_pool=lib_pool,
    )

    rejected = next(a for a in state.attempts if a.expression == "rank(close)")
    assert rejected.passed_guardrails is True, "过了定量护栏的事实须保留"
    assert rejected.reject_category == REJECT_CATEGORY_LIBRARY_CORRELATED
    assert rejected.reject_reason and "与库内因子重复" in rejected.reject_reason
    assert "rank(close)" not in {c["expression"] for c in state.candidates}

    kept = [c for c in state.candidates if c["expression"] == "rank(vol)"]
    assert len(kept) == 1
    assert "max_corr_library" in kept[0]
    assert kept[0]["max_corr_library"] < 0.7


def test_node_guardrails_library_reject_frees_slot_for_orthogonal(tmp_path, monkeypatch):
    """库相关拒绝后，同批后续正交候选仍可入池（top_k 预算内）。"""
    from factorzen.agents.nodes import node_guardrails
    from factorzen.agents.state import AgentState
    from factorzen.discovery.factor_library import build_library_pool
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import HoldoutICResult
    from factorzen.validation.multiple_testing import TrialLedger

    daily = _mk_daily()
    bundle = DataBundle.build(daily)
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.06},
    ])
    lib_pool = build_library_pool("ashare", daily, root=str(tmp_path))
    monkeypatch.setattr(
        "factorzen.validation.holdout.holdout_ic_result",
        lambda fdf, hdf: HoldoutICResult(0.05, 0.5, (0.01, 0.09), n_days=100),
    )

    state = AgentState(seed=1)
    _seed_attempt(state, "rank(close)", ic=0.08)
    _seed_attempt(state, "rank(vol)", ic=0.04)

    node_guardrails(
        state, daily=daily, holdout_df=daily, bundle=bundle,
        ledger=TrialLedger(), top_k=2, lib_pool=lib_pool,
    )
    assert [c["expression"] for c in state.candidates] == ["rank(vol)"]
    assert state.n_library_correlated_rejects >= 1


# ── 3. M1 / team 架构守卫：共用相关函数 ──────────────────────────────────────


def test_library_corr_shared_function_architecture_guard():
    """双路径必须调用同一库相关入口，禁止各自内联 max_correlation(lib_pool)。"""
    shared_names = {"library_orthogonal_check", "max_correlation_detail"}
    for rel in ("agents/nodes.py", "discovery/mining_session.py"):
        tree = ast.parse((_SRC / rel).read_text(encoding="utf-8-sig"))
        called = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    called.add(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    called.add(n.func.attr)
        assert called & shared_names, (
            f"{rel} 未调用共享库相关函数 {shared_names}；实得 calls∩={called & shared_names}"
        )


def test_m1_greedy_skips_library_correlated(tmp_path, monkeypatch):
    """M1 收尾 top-K：与库高相关者跳过，正交者入选并带 max_corr_library。"""
    from factorzen.discovery.mining_session import run_session

    daily = _mk_daily(n_days=80, n_stocks=35)
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
    ])

    # 用极小搜索 + 强制候选列表：monkeypatch RandomSearcher.propose 产出固定表达式
    exprs = ["rank(close)", "rank(vol)", "rank(amount)"]
    idx = {"i": 0}

    class _FakeSearcher:
        def __init__(self, *a, **k):
            pass

        def propose(self):
            from factorzen.discovery.expression import parse_expr
            e = exprs[idx["i"] % len(exprs)]
            idx["i"] += 1
            return parse_expr(e)

    monkeypatch.setattr(
        "factorzen.discovery.mining_session.RandomSearcher", _FakeSearcher,
    )
    res = run_session(
        daily, n_trials=6, top_k=3, seed=1, method="random",
        out_dir=str(tmp_path / "sessions"),
        update_library=False,
        library_orthogonal=True,
        library_root=str(tmp_path),
    )
    cand_exprs = {c["expression"] for c in res["candidates"]}
    assert "rank(close)" not in cand_exprs, "库内同式不得入 M1 top-K"
    # 至少有一个正交候选入选
    assert cand_exprs, "应有正交候选"
    for c in res["candidates"]:
        assert "max_corr_library" in c
        assert c["max_corr_library"] < 0.7
    man = json.loads((Path(res["session_dir"]) / "manifest.json").read_text())
    assert man["library_pool_size"] >= 1
    assert man["n_library_correlated_rejects"] >= 1


# ── 4. 空库零回归 ────────────────────────────────────────────────────────────


def test_empty_library_pool_zero_regression_m1(tmp_path, monkeypatch):
    """无库文件时 library_orthogonal=True 与关开关时候选表达式集合一致。"""
    from factorzen.discovery.mining_session import run_session

    daily = _mk_daily(n_days=60, n_stocks=30)
    exprs = ["rank(close)", "rank(vol)"]
    idx = {"i": 0}

    class _FakeSearcher:
        def __init__(self, *a, **k):
            pass

        def propose(self):
            from factorzen.discovery.expression import parse_expr
            e = exprs[idx["i"] % len(exprs)]
            idx["i"] += 1
            return parse_expr(e)

    monkeypatch.setattr(
        "factorzen.discovery.mining_session.RandomSearcher", _FakeSearcher,
    )

    def _run(*, orthogonal: bool, seed_tag: str):
        idx["i"] = 0
        return run_session(
            daily, n_trials=4, top_k=2, seed=1, method="random",
            out_dir=str(tmp_path / seed_tag),
            update_library=False,
            library_orthogonal=orthogonal,
            library_root=str(tmp_path / "empty_lib"),  # 不存在
        )

    on = _run(orthogonal=True, seed_tag="on")
    off = _run(orthogonal=False, seed_tag="off")
    on_exprs = [c["expression"] for c in on["candidates"]]
    off_exprs = [c["expression"] for c in off["candidates"]]
    assert on_exprs == off_exprs
    # 空库：入池候选不应被强加 max_corr_library（行为与旧一致）
    for c in on["candidates"]:
        assert "max_corr_library" not in c or c.get("max_corr_library") == 0.0


# ── 5. prompt 注入 ───────────────────────────────────────────────────────────


def test_hypothesis_prompt_injects_library_covered():
    from factorzen.agents.roles import hypothesis as hyp_mod

    cap: dict = {}

    def fake(msgs):
        cap["user"] = msgs[1]["content"]
        return json.dumps({"hypotheses": ["x"]})

    hyp_mod.propose_hypotheses(
        fake, known_invalid=[], known_valid=[],
        library_covered=["rank(close)", "ts_mean(vol, 20)"],
    )
    user = cap["user"]
    assert "库内已有" in user
    assert "rank(close)" in user
    assert "正交" in user


def test_hypothesis_prompt_no_library_byte_stable():
    """无 library_covered 时 user prompt 与只传 known 的旧形状一致（无「库内已有」段）。"""
    from factorzen.agents.roles import hypothesis as hyp_mod

    cap: dict = {}

    def fake(msgs):
        cap["user"] = msgs[1]["content"]
        return '{"hypotheses":["x"]}'

    hyp_mod.propose_hypotheses(fake, known_invalid=["a"], known_valid=["b"], n=1)
    user = cap["user"]
    assert "库内已有" not in user
    assert "a" in user and "b" in user


def test_format_library_covered_shared_architecture_guard():
    """双路径（hypothesis / build_agent_messages）共用 format_library_covered。"""
    from factorzen.agents.roles import hypothesis as hyp_mod
    from factorzen.agents.roles.librarian import format_library_covered
    from factorzen.llm import generation as gen_mod

    assert "format_library_covered" in inspect.getsource(hyp_mod.propose_hypotheses)
    assert "format_library_covered" in inspect.getsource(gen_mod.build_agent_messages)
    text = format_library_covered(["rank(close)"])
    assert "库内已有" in text and "rank(close)" in text
    assert format_library_covered(None) == ""
    assert format_library_covered([]) == ""


# ── 6. known_invalid 排除 library_correlated ─────────────────────────────────


def test_known_invalid_excludes_library_correlated(tmp_path):
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIBRARY_CORRELATED

    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([
        {"expression": "rank(close)", "passed": False, "compile_ok": True,
         "ic_train": 0.01, "reject_category": REJECT_CATEGORY_LIBRARY_CORRELATED,
         "reject_reason": "与库内因子重复(corr=0.96, 最相近=rank(close))"},
        {"expression": "rank(vol)", "passed": False, "compile_ok": True,
         "ic_train": 0.001},
    ])
    inv = idx.known_invalid(k=5)
    assert "rank(close)" not in inv
    assert "rank(vol)" in inv


def test_known_invalid_excludes_lift_queue(tmp_path):
    """lift_queue（与旧 gray_zone）不得进 known_invalid 负例回灌。"""
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.discovery.guardrails import (
        REJECT_CATEGORY_GRAY_ZONE,
        REJECT_CATEGORY_LIFT_QUEUE,
    )

    idx = ExperimentIndex(str(tmp_path / "e_lq.jsonl"))
    idx.append([
        {"expression": "rank(amount)", "passed": False, "compile_ok": True,
         "ic_train": 0.008, "reject_category": REJECT_CATEGORY_LIFT_QUEUE,
         "reject_reason": "残差holdout反号(lift队列,待组合裁决)"},
        {"expression": "rank(open)", "passed": False, "compile_ok": True,
         "ic_train": 0.007, "reject_category": REJECT_CATEGORY_GRAY_ZONE,
         "reject_reason": "旧灰区兼容"},
        {"expression": "rank(vol)", "passed": False, "compile_ok": True,
         "ic_train": 0.001},
    ])
    inv = idx.known_invalid(k=5)
    assert "rank(amount)" not in inv
    assert "rank(open)" not in inv
    assert "rank(vol)" in inv


# ── 7. CLI / 常量 ────────────────────────────────────────────────────────────


def test_cli_no_library_orthogonal_flag():
    from factorzen.cli.main import build_parser

    parser = build_parser()
    for cmd in ("search", "agent", "team"):
        args = parser.parse_args(
            ["mine", cmd, "--start", "20240101", "--end", "20240601",
             "--no-library-orthogonal"]
        )
        assert args.no_library_orthogonal is True


def test_reject_category_constant_exists():
    from factorzen.discovery.guardrails import (
        DEFAULT_DUPLICATE_CORR,
        REJECT_CATEGORY_HOLDOUT_COVERAGE,
        REJECT_CATEGORY_LIBRARY_CORRELATED,
        REJECT_CATEGORY_LIFT_QUEUE,
    )
    assert REJECT_CATEGORY_LIBRARY_CORRELATED == "library_correlated"
    assert REJECT_CATEGORY_HOLDOUT_COVERAGE == "holdout_coverage"
    assert REJECT_CATEGORY_LIFT_QUEUE == "lift_queue"
    assert DEFAULT_DUPLICATE_CORR == 0.95


def test_recall_accepts_library_covered():
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import recall

    idx = ExperimentIndex("/tmp/nonexistent_idx_p4.jsonl")  # missing → empty
    r = recall(idx, library_covered=["rank(close)"])
    assert r.library_covered == ["rank(close)"]
    r2 = recall(idx)
    assert r2.library_covered is None
