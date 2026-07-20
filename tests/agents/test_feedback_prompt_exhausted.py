"""W3 B/C/D: lift 拒绝 prompt 注入、exhausted 硬过滤、library 族聚类。"""
from __future__ import annotations

import json
from pathlib import Path

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles.librarian import recall
from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_REJECTED
from factorzen.llm.prompt_fragments import (
    format_library_covered,
    format_library_crowded,
    format_lift_rejected,
)

# ── B1 fragments ─────────────────────────────────────────────────────────────



def test_format_lift_rejected_text():
    text = format_lift_rejected([
        {"expression": "rank(vol)", "lift": 0.0005, "lift_reason": "below_bar"},
        {"expression": "ts_mean(close,5)", "lift": None, "lift_reason": "group_gate_fail"},
    ])
    assert "组合层证明" in text or "无增量" in text
    assert "rank(vol)" in text
    assert "组合增量不足" in text
    assert "组门整体无增量" in text
    assert "lift=" in text



def test_format_library_crowded_text():
    text = format_library_crowded([("holder_num_chg", 9), ("roe", 7)])
    assert "拥挤" in text
    assert "holder_num_chg(9)" in text
    assert "roe(7)" in text


# ── B2 hypothesis / critic ───────────────────────────────────────────────────


def test_propose_structured_injects_lift_rejected():
    from factorzen.agents.roles.hypothesis import propose_structured

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def __call__(self, messages):
            self.calls.append(messages)
            return json.dumps({
                "hypotheses": [{
                    "direction": "x", "mechanism": "m",
                    "expected_sign": 1, "falsification": "f",
                }],
            })

    llm = FakeLLM()
    propose_structured(
        llm,
        known_invalid=[], known_valid=[], n=1,
        lift_rejected=[{"expression": "rank(vol)", "lift": 0.0001, "lift_reason": "below_bar"}],
    )
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "rank(vol)" in blob
    assert "组合" in blob or "lift" in blob.lower() or "增量" in blob


def test_propose_hypotheses_lift_rejected_none_zero_regression():
    from factorzen.agents.roles.hypothesis import propose_hypotheses

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def __call__(self, messages):
            self.calls.append(messages)
            return json.dumps({"hypotheses": ["dir"]})

    llm = FakeLLM()
    propose_hypotheses(llm, known_invalid=[], known_valid=[], n=1, lift_rejected=None)
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "组合层" not in blob and "lift 拒绝" not in blob


def test_critic_optional_lift_rejected():
    from factorzen.agents.roles.critic import critique

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def __call__(self, messages):
            self.calls.append(messages)
            return json.dumps({"verdict": "keep", "reason": "ok"})

    llm = FakeLLM()
    critique(
        {"expression": "rank(close)", "ic_train": 0.02},
        llm,
        lift_rejected=[{"expression": "rank(vol)", "lift": 0.0, "lift_reason": "below_bar"}],
    )
    blob = " ".join(m["content"] for m in llm.calls[0])
    assert "rank(vol)" in blob or "lift" in blob.lower() or "组合" in blob

    llm2 = FakeLLM()
    critique({"expression": "rank(close)"}, llm2)  # 默认 None
    blob2 = " ".join(m["content"] for m in llm2.calls[0])
    assert "组合层" not in blob2



# ── B3 librarian ─────────────────────────────────────────────────────────────


def test_recall_fills_lift_rejected(tmp_path: Path):
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    dw = {"start": "20200101", "end": "20201231", "universe": "csi300", "market": "ashare"}
    idx.append([{
        "expression": "rank(vol)",
        "data_window": dw,
        "reject_category": REJECT_CATEGORY_LIFT_REJECTED,
        "passed": False,
        "compile_ok": True,
        "lift": 0.0001,
        "lift_reason": "below_bar",
        "ts": "2026-01-01T00:00:00",
    }])
    r = recall(idx, k=5, data_window=dw)
    assert r.lift_rejected is not None
    assert any(x["expression"] == "rank(vol)" for x in r.lift_rejected)

    r2 = recall(idx, k=5, data_window={
        "start": "20990101", "end": "20991231", "universe": "x", "market": "ashare",
    })
    assert r2.lift_rejected is None  # 空 → None


def test_recall_exhausted_leaves_raw_names(tmp_path: Path, monkeypatch):
    """C1: RecallResult 带原始 exhausted 叶名，非格式化字符串。"""
    from factorzen.agents.roles import librarian as lib_mod

    monkeypatch.setattr(lib_mod, "EXHAUSTED_MIN_TRIES", 2)
    dw = {"start": "20200101", "end": "20201231", "universe": "csi300", "market": "ashare"}
    idx = ExperimentIndex(str(tmp_path / "e2.jsonl"))
    for i, expr in enumerate([
        "rank(holder_num_chg)",
        "ts_mean(holder_num_chg, 5)",
        "ts_mean(holder_num_chg, 10)",
    ]):
        idx.append([{
            "expression": expr,
            "data_window": dw,
            "passed": False,
            "compile_ok": True,
            "ic_train": 0.01 * (i + 1),
        }])
    r = recall(idx, k=5, data_window=dw, leaf_names=["holder_num_chg", "roe"])
    assert r.exhausted_leaves is not None
    assert "holder_num_chg" in r.exhausted_leaves
    # 格式化文案仍在 leaf_guidance
    assert r.leaf_guidance is not None
    assert any("holder_num_chg" in s for s in (r.leaf_guidance.get("exhausted") or []))


# ── C filter ─────────────────────────────────────────────────────────────────


def test_filter_exhausted_all_exhausted_drop():
    from factorzen.agents.scout_support import filter_exhausted_expressions

    kept, n_drop = filter_exhausted_expressions(
        ["rank(holder_num_chg)", "ts_mean(holder_num_chg, 5)"],
        exhausted={"holder_num_chg"},
        leaf_map=None,
        quota_used={},
        per_leaf_quota=2,
    )
    assert kept == []
    assert n_drop == 2


def test_filter_exhausted_mixed_family_quota():
    from factorzen.agents.scout_support import filter_exhausted_expressions

    quota: dict[str, int] = {}
    # 混族：含 exhausted 叶 + 非 exhausted 叶 → 配额内放行
    kept, n_drop = filter_exhausted_expressions(
        ["div(rank(holder_num_chg), rank(roe))"],
        exhausted={"holder_num_chg"},
        leaf_map=None,
        quota_used=quota,
        per_leaf_quota=2,
    )
    assert kept == ["div(rank(holder_num_chg), rank(roe))"]
    assert n_drop == 0
    assert quota.get("holder_num_chg") == 1

    # 再两条后配额满
    kept2, n2 = filter_exhausted_expressions(
        [
            "div(ts_mean(holder_num_chg, 5), rank(close))",
            "div(ts_mean(holder_num_chg, 10), rank(open))",
        ],
        exhausted={"holder_num_chg"},
        leaf_map=None,
        quota_used=quota,
        per_leaf_quota=2,
    )
    assert n2 == 1  # 第 3 条超配额
    assert len(kept2) == 1
    assert quota["holder_num_chg"] == 2


def test_filter_exhausted_parse_fail_keep():
    from factorzen.agents.scout_support import filter_exhausted_expressions

    kept, n_drop = filter_exhausted_expressions(
        ["this_is_not(valid"],
        exhausted={"holder_num_chg"},
        leaf_map=None,
        quota_used={},
    )
    assert kept == ["this_is_not(valid"]
    assert n_drop == 0


def test_filter_exhausted_none_passthrough():
    from factorzen.agents.scout_support import filter_exhausted_expressions

    exprs = ["rank(vol)", "rank(close)"]
    kept, n = filter_exhausted_expressions(
        exprs, exhausted=None, leaf_map=None, quota_used={},
    )
    assert kept == exprs and n == 0
    kept2, n2 = filter_exhausted_expressions(
        exprs, exhausted=set(), leaf_map=None, quota_used={},
    )
    assert kept2 == exprs and n2 == 0


# ── D library family ─────────────────────────────────────────────────────────


def test_library_covered_by_family(tmp_path: Path):
    from factorzen.discovery.factor_library import (
        FactorRecord,
        library_covered_by_family,
    )

    root = str(tmp_path / "lib")
    recs = [
        FactorRecord(expression="rank(holder_num_chg)", market="ashare",
                     ic_train=0.05, status="active"),
        FactorRecord(expression="ts_mean(holder_num_chg, 5)", market="ashare",
                     ic_train=0.04, status="active"),
        FactorRecord(expression="ts_mean(holder_num_chg, 10)", market="ashare",
                     ic_train=0.03, status="active"),
        FactorRecord(expression="rank(roe)", market="ashare",
                     ic_train=0.06, status="active"),
        FactorRecord(expression="ts_mean(roe, 5)", market="ashare",
                     ic_train=0.02, status="active"),
        FactorRecord(expression="rank(close)", market="ashare",
                     ic_train=0.01, status="active"),
    ]
    # write via save if available, else raw jsonl
    path = Path(root) / "ashare.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in recs) + "\n",
        encoding="utf-8",
    )
    covered, crowded = library_covered_by_family(
        "ashare", per_family=2, max_total=12, crowded_min=3, root=root,
    )
    # holder 族 3 条只留 2；roe 2；close 1
    assert len(covered) <= 5
    # 同叶集 holder 最多 2
    holder_exprs = [e for e in covered if "holder_num_chg" in e]
    assert len(holder_exprs) == 2
    # 最佳 |ic| 的 rank(holder) 应在
    assert any("rank(holder_num_chg)" in e for e in holder_exprs)
    # crowded: holder 出现 3 次 ≥3
    crowded_map = dict(crowded)
    assert crowded_map.get("holder_num_chg") == 3
    # roe 只 2 次 < crowded_min=3 → 不进
    assert "roe" not in crowded_map


def test_format_library_covered_unchanged():
    """旧 fragment 零回归。"""
    assert format_library_covered(None) == ""
    assert format_library_covered(["a", "b"]) == "库内已有(追求与其正交,换方向): a；b"


# ── C wiring + D M5 dual-path behavioral ─────────────────────────────────────


def test_round_exhausted_filter_in_rounds_log(tmp_path: Path, monkeypatch):
    """轮内接线：exhausted 非空 → rounds_log 有 n_exhausted_filtered；None 直通。"""
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.agents.roles import librarian as lib_mod
    from factorzen.agents.team_orchestrator import run_team_agent

    monkeypatch.setattr(lib_mod, "EXHAUSTED_MIN_TRIES", 2)

    # 预置 index：holder_num_chg 挖穿
    idx_path = tmp_path / "experiment_index.jsonl"
    idx = ExperimentIndex(str(idx_path))
    dw = {"start": "20220101", "end": "20220630", "universe": "csi300", "market": "ashare"}
    for expr in [
        "rank(holder_num_chg)",
        "ts_mean(holder_num_chg, 5)",
        "ts_mean(holder_num_chg, 10)",
    ]:
        idx.append([{
            "expression": expr, "data_window": dw,
            "passed": False, "compile_ok": True, "ic_train": 0.01,
        }])

    # 极简日频帧
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 80:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    rng = np.random.default_rng(0)
    for c in [f"{i:06d}.SZ" for i in range(20)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.01
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px, "high": px, "low": px,
                "vol": 1e6, "amount": 1e7,
            })
    daily = pl.DataFrame(rows)

    # scripted：propose → write 产出纯 exhausted 表达式（与 test_team_lift_hook 同款分支）
    def llm_fn(messages):
        text = "\n".join(m["content"] for m in messages)
        if "风控审计员" in text:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in text:
            return json.dumps({
                "expressions": [
                    "rank(holder_num_chg)",
                    "ts_mean(holder_num_chg, 20)",
                ],
            })
        return json.dumps({"hypotheses": ["筹码集中"]})

    result = run_team_agent(
        daily, llm_fn,
        n_rounds=1, seed=1, top_k=3,
        index_path=str(idx_path),
        data_window=dw,
        library_orthogonal=False,
        auto_lift=False,
        heal_rounds=0,
    )
    assert result.rounds_log
    assert "n_exhausted_filtered" in result.rounds_log[0]
    # 两条纯 exhausted 应被过滤
    assert result.rounds_log[0]["n_exhausted_filtered"] >= 1


def test_m5_library_crowded_injected_via_orchestrator(tmp_path: Path, monkeypatch):
    """双路径：M5 外层 scripted-llm 行为测试——library_crowded 进 prompt（非仅 inspect）。"""
    import datetime as dt

    import numpy as np
    import polars as pl

    from factorzen.agents.orchestrator import run_llm_agent
    from factorzen.discovery.factor_library import FactorRecord

    root = tmp_path / "lib"
    root.mkdir()
    recs = []
    for i in range(4):
        # 同叶 close 变体 4 条 → crowded
        recs.append(FactorRecord(
            expression=f"ts_mean(close, {5 + i * 5})",
            market="ashare", ic_train=0.05 - i * 0.005, status="active",
        ))
    (root / "ashare.jsonl").write_text(
        "\n".join(json.dumps(r.__dict__, default=str) for r in recs) + "\n",
        encoding="utf-8",
    )

    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 60:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    rng = np.random.default_rng(1)
    for c in [f"{i:06d}.SZ" for i in range(15)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.01
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px, "high": px, "low": px, "vol": 1e6, "amount": 1e7,
            })
    daily = pl.DataFrame(rows)

    captured: list[str] = []

    def llm_fn(messages):
        blob = "\n".join(m["content"] for m in messages)
        captured.append(blob)
        if "风控" in blob or "审计" in blob:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        return json.dumps({
            "hypothesis": "动量",
            "expressions": ["rank(close)"],
            "rationale": "x",
        })

    run_llm_agent(
        daily, llm_fn,
        n_rounds=1, seed=1, top_k=2,
        library_orthogonal=True,
        library_root=str(root),
        heal_rounds=0,
    )
    all_text = "\n".join(captured)
    # 拥挤叶子文案应出现在生成 prompt
    assert "拥挤" in all_text or "close(" in all_text

