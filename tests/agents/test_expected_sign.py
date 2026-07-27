"""expected_sign 一致性记账：结构化假设的「预期 IC 符号」与实测 train IC 符号对账。

**纯记账，不拦截**——不进护栏、不改 `passed` 判定、不改 Critic 输入。假设声称「预期 IC 为正」
而实测为负却因 |IC| 够大过了护栏，要么机制解释错（样本外可靠性存疑）、要么 Coder 把方向
翻译反了；两种都该留痕，供事后审计与 Critic/Librarian 后续消费。

判别力设计：E2E 用**取反配对**表达式（`X` 与 `neg(X)`，rank IC 逐值互为相反数）当 ground
truth——无需预知 mock 数据上 IC 的真实符号，只需断言「同一 expected_sign 下，配对二者恰有
一个被判 mismatch，且被判的那个 ic_train 与 expected_sign 反号」。
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles.librarian import record
from factorzen.agents.state import AttemptRecord
from factorzen.agents.team_orchestrator import _coerce_expected_sign, _sign_mismatch
from factorzen.discovery.expression import parse_expr, to_expr_string

# 取反配对：rank IC 逐值互为相反数 → 任意 expected_sign 下恰有一个 mismatch
_EXPR_POS = "rank(close)"
_EXPR_NEG = "neg(rank(close))"
_EXPR_BAD = "no_such_leaf_zzz"          # parse 失败 → compile_ok=False、ic_train=None


def _mock_daily() -> pl.DataFrame:
    """40 只 × 180 天。**必须 ≥30 只**——`ic_analysis` 对不足 30 只的截面直接跳过，
    IC 序列为空 → ic_train=None，本文件的对账断言会退化成恒真的 None 检查。"""
    rng = np.random.default_rng(7)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(40)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


def _norm(expr: str) -> str:
    """规范化以便匹配 AttemptRecord.expression（求值层写的是 to_expr_string(node)）。
    不可解析的表达式（_EXPR_BAD）原样返回。"""
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


def _scripted(expected_sign, *, structured: bool = True):
    """按调用序回放 LLM 响应：propose（结构化/裸）→ 之后一律 Critic keep。"""
    if structured:
        first = json.dumps({"hypotheses": [{
            "direction": "方向A", "mechanism": "m",
            "expected_sign": expected_sign, "falsification": "f",
        }]})
    else:
        first = json.dumps({"hypotheses": ["方向A"]})
    seq = [first, json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"]] if i["k"] < len(seq) else seq[-1]
        i["k"] += 1
        return v
    return fn


def _run_round(tmp_path, monkeypatch, *, expected_sign, structured=True,
               exprs=(_EXPR_POS, _EXPR_NEG)):
    """跑一轮 team session，返回 TeamResult。decompose/write 全部打桩，表达式由测试指定。"""
    from factorzen.agents import team_orchestrator as to

    monkeypatch.setattr(to, "decompose_tasks", lambda h, f: [])
    monkeypatch.setattr(
        to, "write_expressions",
        lambda h, f, *, avoid=None, **_kw: list(exprs))
    return to.run_team_agent(
        _mock_daily(), _scripted(expected_sign, structured=structured),
        n_rounds=1, seed=1, index_path=str(tmp_path / "e.jsonl"),
        structured=structured, heal_rounds=0, update_library=False,
        library_orthogonal=False, auto_lift=False, campaign_prior_enabled=False,
    )


def _by_expr(res) -> dict[str, AttemptRecord]:
    return {_norm(a.expression): a for a in res.state.attempts}


# ────────────────────────── ① 判定函数真值表（纯函数） ──────────────────────────


@pytest.mark.parametrize(("ic", "sign", "want"), [
    # 相符 → False（已对账、无冲突）
    (0.05, 1, False),
    (-0.05, -1, False),
    # 相反 → True
    (-0.05, 1, True),
    (0.05, -1, True),
    # 无从判定 → None
    (None, 1, None),          # 未出值（编译失败/预热不足/指纹重复）
    (0.0, 1, None),           # IC 恰为 0：无方向
    (0.05, None, None),       # 非结构化轮 / LLM 未给 expected_sign
    (0.05, 0, None),          # expected_sign 非 ±1
    (float("nan"), 1, None),  # 退化守卫：NaN 既不 >0 也不 <0，不许静默当负号
    (float("inf"), 1, None),
])
def test_sign_mismatch_truth_table(ic, sign, want):
    """判定契约：仅在「实测有方向 ∧ 预期是 ±1」时出 True/False，其余一律 None。"""
    assert _sign_mismatch(ic, sign) is want


@pytest.mark.parametrize(("raw", "want"), [
    (1, 1), (-1, -1), (1.0, 1), (-1.0, -1),
    ("1", 1), ("-1", -1), ("+1", 1),
    (2, 1), (-0.5, -1),            # 非规范幅度只取方向
    (0, None), (0.0, None), (None, None),
    ("正", None), ("", None), (True, None), (False, None),  # bool 不是符号
    (float("nan"), None),
])
def test_coerce_expected_sign(raw, want):
    """LLM 产出未经校验（hypothesis.propose_structured 原样透传 h.get），归一化必须兜住脏值。"""
    assert _coerce_expected_sign(raw) is want


# ────────────────────── ② structured 轮：符号相符 / 相反 / 缺值 ──────────────────────


def test_structured_positive_sign_flags_exactly_the_negated_twin(tmp_path, monkeypatch):
    """expected_sign=+1：取反配对中恰有一个 mismatch，且它是 ic_train<0 的那个。"""
    res = _run_round(tmp_path, monkeypatch, expected_sign=1)
    got = _by_expr(res)
    a, b = got[_norm(_EXPR_POS)], got[_norm(_EXPR_NEG)]

    # ground truth：配对确实互为相反数（保证下面的断言有判别力，而非恒真）
    assert a.ic_train is not None and b.ic_train is not None
    assert a.ic_train == pytest.approx(-b.ic_train, abs=1e-12)
    assert a.ic_train != 0.0

    assert a.expected_sign == 1 and b.expected_sign == 1
    assert {a.sign_mismatch, b.sign_mismatch} == {True, False}, "取反配对必须一正一反"
    flagged = a if a.sign_mismatch else b
    assert flagged.ic_train < 0, "expected_sign=+1 时被标记的必须是实测负 IC 那条"


def test_structured_negative_sign_flips_which_twin_is_flagged(tmp_path, monkeypatch):
    """expected_sign=-1：被标记的换成 ic_train>0 那条——证明判定真读了 expected_sign。"""
    res = _run_round(tmp_path, monkeypatch, expected_sign=-1)
    got = _by_expr(res)
    a, b = got[_norm(_EXPR_POS)], got[_norm(_EXPR_NEG)]

    assert a.expected_sign == -1 and b.expected_sign == -1
    assert {a.sign_mismatch, b.sign_mismatch} == {True, False}
    flagged = a if a.sign_mismatch else b
    assert flagged.ic_train > 0, "expected_sign=-1 时被标记的必须是实测正 IC 那条"


def test_structured_round_without_ic_leaves_mismatch_none(tmp_path, monkeypatch):
    """ic_train 缺失（编译失败）→ sign_mismatch=None；expected_sign 仍照记（区分两种 None）。"""
    res = _run_round(tmp_path, monkeypatch, expected_sign=1,
                     exprs=(_EXPR_POS, _EXPR_BAD))
    bad = next(a for a in res.state.attempts if a.expression == _EXPR_BAD)
    assert bad.compile_ok is False and bad.ic_train is None
    assert bad.expected_sign == 1, "结构化轮的 expected_sign 与是否出值无关"
    assert bad.sign_mismatch is None, "无实测方向 → 无从判定，不许当 False 混进相符池"
    ok = _by_expr(res)[_norm(_EXPR_POS)]
    assert ok.sign_mismatch in (True, False), "同轮出值的那条必须有结论"


# ────────────────────────── ③ 非结构化轮零回归 ──────────────────────────


def test_non_structured_round_records_no_sign_fields(tmp_path, monkeypatch):
    """非 structured 轮无 expected_sign 概念：两个字段全 None，轮计数为 0。"""
    res = _run_round(tmp_path, monkeypatch, expected_sign=None, structured=False)
    assert res.state.attempts, "夹具必须真跑出 attempts，否则断言恒真"
    assert all(a.expected_sign is None for a in res.state.attempts)
    assert all(a.sign_mismatch is None for a in res.state.attempts)
    assert res.rounds_log[0]["n_sign_mismatch"] == 0


def test_attempt_record_defaults_are_none():
    """默认值即「无从判定」——旧构造点（nodes.node_evaluate）不传时不得凭空产生结论。"""
    a = AttemptRecord(iteration=0, hypothesis="h", expression="x", compile_ok=True,
                      ic_train=0.05, passed_guardrails=False, critic_verdict=None,
                      error=None)
    assert a.expected_sign is None and a.sign_mismatch is None


# ────────────────────────── ④ 轮日志计数 ──────────────────────────


def test_rounds_log_counts_only_true_mismatches(tmp_path, monkeypatch):
    """取反配对 + 一条无 IC → 本轮恰好 1 条 mismatch（False 与 None 都不计）。"""
    res = _run_round(tmp_path, monkeypatch, expected_sign=1,
                     exprs=(_EXPR_POS, _EXPR_NEG, _EXPR_BAD))
    n_true = sum(1 for a in res.state.attempts if a.sign_mismatch is True)
    n_false = sum(1 for a in res.state.attempts if a.sign_mismatch is False)
    n_none = sum(1 for a in res.state.attempts if a.sign_mismatch is None)
    assert (n_true, n_false, n_none) == (1, 1, 1), (
        f"配对应一 True 一 False，坏表达式为 None：{[(a.expression, a.sign_mismatch) for a in res.state.attempts]}"
    )
    assert res.rounds_log[0]["n_sign_mismatch"] == 1


# ────────────────────────── ⑤ 落盘 + 旧记录容错 ──────────────────────────


def test_record_writes_sign_fields_only_when_known(tmp_path):
    """「有值才写」惯例：None 不落键（消费方靠 .get 缺省），False 必须落（= 已对账相符）。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    record(idx, [
        AttemptRecord(iteration=0, hypothesis="h", expression="mismatch_expr",
                      compile_ok=True, ic_train=-0.05, passed_guardrails=False,
                      critic_verdict=None, error=None,
                      expected_sign=1, sign_mismatch=True),
        AttemptRecord(iteration=0, hypothesis="h", expression="agree_expr",
                      compile_ok=True, ic_train=0.05, passed_guardrails=False,
                      critic_verdict=None, error=None,
                      expected_sign=1, sign_mismatch=False),
        AttemptRecord(iteration=0, hypothesis="h", expression="plain_expr",
                      compile_ok=True, ic_train=0.05, passed_guardrails=False,
                      critic_verdict=None, error=None),
    ], run_id="t")

    rows = {r["expression"]: r for r in idx.load()}
    assert rows["mismatch_expr"]["expected_sign"] == 1
    assert rows["mismatch_expr"]["sign_mismatch"] is True
    assert rows["agree_expr"]["sign_mismatch"] is False, "相符是结论，不是缺省"
    assert "expected_sign" not in rows["plain_expr"], "非结构化轮不落假值"
    assert "sign_mismatch" not in rows["plain_expr"]


def test_index_readers_tolerate_records_without_sign_fields(tmp_path):
    """旧记录（无新字段）读回不崩：seen/known_valid/known_invalid 全部照常。"""
    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([{
        "expression": "rank(close)", "hypothesis": "h", "ic_train": 0.06,
        "ir_train": 0.4, "n_train": 300, "passed": True, "verdict": "keep",
        "decorrelated": False, "compile_ok": True, "error": None, "run_id": "old",
    }])
    assert "rank(close)" in idx.seen_expressions()
    assert idx.known_valid(k=5) == ["rank(close)"]
    assert idx.known_invalid(k=5) == []


def test_state_to_dict_carries_sign_fields(tmp_path):
    """manifest 溯源：asdict 出来的 attempts 必须带新字段（可复现铁律）。"""
    from factorzen.agents.state import AgentState

    s = AgentState(seed=1)
    s.attempts.append(AttemptRecord(
        iteration=0, hypothesis="h", expression="x", compile_ok=True, ic_train=-0.05,
        passed_guardrails=False, critic_verdict=None, error=None,
        expected_sign=1, sign_mismatch=True))
    d = s.to_dict()["attempts"][0]
    assert d["expected_sign"] == 1 and d["sign_mismatch"] is True
