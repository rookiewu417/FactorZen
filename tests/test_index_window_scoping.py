# tests/test_index_window_scoping.py
"""长期记忆按**数据窗口**分族 + 为跨 session N 累积预留字段。

## 背景：F2（跨 session 多重检验 N 不累积）

`TrialLedger` 每 session 从 0 起，而 Librarian 主动把历史已试表达式喂给 LLM 让它避开，
于是后续 session 在同一搜索空间的剩余部分继续搜索——累计搜索了 120 次，DSR 却按 N=20 判。

**但这是 latent 的**：`run_team_mine` 只有 `fz mine team` 一个调用者，`ops daily` 与
`research run` 都不跑 team 挖掘，`workspace/mine_team/` 从未被创建。跨 session 累积从未发生。

**且 F2 没有验证 oracle**：「跨 session 的 N 应该是多少」是建模立场，不是事实
（对比 F0——M1 真实的 `dsr_pvalue` 可反解校验）。断言「N 累积了」的测试 by construction 恒真，
零判别力。故本文件**不测 N 累积**，只做两件有确定答案的事：

1. **记录前提字段**（`ir_train` / `n_train` / `data_window`）。DSR 的池要的是 IR 不是 IC，
   而 `record()` 此前只落 `ic_train` —— 没有这些字段，**将来永远无法重建历史 IR 池**。
   记录它们不承诺任何统计立场。

2. **按数据窗口分族**。`recall()` 原本从整个 `index_path` 召回，可能横跨多个数据窗口：
   即便统计上按窗口分族，**LLM 已经在拿跨窗口的提示**——一个窗口上「已验证有效」的因子，
   换个窗口未必成立。信息流的族必须与统计族对齐。

族边界 = `(start, end, universe, market)`：PIT 数据对固定窗口不可变
（`get_universe(end, ...)` 取期末快照），同元组 = 同数据 = 同族。
"""
from __future__ import annotations

import logging

from factorzen.agents.experiment_index import ExperimentIndex
from factorzen.agents.roles.librarian import recall, record
from factorzen.agents.state import AttemptRecord

_W1 = {"start": "20220101", "end": "20231229", "universe": "csi800", "market": "ashare"}
_W2 = {"start": "20150101", "end": "20211231", "universe": "csi300", "market": "ashare"}


def _attempt(expr: str, *, ir: float = 0.3, passed: bool = True,
             verdict: str | None = "keep") -> AttemptRecord:
    return AttemptRecord(iteration=0, hypothesis="h", expression=expr, compile_ok=True,
                         ic_train=0.05, passed_guardrails=passed, critic_verdict=verdict,
                         error=None, ir_train=ir, turnover=0.3, n_train=300)


# ── 前提字段：没有它们，将来永远无法重建历史 IR 池 ────────────────────────────


def test_record_persists_ir_train_and_n_train(tmp_path):
    """DSR 的 deflation 池要的是 **IR**，不是 IC。`record()` 此前只落 `ic_train`。

    这两个字段不承诺任何统计立场，只是「将来若要做跨 session N 累积」的前提条件
    （届时须与 sharpe_variance 同源地并入同一个 DeflationBasis，见 R8）。
    """
    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    record(idx, [_attempt("rank(close)", ir=0.42)], run_id="r1", data_window=_W1)

    r = idx.load()[0]
    assert r["ir_train"] == 0.42
    assert r["n_train"] == 300


def test_record_persists_data_window(tmp_path):
    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    record(idx, [_attempt("rank(close)")], run_id="r1", data_window=_W1)

    assert idx.load()[0]["data_window"] == _W1


# ── 按窗口分族 ──────────────────────────────────────────────────────────────


def test_recall_is_scoped_to_the_data_window(tmp_path):
    """一个窗口上「已验证有效」的因子，换个窗口未必成立——不得跨窗口喂给 LLM。"""
    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    record(idx, [_attempt("in_window", ir=0.4)], run_id="r1", data_window=_W1)
    record(idx, [_attempt("other_window", ir=0.4)], run_id="r2", data_window=_W2)

    rec = recall(idx, k=5, data_window=_W1)

    assert "in_window" in rec.seen
    assert "other_window" not in rec.seen, "跨窗口的历史不该进本窗口的去重集"
    assert "other_window" not in rec.known_valid


def test_seen_and_known_lists_filter_by_window(tmp_path):
    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    record(idx, [_attempt("w1_valid", ir=0.4)], run_id="r1", data_window=_W1)
    record(idx, [_attempt("w2_invalid", ir=0.01, passed=False)], run_id="r2", data_window=_W2)

    assert idx.known_valid(k=5, data_window=_W1) == ["w1_valid"]
    assert idx.known_valid(k=5, data_window=_W2) == []
    assert idx.known_invalid(k=5, data_window=_W1) == []
    assert idx.known_invalid(k=5, data_window=_W2) == ["w2_invalid"]


def test_no_window_means_no_filtering(tmp_path):
    """不传 data_window → 不过滤（向后兼容既有调用方与老 index）。"""
    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    record(idx, [_attempt("a", ir=0.4)], run_id="r1", data_window=_W1)
    record(idx, [_attempt("b", ir=0.4)], run_id="r2", data_window=_W2)

    assert idx.seen_expressions() == {"a", "b"}
    assert set(idx.known_valid(k=5)) == {"a", "b"}


def test_legacy_records_without_window_are_excluded_when_filtering(tmp_path, caplog):
    """老记录不知道来自哪个窗口 → 过滤时保守排除，并告警一次（不静默丢数据）。"""
    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    idx.append([{"expression": "legacy", "passed": True, "verdict": "keep",
                 "ic_train": 0.05, "holdout_ic": 0.04, "run_id": "old"}])
    record(idx, [_attempt("fresh", ir=0.4)], run_id="r1", data_window=_W1)

    with caplog.at_level(logging.WARNING, logger="factorzen.agents.experiment_index"):
        valid = idx.known_valid(k=5, data_window=_W1)

    assert valid == ["fresh"], "无窗口标记的老记录不得混进本窗口的召回"
    assert any("data_window" in r.getMessage() for r in caplog.records), \
        "排除老记录必须告警，不能静默"


def test_legacy_records_visible_when_not_filtering(tmp_path):
    """不过滤时老记录照常可见——排除只发生在显式按窗口查询时。"""
    idx = ExperimentIndex(str(tmp_path / "i.jsonl"))
    idx.append([{"expression": "legacy", "passed": True, "verdict": "keep",
                 "ic_train": 0.05, "holdout_ic": 0.04, "run_id": "old"}])

    assert idx.known_valid(k=5) == ["legacy"]


# ── 端到端接线：能力实现了不算，team 路径得真的传下去 ────────────────────────


def test_team_agent_scopes_index_to_its_data_window(tmp_path):
    """`run_team_agent` 必须把 data_window 透传给 Librarian，否则分族形同虚设。"""
    import datetime as dt
    import json

    import numpy as np
    import polars as pl

    from factorzen.agents.team_orchestrator import run_team_agent

    rng = np.random.default_rng(1)
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
    daily = pl.DataFrame(rows)

    def llm(messages):
        system = messages[0]["content"]
        if "verdict" in system:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if '"expressions"' in system:
            return json.dumps({"expressions": ["ts_mean(close,5)"]})
        return json.dumps({"hypotheses": ["动量"]})

    idx_path = str(tmp_path / "i.jsonl")
    run_team_agent(daily, llm, n_rounds=1, seed=1, index_path=idx_path, data_window=_W1)

    recs = ExperimentIndex(idx_path).load()
    assert recs, "本轮应有 attempt 落盘"
    assert all(r.get("data_window") == _W1 for r in recs), \
        "落盘记录必须带上本次运行的数据窗口"
    assert all("ir_train" in r for r in recs)
