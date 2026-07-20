"""合并自 agents 相关碎片测试（test_wiring_caveats.py）。

test_agent_wiring.py：接线测试：能力层(orchestrator) ↔ 接线层(pipeline/CLI) 之间不许漂移
test_agent_ashare_caveats.py：Workstream E：A股机制 + PIT 陷阱 Prompt 注入（研报优化方向①）
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl


# ==== 来自 test_agent_wiring.py ====
def _mock_daily() -> pl.DataFrame:
    rng = np.random.default_rng(1)
    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 180:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{i:06d}.SZ" for i in range(20)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)


# ─────────────────────────── CLI parser 层 ───────────────────────────

def test_parser_mine_agent_exposes_patience_and_heal_rounds():
    from factorzen.cli.main import build_parser

    args = build_parser().parse_args(
        ["mine", "agent", "--start", "20220101", "--end", "20231231",
         "--patience", "3", "--heal-rounds", "1"]
    )
    assert args.patience == 3
    assert args.heal_rounds == 1


def test_parser_mine_team_exposes_structured_patience_heal_rounds():
    from factorzen.cli.main import build_parser

    args = build_parser().parse_args(
        ["mine", "team", "--start", "20220101", "--end", "20231231",
         "--structured", "--patience", "2", "--heal-rounds", "0"]
    )
    assert args.structured is True
    assert args.patience == 2
    assert args.heal_rounds == 0


def test_parser_defaults_are_zero_regression():
    """默认值必须保持既有行为：不早停、非结构化、自愈 2 轮。"""
    from factorzen.cli.main import build_parser

    p = build_parser()
    a = p.parse_args(["mine", "agent", "--start", "20220101", "--end", "20231231"])
    assert a.patience is None          # 跑满 n_rounds
    assert a.heal_rounds == 2

    t = p.parse_args(["mine", "team", "--start", "20220101", "--end", "20231231"])
    assert t.patience is None
    assert t.heal_rounds == 2
    assert t.structured is False


# ─────────────────────────── CLI → pipeline ───────────────────────────

def test_cmd_mine_agent_forwards_patience_and_heal_rounds(monkeypatch):
    from factorzen.cli import main as cli

    captured: dict = {}

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        captured["prepare_lookback"] = lookback_days
        return pl.DataFrame({"ts_code": ["000001.SZ"]})

    def fake_run_agent_mine(daily, **kw):
        captured.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr("factorzen.pipelines.factor_mine_agent.run_agent_mine",
                        fake_run_agent_mine)

    rc = cli.main(["mine", "agent", "--start", "20220101", "--end", "20231231",
                   "--patience", "3", "--heal-rounds", "1"])
    from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
    assert rc == 0
    assert captured["patience"] == 3
    assert captured["heal_rounds"] == 1
    assert captured["prepare_lookback"] == AGENT_WARMUP_LOOKBACK


def test_cmd_mine_team_forwards_structured_patience_heal_rounds(monkeypatch):
    from factorzen.cli import main as cli

    captured: dict = {}

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        captured["prepare_lookback"] = lookback_days
        return pl.DataFrame({"ts_code": ["000001.SZ"]})

    def fake_run_team_mine(daily, **kw):
        captured.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_run_team_mine)

    rc = cli.main(["mine", "team", "--start", "20220101", "--end", "20231231",
                   "--structured", "--patience", "2", "--heal-rounds", "0"])
    assert rc == 0
    assert captured["structured"] is True
    assert captured["patience"] == 2
    assert captured["heal_rounds"] == 0


# ─────────────────────────── pipeline → orchestrator ───────────────────────────

def test_run_agent_mine_forwards_to_orchestrator(monkeypatch, tmp_path):
    from factorzen.agents.orchestrator import AgentResult
    from factorzen.agents.state import AgentState
    from factorzen.pipelines import factor_mine_agent as fma

    captured: dict = {}

    def fake_run_llm_agent(daily, llm_fn, **kw):
        captured.update(kw)
        return AgentResult(state=AgentState(seed=1), candidates=[], n_trials=0)

    monkeypatch.setattr(fma, "run_llm_agent", fake_run_llm_agent)

    fma.run_agent_mine(_mock_daily(), n_rounds=1, seed=1, llm_fn=lambda _m: "{}",
                       out_dir=str(tmp_path), patience=3, heal_rounds=1, export=False)
    assert captured["patience"] == 3
    assert captured["heal_rounds"] == 1


def test_run_team_mine_forwards_to_orchestrator(monkeypatch, tmp_path):
    from factorzen.agents.state import AgentState
    from factorzen.agents.team_orchestrator import TeamResult
    from factorzen.pipelines import factor_mine_team as fmt

    captured: dict = {}

    def fake_run_team_agent(daily, llm_fn, **kw):
        captured.update(kw)
        return TeamResult(state=AgentState(seed=1), candidates=[], n_trials=0)

    monkeypatch.setattr(fmt, "run_team_agent", fake_run_team_agent)

    fmt.run_team_mine(_mock_daily(), n_rounds=1, seed=1, index_path=str(tmp_path / "e.jsonl"),
                      llm_fn=lambda _m: "{}", out_dir=str(tmp_path),
                      structured=True, patience=2, heal_rounds=0, export=False)
    assert captured["structured"] is True
    assert captured["patience"] == 2
    assert captured["heal_rounds"] == 0


def test_team_manifest_records_new_params(monkeypatch, tmp_path):
    """可复现铁律：manifest 必须记下 structured/patience/heal_rounds，否则事后无法重跑。"""
    from factorzen.agents.state import AgentState
    from factorzen.agents.team_orchestrator import TeamResult
    from factorzen.pipelines import factor_mine_team as fmt

    monkeypatch.setattr(fmt, "run_team_agent",
                        lambda *a, **k: TeamResult(state=AgentState(seed=1),
                                                   candidates=[], n_trials=0))

    fmt.run_team_mine(_mock_daily(), n_rounds=1, seed=1,
                      index_path=str(tmp_path / "e.jsonl"), llm_fn=lambda _m: "{}",
                      out_dir=str(tmp_path), run_id="r", structured=True,
                      patience=2, heal_rounds=1, export=False)
    manifest = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert manifest["params"]["structured"] is True
    assert manifest["params"]["patience"] == 2
    assert manifest["params"]["heal_rounds"] == 1


# ─────────────────── RD-Agent 步2：任务分解真正进入流水线 ───────────────────

def test_structured_decomposes_hypothesis_and_drives_coder_per_task(tmp_path, monkeypatch):
    """structured=True：假设先经 decompose_tasks 拆成任务，每个任务各自驱动一次 Coder。

    对齐研报步2「拆两步让每次 LLM 调用专注一件事」；CoSTEER 亦是逐因子独立编码。
    """
    from factorzen.agents import team_orchestrator as to

    decompose_calls: list[str] = []
    write_calls: list[str] = []

    def fake_decompose(hypothesis, llm_fn):
        decompose_calls.append(hypothesis)
        return [{"name": "mom5", "description": "5日动量", "rationale": "趋势延续"},
                {"name": "mom20", "description": "20日动量", "rationale": "中期趋势"}]

    def fake_write(hypothesis, llm_fn, *, avoid=None, **_kw):
        write_calls.append(hypothesis)
        return [f"ts_mean(close,{5 * len(write_calls)})"]

    monkeypatch.setattr(to, "decompose_tasks", fake_decompose)
    monkeypatch.setattr(to, "write_expressions", fake_write)

    seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                       "expected_sign": 1, "falsification": "f"}]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    to.run_team_agent(_mock_daily(), fn, n_rounds=1, seed=1,
                      index_path=str(tmp_path / "e.jsonl"), structured=True, heal_rounds=0)

    assert len(decompose_calls) == 1, "structured=True 必须调用 decompose_tasks（RD-Agent 步2）"
    assert len(write_calls) == 2, f"每个 task 各驱动一次 Coder，实得 {len(write_calls)}"
    assert "mom5" in write_calls[0] and "5日动量" in write_calls[0]
    assert "mom20" in write_calls[1]


def test_decompose_returning_empty_falls_back_to_whole_hypothesis(tmp_path, monkeypatch):
    """LLM 分解失败（空 tasks）→ 降级为对整条假设写表达式，不静默空转。"""
    from factorzen.agents import team_orchestrator as to

    write_calls: list[str] = []

    monkeypatch.setattr(to, "decompose_tasks", lambda h, f: [])

    def fake_write(hypothesis, llm_fn, *, avoid=None, **_kw):
        write_calls.append(hypothesis)
        return ["ts_mean(close,5)"]

    monkeypatch.setattr(to, "write_expressions", fake_write)

    seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                       "expected_sign": 1, "falsification": "f"}]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    to.run_team_agent(_mock_daily(), fn, n_rounds=1, seed=1,
                      index_path=str(tmp_path / "e.jsonl"), structured=True, heal_rounds=0)

    assert len(write_calls) == 1
    assert "动量" in write_calls[0]


def test_non_structured_path_does_not_decompose(tmp_path, monkeypatch):
    """零回归：structured=False（默认）不得调用 decompose_tasks，不增加 LLM 调用。"""
    from factorzen.agents import team_orchestrator as to

    calls: list[str] = []
    monkeypatch.setattr(to, "decompose_tasks",
                        lambda h, f: calls.append(h) or [])  # type: ignore[func-returns-value]

    seq = [json.dumps({"hypotheses": ["动量"]}),
           json.dumps({"expressions": ["ts_mean(close,5)"]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    to.run_team_agent(_mock_daily(), fn, n_rounds=1, seed=1,
                      index_path=str(tmp_path / "e.jsonl"), heal_rounds=0)

    assert calls == []


def test_rounds_log_records_tasks_for_traceability(tmp_path, monkeypatch):
    """实验溯源：structured 轮次的 rounds_log 要留下 task 清单。"""
    from factorzen.agents import team_orchestrator as to

    monkeypatch.setattr(to, "decompose_tasks",
                        lambda h, f: [{"name": "mom5", "description": "5日动量",
                                       "rationale": "趋势"}])
    monkeypatch.setattr(to, "write_expressions",
                        lambda h, f, *, avoid=None, **_kw: ["ts_mean(close,5)"])

    seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                       "expected_sign": 1, "falsification": "f"}]}),
           json.dumps({"verdict": "keep", "reason": "ok"})]
    i = {"k": 0}

    def fn(_m):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v

    res = to.run_team_agent(_mock_daily(), fn, n_rounds=1, seed=1,
                            index_path=str(tmp_path / "e.jsonl"), structured=True, heal_rounds=0)

    assert res.rounds_log[0]["tasks"] == [
        {"name": "mom5", "description": "5日动量", "rationale": "趋势"}
    ]

# ==== 来自 test_agent_ashare_caveats.py ====
def test_caveats_fragment_covers_key_mechanisms():
    from factorzen.llm.prompt_fragments import ASHARE_CAVEATS
    for kw in ["涨跌停", "停牌", "T+1", "PIT", "换手", "风险因子"]:
        assert kw in ASHARE_CAVEATS, f"缺少 {kw}"


def test_build_agent_messages_injects_caveats():
    from factorzen.llm.generation import build_agent_messages
    sys = build_agent_messages(["ts_mean"], ["close"], "", [])[0]["content"]
    assert "涨跌停" in sys and "T+1" in sys


def test_hypothesis_prompt_injects_caveats():
    from factorzen.agents.roles.hypothesis import propose_hypotheses
    cap: dict = {}

    def fake(msgs):
        cap["m"] = msgs
        return '{"hypotheses":["x"]}'
    propose_hypotheses(fake, known_invalid=[], known_valid=[])
    assert "涨跌停" in cap["m"][0]["content"]


def test_coder_prompt_injects_caveats():
    from factorzen.agents.roles.coder import write_expressions
    cap: dict = {}

    def fake(msgs):
        cap["m"] = msgs
        return '{"expressions":["ts_mean(close,5)"]}'
    write_expressions("动量", fake)
    sys = cap["m"][0]["content"]
    assert "PIT" in sys or "涨跌停" in sys
