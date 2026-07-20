"""
test_wiring_caveats.py：合并自 agents 相关碎片测试（test_wiring_caveats.py）。
test_manifest_repro.py：合并自 agents 相关碎片测试（test_manifest_repro.py）。
"""

from __future__ import annotations

import datetime as dt
import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from factorzen.agents.manifest import write_session_manifest
from factorzen.agents.orchestrator import AgentResult
from factorzen.agents.state import AgentState, AttemptRecord
from factorzen.pipelines.factor_mine_agent import run_agent_mine


# ==== 来自 test_wiring_caveats.py ====
# ==== 来自 test_agent_wiring.py ====
def _mock_daily__wiring_caveats() -> pl.DataFrame:
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


def test_agent_wiring_cli_suite(monkeypatch, tmp_path):
    """默认值必须保持既有行为：不早停、非结构化、自愈 2 轮。；test_cmd_mine_agent_forwards_patience_and_heal_rounds；test_cmd_mine_team_forwards_structured_patience_heal_rounds；test_run_agent_mine_forwards_to_orchestrator；test_run_team_mine_forwards_to_orchestrator"""
    # -- 原 test_parser_defaults_are_zero_regression --
    def _section_0_test_parser_defaults_are_zero_regression():
        from factorzen.cli.main import build_parser

        p = build_parser()
        a = p.parse_args(["mine", "agent", "--start", "20220101", "--end", "20231231"])
        assert a.patience is None          # 跑满 n_rounds
        assert a.heal_rounds == 2

        t = p.parse_args(["mine", "team", "--start", "20220101", "--end", "20231231"])
        assert t.patience is None
        assert t.heal_rounds == 2
        assert t.structured is False

    _section_0_test_parser_defaults_are_zero_regression()

    # -- 原 test_cmd_mine_agent_forwards_patience_and_heal_rounds --
    def _section_1_test_cmd_mine_agent_forwards_patience_and_heal_rounds(mp):
        from factorzen.cli import main as cli

        captured: dict = {}

        def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
            captured["prepare_lookback"] = lookback_days
            return pl.DataFrame({"ts_code": ["000001.SZ"]})

        def fake_run_agent_mine(daily, **kw):
            captured.update(kw)
            return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

        mp.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
        mp.setattr("factorzen.pipelines.factor_mine_agent.run_agent_mine",
                            fake_run_agent_mine)

        rc = cli.main(["mine", "agent", "--start", "20220101", "--end", "20231231",
                       "--patience", "3", "--heal-rounds", "1"])
        from factorzen.config.constants import AGENT_WARMUP_LOOKBACK
        assert rc == 0
        assert captured["patience"] == 3
        assert captured["heal_rounds"] == 1
        assert captured["prepare_lookback"] == AGENT_WARMUP_LOOKBACK

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cmd_mine_agent_forwards_patience_and_heal_rounds(mp)

    # -- 原 test_cmd_mine_team_forwards_structured_patience_heal_rounds --
    def _section_2_test_cmd_mine_team_forwards_structured_patience_heal_rounds(mp):
        from factorzen.cli import main as cli

        captured: dict = {}

        def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
            captured["prepare_lookback"] = lookback_days
            return pl.DataFrame({"ts_code": ["000001.SZ"]})

        def fake_run_team_mine(daily, **kw):
            captured.update(kw)
            return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}

        mp.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare)
        mp.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_run_team_mine)

        rc = cli.main(["mine", "team", "--start", "20220101", "--end", "20231231",
                       "--structured", "--patience", "2", "--heal-rounds", "0"])
        assert rc == 0
        assert captured["structured"] is True
        assert captured["patience"] == 2
        assert captured["heal_rounds"] == 0

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cmd_mine_team_forwards_structured_patience_heal_rounds(mp)

    # -- 原 test_run_agent_mine_forwards_to_orchestrator --
    def _section_3_test_run_agent_mine_forwards_to_orchestrator(mp, tmp_path):
        from factorzen.agents.orchestrator import AgentResult
        from factorzen.agents.state import AgentState
        from factorzen.pipelines import factor_mine_agent as fma

        captured: dict = {}

        def fake_run_llm_agent(daily, llm_fn, **kw):
            captured.update(kw)
            return AgentResult(state=AgentState(seed=1), candidates=[], n_trials=0)

        mp.setattr(fma, "run_llm_agent", fake_run_llm_agent)

        fma.run_agent_mine(_mock_daily__wiring_caveats(), n_rounds=1, seed=1, llm_fn=lambda _m: "{}",
                           out_dir=str(tmp_path), patience=3, heal_rounds=1, export=False)
        assert captured["patience"] == 3
        assert captured["heal_rounds"] == 1

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_run_agent_mine_forwards_to_orchestrator(mp, _tp3)

    # -- 原 test_run_team_mine_forwards_to_orchestrator --
    def _section_4_test_run_team_mine_forwards_to_orchestrator(mp, tmp_path):
        from factorzen.agents.state import AgentState
        from factorzen.agents.team_orchestrator import TeamResult
        from factorzen.pipelines import factor_mine_team as fmt

        captured: dict = {}

        def fake_run_team_agent(daily, llm_fn, **kw):
            captured.update(kw)
            return TeamResult(state=AgentState(seed=1), candidates=[], n_trials=0)

        mp.setattr(fmt, "run_team_agent", fake_run_team_agent)

        fmt.run_team_mine(_mock_daily__wiring_caveats(), n_rounds=1, seed=1, index_path=str(tmp_path / "e.jsonl"),
                          llm_fn=lambda _m: "{}", out_dir=str(tmp_path),
                          structured=True, patience=2, heal_rounds=0, export=False)
        assert captured["structured"] is True
        assert captured["patience"] == 2
        assert captured["heal_rounds"] == 0

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_run_team_mine_forwards_to_orchestrator(mp, _tp4)


# ─────────────────────────── CLI → pipeline ───────────────────────────


# ─────────────────────────── pipeline → orchestrator ───────────────────────────


def test_team_manifest_records_new_params(monkeypatch, tmp_path):
    """可复现铁律：manifest 必须记下 structured/patience/heal_rounds，否则事后无法重跑。"""
    from factorzen.agents.state import AgentState
    from factorzen.agents.team_orchestrator import TeamResult
    from factorzen.pipelines import factor_mine_team as fmt

    monkeypatch.setattr(fmt, "run_team_agent",
                        lambda *a, **k: TeamResult(state=AgentState(seed=1),
                                                   candidates=[], n_trials=0))

    fmt.run_team_mine(_mock_daily__wiring_caveats(), n_rounds=1, seed=1,
                      index_path=str(tmp_path / "e.jsonl"), llm_fn=lambda _m: "{}",
                      out_dir=str(tmp_path), run_id="r", structured=True,
                      patience=2, heal_rounds=1, export=False)
    manifest = json.loads((tmp_path / "r" / "manifest.json").read_text())
    assert manifest["params"]["structured"] is True
    assert manifest["params"]["patience"] == 2
    assert manifest["params"]["heal_rounds"] == 1


# ─────────────────── RD-Agent 步2：任务分解真正进入流水线 ───────────────────

def test_structured_decompose_wiring_suite(tmp_path, monkeypatch):
    """structured=True：假设先经 decompose_tasks 拆成任务，每个任务各自驱动一次 Coder。；LLM 分解失败（空 tasks）→ 降级为对整条假设写表达式，不静默空转。；零回归：structured=False（默认）不得调用 decompose_tasks，不增加 LLM 调用。；实验溯源：structured 轮次的 rounds_log 要留下 task 清单。"""
    # -- 原 test_structured_decomposes_hypothesis_and_drives_coder_per_task --
    def _section_0_test_structured_decomposes_hypothesis_and_drives_coder_per_task(tmp_path, mp):
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

        mp.setattr(to, "decompose_tasks", fake_decompose)
        mp.setattr(to, "write_expressions", fake_write)

        seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                           "expected_sign": 1, "falsification": "f"}]}),
               json.dumps({"verdict": "keep", "reason": "ok"})]
        i = {"k": 0}

        def fn(_m):
            v = seq[i["k"] % len(seq)]
            i["k"] += 1
            return v

        to.run_team_agent(_mock_daily__wiring_caveats(), fn, n_rounds=1, seed=1,
                          index_path=str(tmp_path / "e.jsonl"), structured=True, heal_rounds=0)

        assert len(decompose_calls) == 1, "structured=True 必须调用 decompose_tasks（RD-Agent 步2）"
        assert len(write_calls) == 2, f"每个 task 各驱动一次 Coder，实得 {len(write_calls)}"
        assert "mom5" in write_calls[0] and "5日动量" in write_calls[0]
        assert "mom20" in write_calls[1]

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_structured_decomposes_hypothesis_and_drives_coder_per_task(_tp0, mp)

    # -- 原 test_decompose_returning_empty_falls_back_to_whole_hypothesis --
    def _section_1_test_decompose_returning_empty_falls_back_to_whole_hypothesis(tmp_path, mp):
        from factorzen.agents import team_orchestrator as to

        write_calls: list[str] = []

        mp.setattr(to, "decompose_tasks", lambda h, f: [])

        def fake_write(hypothesis, llm_fn, *, avoid=None, **_kw):
            write_calls.append(hypothesis)
            return ["ts_mean(close,5)"]

        mp.setattr(to, "write_expressions", fake_write)

        seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                           "expected_sign": 1, "falsification": "f"}]}),
               json.dumps({"verdict": "keep", "reason": "ok"})]
        i = {"k": 0}

        def fn(_m):
            v = seq[i["k"] % len(seq)]
            i["k"] += 1
            return v

        to.run_team_agent(_mock_daily__wiring_caveats(), fn, n_rounds=1, seed=1,
                          index_path=str(tmp_path / "e.jsonl"), structured=True, heal_rounds=0)

        assert len(write_calls) == 1
        assert "动量" in write_calls[0]

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_decompose_returning_empty_falls_back_to_whole_hypothesis(_tp1, mp)

    # -- 原 test_non_structured_path_does_not_decompose --
    def _section_2_test_non_structured_path_does_not_decompose(tmp_path, mp):
        from factorzen.agents import team_orchestrator as to

        calls: list[str] = []
        mp.setattr(to, "decompose_tasks",
                            lambda h, f: calls.append(h) or [])  # type: ignore[func-returns-value]

        seq = [json.dumps({"hypotheses": ["动量"]}),
               json.dumps({"expressions": ["ts_mean(close,5)"]}),
               json.dumps({"verdict": "keep", "reason": "ok"})]
        i = {"k": 0}

        def fn(_m):
            v = seq[i["k"] % len(seq)]
            i["k"] += 1
            return v

        to.run_team_agent(_mock_daily__wiring_caveats(), fn, n_rounds=1, seed=1,
                          index_path=str(tmp_path / "e.jsonl"), heal_rounds=0)

        assert calls == []

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_non_structured_path_does_not_decompose(_tp2, mp)

    # -- 原 test_rounds_log_records_tasks_for_traceability --
    def _section_3_test_rounds_log_records_tasks_for_traceability(tmp_path, mp):
        from factorzen.agents import team_orchestrator as to

        mp.setattr(to, "decompose_tasks",
                            lambda h, f: [{"name": "mom5", "description": "5日动量",
                                           "rationale": "趋势"}])
        mp.setattr(to, "write_expressions",
                            lambda h, f, *, avoid=None, **_kw: ["ts_mean(close,5)"])

        seq = [json.dumps({"hypotheses": [{"direction": "动量", "mechanism": "m",
                                           "expected_sign": 1, "falsification": "f"}]}),
               json.dumps({"verdict": "keep", "reason": "ok"})]
        i = {"k": 0}

        def fn(_m):
            v = seq[i["k"] % len(seq)]
            i["k"] += 1
            return v

        res = to.run_team_agent(_mock_daily__wiring_caveats(), fn, n_rounds=1, seed=1,
                                index_path=str(tmp_path / "e.jsonl"), structured=True, heal_rounds=0)

        assert res.rounds_log[0]["tasks"] == [
            {"name": "mom5", "description": "5日动量", "rationale": "趋势"}
        ]

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_rounds_log_records_tasks_for_traceability(_tp3, mp)


# ==== 来自 test_agent_ashare_caveats.py ====
def test_ashare_caveats_prompt_suite():
    """test_caveats_fragment_covers_key_mechanisms；test_build_agent_messages_injects_caveats；test_hypothesis_prompt_injects_caveats；test_coder_prompt_injects_caveats"""
    # -- 原 test_caveats_fragment_covers_key_mechanisms --
    def _section_0_test_caveats_fragment_covers_key_mechanisms():
        from factorzen.llm.prompt_fragments import ASHARE_CAVEATS
        for kw in ["涨跌停", "停牌", "T+1", "PIT", "换手", "风险因子"]:
            assert kw in ASHARE_CAVEATS, f"缺少 {kw}"

    _section_0_test_caveats_fragment_covers_key_mechanisms()

    # -- 原 test_build_agent_messages_injects_caveats --
    def _section_1_test_build_agent_messages_injects_caveats():
        from factorzen.llm.generation import build_agent_messages
        sys = build_agent_messages(["ts_mean"], ["close"], "", [])[0]["content"]
        assert "涨跌停" in sys and "T+1" in sys

    _section_1_test_build_agent_messages_injects_caveats()

    # -- 原 test_hypothesis_prompt_injects_caveats --
    def _section_2_test_hypothesis_prompt_injects_caveats():
        from factorzen.agents.roles.hypothesis import propose_hypotheses
        cap: dict = {}

        def fake(msgs):
            cap["m"] = msgs
            return '{"hypotheses":["x"]}'
        propose_hypotheses(fake, known_invalid=[], known_valid=[])
        assert "涨跌停" in cap["m"][0]["content"]

    _section_2_test_hypothesis_prompt_injects_caveats()

    # -- 原 test_coder_prompt_injects_caveats --
    def _section_3_test_coder_prompt_injects_caveats():
        from factorzen.agents.roles.coder import write_expressions
        cap: dict = {}

        def fake(msgs):
            cap["m"] = msgs
            return '{"expressions":["ts_mean(close,5)"]}'
        write_expressions("动量", fake)
        sys = cap["m"][0]["content"]
        assert "PIT" in sys or "涨跌停" in sys

    _section_3_test_coder_prompt_injects_caveats()


# ==== 来自 test_manifest_repro.py ====
# ==== 来自 test_agent_manifest.py ====
def test_manifest_repro_suite(tmp_path, monkeypatch):
    """test_manifest_written_with_audit_trail；没有数据窗口与 universe，manifest 就不可复现（铁律#3）。；注入 llm_fn 时不去读 env，但必须标记出来——否则读者会误以为用了 .env 里的模型。；默认路径下必须记录实际使用的 model/provider/temperature —— LLM 挖掘强依赖它们。；第二 profile（sub2api/openai）必须进 manifest，事后才能复现上游。"""
    # -- 原 test_manifest_written_with_audit_trail --
    def _section_0_test_manifest_written_with_audit_trail(tmp_path):
        s = AgentState(seed=42)
        s.attempts = [AttemptRecord(0, "h", "rank(close)", True, 0.04, True, "keep", None)]
        s.candidates = [{"expression": "rank(close)", "holdout_ic": 0.03, "dsr": 0.8}]
        res = AgentResult(state=s, candidates=s.candidates, n_trials=5)
        p = write_session_manifest(res, out_dir=str(tmp_path), run_id="t1",
                                   params={"n_rounds": 3, "seed": 42})
        m = json.loads(Path(p).read_text())
        assert m["seed"] == 42 and m["n_trials"] == 5
        assert m["params"]["n_rounds"] == 3
        assert m["attempts"][0]["expression"] == "rank(close)"   # 全程尝试可审计
        assert m["candidates"][0]["dsr"] == 0.8
        assert "git_sha" in m

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_manifest_written_with_audit_trail(_tp0)

    # -- 原 test_manifest_records_data_window --
    def _section_1_test_manifest_records_data_window():
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            run_agent_mine(_mock_daily__manifest_repro(), n_rounds=1, seed=42, out_dir=td, llm_fn=_llm(),
                           run_id="t", export=False, data_window=_WINDOW,
                           command="fz mine agent --start 20220101")
            p = json.loads((Path(td) / "t" / "manifest.json").read_text())["params"]

        assert p["start"] == "20220101"
        assert p["end"] == "20231229"
        assert p["universe"] == "csi800"
        assert p["market"] == "ashare"
        assert p["command"] == "fz mine agent --start 20220101"

    _section_1_test_manifest_records_data_window()

    # -- 原 test_manifest_records_llm_identity_when_injected --
    def _section_2_test_manifest_records_llm_identity_when_injected():
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            run_agent_mine(_mock_daily__manifest_repro(), n_rounds=1, seed=42, out_dir=td, llm_fn=_llm(),
                           run_id="t", export=False)
            p = json.loads((Path(td) / "t" / "manifest.json").read_text())["params"]

        assert p["llm"] == {"injected": True}

    _section_2_test_manifest_records_llm_identity_when_injected()

    # -- 原 test_llm_meta_reads_model_and_provider_from_config --
    def _section_3_test_llm_meta_reads_model_and_provider_from_config(mp):
        from factorzen.llm.config import LLMConfig
        from factorzen.pipelines import factor_mine_agent as mod

        fake = LLMConfig(enabled=True, base_url="https://x/v1", api_key="sk-SECRET-TOKEN-XYZ",
                         model="DeepSeek-V4-Pro", provider="DeepSeek", temperature=0.2,
                         flavor="aiping", profile=None)
        mp.setattr(mod, "load_llm_config", lambda **_kw: fake)

        meta = mod._llm_meta(None)

        assert meta["model"] == "DeepSeek-V4-Pro"
        assert meta["provider"] == "DeepSeek"
        assert meta["temperature"] == 0.2
        assert meta["flavor"] == "aiping"
        assert meta["profile"] is None
        assert "api_key" not in meta, "凭据不得进 manifest"
        assert fake.api_key not in json.dumps(meta, ensure_ascii=False)

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_llm_meta_reads_model_and_provider_from_config(mp)

    # -- 原 test_llm_meta_records_profile_and_flavor_for_openai_gateway --
    def _section_4_test_llm_meta_records_profile_and_flavor_for_openai_gateway(mp):
        from factorzen.llm.config import LLMConfig
        from factorzen.pipelines import factor_mine_agent as mod

        fake = LLMConfig(
            enabled=True,
            base_url="http://localhost:8080/v1",
            api_key="sk-PLACEHOLDER",
            model="gpt-5.4",
            flavor="openai",
            profile="sub2api",
        )
        mp.setattr(mod, "load_llm_config", lambda **_kw: fake)

        meta = mod._llm_meta(None)

        assert meta["model"] == "gpt-5.4"
        assert meta["flavor"] == "openai"
        assert meta["profile"] == "sub2api"
        assert "api_key" not in meta
        assert "sk-PLACEHOLDER" not in json.dumps(meta, ensure_ascii=False)

    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_llm_meta_records_profile_and_flavor_for_openai_gateway(mp)


# ==== 来自 test_mining_manifest_reproducibility.py ====
def _mock_daily__manifest_repro(n_stocks=40, n_days=180, seed=1):
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


def _llm():
    st = {"round": -1}

    def fn(messages):
        system = messages[0]["content"]
        if "consistent" in system:
            return json.dumps({"consistent": True, "reason": "ok"})
        if "verdict" in system:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        st["round"] += 1
        w = 4 + st["round"]
        return json.dumps({"hypothesis": f"h{w}", "expressions": [f"ts_mean(close,{w})"],
                           "rationale": "r"})
    return fn


_WINDOW = {"start": "20220101", "end": "20231229", "universe": "csi800", "market": "ashare"}


def test_default_run_id_carries_timestamp():
    """同 seed 重跑不得静默覆盖上一次的产物。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_agent_mine(_mock_daily__manifest_repro(), n_rounds=1, seed=42, out_dir=td,
                             llm_fn=_llm(), export=False)
        name = Path(res["run_dir"]).name

    assert re.fullmatch(r"\d{8}_\d{6}_agent_42_1r", name), f"run_id 应带时间戳，实得 {name}"


# test_exported_dir_is_cleared_before_write 已删除：exported/*.py 桥废除，
# agent/team 不再写 exported/，清理契约无对象。


def test_manifest_is_strict_json_even_when_pbo_is_nan():
    """`pool_pbo` 在候选<2 时返回 nan；`json.dumps` 会写出裸 `NaN`，那不是合法 JSON。

    Python 的 json.loads 宽容地接受它，但标准解析器（其它语言、jq、前端）会直接失败。
    manifest 是跨工具消费的产物，必须是严格合法的 JSON。
    """
    import tempfile

    def _strict(c):
        raise ValueError(f"非法 JSON 常量: {c}")

    with tempfile.TemporaryDirectory() as td:
        # 1 轮 → 候选必 <2 → state.pbo 为 nan
        run_agent_mine(_mock_daily__manifest_repro(), n_rounds=1, seed=42, out_dir=td, llm_fn=_llm(),
                       run_id="t", export=False)
        raw = (Path(td) / "t" / "manifest.json").read_text()

    assert "NaN" not in raw, "manifest 不得含裸 NaN（非法 JSON）"
    m = json.loads(raw, parse_constant=_strict)      # 严格模式：遇 NaN/Infinity 即抛
    assert m["pbo"] is None, "nan 应序列化为 null"


def test_dump_manifest_sanitizes_nan_nested_in_tree(tmp_path):
    """nan 不只出现在顶层 pbo：attempts[].ir_train、candidates[].dsr 都可能是 nan。"""
    from factorzen.agents.manifest import dump_manifest

    path = tmp_path / "m.json"
    dump_manifest({
        "pbo": float("nan"),
        "attempts": [{"ir_train": float("nan"), "ic_train": 0.03}],
        "candidates": [{"dsr": float("inf"), "holdout_ic": -0.05}],
    }, path)

    raw = path.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    m = json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert m["attempts"][0]["ir_train"] is None
    assert m["attempts"][0]["ic_train"] == 0.03      # 正常值不受影响
    assert m["candidates"][0]["dsr"] is None
    assert m["candidates"][0]["holdout_ic"] == -0.05


# ── CLI 接线（能力实现了不算，用户得能触达）─────────────────────────────────


def test_cli_forwards_data_window_and_command_to_agent_pipeline(monkeypatch):
    """从 CLI 最外层驱动：handler 必须把 start/end/universe 透传下去，而非喂完数据就丢弃。"""
    from factorzen.cli import main as cli

    captured: dict = {}
    monkeypatch.setattr(cli, "__name__", cli.__name__)
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily",
                        lambda *a, **k: _mock_daily__manifest_repro())
    monkeypatch.setattr("factorzen.pipelines.factor_mine_agent.run_agent_mine",
                        lambda daily, **kw: captured.update(kw) or
                        {"run_dir": "x", "n_candidates": 0, "n_trials": 0, "candidates": []})

    cli.main(["mine", "agent", "--start", "20220101", "--end", "20231229",
              "--universe", "csi800", "--iterations", "1"])

    # data_window 自带 membership 溯源三键（mock prepare 未填 out_meta → 占位 None）
    assert captured["data_window"] == {"start": "20220101", "end": "20231229",
                                       "universe": "csi800", "market": "ashare",
                                       "membership_mode": None,
                                       "membership_hash": None,
                                       "membership_n_rows": None}
    assert "mine" in captured["command"] and "agent" in captured["command"]


def test_cli_forwards_data_window_and_command_to_team_pipeline(monkeypatch):
    from factorzen.cli import main as cli

    captured: dict = {}
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily",
                        lambda *a, **k: _mock_daily__manifest_repro())
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine",
                        lambda daily, **kw: captured.update(kw) or
                        {"run_dir": "x", "n_candidates": 0, "n_trials": 0, "candidates": []})

    cli.main(["mine", "team", "--start", "20220101", "--end", "20231229",
              "--universe", "csi800", "--iterations", "1"])

    assert captured["data_window"]["universe"] == "csi800"
    assert captured["command"]

# ==== 来自 test_membership_manifest.py ====
# ── 与 test_universe_membership 对齐的假日历 / 成分 ──────────────────────
_JAN_DATES = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
_FEB_DATES = [date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 5)]
_ALL_TRADE_DATES = _JAN_DATES + _FEB_DATES
_JAN_STR = [d.strftime("%Y%m%d") for d in _JAN_DATES]
_FEB_STR = [d.strftime("%Y%m%d") for d in _FEB_DATES]


def _mock_trade_dates(start: str, end: str) -> list[date]:
    return [d for d in _ALL_TRADE_DATES if start <= d.strftime("%Y%m%d") <= end]


def _members_by_month(index_code: str, date_str: str) -> list[str]:
    ym = date_str[:6]
    if index_code == "000300.SH":
        if ym == "202401":
            return ["A.SZ", "B.SZ"]
        if ym == "202402":
            return ["B.SZ", "C.SZ"]
    return []


def _synthetic_daily_frame() -> pl.DataFrame:
    warmup = [date(2023, 12, 29)]
    days = warmup + _ALL_TRADE_DATES
    codes = ["A.SZ", "B.SZ", "C.SZ"]
    rows = []
    for c in codes:
        for d in days:
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": c,
                    "close": 10.0,
                    "close_adj": 10.0,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "vol": 1e5,
                    "amount": 1e6,
                }
            )
    return pl.DataFrame(rows)


def _patch_prepare_stack(monkeypatch, daily: pl.DataFrame, *, end_universe=None):
    """mock FactorDataContext + attach_* + calendar/members（同 test_universe_membership）。"""
    import factorzen.daily.data.context as ctx_mod
    import factorzen.pipelines.factor_mine as fm

    monkeypatch.setattr(
        "factorzen.core.calendar.get_trade_dates", _mock_trade_dates
    )
    monkeypatch.setattr(
        "factorzen.core.universe._load_index_members", _members_by_month
    )

    def _batch(index_code: str, day_strs: list[str]) -> pl.DataFrame:
        rows: list[dict[str, str]] = []
        for d in day_strs:
            for c in _members_by_month(index_code, d):
                rows.append({"trade_date": d, "ts_code": c})
        if not rows:
            return pl.DataFrame(schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8})
        return pl.DataFrame(rows)

    monkeypatch.setattr(
        "factorzen.core.universe._batch_index_membership", _batch
    )

    class _FakeCtx:
        def __init__(self, **kw):
            self.kw = kw
            _FakeCtx.last_kw = kw

        @property
        def daily(self):
            uni = self.kw.get("universe")
            df = daily
            if uni is not None:
                df = df.filter(pl.col("ts_code").is_in(list(uni)))
            return df.lazy()

        @property
        def daily_basic(self):
            return pl.DataFrame(
                {
                    "trade_date": pl.Series([], dtype=pl.Date),
                    "ts_code": pl.Series([], dtype=pl.Utf8),
                }
            ).lazy()

    _FakeCtx.last_kw = {}
    monkeypatch.setattr(ctx_mod, "FactorDataContext", _FakeCtx)
    monkeypatch.setattr(
        "factorzen.daily.data.pit.attach_fundamentals", lambda d: d
    )
    monkeypatch.setattr("factorzen.daily.data.pit.attach_holders", lambda d: d)
    monkeypatch.setattr("factorzen.daily.data.flows.attach_flows", lambda d: d)

    if end_universe is not None:
        def _fake_get_universe(date_str, universe_name="all_a"):
            return pl.DataFrame({"ts_code": end_universe})

        monkeypatch.setattr(
            "factorzen.core.universe.get_universe", _fake_get_universe
        )

    return fm, _FakeCtx


# ═══════════════════════════════════════════════════════════════════════════
# 1–4：prepare_mining_daily out_meta
# ═══════════════════════════════════════════════════════════════════════════


def test_membership_manifest_suite(monkeypatch, tmp_path):
    """out_meta 传入 + membership 成功 → mode=pit、hash 与直算一致、n_rows 正确。；membership 构造抛异常 → fail closed（不再写 asof_fallback 可入库 meta）。；universe=None → mode=None（未过滤）。；out_meta=None → 不炸、行为与任务 H 一致（回归）。；CLI 装配 prep_meta → data_window → agent manifest.params 含 membership_mode/hash。"""
    # -- 原 test_out_meta_pit_success --
    def _section_0_test_out_meta_pit_success(mp):
        from factorzen.core.universe import get_universe_membership, membership_hash

        daily = _synthetic_daily_frame()
        fm, _ = _patch_prepare_stack(mp, daily)

        out_meta: dict = {}
        out = fm.prepare_mining_daily(
            "20240102", "20240205", universe="csi300", out_meta=out_meta
        )

        assert "in_universe" in out.columns
        assert out_meta["membership_mode"] == "pit"
        assert out_meta["universe"] == "csi300"
        assert out_meta["membership_hash"] is not None

        mem = get_universe_membership("20240102", "20240205", "csi300")
        assert out_meta["membership_hash"] == membership_hash(mem)
        assert out_meta["membership_n_rows"] == mem.height
        assert out_meta["membership_n_rows"] > 0

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_out_meta_pit_success(mp)

    # -- 原 test_out_meta_membership_failure_fails_closed --
    def _section_1_test_out_meta_membership_failure_fails_closed(mp):
        daily = _synthetic_daily_frame()
        fm, _ = _patch_prepare_stack(
            mp, daily, end_universe=["B.SZ", "C.SZ"]
        )

        def _boom(*a, **k):
            raise RuntimeError("mock membership failure")

        mp.setattr(
            "factorzen.core.universe.get_universe_membership", _boom
        )

        out_meta: dict = {}
        with pytest.raises(ValueError, match=r"PIT membership|拒绝回退"):
            fm.prepare_mining_daily(
                "20240102", "20240205", universe="csi300", out_meta=out_meta
            )
        # 抛错前未写入可入库 meta（fail closed 不落 asof_fallback 产物）
        assert out_meta == {}

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_out_meta_membership_failure_fails_closed(mp)

    # -- 原 test_out_meta_universe_none --
    def _section_2_test_out_meta_universe_none(mp):
        daily = _synthetic_daily_frame()
        fm, _ = _patch_prepare_stack(mp, daily)

        out_meta: dict = {}
        out = fm.prepare_mining_daily(
            "20240102", "20240205", universe=None, out_meta=out_meta
        )

        assert "in_universe" not in out.columns
        assert out_meta["membership_mode"] is None
        assert out_meta["membership_hash"] is None
        assert out_meta.get("membership_n_rows") is None
        assert out_meta["universe"] is None

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_out_meta_universe_none(mp)

    # -- 原 test_out_meta_none_no_crash_behavior_unchanged --
    def _section_3_test_out_meta_none_no_crash_behavior_unchanged(mp):
        daily = _synthetic_daily_frame()
        n_raw = daily.height
        fm, FakeCtx = _patch_prepare_stack(mp, daily)

        out = fm.prepare_mining_daily(
            "20240102", "20240205", universe="csi300", out_meta=None
        )

        assert out.height == n_raw
        assert "in_universe" in out.columns
        assert set(FakeCtx.last_kw["universe"]) == {"A.SZ", "B.SZ", "C.SZ"}
        # 调出：A 1 月 True、2 月 False
        a = out.filter(pl.col("ts_code") == "A.SZ")
        assert a.filter(pl.col("trade_date").is_in(_JAN_DATES))["in_universe"].all()
        assert not a.filter(pl.col("trade_date").is_in(_FEB_DATES))["in_universe"].any()

    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_out_meta_none_no_crash_behavior_unchanged(mp)

    # -- 原 test_agent_manifest_includes_membership_fields --
    def _section_4_test_agent_manifest_includes_membership_fields(mp, tmp_path):
        from factorzen.cli import main as cli_main
        from factorzen.core.universe import get_universe_membership, membership_hash
        from factorzen.pipelines.factor_mine_agent import run_agent_mine

        daily = _synthetic_daily_frame()
        _patch_prepare_stack(mp, daily)

        args = SimpleNamespace(
            market="ashare",
            start="20240102",
            end="20240205",
            universe="csi300",
            symbols=None,
            top_n=50,
            freq="daily",
        )
        frame, profile, prep_meta = cli_main._prepare_agent_mining_data(args)
        assert frame is not None
        assert profile is None
        assert prep_meta["membership_mode"] == "pit"
        assert prep_meta["membership_hash"] is not None

        mem = get_universe_membership("20240102", "20240205", "csi300")
        assert prep_meta["membership_hash"] == membership_hash(mem)

        # 与 _cmd_mine_agent 同口径：data_window 并入 membership_* 后进 params
        data_window = {
            **cli_main._data_window(args),
            "membership_mode": prep_meta.get("membership_mode"),
            "membership_hash": prep_meta.get("membership_hash"),
            "membership_n_rows": prep_meta.get("membership_n_rows"),
        }

        def _llm(messages):
            system = messages[0]["content"]
            if "consistent" in system:
                return json.dumps({"consistent": True, "reason": "ok"})
            if "verdict" in system:
                return json.dumps({"verdict": "keep", "reason": "ok"})
            return json.dumps(
                {
                    "hypothesis": "h",
                    "expressions": ["ts_mean(close,5)"],
                    "rationale": "r",
                }
            )

        # 评估路径 mock 掉，避免合成帧不够评估维度
        mp.setattr(
            "factorzen.agents.orchestrator.run_llm_agent",
            lambda *a, **k: _FakeAgentResult(),
        )

        res = run_agent_mine(
            frame,
            n_rounds=1,
            seed=1,
            out_dir=str(tmp_path),
            llm_fn=_llm,
            run_id="mem_t",
            export=False,
            data_window=data_window,
            command="fz mine agent --universe csi300",
            eval_start="20240102",
        )
        man = json.loads(
            (Path(res["run_dir"]) / "manifest.json").read_text(encoding="utf-8")
        )
        params = man["params"]
        assert params["membership_mode"] == "pit"
        assert params["membership_hash"] == prep_meta["membership_hash"]
        assert params["membership_n_rows"] == prep_meta["membership_n_rows"]
        assert params["universe"] == "csi300"
        assert params["start"] == "20240102"

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_agent_manifest_includes_membership_fields(mp, _tp4)


# ═══════════════════════════════════════════════════════════════════════════
# 5：agent 路径 manifest 端到端（mock 到 manifest 组装层）
# ═══════════════════════════════════════════════════════════════════════════


class _FakeAgentResult:
    """最小 AgentResult 桩：让 write_session_manifest 能落盘。"""

    def __init__(self):
        from factorzen.agents.state import AgentState

        self.state = AgentState(seed=1)
        self.candidates: list = []
        self.n_trials = 0
        self.sharpe_variance = 0.0


def test_run_mine_patches_session_manifest(monkeypatch, tmp_path):
    """run_mine 在 run_session 无注入口时，读-补-写 session manifest 的 membership_*。"""
    import factorzen.pipelines.factor_mine as fm

    daily = _synthetic_daily_frame()
    _patch_prepare_stack(monkeypatch, daily)

    session_dir = tmp_path / "session_1_random"
    session_dir.mkdir(parents=True)
    # run_session 写的骨架 manifest（无 membership）
    (session_dir / "manifest.json").write_text(
        json.dumps({"seed": 1, "method": "random", "n_trials": 0}, indent=2),
        encoding="utf-8",
    )

    def _fake_run_session(frame, **kw):
        return {
            "candidates": [],
            "session_dir": str(session_dir),
            "n_trials": 0,
            "n_scored": 0,
        }

    monkeypatch.setattr(fm, "run_session", _fake_run_session)

    result = fm.run_mine(
        start="20240102", end="20240205", universe="csi300", n_trials=1
    )
    assert result["session_dir"] == str(session_dir)

    man = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert man["membership_mode"] == "pit"
    assert man["membership_hash"] is not None
    assert man["membership_n_rows"] is not None
    assert man["universe"] == "csi300"
    # 原有字段保留
    assert man["seed"] == 1
    assert man["method"] == "random"

# ==== 来自 test_agent_state.py ====

def test_agent_state_defaults_and_serializable():
    import json
    s = AgentState(seed=42)
    assert s.iteration == 0 and s.attempts == [] and s.candidates == []
    assert s.seen_expressions == set() and s.negative_examples == []
    s.attempts.append(AttemptRecord(iteration=0, hypothesis="h", expression="rank(close)",
                                    compile_ok=True, ic_train=0.05, passed_guardrails=True,
                                    critic_verdict="keep", error=None))
    s.seen_expressions.add("rank(close)")
    d = s.to_dict()  # set 转 list，dataclass 转 dict
    assert json.dumps(d)  # 不抛 = JSON 可序列化
    assert d["attempts"][0]["expression"] == "rank(close)"
    assert "rank(close)" in d["seen_expressions"]
