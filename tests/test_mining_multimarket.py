"""Phase 1：M5/M6 LLM 挖掘多市场化（crypto）+ A 股逐字节零回归。

覆盖：
- 1.1 Prompt 市场化：market_caveats / signal_families / build_agent_messages / coder 语法 prompt
  在 crypto 下含 crypto 约束+叶子、不含 A 股专有叶子；A 股默认路径与改前 golden 逐字节相同。
- 1.2 生成与评估层吃 profile：AgentContext.from_profile；evaluation 三入口 profile-gating；
  parse_expr/warmup/评估三路径对 crypto 表达式 leaf_map 一致可解析。
- 1.3 experiment_index 按 market 分族。
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

_GOLDEN = json.loads((Path(__file__).parent / "golden_ashare_prompts.json").read_text())


# ── 1.1 A 股逐字节零回归（golden 对照，改前捕获） ──────────────────────────────
def test_build_agent_messages_ashare_byte_identical():
    from factorzen.llm.generation import build_agent_messages
    m = build_agent_messages(["ts_mean", "ts_std"], ["close", "vol"], "FB", ["neg1"])
    assert m[0]["content"] == _GOLDEN["bam_sys"]
    assert m[1]["content"] == _GOLDEN["bam_user"]
    # market="ashare" 显式 == 默认（证明 default 分支等价）
    m2 = build_agent_messages(["ts_mean", "ts_std"], ["close", "vol"], "FB", ["neg1"],
                              market="ashare")
    assert m2[0]["content"] == _GOLDEN["bam_sys"]


def test_build_agent_messages_ashare_budget_byte_identical():
    from factorzen.llm.generation import build_agent_messages
    m = build_agent_messages(["ts_mean"], ["close"], "", [],
                             leaf_budgets={"north_ratio": 238})
    assert m[0]["content"] == _GOLDEN["bam_budget_sys"]


def test_coder_syntax_prompt_ashare_byte_identical():
    from factorzen.agents.roles.coder import _syntax_prompt
    assert _syntax_prompt() == _GOLDEN["coder_syntax"]
    assert _syntax_prompt({"north_ratio": 238}) == _GOLDEN["coder_syntax_budget"]
    # 显式 market/leaf_names=None 亦等价
    assert _syntax_prompt(market="ashare", leaf_names=None) == _GOLDEN["coder_syntax"]


def test_hypothesis_prompts_ashare_byte_identical():
    from factorzen.agents.roles.hypothesis import propose_hypotheses, propose_structured
    cap: dict = {}

    def fake(msgs):
        cap["sys"] = msgs[0]["content"]
        cap["user"] = msgs[1]["content"]
        return '{"hypotheses":["x"]}'
    propose_hypotheses(fake, known_invalid=["a"], known_valid=["b"], feedback="fb", n=2)
    assert cap["sys"] == _GOLDEN["hyp_sys"]
    assert cap["user"] == _GOLDEN["hyp_user"]

    def fake2(msgs):
        cap["s2"] = msgs[0]["content"]
        return '{"hypotheses":[{"direction":"d"}]}'
    propose_structured(fake2, known_invalid=[], known_valid=[])
    assert cap["s2"] == _GOLDEN["struct_sys"]


def test_signal_families_ashare_byte_identical():
    from factorzen.agents.roles.hypothesis import signal_families
    assert signal_families() == _GOLDEN["signal_families"]
    assert signal_families("ashare") == _GOLDEN["signal_families"]


# ── 1.1 crypto prompt 市场化 ───────────────────────────────────────────────────
def test_market_caveats_crypto_vs_ashare():
    from factorzen.llm.prompt_fragments import ASHARE_CAVEATS, market_caveats
    cr = market_caveats("crypto")
    for kw in ["funding", "open_interest", "taker_buy_ratio", "T+0", "24/7", "PIT"]:
        assert kw in cr, f"crypto caveats 缺 {kw}"
    # crypto caveats 自包含：不引用 A 股规则口径（T+1/停牌），也不广告 A 股专有叶子
    assert "T+1" not in cr and "north_ratio" not in cr and "roe" not in cr
    assert market_caveats("ashare") == ASHARE_CAVEATS  # ashare 逐字节
    # 未知市场 → 通用兜底（含 PIT），不抛
    assert "PIT" in market_caveats("does_not_exist")


def test_build_agent_messages_crypto_leaves_and_caveats():
    from factorzen.llm.generation import build_agent_messages
    sys = build_agent_messages(
        ["ts_mean"], ["close", "funding_rate", "open_interest", "taker_buy_ratio"],
        market="crypto")[0]["content"]
    assert "funding" in sys and "open_interest" in sys and "T+0" in sys
    # A 股专有叶子不得泄漏进 crypto prompt（不广告不存在的叶子——能力层↔接线层漂移）
    assert "north_ratio" not in sys and "roe" not in sys and "T+1" not in sys


def test_coder_syntax_prompt_crypto():
    from factorzen.agents.roles.coder import _syntax_prompt
    sys = _syntax_prompt(market="crypto",
                         leaf_names=["close", "funding_rate", "open_interest"])
    assert "funding_rate" in sys and "T+0" in sys
    assert "north_ratio" not in sys and "roe" not in sys


def test_signal_families_crypto():
    from factorzen.agents.roles.hypothesis import signal_families
    fam = signal_families("crypto")
    assert "funding" in fam or "资金费率" in fam
    assert "open_interest" in fam or "持仓量" in fam
    assert "北向" not in fam and "roe" not in fam


def test_propose_structured_crypto_injects_crypto_market():
    from factorzen.agents.roles.hypothesis import propose_structured
    cap: dict = {}

    def fake(msgs):
        cap["sys"] = msgs[0]["content"]
        return '{"hypotheses":[{"direction":"d"}]}'
    propose_structured(fake, known_invalid=[], known_valid=[], market="crypto")
    assert "funding" in cap["sys"] and "T+0" in cap["sys"]
    assert "涨跌停" not in cap["sys"]


# ── 1.2 生成/评估层吃 profile ──────────────────────────────────────────────────
class _CryptoProfileStub:
    """轻量 crypto profile：evaluation/AgentContext 只用 .name + .factors。"""
    name = "crypto"

    def __init__(self):
        from factorzen.markets.crypto.factors import CryptoFactorSet
        self.factors = CryptoFactorSet()


def _crypto_daily(n_syms: int = 40, n_days: int = 90) -> pl.DataFrame:
    """合成 crypto 挖掘帧：多标的截面 + funding_rate/open_interest 叶子（≥30 只满足 IC 截面门）。"""
    import datetime as dt

    import numpy as np
    rng = np.random.default_rng(7)
    base = dt.date(2024, 1, 1)
    rows = []
    for s in range(n_syms):
        price = 100.0 + s * 10
        for d in range(n_days):
            price *= 1.0 + float(rng.normal(0, 0.02))
            vol = float(rng.uniform(1e3, 1e5))
            rows.append({
                "ts_code": f"SYM{s}USDT",
                "trade_date": base + dt.timedelta(days=d),
                "open": price * 0.99, "high": price * 1.01, "low": price * 0.98,
                "close": price, "vol": vol, "amount": price * vol,
                "funding_rate": float(rng.normal(0.0001, 0.0002)),
                "open_interest": float(rng.uniform(1e6, 1e7)),
                "taker_buy_volume": vol * float(rng.uniform(0.4, 0.6)),
            })
    return pl.DataFrame(rows)


def test_agent_context_from_profile_crypto_vs_default():
    from factorzen.agents.nodes import AgentContext
    from factorzen.discovery.operators import LEAF_FEATURES, OPERATORS
    # 默认（None）= A 股，零回归
    d = AgentContext.from_profile(None)
    assert d.market == "ashare" and d.leaf_map is None
    assert d.leaf_names == list(LEAF_FEATURES.keys())
    assert d.op_names == list(OPERATORS.keys())
    # crypto
    c = AgentContext.from_profile(_CryptoProfileStub())
    assert c.market == "crypto"
    assert "funding_rate" in c.leaf_names and "open_interest" in c.leaf_names
    assert c.leaf_map is not None and c.leaf_map["funding_rate"] == "funding_rate"
    # 算子集市场无关
    assert c.op_names == list(OPERATORS.keys())


def test_evaluate_expressions_crypto_profile_parses_funding():
    from factorzen.agents.evaluation import evaluate_expressions
    from factorzen.discovery.scoring import DataBundle
    daily = _crypto_daily()
    bundle = DataBundle.build(daily)
    prof = _CryptoProfileStub()
    # crypto profile 下 funding_rate 表达式可解析可求值
    res = evaluate_expressions(["ts_mean(funding_rate, 5)", "ts_zscore(open_interest, 10)"],
                               daily, bundle, profile=prof)
    assert all(r["compile_ok"] for r in res), res
    assert any(r["ic_train"] is not None for r in res)
    # 无 profile（A 股默认）→ funding_rate 未知叶子 → compile_ok False（零回归的排斥面）
    res_a = evaluate_expressions(["ts_mean(funding_rate, 5)"], daily, bundle)
    assert res_a[0]["compile_ok"] is False


def test_crypto_leaf_map_parity_across_parse_warmup_eval():
    """crypto 表达式在评估/预热门/预算三条路径 leaf_map 一致（parse_expr / warmup_shortfall /
    leaf_warmup_budgets 都吃同一 crypto leaf_map）。"""
    import datetime as dt

    from factorzen.agents.evaluation import _preprocess_daily
    from factorzen.discovery.expression import (
        leaf_warmup_budgets,
        parse_expr,
        warmup_shortfall,
    )
    prof = _CryptoProfileStub()
    leaf_map = prof.factors.leaf_features()
    daily = _crypto_daily(n_days=60)
    prepped = _preprocess_daily(daily, prof)
    eval_start = dt.date(2024, 2, 1)
    # parse
    node = parse_expr("ts_mean(funding_rate, 20)", leaf_map)
    # 预热门 have 与预算表逐值一致（同 leaf_map）
    budgets = leaf_warmup_budgets(prepped, eval_start, ["funding_rate"], leaf_map=leaf_map)
    from factorzen.discovery.expression import warmup_bars_by_leaf
    have = warmup_bars_by_leaf(node, prepped, eval_start, leaf_map)["funding_rate"]
    assert have == budgets["funding_rate"]
    # warmup_shortfall 用同一 leaf_map（窗口 20 < have → 不欠预热）
    assert warmup_shortfall(node, prepped, eval_start, leaf_map) is None


def test_make_health_check_crypto_profile_funding_healthy():
    from factorzen.agents.evaluation import make_health_check
    prof = _CryptoProfileStub()
    daily = _crypto_daily(n_days=60)
    check = make_health_check(daily, profile=prof, leaf_map=prof.factors.leaf_features())
    # crypto 叶子表达式健康（None），不被误判解析失败
    assert check("ts_mean(funding_rate, 10)") is None
    # A 股默认（无 leaf_map）→ funding_rate 判解析失败
    check_a = make_health_check(daily)
    assert check_a("ts_mean(funding_rate, 10)") is not None


# ── 1.3 experiment_index 按 market 分族（A 股 known_invalid 不得泄漏进 crypto recall） ──
def test_experiment_index_scoped_by_market(tmp_path):
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import recall
    idx = ExperimentIndex(str(tmp_path / "idx.jsonl"))
    aw = {"start": "20240101", "end": "20241231", "universe": "csi800", "market": "ashare"}
    cw = {"start": "20240101", "end": "20241231", "universe": None, "market": "crypto"}
    # A 股一条「已验证无效」记录 + crypto 一条「已验证有效」记录
    idx.append([
        {"expression": "ts_mean(north_ratio, 20)", "hypothesis": "h", "ic_train": 0.001,
         "ir_train": 0.01, "n_train": 100, "passed": False, "verdict": None,
         "decorrelated": False, "compile_ok": True, "error": None, "data_window": aw,
         "run_id": "a"},
        {"expression": "ts_mean(funding_rate, 20)", "hypothesis": "h", "ic_train": 0.05,
         "ir_train": 0.3, "n_train": 100, "passed": True, "verdict": "keep", "holdout_ic": 0.04,
         "decorrelated": False, "compile_ok": True, "error": None, "data_window": cw,
         "run_id": "c"},
    ])
    a_rec = recall(idx, data_window=aw)
    c_rec = recall(idx, data_window=cw)
    # A 股的 north_ratio 负例不得出现在 crypto 族的任何召回里
    assert any("north_ratio" in e for e in a_rec.known_invalid)
    assert all("north_ratio" not in e for e in c_rec.known_invalid + c_rec.known_valid)
    assert all("north_ratio" not in e for e in c_rec.seen)
    # crypto 的有效因子只在 crypto 族可见
    assert any("funding_rate" in e for e in c_rec.known_valid)
    assert all("funding_rate" not in e for e in a_rec.known_valid + a_rec.known_invalid)


# ── 裸 JSON 数组容错（crypto smoke 实测：DeepSeek 常返回顶层数组而非包装对象，
#    旧解析直接丢整轮假设——4/6 与 4/4 轮「Hypothesis 未产出假设」的根因） ──────────
_BARE_STRUCT_ARRAY = (
    '[\n  {"direction": "d1", "mechanism": "m1", "expected_sign": 1, "falsification": "f1"},\n'
    '  {"direction": "d2", "mechanism": "m2", "expected_sign": -1, "falsification": "f2"}\n]'
)


def test_propose_structured_accepts_bare_json_array():
    from factorzen.agents.roles.hypothesis import propose_structured
    out = propose_structured(lambda _m: _BARE_STRUCT_ARRAY,
                             known_invalid=[], known_valid=[], n=2, market="crypto")
    assert [h["direction"] for h in out] == ["d1", "d2"]


def test_propose_structured_accepts_fenced_bare_array():
    from factorzen.agents.roles.hypothesis import propose_structured
    out = propose_structured(lambda _m: "```json\n" + _BARE_STRUCT_ARRAY + "\n```",
                             known_invalid=[], known_valid=[])
    assert len(out) == 2 and out[1]["expected_sign"] == -1


def test_propose_hypotheses_accepts_bare_string_array():
    from factorzen.agents.roles.hypothesis import propose_hypotheses
    out = propose_hypotheses(lambda _m: '["方向1", "方向2"]',
                             known_invalid=[], known_valid=[], n=2)
    assert out == ["方向1", "方向2"]


def test_write_expressions_accepts_bare_string_array():
    from factorzen.agents.roles.coder import write_expressions
    out = write_expressions("h", lambda _m: '["ts_mean(close,5)", "rank(vol)"]')
    assert out == ["ts_mean(close,5)", "rank(vol)"]


def test_decompose_tasks_accepts_bare_dict_array():
    from factorzen.agents.roles.coder import decompose_tasks
    raw = '[{"name": "n1", "description": "d1", "rationale": "r1"}]'
    out = decompose_tasks("h", lambda _m: raw)
    assert out == [{"name": "n1", "description": "d1", "rationale": "r1"}]


def test_wrapped_object_still_wins_over_array_fallback():
    """包装对象路径零回归：正常 {"hypotheses": [...]} 响应不受数组回退影响。"""
    from factorzen.agents.roles.hypothesis import propose_hypotheses
    out = propose_hypotheses(lambda _m: '{"hypotheses": ["a"]}',
                             known_invalid=[], known_valid=[])
    assert out == ["a"]


# ── 1.3 CLI 接线：--market crypto 装配帧 + 透传 profile；ashare 保持 profile=None ──
def _team_args(**over):
    import argparse
    base = dict(start="20240301", end="20241231", universe=None, market="ashare",
                symbols=None, top_n=50, iterations=2, top_k=5, seed=42,
                index_path="/tmp/e.jsonl", structured=True, patience=None, heal_rounds=0,
                hypotheses_per_round=1, freq="daily", command_line="mine team")
    base.update(over)
    return argparse.Namespace(**base)


def test_cmd_mine_team_ashare_passes_profile_none(monkeypatch):
    from factorzen.cli import main as cli
    cap: dict = {}
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily",
                        lambda start, end, universe=None, lookback_days=None, **kw: _mock_ashare_daily())

    def fake_team_mine(daily, **kw):
        cap.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_team_mine)
    rc = cli._cmd_mine_team(_team_args(market="ashare"))
    assert rc == 0
    assert cap["profile"] is None            # A 股零回归：不带 profile
    assert cap["eval_start"] == "20240301"


def test_cmd_mine_team_crypto_assembles_and_threads_profile(monkeypatch):
    from factorzen.cli import main as cli
    cap: dict = {}
    fake_profile = _CryptoProfileStub()
    fake_profile.base_freq = "daily"
    fake_profile.provider = object()

    monkeypatch.setattr("factorzen.markets.crypto.profile.build_crypto_profile",
                        lambda **_k: fake_profile)

    def fake_build(provider, symbols, start, end, freq):
        cap["build"] = dict(symbols=symbols, start=start, end=end, freq=freq)
        return _crypto_daily(n_days=40)
    monkeypatch.setattr("factorzen.markets.crypto.mining.build_crypto_daily", fake_build)

    def fake_team_mine(daily, **kw):
        cap.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_team_mine)

    rc = cli._cmd_mine_team(_team_args(market="crypto", symbols="BTCUSDT,ETHUSDT"))
    assert rc == 0
    # profile 透传（非 None）+ eval_start=挖掘窗口 start（预热前缀边界）
    assert cap["profile"] is fake_profile
    assert cap["eval_start"] == "20240301"
    # 预热前缀：build_crypto_daily 的 start 明显早于挖掘窗口 start（AGENT_WARMUP_LOOKBACK 自然日）
    assert cap["build"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert cap["build"]["start"] < "20240301"
    # data_window.market 如实记录 crypto
    assert cap["data_window"]["market"] == "crypto"


def _mock_ashare_daily() -> pl.DataFrame:
    import datetime as dt
    rows = []
    base = dt.date(2024, 1, 1)
    for s in range(35):
        for d in range(40):
            rows.append({"ts_code": f"{s:06d}.SZ", "trade_date": base + dt.timedelta(days=d),
                         "close": 10.0 + d, "open": 10.0, "high": 11.0, "low": 9.0,
                         "vol": 1e5, "amount": 1e6})
    return pl.DataFrame(rows)


# ── Phase 3 US CLI 接线：--market us 装配后复权帧 + 透传 profile（价量族，无 A 股叶子泄漏） ──
class _USProfileStub:
    name = "us"

    def __init__(self):
        from factorzen.markets.us.factors import USFactorSet
        self.factors = USFactorSet()


def _us_daily(n_syms: int = 35, n_days: int = 40) -> pl.DataFrame:
    import datetime as dt
    base = dt.date(2024, 1, 1)
    rows = []
    for s in range(n_syms):
        for d in range(n_days):
            rows.append({"ts_code": f"US{s:03d}", "trade_date": base + dt.timedelta(days=d),
                         "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0 + d,
                         "vol": 1e5, "amount": 1e6})
    return pl.DataFrame(rows)


def test_cmd_mine_team_us_assembles_and_threads_profile(monkeypatch):
    from factorzen.cli import main as cli
    cap: dict = {}
    fake_profile = _USProfileStub()
    fake_profile.base_freq = "daily"
    fake_profile.provider = object()

    class _U:
        def snapshot(self, d):
            return ["AAPL", "MSFT"]
    fake_profile.universe = _U()
    monkeypatch.setattr("factorzen.markets.us.profile.build_us_profile", lambda **_k: fake_profile)

    def fake_build(provider, symbols, start, end, freq="daily"):
        cap["build"] = dict(symbols=symbols, start=start, end=end, freq=freq)
        return _us_daily()
    monkeypatch.setattr("factorzen.markets.us.mining.build_us_daily", fake_build)

    def fake_team_mine(daily, **kw):
        cap.update(kw)
        return {"n_candidates": 0, "n_trials": 0, "run_dir": "x"}
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine", fake_team_mine)

    rc = cli._cmd_mine_team(_team_args(market="us", symbols=None))
    assert rc == 0
    assert cap["profile"] is fake_profile          # profile 透传（非 None）
    assert cap["eval_start"] == "20240301"          # eval_start=挖掘窗口 start（预热边界）
    assert cap["build"]["symbols"] == ["AAPL", "MSFT"]  # 缺 --symbols → universe 静态快照
    assert cap["build"]["start"] < "20240301"       # 预热前缀：早于挖掘窗口 start
    assert cap["data_window"]["market"] == "us"     # manifest 如实记录 us


def test_us_leaf_map_has_no_ashare_leaves():
    # us 叶子仅价量族，A 股专有叶子（north_ratio/roe/net_mf_amount）零泄漏
    from factorzen.markets.us.factors import USFactorSet
    leaves = set(USFactorSet().leaf_features())
    assert {"north_ratio", "roe", "net_mf_amount", "funding_rate", "oi"}.isdisjoint(leaves)
    assert {"close", "vwap", "log_vol", "ret_1d", "amount"}.issubset(leaves)
