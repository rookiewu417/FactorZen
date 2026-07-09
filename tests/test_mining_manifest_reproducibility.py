# tests/test_mining_manifest_reproducibility.py
"""P1: manifest 可复现（铁律#3）——「事后拿 manifest 能重跑出同样结果」。

修复前 `params` 只有 `{n_rounds, seed, top_k, holdout_ratio, patience, heal_rounds}`：

- **缺 start / end / universe / market**。CLI handler 明明拿到了 `args.start/end/universe`，
  喂给 `prepare_mining_daily` 之后**直接丢弃**——事后无从得知这批因子挖自哪段数据、哪个票池。
- **缺 LLM model / provider / temperature**。LLM 挖掘的结果强依赖模型，没有它们 manifest
  根本不可复现；也无法证明某批因子是哪个模型挖的。
- **缺 command**。
- `run_id = f"agent_{seed}_{n_rounds}r"` 不含时间戳也不含数据窗口 → 同 seed 重跑**静默覆盖**
  上一次的 manifest.json / candidates.csv。
- `exported/` 里上次 run 的多余因子文件**残留**，被下游消费。
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.pipelines.factor_mine_agent import run_agent_mine


def _mock_daily(n_stocks=40, n_days=180, seed=1):
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


def test_manifest_records_data_window():
    """没有数据窗口与 universe，manifest 就不可复现（铁律#3）。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_agent_mine(_mock_daily(), n_rounds=1, seed=42, out_dir=td, llm_fn=_llm(),
                       run_id="t", export=False, data_window=_WINDOW,
                       command="fz mine agent --start 20220101")
        p = json.loads((Path(td) / "t" / "manifest.json").read_text())["params"]

    assert p["start"] == "20220101"
    assert p["end"] == "20231229"
    assert p["universe"] == "csi800"
    assert p["market"] == "ashare"
    assert p["command"] == "fz mine agent --start 20220101"


def test_manifest_records_llm_identity_when_injected():
    """注入 llm_fn 时不去读 env，但必须标记出来——否则读者会误以为用了 .env 里的模型。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_agent_mine(_mock_daily(), n_rounds=1, seed=42, out_dir=td, llm_fn=_llm(),
                       run_id="t", export=False)
        p = json.loads((Path(td) / "t" / "manifest.json").read_text())["params"]

    assert p["llm"] == {"injected": True}


def test_llm_meta_reads_model_and_provider_from_config(monkeypatch):
    """默认路径下必须记录实际使用的 model/provider/temperature —— LLM 挖掘强依赖它们。"""
    from factorzen.llm.config import LLMConfig
    from factorzen.pipelines import factor_mine_agent as mod

    fake = LLMConfig(enabled=True, base_url="https://x/v1", api_key="sk-SECRET-TOKEN-XYZ",
                     model="DeepSeek-V4-Pro", provider="DeepSeek", temperature=0.2)
    monkeypatch.setattr(mod, "load_llm_config", lambda **_kw: fake)

    meta = mod._llm_meta(None)

    assert meta["model"] == "DeepSeek-V4-Pro"
    assert meta["provider"] == "DeepSeek"
    assert meta["temperature"] == 0.2
    assert "api_key" not in meta, "凭据不得进 manifest"
    assert fake.api_key not in json.dumps(meta, ensure_ascii=False)


def test_default_run_id_carries_timestamp():
    """同 seed 重跑不得静默覆盖上一次的产物。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_agent_mine(_mock_daily(), n_rounds=1, seed=42, out_dir=td,
                             llm_fn=_llm(), export=False)
        name = Path(res["run_dir"]).name

    assert re.fullmatch(r"agent_42_1r_\d{8}_\d{6}", name), f"run_id 应带时间戳，实得 {name}"


def test_exported_dir_is_cleared_before_write():
    """上次 run 的陈旧因子文件必须清掉，否则下游会消费到它们。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        stale = Path(td) / "t" / "exported"
        stale.mkdir(parents=True)
        (stale / "agent_t_99.py").write_text("# 上一次 run 的残留（本次只产出 0..N-1）")

        run_agent_mine(_mock_daily(), n_rounds=1, seed=42, out_dir=td, llm_fn=_llm(),
                       run_id="t", export=True)

        assert not (stale / "agent_t_99.py").exists(), "exported/ 写前应清理陈旧因子"


# ── CLI 接线（能力实现了不算，用户得能触达）─────────────────────────────────


def test_cli_forwards_data_window_and_command_to_agent_pipeline(monkeypatch):
    """从 CLI 最外层驱动：handler 必须把 start/end/universe 透传下去，而非喂完数据就丢弃。"""
    from factorzen.cli import main as cli

    captured: dict = {}
    monkeypatch.setattr(cli, "__name__", cli.__name__)
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily",
                        lambda *a, **k: _mock_daily())
    monkeypatch.setattr("factorzen.pipelines.factor_mine_agent.run_agent_mine",
                        lambda daily, **kw: captured.update(kw) or
                        {"run_dir": "x", "n_candidates": 0, "n_trials": 0, "candidates": []})

    cli.main(["mine", "agent", "--start", "20220101", "--end", "20231229",
              "--universe", "csi800", "--iterations", "1"])

    assert captured["data_window"] == {"start": "20220101", "end": "20231229",
                                       "universe": "csi800", "market": "ashare"}
    assert "mine" in captured["command"] and "agent" in captured["command"]


def test_cli_forwards_data_window_and_command_to_team_pipeline(monkeypatch):
    from factorzen.cli import main as cli

    captured: dict = {}
    monkeypatch.setattr("factorzen.pipelines.factor_mine.prepare_mining_daily",
                        lambda *a, **k: _mock_daily())
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine",
                        lambda daily, **kw: captured.update(kw) or
                        {"run_dir": "x", "n_candidates": 0, "n_trials": 0, "candidates": []})

    cli.main(["mine", "team", "--start", "20220101", "--end", "20231229",
              "--universe", "csi800", "--iterations", "1"])

    assert captured["data_window"]["universe"] == "csi800"
    assert captured["command"]
