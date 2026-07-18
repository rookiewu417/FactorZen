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
- （历史）`exported/` 陈旧残留——Batch 2 已废除 exported/*.py 桥，相关断言删除。
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
                     model="DeepSeek-V4-Pro", provider="DeepSeek", temperature=0.2,
                     flavor="aiping", profile=None)
    monkeypatch.setattr(mod, "load_llm_config", lambda **_kw: fake)

    meta = mod._llm_meta(None)

    assert meta["model"] == "DeepSeek-V4-Pro"
    assert meta["provider"] == "DeepSeek"
    assert meta["temperature"] == 0.2
    assert meta["flavor"] == "aiping"
    assert meta["profile"] is None
    assert "api_key" not in meta, "凭据不得进 manifest"
    assert fake.api_key not in json.dumps(meta, ensure_ascii=False)


def test_llm_meta_records_profile_and_flavor_for_openai_gateway(monkeypatch):
    """第二 profile（sub2api/openai）必须进 manifest，事后才能复现上游。"""
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
    monkeypatch.setattr(mod, "load_llm_config", lambda **_kw: fake)

    meta = mod._llm_meta(None)

    assert meta["model"] == "gpt-5.4"
    assert meta["flavor"] == "openai"
    assert meta["profile"] == "sub2api"
    assert "api_key" not in meta
    assert "sk-PLACEHOLDER" not in json.dumps(meta, ensure_ascii=False)


def test_default_run_id_carries_timestamp():
    """同 seed 重跑不得静默覆盖上一次的产物。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_agent_mine(_mock_daily(), n_rounds=1, seed=42, out_dir=td,
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
        run_agent_mine(_mock_daily(), n_rounds=1, seed=42, out_dir=td, llm_fn=_llm(),
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
                        lambda *a, **k: _mock_daily())
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
                        lambda *a, **k: _mock_daily())
    monkeypatch.setattr("factorzen.pipelines.factor_mine_team.run_team_mine",
                        lambda daily, **kw: captured.update(kw) or
                        {"run_dir": "x", "n_candidates": 0, "n_trials": 0, "candidates": []})

    cli.main(["mine", "team", "--start", "20220101", "--end", "20231229",
              "--universe", "csi800", "--iterations", "1"])

    assert captured["data_window"]["universe"] == "csi800"
    assert captured["command"]
