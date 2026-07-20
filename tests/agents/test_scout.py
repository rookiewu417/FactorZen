"""B-W2：Feature Scout 角色 + scout_support 编排 + team/agent 接线（全离线 mock）。"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.agents.nodes import AgentContext
from factorzen.agents.roles.feature_scout import propose_intraday_features
from factorzen.agents.scout_support import (
    ScoutState,
    promote_admitted_exprs,
    run_scout_round,
)
from factorzen.discovery.intraday_expr import make_expr_spec

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


def test_propose_valid_json_array():
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


def test_propose_malformed_returns_empty():
    def llm_fn(messages):
        return "这不是 JSON 也不是数组"

    assert propose_intraday_features(llm_fn, k=3, avoid=[], known_features="") == []


def test_propose_mixed_skips_non_dict_and_caps_k():
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


def test_propose_llm_raises_returns_empty():
    def llm_fn(messages):
        raise RuntimeError("network")

    assert propose_intraday_features(llm_fn, k=1, avoid=[], known_features="") == []


def test_propose_wrapped_features_key():
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


# ── run_scout_round ──────────────────────────────────────────────────────


def test_run_scout_round_injects_and_audits(monkeypatch):
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

    monkeypatch.setattr(
        "factorzen.agents.scout_support.materialize_expr_features", fake_mat,
    )
    monkeypatch.setattr(
        "factorzen.agents.scout_support.screen_expr_panel", fake_screen,
    )
    # 跳过 leaf_health 死叶（合成帧覆盖可能为 0）
    monkeypatch.setattr(
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


def test_run_scout_round_max_leaves_skips_llm(monkeypatch):
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


def test_promote_only_referenced(tmp_path: Path, monkeypatch):
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

    monkeypatch.setattr(
        "factorzen.agents.scout_support.register_expr_features", fake_reg,
    )
    monkeypatch.setattr(
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


def test_team_flag_off_no_scout_block(tmp_path: Path):
    """flag-off：不建 ScoutState，result.intraday_scout 为 None（行为与改前一致）。"""
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



def test_cli_parser_intraday_scout_flags():
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


def test_cli_intraday_scout_non_ashare_returns_2(monkeypatch, capsys):
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


def test_cli_intraday_scout_implies_leaves(monkeypatch, capsys):
    """--intraday-scout 隐含 intraday_leaves=True 再进 prepare。"""
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

    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine.prepare_mining_daily", fake_prepare,
    )
    monkeypatch.setattr(
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
