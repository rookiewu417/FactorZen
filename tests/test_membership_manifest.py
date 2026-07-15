"""任务 J：manifest membership 溯源接线（收任务 H 的钩子）。

全 mock 离线；fixture 手法复用 test_universe_membership。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

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


def test_out_meta_pit_success(monkeypatch):
    """out_meta 传入 + membership 成功 → mode=pit、hash 与直算一致、n_rows 正确。"""
    from factorzen.core.universe import get_universe_membership, membership_hash

    daily = _synthetic_daily_frame()
    fm, _ = _patch_prepare_stack(monkeypatch, daily)

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


def test_out_meta_membership_failure_fails_closed(monkeypatch):
    """membership 构造抛异常 → fail closed（不再写 asof_fallback 可入库 meta）。"""
    daily = _synthetic_daily_frame()
    fm, _ = _patch_prepare_stack(
        monkeypatch, daily, end_universe=["B.SZ", "C.SZ"]
    )

    def _boom(*a, **k):
        raise RuntimeError("mock membership failure")

    monkeypatch.setattr(
        "factorzen.core.universe.get_universe_membership", _boom
    )

    out_meta: dict = {}
    with pytest.raises(ValueError, match=r"PIT membership|拒绝回退"):
        fm.prepare_mining_daily(
            "20240102", "20240205", universe="csi300", out_meta=out_meta
        )
    # 抛错前未写入可入库 meta（fail closed 不落 asof_fallback 产物）
    assert out_meta == {}


def test_out_meta_universe_none(monkeypatch):
    """universe=None → mode=None（未过滤）。"""
    daily = _synthetic_daily_frame()
    fm, _ = _patch_prepare_stack(monkeypatch, daily)

    out_meta: dict = {}
    out = fm.prepare_mining_daily(
        "20240102", "20240205", universe=None, out_meta=out_meta
    )

    assert "in_universe" not in out.columns
    assert out_meta["membership_mode"] is None
    assert out_meta["membership_hash"] is None
    assert out_meta.get("membership_n_rows") is None
    assert out_meta["universe"] is None


def test_out_meta_none_no_crash_behavior_unchanged(monkeypatch):
    """out_meta=None → 不炸、行为与任务 H 一致（回归）。"""
    daily = _synthetic_daily_frame()
    n_raw = daily.height
    fm, FakeCtx = _patch_prepare_stack(monkeypatch, daily)

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


# ═══════════════════════════════════════════════════════════════════════════
# 5：agent 路径 manifest 端到端（mock 到 manifest 组装层）
# ═══════════════════════════════════════════════════════════════════════════


def test_agent_manifest_includes_membership_fields(monkeypatch, tmp_path):
    """CLI 装配 prep_meta → data_window → agent manifest.params 含 membership_mode/hash。"""
    from factorzen.cli import main as cli_main
    from factorzen.core.universe import get_universe_membership, membership_hash
    from factorzen.pipelines.factor_mine_agent import run_agent_mine

    daily = _synthetic_daily_frame()
    _patch_prepare_stack(monkeypatch, daily)

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
    monkeypatch.setattr(
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
