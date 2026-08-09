"""session 末 lift 钩子子进程化：裁决结果与进程内逐值一致 + 失败必须降级不崩 session。

背景（2026-07-29 全 A campaign 实测）：20 轮跑完后主进程常驻 ~17G（prepped 全宽帧 +
库池 + ResidualProjector + session state），钩子在**同一进程**里重新物化 79 条 active
基线 → RSS 22.85G → exit=137，整场挖掘产出陪葬（`.artifacts/campaign_newbase_1/`）。

判别力来源：
- **真跑子进程**（`python -m factorzen.agents.lift_hook_worker`），不是把 worker
  在进程内调一遍充数；两条路径跑在**两份内容相同但物理独立**的库上，逐值比对
  lift/passed/准入计数——库共用会让「第二次跑」看见第一次的写入，断言恒真。
- 降级：Popen 被替身成非零退出，断言 ①不抛 ②lift_error 有值 ③队列真的落了盘
  ④**没有**回落进程内偷偷跑完（回落等于把刚绕开的 OOM 搬回主进程）。
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl

MARKET = "ashare"
_ACTIVE_EXPRS = ("rank(close)", "rank(vol)", "rank(amount)")


def _prepped(n_days: int = 300, n_stocks: int = 40, seed: int = 5) -> pl.DataFrame:
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
        px = 10.0 + rng.uniform(0, 5)
        for day in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": day,
                "ts_code": c,
                "close": px,
                "close_adj": px,
                "open_adj": px * (1 + rng.standard_normal() * 0.003),
                "high_adj": px * 1.01,
                "low_adj": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def _seed_library(root) -> None:
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    recs = [
        FactorRecord(
            expression=e, market=MARKET, status="active",
            admission_track="single", ic_train=0.05 - 0.01 * i, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        )
        for i, e in enumerate(_ACTIVE_EXPRS)
    ]
    _save_library(MARKET, recs, root=str(root))


_QUEUE = [
    {"expression": "rank(low)", "ic_train": 0.03, "residual_ic_train": 0.02},
    {"expression": "rank(high)", "ic_train": 0.02, "residual_ic_train": 0.015},
]

_HOLDOUT_START = "2022-06-01"
_HOLDOUT_END = "2022-08-01"


def _core_kwargs(prepped, lib_root):
    return dict(
        daily=prepped,
        prepped=prepped,
        holdout_start=_HOLDOUT_START,
        holdout_end=_HOLDOUT_END,
        profile=None,
        profile_name=None,
        leaf_map=None,
        market=MARKET,
        library_root=str(lib_root),
        seed=7,
        horizon=5,
        lift_se_mult=1.645,
        lift_workers=1,
        data_window={"start": "20220101", "end": "20221231", "universe": None},
        run_id="t_run",
        index=None,
        exec_price_col=None,
    )


def _numeric_view(meta: dict) -> list[tuple]:
    rows = []
    for r in sorted(meta.get("lift_results") or [], key=lambda x: str(x.get("expression"))):
        rows.append((
            r.get("expression"), r.get("lift"), r.get("lift_se"),
            r.get("lift_second_half"), r.get("baseline"), r.get("passed"),
            r.get("error"),
        ))
    return rows


# ── 1. 子进程 vs 进程内：逐值一致 ──────────────────────────────────────────────


def test_subprocess_hook_matches_inprocess(tmp_path):
    """真 spawn 一次，与进程内 `_run_lift_hook_core` 的裁决逐值比对（两份独立库）。"""
    from factorzen.agents.team_orchestrator import (
        _empty_lift_meta,
        _lift_hook_via_subprocess,
        _run_lift_hook_core,
    )

    prepped = _prepped()
    root_a = tmp_path / "lib_a" / "factor_library"
    root_b = tmp_path / "lib_b" / "factor_library"
    _seed_library(root_a)
    _seed_library(root_b)

    in_proc = _run_lift_hook_core(
        list(_QUEUE), _empty_lift_meta(n_lift_queue=len(_QUEUE)),
        **_core_kwargs(prepped, root_a),
    )
    sub = _lift_hook_via_subprocess(
        list(_QUEUE), _empty_lift_meta(n_lift_queue=len(_QUEUE)),
        prepped=prepped,
        holdout_start=_HOLDOUT_START,
        holdout_end=_HOLDOUT_END,
        leaf_map=None,
        market=MARKET,
        library_root=str(root_b),
        seed=7,
        horizon=5,
        lift_se_mult=1.645,
        lift_workers=1,
        data_window={"start": "20220101", "end": "20221231", "universe": None},
        run_id="t_run",
        index_path=None,
        exec_price_col=None,
        profile_name=None,
    )
    assert sub.get("lift_error") is None, sub.get("lift_error")
    assert in_proc.get("lift_error") is None, in_proc.get("lift_error")
    # 队列非空且真跑过：否则下面的逐值比对是「空 == 空」的恒真
    assert in_proc["n_lift_evaluated"] >= 1
    assert _numeric_view(in_proc), "进程内没产出任何 lift 行，比对无判别力"
    assert _numeric_view(sub) == _numeric_view(in_proc)
    assert sub["n_lift_evaluated"] == in_proc["n_lift_evaluated"]
    assert sub["lift_admissions"] == in_proc["lift_admissions"]
    assert sub["n_lift_queue"] == len(_QUEUE)
    assert (sub.get("lift_group") or {}).get("lift") == (
        (in_proc.get("lift_group") or {}).get("lift")
    )
    # 两份库文件应当被写成同样的内容（准入是两条路径共同的终点）
    a = (root_a / "ashare.jsonl").read_text(encoding="utf-8")
    b = (root_b / "ashare.jsonl").read_text(encoding="utf-8")
    assert len(a.splitlines()) == len(b.splitlines())


def test_worker_payload_roundtrip_is_the_same_core(tmp_path):
    """worker 只是「读帧 + 调同一个 core」——payload 直调结果必须与进程内一致。"""
    from factorzen.agents.lift_hook_worker import run_payload
    from factorzen.agents.team_orchestrator import _empty_lift_meta, _run_lift_hook_core

    prepped = _prepped()
    root_a = tmp_path / "a" / "factor_library"
    root_b = tmp_path / "b" / "factor_library"
    _seed_library(root_a)
    _seed_library(root_b)
    frame = tmp_path / "prepped.arrow"
    prepped.write_ipc(str(frame), compression="lz4")

    in_proc = _run_lift_hook_core(
        list(_QUEUE), _empty_lift_meta(n_lift_queue=len(_QUEUE)),
        **_core_kwargs(prepped, root_a),
    )
    out = run_payload({
        "prepped_path": str(frame),
        "queue": list(_QUEUE),
        "holdout_start": _HOLDOUT_START,
        "holdout_end": _HOLDOUT_END,
        "leaf_map": None,
        "market": MARKET,
        "library_root": str(root_b),
        "seed": 7,
        "horizon": 5,
        "lift_se_mult": 1.645,
        "lift_workers": 1,
        "data_window": {"start": "20220101", "end": "20221231", "universe": None},
        "run_id": "t_run",
        "index_path": None,
        "exec_price_col": None,
        "profile_name": None,
    })
    assert out.get("lift_error") is None, out.get("lift_error")
    assert _numeric_view(out) == _numeric_view(in_proc)


def test_worker_business_error_becomes_lift_error_not_crash(tmp_path):
    """子进程内的业务异常 → lift_error（与进程内同格式），**不**让 worker 非零退出。"""
    from factorzen.agents.lift_hook_worker import run_payload

    out = run_payload({
        "prepped_path": str(tmp_path / "does_not_exist.arrow"),
        "queue": list(_QUEUE),
        "holdout_start": _HOLDOUT_START,
        "holdout_end": _HOLDOUT_END,
        "market": MARKET,
        "library_root": str(tmp_path / "factor_library"),
        "horizon": 5,
    })
    assert out["lift_error"]
    assert out["n_lift_queue"] == len(_QUEUE)
    assert out["lift_results"] == []


# ── 2. 失败必须降级：队列落盘 + 提示，且绝不回落进程内 ─────────────────────────


def test_subprocess_failure_degrades_to_queue_dump(tmp_path, monkeypatch, capsys):
    import subprocess as _sp

    from factorzen.agents.team_orchestrator import (
        _empty_lift_meta,
        _lift_hook_via_subprocess,
        lift_hook_queue_dump_path,
    )

    lib_root = tmp_path / "ws" / "factor_library"
    _seed_library(lib_root)
    core_calls = {"n": 0}
    import factorzen.agents.team_orchestrator as to

    real_core = to._run_lift_hook_core

    def counted(*a, **kw):
        core_calls["n"] += 1
        return real_core(*a, **kw)

    monkeypatch.setattr(to, "_run_lift_hook_core", counted)

    class _FakeProc:
        stdout = iter(["[mine-team] lift 钩子 ▸ 子进程日志一行\n"])

        def wait(self):
            return 137

    monkeypatch.setattr(_sp, "Popen", lambda *a, **kw: _FakeProc())

    meta = _lift_hook_via_subprocess(
        list(_QUEUE), _empty_lift_meta(n_lift_queue=len(_QUEUE)),
        prepped=_prepped(n_days=40, n_stocks=8),
        holdout_start=_HOLDOUT_START, holdout_end=_HOLDOUT_END,
        leaf_map=None, market=MARKET, library_root=str(lib_root),
        seed=7, horizon=5, lift_se_mult=1.645, lift_workers=1,
        data_window=None, run_id="t_run", index_path=None,
        exec_price_col=None, profile_name=None,
    )
    assert meta["lift_error"] and "137" in meta["lift_error"]
    assert core_calls["n"] == 0, "子进程失败后不得回落进程内执行（那正是 OOM 的来源）"
    dump = lift_hook_queue_dump_path(str(lib_root), "t_run")
    assert dump.is_file()
    assert json.loads(dump.read_text(encoding="utf-8")) == _QUEUE
    out = capsys.readouterr().out
    assert "lift 钩子 ▸ 失败（已记 lift_error）" in out
    assert "lift-test" in out, "降级提示里必须给出手动补跑命令"
    # 子进程的日志被逐字转发（钩子日志格式保持）
    assert "[mine-team] lift 钩子 ▸ 子进程日志一行" in out


def test_release_frames_runs_before_spawn(tmp_path, monkeypatch):
    """主进程必须在 spawn **之前**松开大帧，否则两边内存照旧叠加。"""
    import subprocess as _sp

    from factorzen.agents.team_orchestrator import (
        _empty_lift_meta,
        _lift_hook_via_subprocess,
    )

    order: list[str] = []
    lib_root = tmp_path / "ws" / "factor_library"
    _seed_library(lib_root)

    class _FakeProc:
        stdout = iter(())

        def wait(self):
            order.append("spawn")
            return 1

    monkeypatch.setattr(_sp, "Popen", lambda *a, **kw: _FakeProc())
    _lift_hook_via_subprocess(
        list(_QUEUE), _empty_lift_meta(n_lift_queue=len(_QUEUE)),
        prepped=_prepped(n_days=30, n_stocks=6),
        holdout_start=_HOLDOUT_START, holdout_end=_HOLDOUT_END,
        leaf_map=None, market=MARKET, library_root=str(lib_root),
        seed=7, horizon=5, lift_se_mult=1.645, lift_workers=1,
        data_window=None, run_id="r", index_path=None,
        exec_price_col=None, profile_name=None,
        release_frames=lambda: order.append("release"),
    )
    assert order == ["release", "spawn"]


# ── 3. 注入物在场 → 自动退回进程内（注入不可序列化，且语义是「调用方自带基线」）──


def test_injection_forces_inprocess(tmp_path, monkeypatch):
    import factorzen.agents.team_orchestrator as to
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_QUEUE

    lib_root = tmp_path / "ws" / "factor_library"
    _seed_library(lib_root)
    prepped = _prepped(n_days=40, n_stocks=8)

    state = AgentState(seed=1)
    state.attempts = [
        AttemptRecord(
            iteration=1, hypothesis="h", expression="rank(low)",
            compile_ok=True, ic_train=0.03, passed_guardrails=False,
            critic_verdict=None, error=None,
            reject_category=REJECT_CATEGORY_LIFT_QUEUE,
        ),
    ]

    spawned = {"n": 0}
    monkeypatch.setattr(
        to, "_lift_hook_via_subprocess",
        lambda *a, **kw: spawned.__setitem__("n", spawned["n"] + 1) or {},
    )
    seen = {"n": 0}
    real_core = to._run_lift_hook_core
    monkeypatch.setattr(
        to, "_run_lift_hook_core",
        lambda *a, **kw: (seen.__setitem__("n", seen["n"] + 1), real_core(*a, **kw))[1],
    )

    holdout = prepped.select(["trade_date"]).unique().sort("trade_date")
    to._session_end_auto_lift(
        state,
        daily=prepped, holdout_df=holdout, profile=None, ctx=None,
        market=MARKET, library_root=str(lib_root), seed=1, horizon=5,
        prepped=prepped,
        # 任一注入在场 → 必须留在进程内
        active_factor_dfs={"rank(close)": prepped.select(
            ["trade_date", "ts_code", pl.col("close").alias("factor_value")]
        )},
        subprocess_mode=True,
    )
    assert spawned["n"] == 0, "有注入物时不得 spawn"
    assert seen["n"] == 1
