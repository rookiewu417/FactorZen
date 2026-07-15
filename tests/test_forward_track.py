"""forward_track / forward_review：probation→active 的 paper forward 确认机制。

全 mock 离线；TDD 核心锁死 PIT 口径 ic(t)=spearman(factor(t-1), ret(t))。
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.core.stats import spearman_avg_rank

# ── 构造工具 ─────────────────────────────────────────────────────────────────


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _write_lib(root: Path, market: str, rows: list[dict]) -> Path:
    """写库 jsonl（原始 dict，保留未知字段）。"""
    p = root / f"{market}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return p


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _daily_3day() -> tuple[pl.DataFrame, date, date, date]:
    """3 交易日小帧；收盘价设计使 factor(t-1)×ret(t) 与 factor(t)×ret(t) 的 IC 不等。

    Day1 / Day2 / Day3；as_of=Day3 时：
    - factor(t-1)=Day2 close：10, 20, 15
    - ret(t)=Day3/Day2-1：0.2, -0.05, 1.0
    - factor(t)=Day3 close：12, 19, 30  （反例：同日截面）
    """
    d1 = date(2024, 1, 2)
    d2 = date(2024, 1, 3)
    d3 = date(2024, 1, 4)
    rows = []
    # (day, code, close)
    data = [
        (d1, "000001.SZ", 9.0),
        (d1, "000002.SZ", 19.0),
        (d1, "000003.SZ", 14.0),
        (d2, "000001.SZ", 10.0),
        (d2, "000002.SZ", 20.0),
        (d2, "000003.SZ", 15.0),
        (d3, "000001.SZ", 12.0),
        (d3, "000002.SZ", 19.0),
        (d3, "000003.SZ", 30.0),
    ]
    for d, code, close in data:
        rows.append({
            "trade_date": d,
            "ts_code": code,
            "close": close,
            "close_adj": close,
            "open": close,
            "open_adj": close,
            "high": close,
            "high_adj": close,
            "low": close,
            "low_adj": close,
            "vol": 1000.0,
            "amount": 10000.0,
        })
    return pl.DataFrame(rows), d1, d2, d3


def _lib_row(expr: str, *, status="probation", ic_train=0.05,
             updated_at="2024-01-01", **extra) -> dict:
    d = {
        "expression": expr,
        "market": "ashare",
        "status": status,
        "ic_train": ic_train,
        "updated_at": updated_at,
        "added_at": updated_at,
        "admission_track": "lift",
    }
    d.update(extra)
    return d


# ── 1. PIT 口径锁死 ──────────────────────────────────────────────────────────


def test_pit_ic_uses_factor_t_minus_1_and_ret_t(tmp_path):
    """ic(as_of)=spearman(factor(t-1), ret(t))；与 factor(t)×ret(t) 不等。"""
    from factorzen.discovery.forward_track import record_forward_ics

    daily, _d1, _d2, d3 = _daily_3day()
    as_of = _yyyymmdd(d3)
    expr = "close"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=0.05),
    ])

    out = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily, lookback_days=5,
        now=as_of,  # 历史 fixture：注入 now 避免被反回灌拒
    )
    assert out["recorded"] == 1
    assert out["failed"] == 0

    rows = _read_jsonl(tmp_path / "forward_track" / "ashare.jsonl")
    assert len(rows) == 1
    assert rows[0]["date"] == as_of
    assert rows[0]["expression"] == expr
    got = rows[0]["ic"]
    assert got is not None

    # 手工期望：factor(t-1)=Day2 close，ret(t)=Day3/Day2-1
    f_prev = np.array([10.0, 20.0, 15.0])
    ret_t = np.array([12 / 10 - 1, 19 / 20 - 1, 30 / 15 - 1])
    expected = spearman_avg_rank(f_prev, ret_t)
    assert expected is not None
    assert abs(got - expected) < 1e-9

    # 反例：factor(t)×ret(t) 不应等于 PIT 口径
    f_same = np.array([12.0, 19.0, 30.0])
    wrong = spearman_avg_rank(f_same, ret_t)
    assert wrong is not None
    assert abs(got - wrong) > 1e-6, "PIT 与同日截面 IC 必须可区分"


# ── 2. 幂等 ──────────────────────────────────────────────────────────────────


def test_record_idempotent_skips_existing(tmp_path):
    """同 (date, expression, universe) 重跑 → recorded=0、skipped_existing=N。"""
    from factorzen.discovery.forward_track import record_forward_ics

    daily, _d1, _d2, d3 = _daily_3day()
    as_of = _yyyymmdd(d3)
    _write_lib(tmp_path, "ashare", [
        _lib_row("close", status="probation"),
        _lib_row("open", status="active"),
    ])

    out1 = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily, now=as_of,
    )
    assert out1["recorded"] == 2
    assert out1["skipped_existing"] == 0

    out2 = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily, now=as_of,
    )
    assert out2["recorded"] == 0
    assert out2["skipped_existing"] == 2
    rows = _read_jsonl(tmp_path / "forward_track" / "ashare.jsonl")
    assert len(rows) == 2


# ── 3. review 门槛 ───────────────────────────────────────────────────────────


def _seed_forward_ics(root: Path, market: str, expr: str, ics: list[float],
                      start: date | None = None) -> None:
    """写连续交易日的 forward jsonl（ic 序列）。"""
    start = start or date(2024, 3, 1)
    path = root / "forward_track" / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, ic in enumerate(ics):
        d = start + timedelta(days=i)  # 日历日即可，比较用字符串序
        lines.append(json.dumps({
            "date": _yyyymmdd(d),
            "expression": expr,
            "ic": ic,
            "n_stocks": 50,
            "status_at_record": "probation",
        }, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_review_hold_when_n_lt_min_days(tmp_path):
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(close)"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=0.05, updated_at="2024-01-01"),
    ])
    _seed_forward_ics(tmp_path, "ashare", expr, [0.02] * 10)

    rows = forward_review("ashare", root=str(tmp_path), min_days=60)
    assert len(rows) == 1
    assert rows[0]["decision"] == "hold"
    assert rows[0]["n_days"] == 10
    assert rows[0]["reason"] in ("insufficient_days", None) or "insuff" in str(
        rows[0].get("reason") or ""
    )


def test_review_promote_significant_positive(tmp_path):
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(close)"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=0.05, updated_at="2024-01-01"),
    ])
    # 强正 IC，块 SE 相对均值小 → promote
    _seed_forward_ics(tmp_path, "ashare", expr, [0.05] * 80)

    rows = forward_review(
        "ashare", root=str(tmp_path), min_days=60, se_mult=1.645, block_days=20,
    )
    assert rows[0]["decision"] == "promote"
    assert rows[0]["mean"] is not None and rows[0]["mean"] > 0
    assert rows[0]["ci_low"] is not None and rows[0]["ci_low"] > 0


def test_review_demote_significant_negative(tmp_path):
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(close)"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=0.05, updated_at="2024-01-01"),
    ])
    _seed_forward_ics(tmp_path, "ashare", expr, [-0.05] * 80)

    rows = forward_review(
        "ashare", root=str(tmp_path), min_days=60, se_mult=1.645, block_days=20,
    )
    assert rows[0]["decision"] == "demote"


def test_review_hold_near_zero(tmp_path):
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(close)"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=0.05, updated_at="2024-01-01"),
    ])
    # 接近 0 且有噪声 → CI 跨 0 → hold
    rng = np.random.default_rng(0)
    ics = (rng.normal(0.0, 0.01, size=80)).tolist()
    _seed_forward_ics(tmp_path, "ashare", expr, ics)

    rows = forward_review(
        "ashare", root=str(tmp_path), min_days=60, se_mult=1.645, block_days=20,
    )
    assert rows[0]["decision"] == "hold"


# ── 4. 负方向因子 ────────────────────────────────────────────────────────────


def test_review_negative_ic_train_sign_flip_promote(tmp_path):
    """ic_train<0 且 forward ic 全负 → adj 后为正 → promote。"""
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(volume)"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=-0.04, updated_at="2024-01-01"),
    ])
    _seed_forward_ics(tmp_path, "ashare", expr, [-0.05] * 80)

    rows = forward_review(
        "ashare", root=str(tmp_path), min_days=60, se_mult=1.645, block_days=20,
    )
    assert rows[0]["decision"] == "promote"
    assert rows[0]["mean"] is not None and rows[0]["mean"] > 0


def test_review_missing_sign_holds(tmp_path):
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(close)"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=None, updated_at="2024-01-01"),
    ])
    _seed_forward_ics(tmp_path, "ashare", expr, [0.05] * 80)

    rows = forward_review("ashare", root=str(tmp_path), min_days=60)
    assert rows[0]["decision"] == "hold"
    assert rows[0]["reason"] == "missing_sign"


def test_review_admission_ic_preferred_over_ic_train(tmp_path):
    """admission_ic 优先于 ic_train：admission_ic>0 + forward 正 → promote（即使 ic_train 负）。"""
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(close)"
    _write_lib(tmp_path, "ashare", [
        _lib_row(
            expr, status="probation", ic_train=-0.04, updated_at="2024-01-01",
            admission_ic=0.05,
        ),
    ])
    _seed_forward_ics(tmp_path, "ashare", expr, [0.05] * 80)

    rows = forward_review(
        "ashare", root=str(tmp_path), min_days=60, se_mult=1.645, block_days=20,
    )
    assert rows[0]["decision"] == "promote"
    assert rows[0]["reason"] != "missing_sign"
    assert rows[0]["mean"] is not None and rows[0]["mean"] > 0


def test_e2e_lift_admission_ic_enables_promote_chain(tmp_path):
    """生产形态：lift row 无 ic_train、有 admission_ic → upsert probation → forward promote。

    对照：admission_ic 与 ic_train 皆无 → 仍 hold/missing_sign（修复点在方向传递）。
    """
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions
    from factorzen.discovery.forward_track import forward_review

    good_expr = "rank(close)"
    bad_expr = "rank(open)"
    meta = {
        "session_dir": "sess/e2e",
        "run_id": "run_e2e",
        "universe": "csi300",
        "eval_start": "20200101",
        "eval_end": "20260101",
        "horizon": 5,
        "git_sha": "deadbeef",
        "now": "2024-01-01",
    }
    # production 形态 lift 结果行：有 admission_ic，无 ic_train（候选挖掘字段未透传时）
    good_row = {
        "expression": good_expr,
        "lift": 0.005,
        "lift_se": 0.001,
        "lift_first_half": 0.004,
        "lift_second_half": 0.004,
        "baseline": 0.04,
        "admission_ic": 0.05,
        # 故意不设 ic_train
    }
    bad_row = {
        "expression": bad_expr,
        "lift": 0.005,
        "lift_se": 0.001,
        "lift_first_half": 0.004,
        "lift_second_half": 0.004,
        "baseline": 0.04,
        # 无 admission_ic、无 ic_train → 方向缺失
    }
    out = upsert_lift_admissions(
        [good_row, bad_row],
        market="ashare",
        root=str(tmp_path),
        meta=meta,
        allow_active=False,  # 运营 cap：decision=active 也落 probation
    )
    assert out["added_probation"] == 2
    assert out["errors"] == []

    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert lib[good_expr].status == "probation"
    assert lib[good_expr].admission_ic is not None
    assert abs(lib[good_expr].admission_ic - 0.05) < 1e-12
    assert lib[good_expr].ic_train is None
    assert lib[bad_expr].status == "probation"
    assert lib[bad_expr].admission_ic is None
    assert lib[bad_expr].ic_train is None

    # 两因子各 80 天全正 forward IC（universe 须与准入 meta 一致，review 按口径过滤）
    path = tmp_path / "forward_track" / "ashare.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    start = date(2024, 3, 1)
    for i in range(80):
        d = _yyyymmdd(start + timedelta(days=i))
        for expr in (good_expr, bad_expr):
            lines.append(json.dumps({
                "date": d, "expression": expr, "ic": 0.05,
                "n_stocks": 50, "status_at_record": "probation",
                "universe": "csi300",
            }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = forward_review(
        "ashare", root=str(tmp_path), min_days=60, se_mult=1.645, block_days=20,
    )
    by = {r["expression"]: r for r in rows}
    assert by[good_expr]["decision"] == "promote", by[good_expr]
    assert by[good_expr]["reason"] != "missing_sign"
    assert by[bad_expr]["decision"] == "hold"
    assert by[bad_expr]["reason"] == "missing_sign"


def test_factor_record_admission_ic_from_dict_compat():
    """旧 FactorRecord 行（无 admission_ic）反序列化不得报错 → None。"""
    from factorzen.discovery.factor_library import FactorRecord

    rec = FactorRecord.from_dict({
        "expression": "rank(close)",
        "market": "ashare",
        "status": "probation",
        "ic_train": None,
    })
    assert rec.admission_ic is None
    d = rec.to_dict()
    assert "admission_ic" in d
    assert d["admission_ic"] is None


# ── 5. apply 状态机 ──────────────────────────────────────────────────────────


def test_apply_promote_and_demote_and_untouched(tmp_path):
    from factorzen.discovery.forward_track import forward_review

    promote_expr = "rank(close)"
    demote_expr = "rank(open)"
    single_active = "ts_mean(close, 5)"
    correlated = "rank(high)"

    _write_lib(tmp_path, "ashare", [
        _lib_row(promote_expr, status="probation", ic_train=0.05, updated_at="2024-01-01"),
        _lib_row(demote_expr, status="probation", ic_train=0.05, updated_at="2024-01-01"),
        _lib_row(single_active, status="active", ic_train=0.08,
                 admission_track="single", updated_at="2024-01-01"),
        _lib_row(correlated, status="correlated", ic_train=0.06,
                 correlated_with=single_active, updated_at="2024-01-01"),
    ])
    # 两条 forward 序列写在同一文件
    path = tmp_path / "forward_track" / "ashare.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    start = date(2024, 3, 1)
    for i in range(80):
        d = _yyyymmdd(start + timedelta(days=i))
        lines.append(json.dumps({
            "date": d, "expression": promote_expr, "ic": 0.05,
            "n_stocks": 50, "status_at_record": "probation",
        }))
        lines.append(json.dumps({
            "date": d, "expression": demote_expr, "ic": -0.05,
            "n_stocks": 50, "status_at_record": "probation",
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lib_path = tmp_path / "ashare.jsonl"
    before = lib_path.read_text(encoding="utf-8")
    mtime_before = lib_path.stat().st_mtime

    # dry-run：不写盘
    dry = forward_review(
        "ashare", root=str(tmp_path), min_days=60, se_mult=1.645, apply=False,
    )
    assert {r["expression"]: r["decision"] for r in dry}[promote_expr] == "promote"
    assert {r["expression"]: r["decision"] for r in dry}[demote_expr] == "demote"
    assert lib_path.read_text(encoding="utf-8") == before
    assert lib_path.stat().st_mtime == mtime_before

    rows = forward_review(
        "ashare", root=str(tmp_path), min_days=60, se_mult=1.645, apply=True,
    )
    by_expr = {r["expression"]: r for r in rows}
    assert by_expr[promote_expr]["decision"] == "promote"
    assert by_expr[demote_expr]["decision"] == "demote"

    lib = {r["expression"]: r for r in _read_jsonl(lib_path)}
    assert lib[promote_expr]["status"] == "active"
    assert lib[promote_expr].get("forward_confirmed_at")
    assert lib[promote_expr].get("forward_n_days") == 80
    assert lib[demote_expr]["status"] == "no_lift"
    # single-track active / correlated 不动
    assert lib[single_active]["status"] == "active"
    assert lib[single_active].get("admission_track") == "single"
    assert lib[correlated]["status"] == "correlated"


# ── 6. updated_at 过滤 ───────────────────────────────────────────────────────


def test_review_filters_ics_before_updated_at(tmp_path):
    """进入 probation 之前的 forward 记录不计入。"""
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(close)"
    # updated_at = 2024-04-01；之前的 100 天不应计入
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=0.05, updated_at="2024-04-01"),
    ])
    path = tmp_path / "forward_track" / "ashare.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    # 进入前：强正 IC 100 天，全部严格早于 updated_at（若误计入会 promote）
    for i in range(100):
        d = date(2024, 1, 1) + timedelta(days=i)  # 20240101–20240409 会越界；封顶到 0331
        if d >= date(2024, 4, 1):
            break
        lines.append(json.dumps({
            "date": _yyyymmdd(d), "expression": expr, "ic": 0.08,
            "n_stocks": 50, "status_at_record": "probation",
        }))
    # 进入后：仅 5 天（date > updated_at）
    for i in range(5):
        d = date(2024, 4, 2) + timedelta(days=i)
        lines.append(json.dumps({
            "date": _yyyymmdd(d), "expression": expr, "ic": 0.08,
            "n_stocks": 50, "status_at_record": "probation",
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = forward_review("ashare", root=str(tmp_path), min_days=60)
    assert rows[0]["n_days"] == 5
    assert rows[0]["decision"] == "hold"


# ── 7. CLI 透传 ──────────────────────────────────────────────────────────────


def test_cli_forward_track_parser_and_handler(tmp_path, monkeypatch, capsys):
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    calls = []

    def fake_record(market, as_of, **kw):
        calls.append({"market": market, "as_of": as_of, **kw})
        return {"recorded": 2, "skipped_existing": 1, "failed": 0}

    monkeypatch.setattr(
        "factorzen.discovery.forward_track.record_forward_ics", fake_record,
    )
    monkeypatch.setattr(
        "factorzen.discovery.backtest_window.latest_data_date",
        lambda m: date(2024, 6, 15),
    )

    args = build_parser().parse_args([
        "factor-library", "forward-track",
        "--market", "ashare",
        "--root", str(tmp_path),
    ])
    rc = cli_main._cmd_factor_library_forward_track(args)
    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["market"] == "ashare"
    assert calls[0]["as_of"] == "20240615"
    assert calls[0]["root"] == str(tmp_path)
    out = capsys.readouterr().out
    assert "recorded" in out or "2" in out


def test_cli_forward_track_date_override(tmp_path, monkeypatch):
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    calls = []
    monkeypatch.setattr(
        "factorzen.discovery.forward_track.record_forward_ics",
        lambda market, as_of, **kw: calls.append((market, as_of, kw)) or {
            "recorded": 0, "skipped_existing": 0, "failed": 0,
        },
    )
    args = build_parser().parse_args([
        "factor-library", "forward-track",
        "--market", "ashare",
        "--date", "20240104",
        "--root", str(tmp_path),
    ])
    rc = cli_main._cmd_factor_library_forward_track(args)
    assert rc == 0
    assert calls[0][1] == "20240104"


def test_cli_forward_review_parser_and_apply(tmp_path, monkeypatch, capsys):
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    calls = []

    def fake_review(market, **kw):
        calls.append({"market": market, **kw})
        return [{
            "expression": "rank(close)",
            "decision": "promote",
            "n_days": 80,
            "mean": 0.04,
            "se": 0.01,
            "ci_low": 0.02,
            "reason": None,
        }]

    monkeypatch.setattr(
        "factorzen.discovery.forward_track.forward_review", fake_review,
    )

    args = build_parser().parse_args([
        "factor-library", "forward-review",
        "--market", "ashare",
        "--min-days", "60",
        "--se-mult", "1.645",
        "--root", str(tmp_path),
        "--apply",
    ])
    rc = cli_main._cmd_factor_library_forward_review(args)
    assert rc == 0
    assert calls[0]["apply"] is True
    assert calls[0]["min_days"] == 60
    assert math.isclose(calls[0]["se_mult"], 1.645)
    assert calls[0]["root"] == str(tmp_path)
    out = capsys.readouterr().out
    assert "promote" in out


def test_forward_fields_survive_library_roundtrip(tmp_path):
    """forward_confirmed_at/forward_n_days 必须进 FactorRecord schema。

    否则 from_dict 丢弃未知字段 → 任一次 load→save 循环把 apply 写入的
    确认痕迹静默洗掉（本用例即该回归的锁）。
    """
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
    )

    rec = FactorRecord(
        expression="rank(close)", market="ashare", status="active",
        admission_track="lift", forward_confirmed_at="2026-07-14",
        forward_n_days=75, added_at="2026-07-01", updated_at="2026-07-14",
    )
    _save_library("ashare", [rec], root=str(tmp_path))
    back = load_library("ashare", root=str(tmp_path))
    assert back[0].forward_confirmed_at == "2026-07-14"
    assert back[0].forward_n_days == 75
    # 再走一轮 load→save（模拟后续 upsert/rebuild 的写盘循环）
    _save_library("ashare", back, root=str(tmp_path))
    again = load_library("ashare", root=str(tmp_path))
    assert again[0].forward_confirmed_at == "2026-07-14"
    assert again[0].forward_n_days == 75


def test_assemble_universe_follows_admission_mode(monkeypatch, tmp_path):
    """单一 universe 库：生产装配透传 record.universe（非全 A）。

    首跑实测 n_stocks=5511（全 A）暴露：csi300 准入的因子在全 A 截面上的
    forward IC 是另一个统计量，不能用于裁决。
    """
    from factorzen.discovery import forward_track as ft
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    recs = [
        FactorRecord(expression="rank(close)", market="ashare", status="active",
                     universe="csi300", ic_train=0.02,
                     added_at="2026-07-01", updated_at="2026-07-01"),
        FactorRecord(expression="rank(vol)", market="ashare", status="probation",
                     universe="csi300", ic_train=0.02,
                     added_at="2026-07-01", updated_at="2026-07-01"),
    ]
    _save_library("ashare", recs, root=str(tmp_path))

    captured: dict = {}

    def fake_prepare(start, end, universe=None, lookback_days=None, **kw):
        captured["universe"] = universe
        raise RuntimeError("stop after capture")  # 只验证透传，不真装配

    monkeypatch.setattr(ft, "prepare_mining_daily", fake_prepare)
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="stop after capture"):
        ft.record_forward_ics(
            "ashare", "20260605", root=str(tmp_path), now="20260605",
        )
    assert captured["universe"] == "csi300"

    # 显式 --universe 覆盖各因子准入口径
    with _pytest.raises(RuntimeError, match="stop after capture"):
        ft.record_forward_ics(
            "ashare", "20260605", root=str(tmp_path),
            universe="csi800", now="20260605",
        )
    assert captured["universe"] == "csi800"


# ── 8. S4/P6：按 universe 分组 + ledger 三元键 + review 不混截面 ─────────────


def test_record_groups_by_universe_and_assembles_each(monkeypatch, tmp_path):
    """混口径库：按 record.universe 分组装配，csi500 不在 csi300 众数截面上算 IC。"""
    from factorzen.discovery import forward_track as ft
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    recs = [
        FactorRecord(expression="close", market="ashare", status="probation",
                     universe="csi300", ic_train=0.02,
                     added_at="2024-01-01", updated_at="2024-01-01"),
        FactorRecord(expression="open", market="ashare", status="probation",
                     universe="csi300", ic_train=0.02,
                     added_at="2024-01-01", updated_at="2024-01-01"),
        FactorRecord(expression="high", market="ashare", status="active",
                     universe="csi300", ic_train=0.02,
                     added_at="2024-01-01", updated_at="2024-01-01"),
        FactorRecord(expression="low", market="ashare", status="probation",
                     universe="csi500", ic_train=0.02,
                     added_at="2024-01-01", updated_at="2024-01-01"),
    ]
    _save_library("ashare", recs, root=str(tmp_path))

    daily_300, _d1, _d2, d3 = _daily_3day()
    # csi500 截面：仅 2 只股票、价序列不同，确保 IC 可与 csi300 区分
    d1, d2, d3d = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    daily_500 = pl.DataFrame([
        {"trade_date": d1, "ts_code": "600001.SH", "close": 5.0, "close_adj": 5.0,
         "open": 5.0, "open_adj": 5.0, "high": 5.0, "high_adj": 5.0,
         "low": 4.0, "low_adj": 4.0, "vol": 100.0, "amount": 1000.0},
        {"trade_date": d1, "ts_code": "600002.SH", "close": 8.0, "close_adj": 8.0,
         "open": 8.0, "open_adj": 8.0, "high": 8.0, "high_adj": 8.0,
         "low": 7.0, "low_adj": 7.0, "vol": 100.0, "amount": 1000.0},
        {"trade_date": d2, "ts_code": "600001.SH", "close": 6.0, "close_adj": 6.0,
         "open": 6.0, "open_adj": 6.0, "high": 6.0, "high_adj": 6.0,
         "low": 5.0, "low_adj": 5.0, "vol": 100.0, "amount": 1000.0},
        {"trade_date": d2, "ts_code": "600002.SH", "close": 9.0, "close_adj": 9.0,
         "open": 9.0, "open_adj": 9.0, "high": 9.0, "high_adj": 9.0,
         "low": 8.0, "low_adj": 8.0, "vol": 100.0, "amount": 1000.0},
        {"trade_date": d3d, "ts_code": "600001.SH", "close": 7.0, "close_adj": 7.0,
         "open": 7.0, "open_adj": 7.0, "high": 7.0, "high_adj": 7.0,
         "low": 6.0, "low_adj": 6.0, "vol": 100.0, "amount": 1000.0},
        {"trade_date": d3d, "ts_code": "600002.SH", "close": 10.0, "close_adj": 10.0,
         "open": 10.0, "open_adj": 10.0, "high": 10.0, "high_adj": 10.0,
         "low": 9.0, "low_adj": 9.0, "vol": 100.0, "amount": 1000.0},
    ])
    as_of = _yyyymmdd(d3)
    assemble_calls: list[dict] = []

    def fake_assemble(market, as_of_arg, lookback_days, universe=None):
        assemble_calls.append({
            "market": market, "as_of": as_of_arg,
            "lookback_days": lookback_days, "universe": universe,
        })
        if universe == "csi300":
            return daily_300
        if universe == "csi500":
            return daily_500
        raise AssertionError(f"unexpected universe {universe!r}")

    monkeypatch.setattr(ft, "_assemble_daily", fake_assemble)

    out = ft.record_forward_ics(
        "ashare", as_of, root=str(tmp_path), now=as_of,
    )
    assert out["recorded"] == 4
    assert out["failed"] == 0

    # 每 universe 各装配一次
    unis = sorted(c["universe"] for c in assemble_calls)
    assert unis == ["csi300", "csi500"]
    assert len(assemble_calls) == 2

    rows = _read_jsonl(tmp_path / "forward_track" / "ashare.jsonl")
    by = {(r["expression"], r.get("universe")): r for r in rows}
    assert set(by) == {
        ("close", "csi300"), ("open", "csi300"),
        ("high", "csi300"), ("low", "csi500"),
    }
    # csi500 因子截面只有 2 只股票
    assert by[("low", "csi500")]["n_stocks"] == 2
    # csi300 因子截面 3 只
    assert by[("close", "csi300")]["n_stocks"] == 3


def test_ledger_key_includes_universe(tmp_path):
    """同 (date, expr) 不同 universe 的 row 都能写入；row 含 universe 字段。"""
    from factorzen.discovery.forward_track import record_forward_ics

    daily, _d1, _d2, d3 = _daily_3day()
    as_of = _yyyymmdd(d3)
    expr = "close"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", universe="csi300"),
    ])
    out1 = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily, now=as_of,
    )
    assert out1["recorded"] == 1
    rows = _read_jsonl(tmp_path / "forward_track" / "ashare.jsonl")
    assert rows[0].get("universe") == "csi300"

    # 同 date/expr、另一 universe 的记录不得被幂等跳过
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", universe="csi500"),
    ])
    out2 = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily, now=as_of,
    )
    assert out2["recorded"] == 1
    assert out2["skipped_existing"] == 0
    rows = _read_jsonl(tmp_path / "forward_track" / "ashare.jsonl")
    assert len(rows) == 2
    unis = {r.get("universe") for r in rows}
    assert unis == {"csi300", "csi500"}
    assert all(r["expression"] == expr and r["date"] == as_of for r in rows)


def test_review_filters_forward_ics_by_universe(tmp_path):
    """同 expr 两套 universe IC：review 只取 rec.universe 对应序列，不混用。"""
    from factorzen.discovery.forward_track import forward_review

    expr = "rank(close)"
    # csi500 record；ledger 混有 csi300（强正 80 天）与 csi500（仅 5 天）
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", ic_train=0.05,
                 updated_at="2024-01-01", universe="csi500"),
    ])
    path = tmp_path / "forward_track" / "ashare.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    start = date(2024, 3, 1)
    for i in range(80):
        d = _yyyymmdd(start + timedelta(days=i))
        lines.append(json.dumps({
            "date": d, "expression": expr, "ic": 0.05,
            "n_stocks": 50, "status_at_record": "probation",
            "universe": "csi300",
        }))
    for i in range(5):
        d = _yyyymmdd(start + timedelta(days=i))
        lines.append(json.dumps({
            "date": d, "expression": expr, "ic": 0.01,
            "n_stocks": 40, "status_at_record": "probation",
            "universe": "csi500",
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = forward_review("ashare", root=str(tmp_path), min_days=60)
    assert len(rows) == 1
    # 若混用 csi300 的 80 天会 promote；正确只计 csi500 的 5 天 → hold
    assert rows[0]["n_days"] == 5
    assert rows[0]["decision"] == "hold"
    assert rows[0]["reason"] in ("insufficient_days", None) or "insuff" in str(
        rows[0].get("reason") or ""
    )


# ── 9. S5/P8：非 A 股 fail-closed + 全失败非零退出 ───────────────────────────


def test_record_non_ashare_fail_closed(tmp_path):
    """生产路径 daily=None 且 market!=ashare → 不得静默走 A 股装配。"""
    import pytest

    from factorzen.discovery.factor_library import FactorRecord, _save_library
    from factorzen.discovery.forward_track import record_forward_ics

    _save_library("crypto", [
        FactorRecord(expression="close", market="crypto", status="probation",
                     ic_train=0.02, added_at="2024-01-01", updated_at="2024-01-01"),
    ], root=str(tmp_path))

    with pytest.raises(ValueError, match=r"fail closed|暂未接入"):
        record_forward_ics(
            "crypto", "20240104", root=str(tmp_path), daily=None, now="20240104",
        )


def test_cli_forward_track_non_ashare_nonzero(tmp_path, monkeypatch, capsys):
    """CLI crypto 无注入 → 返回码非 0（fail closed）。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    _save_library("crypto", [
        FactorRecord(expression="close", market="crypto", status="probation",
                     ic_train=0.02, added_at="2024-01-01", updated_at="2024-01-01"),
    ], root=str(tmp_path))

    args = build_parser().parse_args([
        "factor-library", "forward-track",
        "--market", "crypto",
        "--date", "20240104",
        "--root", str(tmp_path),
    ])
    rc = cli_main._cmd_factor_library_forward_track(args)
    assert rc != 0
    err = capsys.readouterr().err
    assert "crypto" in err or "fail" in err.lower() or "暂未" in err


def test_cli_forward_track_all_failed_returns_1(tmp_path, monkeypatch, capsys):
    """全部 materialize 失败 → CLI return 1。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    monkeypatch.setattr(
        "factorzen.discovery.forward_track.record_forward_ics",
        lambda *a, **k: {"recorded": 3, "skipped_existing": 0, "failed": 3},
    )
    args = build_parser().parse_args([
        "factor-library", "forward-track",
        "--market", "ashare",
        "--date", "20240104",
        "--root", str(tmp_path),
    ])
    rc = cli_main._cmd_factor_library_forward_track(args)
    assert rc == 1
    out = capsys.readouterr().out + capsys.readouterr().err
    # 至少打印了结果摘要
    assert "failed" in out or "3" in out


def test_injected_daily_zero_regression_single_universe(tmp_path):
    """注入 daily 单 universe 路径：行为等价旧路径（记 IC + 幂等）。"""
    from factorzen.discovery.forward_track import record_forward_ics

    daily, _d1, _d2, d3 = _daily_3day()
    as_of = _yyyymmdd(d3)
    expr = "close"
    _write_lib(tmp_path, "ashare", [
        _lib_row(expr, status="probation", universe="csi300", ic_train=0.05),
    ])
    out = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily, lookback_days=5,
        now=as_of,
    )
    assert out["recorded"] == 1
    assert out["failed"] == 0
    rows = _read_jsonl(tmp_path / "forward_track" / "ashare.jsonl")
    assert rows[0]["ic"] is not None
    assert rows[0].get("universe") == "csi300"

    out2 = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily, lookback_days=5,
        now=as_of,
    )
    assert out2["recorded"] == 0
    assert out2["skipped_existing"] == 1


# ── 10. S6：wall-clock 反回灌 + provenance ───────────────────────────────────


def test_reject_historical_backfill(tmp_path):
    """as_of 距 now 过远 → ValueError；allow_backfill=True 时放行。"""
    import pytest

    from factorzen.discovery.forward_track import record_forward_ics

    daily, _d1, _d2, _d3 = _daily_3day()
    as_of = "20240101"
    # daily 帧是 20240102–04；用 allow_backfill 路径时 as_of 需在帧内才有 IC，
    # 此处只断言「反回灌」门禁：先拒旧日，再 allow 后不因门禁失败。
    _write_lib(tmp_path, "ashare", [
        _lib_row("close", status="probation"),
    ])

    with pytest.raises(ValueError, match=r"回灌|max_backfill|allow-backfill"):
        record_forward_ics(
            "ashare", as_of, root=str(tmp_path), daily=daily,
            now="20240301",  # lag=60 > 10
        )

    # allow_backfill 跳过旧日拒绝（as_of 可能无前序交易日 → failed 可 >0，但不抛）
    out = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily,
        now="20240301", allow_backfill=True,
    )
    assert "error" not in out
    assert out["recorded"] >= 1


def test_reject_future_as_of(tmp_path):
    """as_of > now → 拒绝未来日。"""
    import pytest

    from factorzen.discovery.forward_track import record_forward_ics

    daily, _d1, _d2, _d3 = _daily_3day()
    _write_lib(tmp_path, "ashare", [
        _lib_row("close", status="probation"),
    ])

    with pytest.raises(ValueError, match=r"未来|future"):
        record_forward_ics(
            "ashare", "20240310", root=str(tmp_path), daily=daily,
            now="20240301",
        )


def test_fresh_record_writes_provenance(tmp_path):
    """lag ≤ max_backfill_days → 正常写入；row 含 recorded_at/git_sha/status_at_record。"""
    from factorzen.discovery.forward_track import record_forward_ics

    daily, _d1, _d2, d3 = _daily_3day()
    as_of = _yyyymmdd(d3)  # 20240104
    _write_lib(tmp_path, "ashare", [
        _lib_row("close", status="probation"),
    ])

    out = record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily,
        now="20240106",  # lag=2 ≤ 10
    )
    assert out["recorded"] == 1
    assert out["failed"] == 0

    rows = _read_jsonl(tmp_path / "forward_track" / "ashare.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row.get("recorded_at"), "recorded_at 必须非空"
    assert row.get("git_sha"), "git_sha 必须非空"
    assert row.get("status_at_record") == "probation"
    assert row.get("as_of_lag_days") == 2
    assert "T" in row["recorded_at"]  # ISO wall-clock


def test_cli_forward_track_stale_date_nonzero(tmp_path, monkeypatch, capsys):
    """CLI --date 远古、无 --allow-backfill → 非零；带 --allow-backfill → 0。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    # 空库亦可：门禁在装配前，allow_backfill 后 to_eval 为空 → recorded=0 仍 0 退出
    args_reject = build_parser().parse_args([
        "factor-library", "forward-track",
        "--market", "ashare",
        "--date", "20200101",
        "--root", str(tmp_path),
    ])
    rc_reject = cli_main._cmd_factor_library_forward_track(args_reject)
    assert rc_reject != 0
    err = capsys.readouterr().err
    assert "回灌" in err or "backfill" in err.lower() or "max_backfill" in err or "失败" in err

    args_ok = build_parser().parse_args([
        "factor-library", "forward-track",
        "--market", "ashare",
        "--date", "20200101",
        "--allow-backfill",
        "--root", str(tmp_path),
    ])
    rc_ok = cli_main._cmd_factor_library_forward_track(args_ok)
    assert rc_ok == 0


def test_provenance_persisted_on_disk(tmp_path):
    """读回 ledger 行断言含 recorded_at/git_sha。"""
    from factorzen.discovery.forward_track import record_forward_ics

    daily, _d1, _d2, d3 = _daily_3day()
    as_of = _yyyymmdd(d3)
    _write_lib(tmp_path, "ashare", [
        _lib_row("close", status="probation", universe="csi300"),
    ])
    record_forward_ics(
        "ashare", as_of, root=str(tmp_path), daily=daily, now=as_of,
    )
    path = tmp_path / "forward_track" / "ashare.jsonl"
    raw = path.read_text(encoding="utf-8")
    assert "recorded_at" in raw
    assert "git_sha" in raw
    rows = _read_jsonl(path)
    assert rows[0]["recorded_at"]
    assert rows[0]["git_sha"]
    assert rows[0].get("command") == "forward-track"
