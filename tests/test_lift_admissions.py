"""lift 准入轨道：upsert_lift_admissions + rebuild lift 复审 + 库兼容。"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

# ── helpers ──────────────────────────────────────────────────────────────────

def _lift_row(expr, *, lift, lift_se=0.0, lift_second_half=0.01,
              lift_first_half=0.01, baseline=0.04, **extra):
    d = {
        "expression": expr,
        "lift": lift,
        "lift_se": lift_se,
        "lift_first_half": lift_first_half,
        "lift_second_half": lift_second_half,
        "baseline": baseline,
    }
    d.update(extra)
    return d


def _meta(**kw):
    base = {
        "session_dir": "sess/abc",
        "run_id": "run42",
        "universe": "csi300",
        "eval_start": "20200101",
        "eval_end": "20260101",
        "horizon": 5,
        "git_sha": "deadbeef",
        "now": "2026-07-14",
    }
    base.update(kw)
    return base


def _daily_tiny(n_days=20, n_stocks=5):
    rows = []
    d0 = date(2024, 1, 2)
    for i in range(n_days * 2):
        d = d0 + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        if len({r["trade_date"] for r in rows}) >= n_days:
            break
        for s in range(n_stocks):
            rows.append({
                "trade_date": d, "ts_code": f"{s:06d}.SH",
                "close": 10.0 + s, "close_adj": 10.0 + s,
                "open_adj": 10.0, "high_adj": 10.1, "low_adj": 9.9,
                "open": 10.0, "high": 10.1, "low": 9.9, "pre_close": 10.0,
                "vol": 1e5, "amount": 1e7,
            })
    return pl.DataFrame(rows)


# ── upsert_lift_admissions ───────────────────────────────────────────────────

def test_upsert_lift_admissions_three_states(tmp_path):
    """active / probation / reject 三态计数与落盘字段。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    rows = [
        # lift ≥ DEFAULT_LIFT_THRESHOLD 且 second_half > 0 → active
        _lift_row("rank(close)", lift=0.005, lift_se=0.001, lift_second_half=0.004,
                  ic_train=0.02, holdout_ic=0.01),
        # lift ≥ 门槛但 second_half ≤ 0 → probation
        _lift_row("rank(open)", lift=0.004, lift_se=0.001, lift_second_half=-0.001,
                  ic_train=0.01),
        # lift 过低 → reject
        _lift_row("rank(vol)", lift=0.0001, lift_se=0.0, lift_second_half=0.01),
    ]
    out = upsert_lift_admissions(
        rows, market="ashare", root=str(tmp_path), meta=_meta(),
    )
    assert out["added_active"] == 1
    assert out["added_probation"] == 1
    assert out["rejected"] == 1
    assert out["errors"] == []

    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert set(lib) == {"rank(close)", "rank(open)"}
    a = lib["rank(close)"]
    assert a.status == "active"
    assert a.admission_track == "lift"
    assert abs(a.lift - 0.005) < 1e-12
    assert abs(a.lift_se - 0.001) < 1e-12
    assert abs(a.lift_second_half - 0.004) < 1e-12
    assert abs(a.lift_first_half - 0.01) < 1e-12
    assert abs(a.lift_baseline - 0.04) < 1e-12
    assert abs(a.ic_train - 0.02) < 1e-12
    assert a.source_run_id == "run42"
    assert a.source_session_dir == "sess/abc"
    assert a.universe == "csi300"
    assert a.eval_start == "20200101" and a.eval_end == "20260101"
    assert a.horizon == 5
    assert a.git_sha == "deadbeef"

    p = lib["rank(open)"]
    assert p.status == "probation"
    assert p.admission_track == "lift"


def test_upsert_lift_admissions_bad_row_tolerant(tmp_path):
    """坏行容错：一行炸不崩整批。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    rows = [
        "not-a-dict",
        _lift_row("rank(close)", lift=0.01, lift_se=0.0, lift_second_half=0.01),
        {"lift": 0.01},  # missing expression
    ]
    out = upsert_lift_admissions(
        rows, market="ashare", root=str(tmp_path), meta=_meta(),
    )
    assert out["added_active"] == 1
    assert len(out["errors"]) == 2
    lib = load_library("ashare", root=str(tmp_path))
    assert len(lib) == 1 and lib[0].expression == "rank(close)"


def test_upsert_lift_admissions_duplicate_updates(tmp_path):
    """重复 expression：更新指标与 status，不重复添加。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    upsert_lift_admissions(
        [_lift_row("rank(close)", lift=0.01, lift_se=0.0, lift_second_half=0.01,
                   ic_train=0.02)],
        market="ashare", root=str(tmp_path),
        meta=_meta(now="2026-07-01"),
    )
    out2 = upsert_lift_admissions(
        # 后半段变负 → probation；指标刷新
        [_lift_row("rank(close)", lift=0.008, lift_se=0.0, lift_second_half=-0.001,
                   ic_train=0.03)],
        market="ashare", root=str(tmp_path),
        meta=_meta(now="2026-07-14", run_id="run99"),
    )
    assert out2["added_probation"] == 1 and out2["added_active"] == 0
    lib = load_library("ashare", root=str(tmp_path))
    assert len(lib) == 1
    r = lib[0]
    assert r.status == "probation"
    assert r.added_at == "2026-07-01"
    assert r.updated_at == "2026-07-14"
    assert abs(r.ic_train - 0.03) < 1e-12
    assert abs(r.lift - 0.008) < 1e-12
    assert r.source_run_id == "run99"


def test_upsert_lift_admissions_meta_provenance(tmp_path):
    """meta provenance 字段落盘。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    upsert_lift_admissions(
        [_lift_row("ts_mean(close, 5)", lift=0.01, lift_se=0.0, lift_second_half=0.02)],
        market="crypto", root=str(tmp_path),
        meta=_meta(
            session_dir="/tmp/sess",
            run_id="lift_batch_1",
            universe="perp",
            eval_start="20210101",
            eval_end="20240601",
            horizon=10,
            git_sha="abc123",
            now="2026-07-14",
        ),
    )
    r = load_library("crypto", root=str(tmp_path))[0]
    assert r.market == "crypto"
    assert r.source_session_dir == "/tmp/sess"
    assert r.source_run_id == "lift_batch_1"
    assert r.universe == "perp"
    assert r.eval_start == "20210101" and r.eval_end == "20240601"
    assert r.horizon == 10
    assert r.git_sha == "abc123"
    assert r.admission_track == "lift"


def test_old_jsonl_missing_new_fields_loads(tmp_path):
    """旧 jsonl 无新字段 → 读取时默认值兼容。"""
    from factorzen.discovery.factor_library import FactorRecord, load_library

    path = Path(tmp_path) / "ashare.jsonl"
    # 故意只写旧字段
    old = {
        "expression": "rank(close)",
        "market": "ashare",
        "ic_train": 0.05,
        "holdout_ic": 0.04,
        "status": "active",
        "added_at": "2026-01-01",
        "updated_at": "2026-01-01",
    }
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    lib = load_library("ashare", root=str(tmp_path))
    assert len(lib) == 1
    r = lib[0]
    assert r.admission_track == "single"
    assert r.lift_se is None
    assert r.lift_first_half is None
    assert r.lift_second_half is None
    assert r.holdout_n_days is None
    # 写回 round-trip 带上新字段默认值
    d = r.to_dict()
    assert d["admission_track"] == "single"
    r2 = FactorRecord.from_dict(d)
    assert r2.admission_track == "single" and r2.status == "active"


def test_single_upsert_persists_holdout_n_days(tmp_path):
    """single 轨 upsert：调用方传 n_holdout_days / holdout_n_days 则落盘。"""
    from factorzen.discovery.factor_library import load_library, upsert

    upsert(
        "ashare",
        [{"expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
          "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 291}],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        run_id="r", session_dir="s", git_sha="a", now="2026-07-14",
        root=str(tmp_path),
    )
    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.holdout_n_days == 291
    assert r.admission_track == "single"


# ── build_library_pool 与 admission_track ────────────────────────────────────

def test_build_library_pool_by_status_not_track(tmp_path):
    """admission_track 不影响池构建，只有 status 起作用。

    - 默认 statuses=("active",)：含 lift 轨 active，不含 probation/no_lift
    - statuses=("active","probation") 可选入 probation
    """
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        build_library_pool,
    )

    recs = [
        FactorRecord(
            expression="rank(close)", market="ashare", status="active",
            admission_track="single", ic_train=0.05,
        ),
        FactorRecord(
            expression="rank(open)", market="ashare", status="active",
            admission_track="lift", ic_train=0.04, lift=0.01,
        ),
        FactorRecord(
            expression="rank(vol)", market="ashare", status="probation",
            admission_track="lift", ic_train=0.01, lift=0.002,
        ),
        FactorRecord(
            expression="rank(high)", market="ashare", status="no_lift",
            admission_track="lift", ic_train=0.01, lift=0.0,
        ),
    ]
    _save_library("ashare", recs, root=str(tmp_path))
    daily = _daily_tiny()

    pool = build_library_pool("ashare", daily, root=str(tmp_path))
    assert "rank(close)" in pool  # single active
    assert "rank(open)" in pool   # lift active 参与默认池
    assert "rank(vol)" not in pool
    assert "rank(high)" not in pool

    pool2 = build_library_pool(
        "ashare", daily, root=str(tmp_path), statuses=("active", "probation"),
    )
    assert "rank(vol)" in pool2
    assert "rank(high)" not in pool2  # no_lift 仍排除


# ── rebuild lift 复审 ────────────────────────────────────────────────────────

def test_rebuild_lift_review_with_mock_runner(tmp_path):
    """混合库：2 single active + 1 lift active + 1 lift probation；
    mock lift_runner → single 不变、lift 按结果转 active/probation/no_lift、manifest 计数。
    """
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        rebuild,
    )

    recs = [
        FactorRecord(
            expression="rank(close)", market="ashare", status="active",
            admission_track="single", ic_train=0.05, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ),
        FactorRecord(
            expression="rank(open)", market="ashare", status="active",
            admission_track="single", ic_train=0.04, holdout_ic=0.03,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ),
        FactorRecord(
            expression="rank(vol)", market="ashare", status="active",
            admission_track="lift", ic_train=0.01, holdout_ic=0.0,
            lift=0.01, lift_se=0.001, lift_second_half=0.005,
            added_at="2026-07-02", updated_at="2026-07-02",
        ),
        FactorRecord(
            expression="rank(high)", market="ashare", status="probation",
            admission_track="lift", ic_train=0.008, holdout_ic=-0.001,
            lift=0.003, lift_se=0.001, lift_second_half=-0.001,
            added_at="2026-07-02", updated_at="2026-07-02",
        ),
    ]
    _save_library("ashare", recs, root=str(tmp_path))

    def evaluate(exprs):
        # single 源只返回 close/open
        out = []
        for e in exprs:
            if e == "rank(close)":
                out.append({
                    "expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
                    "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 100,
                })
            elif e == "rank(open)":
                out.append({
                    "expression": "rank(open)", "ic_train": 0.04, "holdout_ic": 0.03,
                    "dsr_pvalue": 0.2, "n_train": 100, "n_holdout_days": 100,
                })
        return out

    calls: list[str] = []

    def lift_runner(cands, *, active_factor_dfs=None, combine_fn=None, **kw):
        expr = cands[0]["expression"]
        calls.append(expr)
        # 新池应含 single active
        assert active_factor_dfs is not None
        if expr == "rank(vol)":
            # 复审仍 active
            return [_lift_row(expr, lift=0.006, lift_se=0.001, lift_second_half=0.003,
                              baseline=0.05)]
        if expr == "rank(high)":
            # 复审无增量 → no_lift
            return [_lift_row(expr, lift=0.0001, lift_se=0.0, lift_second_half=0.01,
                              baseline=0.05)]
        return [_lift_row(expr, lift=0.0)]

    rebuild(
        "ashare",
        sources=["rank(close)", "rank(open)"],
        eval_window=("20200101", "20260101"),
        universe="csi300", horizon=1,
        evaluate=evaluate, git_sha="rebuild1", now="2026-07-14",
        root=str(tmp_path), fresh=True,
        lift_runner=lift_runner,
        active_factor_dfs={
            "rank(close)": pl.DataFrame({
                "trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SH"],
                "factor_value": [1.0],
            }),
            "rank(open)": pl.DataFrame({
                "trade_date": [date(2024, 1, 2)], "ts_code": ["000001.SH"],
                "factor_value": [2.0],
            }),
        },
    )

    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    # single 轨仍 active
    assert lib["rank(close)"].status == "active"
    assert lib["rank(close)"].admission_track == "single"
    assert lib["rank(open)"].status == "active"
    assert lib["rank(open)"].admission_track == "single"
    # lift 轨按 mock
    assert lib["rank(vol)"].status == "active"
    assert lib["rank(vol)"].admission_track == "lift"
    assert abs(lib["rank(vol)"].lift - 0.006) < 1e-12
    assert lib["rank(high)"].status == "no_lift"
    assert lib["rank(high)"].admission_track == "lift"
    assert set(calls) == {"rank(vol)", "rank(high)"}

    man = json.loads((Path(tmp_path) / "rebuild_ashare_manifest.json").read_text())
    assert man["n_lift_reviewed"] == 2
    assert man["n_lift_active"] == 1
    assert man["n_lift_probation"] == 0
    assert man["n_lift_demoted"] == 1
    assert man["n_lift_evaluated"] == 2
    assert "lift_review_error" not in man


def test_rebuild_lift_review_exception_preserves_single(tmp_path):
    """复审抛异常 → single 轨结果完好 + error 记录 + lift 轨保持原状。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        rebuild,
    )

    recs = [
        FactorRecord(
            expression="rank(close)", market="ashare", status="active",
            admission_track="single", ic_train=0.05, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ),
        FactorRecord(
            expression="rank(vol)", market="ashare", status="active",
            admission_track="lift", ic_train=0.01, lift=0.01,
            lift_second_half=0.005, added_at="2026-07-02", updated_at="2026-07-02",
        ),
    ]
    _save_library("ashare", recs, root=str(tmp_path))

    def evaluate(exprs):
        return [{
            "expression": "rank(close)", "ic_train": 0.06, "holdout_ic": 0.05,
            "dsr_pvalue": 0.1, "n_train": 120, "n_holdout_days": 100,
        }]

    def boom_runner(cands, **kw):
        raise RuntimeError("lgbm exploded")

    rebuild(
        "ashare", sources=["rank(close)"],
        eval_window=("20200101", "20260101"),
        universe="u", horizon=1, evaluate=evaluate, git_sha="x",
        now="2026-07-14", root=str(tmp_path), fresh=True,
        lift_runner=boom_runner,
    )
    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    # single 轨完好（指标已更新）
    assert lib["rank(close)"].status == "active"
    assert abs(lib["rank(close)"].ic_train - 0.06) < 1e-12
    # lift 轨保持原状
    assert lib["rank(vol)"].status == "active"
    assert lib["rank(vol)"].admission_track == "lift"
    assert abs(lib["rank(vol)"].lift - 0.01) < 1e-12

    man = json.loads((Path(tmp_path) / "rebuild_ashare_manifest.json").read_text())
    assert "lift_review_error" in man
    assert "lgbm exploded" in man["lift_review_error"]
    assert man["n_lift_evaluated"] == 0


def test_rebuild_lift_to_probation(tmp_path):
    """lift 复审：总量过但 second_half≤0 → probation。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        rebuild,
    )

    _save_library("ashare", [
        FactorRecord(
            expression="rank(close)", market="ashare", status="active",
            admission_track="single", ic_train=0.05, holdout_ic=0.04,
            n_train=100, added_at="2026-07-01", updated_at="2026-07-01",
        ),
        FactorRecord(
            expression="rank(vol)", market="ashare", status="active",
            admission_track="lift", ic_train=0.01, lift=0.01,
            added_at="2026-07-02", updated_at="2026-07-02",
        ),
    ], root=str(tmp_path))

    def evaluate(exprs):
        return [{
            "expression": "rank(close)", "ic_train": 0.05, "holdout_ic": 0.04,
            "n_train": 100, "n_holdout_days": 100,
        }]

    def runner(cands, **kw):
        return [_lift_row(
            cands[0]["expression"], lift=0.005, lift_se=0.001, lift_second_half=0.0,
        )]

    rebuild(
        "ashare", sources=["rank(close)"],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        evaluate=evaluate, git_sha="x", now="2026-07-14", root=str(tmp_path),
        lift_runner=runner, active_factor_dfs={},
    )
    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert lib["rank(vol)"].status == "probation"
    man = json.loads((Path(tmp_path) / "rebuild_ashare_manifest.json").read_text())
    assert man["n_lift_probation"] == 1
    assert man["n_lift_demoted"] == 0
    assert man["n_lift_active"] == 0


def test_upsert_lift_admissions_never_overwrites_single_track_active(tmp_path):
    """single 轨 active 记录不被 lift 批次覆盖/降级（与 rebuild 侧守卫同语义）。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    single = FactorRecord(
        expression="rank(close)", market="ashare", status="active",
        ic_train=0.03, holdout_ic=0.02,
        eval_start="20200101", eval_end="20260101", universe="csi300",
        horizon=5, added_at="2026-07-01", updated_at="2026-07-01",
    )
    _save_library("ashare", [single], root=str(tmp_path))

    out = upsert_lift_admissions(
        # 同表达式的 lift 行本会判 probation（second_half<0）——不得改写 single 记录
        [_lift_row("rank(close)", lift=0.004, lift_se=0.001,
                   lift_second_half=-0.001)],
        market="ashare", root=str(tmp_path), meta=_meta(),
    )
    assert out["added_active"] == 0 and out["added_probation"] == 0
    assert out.get("skipped_single_track", 0) == 1

    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    rec = lib["rank(close)"]
    assert rec.status == "active"
    assert (rec.admission_track or "single") == "single"
    assert rec.ic_train == 0.03  # 指标未被改写


# ── C1: lift_admission SE 契约 ───────────────────────────────────────────────

def test_lift_admission_nonfinite_se_rejects():
    """SE 缺失/非有限 → reject；finite 0.0 仍合法，过门可 active/probation。"""
    from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
    from factorzen.discovery.lift_test import lift_admission

    thr = DEFAULT_LIFT_THRESHOLD
    assert lift_admission({
        "lift": thr, "lift_se": None, "lift_second_half": 0.01,
    }, threshold=thr) == "reject"
    assert lift_admission({
        "lift": thr, "lift_se": float("nan"), "lift_second_half": 0.01,
    }, threshold=thr) == "reject"
    assert lift_admission({
        "lift": thr, "lift_se": float("inf"), "lift_second_half": 0.01,
    }, threshold=thr) == "reject"
    # finite 0.0 合法：bar = threshold
    assert lift_admission({
        "lift": thr, "lift_se": 0.0, "lift_second_half": 0.01,
    }, threshold=thr) == "active"
    assert lift_admission({
        "lift": thr, "lift_se": 0.0, "lift_second_half": -0.001,
    }, threshold=thr) == "probation"


# ── C2: upsert 复测 reject 降级 no_lift ──────────────────────────────────────

def test_upsert_lift_reject_demotes_lift_active(tmp_path):
    """lift 轨 active 复测 reject → no_lift；写回指标 + demoted 计数 + 落盘一致。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    prev = FactorRecord(
        expression="rank(vol)", market="ashare", status="active",
        admission_track="lift", ic_train=0.01, lift=0.01, lift_se=0.001,
        lift_second_half=0.005, lift_first_half=0.008, lift_baseline=0.04,
        added_at="2026-07-01", updated_at="2026-07-01",
        source_run_id="old_run", source_session_dir="old_sess",
    )
    _save_library("ashare", [prev], root=str(tmp_path))

    out = upsert_lift_admissions(
        [_lift_row(
            "rank(vol)", lift=-0.01, lift_se=0.002, lift_second_half=-0.02,
            lift_first_half=-0.005, baseline=0.05,
        )],
        market="ashare", root=str(tmp_path),
        meta=_meta(now="2026-07-14", run_id="retest99"),
    )
    assert out.get("demoted_no_lift", 0) == 1
    assert out["rejected"] == 0
    assert out["added_active"] == 0 and out["added_probation"] == 0

    lib = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    r = lib["rank(vol)"]
    assert r.status == "no_lift"
    assert r.admission_track == "lift"
    assert abs(r.lift - (-0.01)) < 1e-12
    assert abs(r.lift_se - 0.002) < 1e-12
    assert abs(r.lift_first_half - (-0.005)) < 1e-12
    assert abs(r.lift_second_half - (-0.02)) < 1e-12
    assert abs(r.lift_baseline - 0.05) < 1e-12
    assert r.added_at == "2026-07-01"
    assert r.updated_at == "2026-07-14"
    assert r.source_run_id == "retest99"


def test_upsert_lift_reject_demotes_lift_probation(tmp_path):
    """lift 轨 probation 复测 reject → no_lift。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    prev = FactorRecord(
        expression="rank(high)", market="ashare", status="probation",
        admission_track="lift", ic_train=0.008, lift=0.003, lift_se=0.001,
        lift_second_half=-0.001, added_at="2026-07-02", updated_at="2026-07-02",
    )
    _save_library("ashare", [prev], root=str(tmp_path))

    out = upsert_lift_admissions(
        [_lift_row("rank(high)", lift=-0.01, lift_se=0.0, lift_second_half=0.01)],
        market="ashare", root=str(tmp_path), meta=_meta(),
    )
    assert out.get("demoted_no_lift", 0) == 1
    assert out["rejected"] == 0
    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.expression == "rank(high)"
    assert r.status == "no_lift"
    assert r.admission_track == "lift"


def test_upsert_lift_reject_does_not_touch_single_track(tmp_path):
    """single 轨 active 复测 reject → 状态不变、计 rejected、无 demoted。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    single = FactorRecord(
        expression="rank(close)", market="ashare", status="active",
        admission_track="single", ic_train=0.03, holdout_ic=0.02,
        added_at="2026-07-01", updated_at="2026-07-01",
    )
    _save_library("ashare", [single], root=str(tmp_path))

    out = upsert_lift_admissions(
        [_lift_row("rank(close)", lift=-0.01, lift_se=0.0, lift_second_half=0.01)],
        market="ashare", root=str(tmp_path), meta=_meta(),
    )
    assert out["rejected"] == 1
    assert out.get("demoted_no_lift", 0) == 0
    assert out.get("skipped_single_track", 0) == 0  # reject 路径不计 skip

    r = load_library("ashare", root=str(tmp_path))[0]
    assert r.status == "active"
    assert (r.admission_track or "single") == "single"
    assert r.ic_train == 0.03


def test_upsert_lift_reject_new_expression_not_stored(tmp_path):
    """新表达式 reject → 不入库、计 rejected。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    out = upsert_lift_admissions(
        [_lift_row("rank(low)", lift=-0.01, lift_se=0.0, lift_second_half=0.01)],
        market="ashare", root=str(tmp_path), meta=_meta(),
    )
    assert out["rejected"] == 1
    assert out.get("demoted_no_lift", 0) == 0
    assert out["added_active"] == 0 and out["added_probation"] == 0
    assert load_library("ashare", root=str(tmp_path)) == []
