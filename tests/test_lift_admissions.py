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
    # allow_active=True：测 lift_admission 三态 decision→status 映射（默认 cap 见 test_lift_probation_cap）
    out = upsert_lift_admissions(
        rows, market="ashare", root=str(tmp_path), meta=_meta(),
        allow_active=True,
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
        allow_active=True,
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

    def lift_runner(cands, *, active_factor_dfs=None, **kw):
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


def test_rebuild_lift_review_refreshes_admission_ic(tmp_path):
    """复审必须把 runner 现算的 ``admission_ic`` 写回记录（方向权威，不能只留旧值）。

    真实事故：库内 2 条 lift 轨 probation 的 ``admission_ic`` 是 trade_date 格式 P0
    留下的哨兵 ``0.0``；``run_lift_tests`` 每次复审都现算正确值供 ``lift_admission``
    判门，但写回循环只搬 ``lift_*`` 字段 → 哨兵永不刷新 →
    ``forward_review`` 的 ``_sign_from_ic_train(0.0)`` 恒 None → 永判 ``missing_sign``，
    记录卡死 probation 无法晋升/降级。
    """
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
            # 哨兵 0.0：方向解析不出来
            expression="rank(vol)", market="ashare", status="probation",
            admission_track="lift", ic_train=0.01, lift=0.01, admission_ic=0.0,
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
            cands[0]["expression"], lift=0.006, lift_se=0.001,
            lift_second_half=0.003, admission_ic=-0.031,
        )]

    rebuild(
        "ashare", sources=["rank(close)"],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        evaluate=evaluate, git_sha="x", now="2026-07-14", root=str(tmp_path),
        lift_runner=runner, active_factor_dfs={},
    )
    rec = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}["rank(vol)"]
    # 取负值断言：既证明「刷新了」，又证明没被 ic_train(+0.01) 或 |值| 冒名顶替
    assert rec.admission_ic == -0.031, rec.admission_ic

    # 方向已可解析（哨兵 0.0 会返回 None → 卡死 missing_sign）
    from factorzen.discovery.forward_track import _sign_from_ic_train

    assert _sign_from_ic_train(rec.admission_ic) == -1.0


def test_rebuild_lift_review_keeps_admission_ic_when_row_lacks_it(tmp_path):
    """runner 未给 ``admission_ic``（如 error 行）→ 保留旧值，不写 None 抹掉方向。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        rebuild,
    )

    _save_library("ashare", [
        FactorRecord(
            expression="rank(vol)", market="ashare", status="active",
            admission_track="lift", ic_train=0.01, lift=0.01, admission_ic=0.042,
            added_at="2026-07-02", updated_at="2026-07-02",
        ),
    ], root=str(tmp_path))

    rebuild(
        "ashare", sources=[],
        eval_window=("20200101", "20260101"), universe="u", horizon=1,
        evaluate=lambda exprs: [], git_sha="x", now="2026-07-14", root=str(tmp_path),
        lift_runner=lambda cands, **kw: [_lift_row(
            cands[0]["expression"], lift=0.006, lift_se=0.001, lift_second_half=0.003,
        )],
        active_factor_dfs={},
    )
    rec = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}["rank(vol)"]
    assert rec.admission_ic == 0.042, rec.admission_ic


def test_rebuild_lift_review_eval_failure_does_not_demote(tmp_path):
    """求值失败（error 行 / lift=None）**不得**当作「无增量」降级。

    实际事故：``factor-library rebuild`` 的 CLI 路径不开分钟叶子，含 ``i_*`` 叶子的
    lift 记录物化必失败 → row 带 ``error=materialize_failed``、``lift=None`` →
    旧代码交给 ``lift_admission`` 判 reject → 2 条 probation 静默变 ``no_lift``。
    「算不出来」与「算出来没用」是两件事，前者必须保持原状并大声报错。
    """
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        rebuild,
    )

    _save_library("ashare", [
        FactorRecord(
            expression="ts_mean(neg(abs(i_ret_open30)), 20)", market="ashare",
            status="probation", admission_track="lift", admission_decision="active",
            ic_train=0.05, lift=0.0016, lift_se=0.00099, admission_ic=0.0,
            added_at="2026-07-17", updated_at="2026-07-17",
        ),
    ], root=str(tmp_path))

    def runner(cands, **kw):
        # run_lift_tests 的物化失败行形态：error + 整套 lift 字段为 None
        return [{
            "expression": cands[0]["expression"], "error": "materialize_failed",
            "lift": None, "lift_se": None, "lift_first_half": None,
            "lift_second_half": None, "admission_ic": None,
            "lift_metric": "residual_ic_v1",
        }]

    res = rebuild(
        "ashare", sources=[],
        eval_window=("20200101", "20260410"), universe="csi800", horizon=5,
        evaluate=lambda exprs: [], git_sha="x", now="2026-07-19", root=str(tmp_path),
        lift_runner=runner, active_factor_dfs={},
    )
    rec = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}[
        "ts_mean(neg(abs(i_ret_open30)), 20)"
    ]
    assert rec.status == "probation", rec.status
    assert rec.admission_decision == "active", rec.admission_decision
    # 求值失败必须可见：既进结果对象，也进 manifest
    assert rec.expression in res.lift_eval_failed, res.lift_eval_failed
    man = json.loads((Path(tmp_path) / "rebuild_ashare_manifest.json").read_text())
    assert man["n_lift_eval_failed"] == 1
    assert man["n_lift_demoted"] == 0


def test_rebuild_lift_review_empty_rows_does_not_demote(tmp_path):
    """runner 返回空 list（同样是「没算出来」）→ 保持原状 + 计入求值失败。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        rebuild,
    )

    _save_library("ashare", [
        FactorRecord(
            expression="rank(vol)", market="ashare", status="active",
            admission_track="lift", admission_decision="active",
            ic_train=0.05, lift=0.01, added_at="2026-07-17", updated_at="2026-07-17",
        ),
    ], root=str(tmp_path))

    res = rebuild(
        "ashare", sources=[],
        eval_window=("20200101", "20260410"), universe="csi800", horizon=5,
        evaluate=lambda exprs: [], git_sha="x", now="2026-07-19", root=str(tmp_path),
        lift_runner=lambda cands, **kw: [], active_factor_dfs={},
    )
    rec = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}["rank(vol)"]
    assert rec.status == "active", rec.status
    assert res.lift_eval_failed == ["rank(vol)"], res.lift_eval_failed


def test_rebuild_lift_review_still_demotes_on_real_no_lift(tmp_path):
    """对照：真算出来了但增量不够 → 照常降级 no_lift（守卫没把真降级也堵死）。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        rebuild,
    )

    _save_library("ashare", [
        FactorRecord(
            expression="rank(vol)", market="ashare", status="probation",
            admission_track="lift", admission_decision="probation",
            ic_train=0.05, lift=0.01, added_at="2026-07-17", updated_at="2026-07-17",
        ),
    ], root=str(tmp_path))

    res = rebuild(
        "ashare", sources=[],
        eval_window=("20200101", "20260410"), universe="csi800", horizon=5,
        evaluate=lambda exprs: [], git_sha="x", now="2026-07-19", root=str(tmp_path),
        lift_runner=lambda cands, **kw: [_lift_row(
            cands[0]["expression"], lift=0.0, lift_se=0.001,
        )],
        active_factor_dfs={},
    )
    rec = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}["rank(vol)"]
    assert rec.status == "no_lift", rec.status
    assert res.lift_eval_failed == [], res.lift_eval_failed


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


# ── P9：准入 provenance 落盘可重放 ───────────────────────────────────────────


def test_upsert_lift_admissions_persists_admission_provenance(tmp_path):
    """run_lift_tests row → upsert → 读回 FactorRecord 字段与 row 一致。

    residual_ic_v1：无 combine_fn/cv_params；cv_* 键保留值为 None。
    """
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions
    from factorzen.discovery.lift_test import LiftEvalContext, run_lift_tests

    dates = []
    d = date(2024, 1, 2)
    while len(dates) < 50:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    n_stocks = 40  # residual 日守卫 max(30, k+10)
    active = {
        "lib_b": pl.DataFrame({
            "trade_date": [dd for dd in dates for _ in range(n_stocks)],
            "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
            "factor_value": [float(s + 1) for _ in dates for s in range(n_stocks)],
        }),
        "lib_a": pl.DataFrame({
            "trade_date": [dd for dd in dates for _ in range(n_stocks)],
            "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
            "factor_value": [float(s) for _ in dates for s in range(n_stocks)],
        }),
    }
    ret = pl.DataFrame({
        "trade_date": [dd for dd in dates for _ in range(n_stocks)],
        "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
        "ret": [0.01 * s for _ in dates for s in range(n_stocks)],
    })
    cand = pl.DataFrame({
        "trade_date": [dd for dd in dates for _ in range(n_stocks)],
        "ts_code": [f"{s:04d}.SZ" for _ in dates for s in range(n_stocks)],
        "factor_value": [float(s) + 0.5 for _ in dates for s in range(n_stocks)],
    })

    ctx = LiftEvalContext(
        market="ashare",
        prepped=pl.DataFrame({"trade_date": ["x"], "ts_code": ["y"], "close": [1.0]}),
        leaf_map=None,
        horizon=5,
        admission_start="20240120",
        admission_end="20240315",
        profile_name="ashare_v1",
    )
    rows = run_lift_tests(
        [{"expression": "rank(close)", "residual_ic_train": 0.02, "ic_train": 0.03}],
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        block_days=12,
        threshold=0.001,
        ctx=ctx,
        lift_workers=1,
    )
    # 强制 passed 以便 upsert 写入（本测只关心 provenance 落盘）
    rows[0]["lift"] = 0.05
    rows[0]["lift_se"] = 0.001
    rows[0]["lift_first_half"] = 0.04
    rows[0]["lift_second_half"] = 0.06
    rows[0]["passed"] = True
    row = rows[0]

    upsert_lift_admissions(
        [row],
        market="ashare",
        root=str(tmp_path),
        meta=_meta(horizon=5, now="2026-07-14"),
        threshold=0.001,
        se_mult=1.645,
        allow_active=True,
    )
    rec = load_library("ashare", root=str(tmp_path))[0]

    assert rec.admission_start == row["admission_start"] == "20240120"
    assert rec.admission_end == row["admission_end"] == "20240315"
    assert rec.scored_start == row["scored_start"]
    assert rec.scored_end == row["scored_end"]
    assert rec.block_days == row["block_days"] == 12
    # residual_ic_v1：CV 键保留、值为 None（FactorRecord schema 不动）
    assert rec.cv_train_days == row["cv_train_days"] is None
    assert rec.cv_test_days == row["cv_test_days"] is None
    assert rec.baseline_hash == row["baseline_hash"]
    assert rec.baseline_hash is not None
    assert rec.profile_name == row["profile_name"] == "ashare_v1"
    assert rec.frequency == row["frequency"] == "daily"
    assert rec.horizon == 5
    # threshold 来自 row；se_mult 由 upsert 入参注入
    assert rec.lift_threshold == 0.001
    assert rec.lift_se_mult == 1.645


def test_old_jsonl_missing_admission_provenance_fields():
    """旧 jsonl 无 P9 字段 → from_dict 不报错、新字段默认 None。"""
    from factorzen.discovery.factor_library import FactorRecord

    old = {
        "expression": "rank(close)",
        "market": "ashare",
        "ic_train": 0.05,
        "status": "active",
        "admission_track": "lift",
        "lift": 0.01,
        "horizon": 5,
        "eval_start": "20200101",
        "eval_end": "20240101",
    }
    r = FactorRecord.from_dict(old)
    assert r.admission_start is None
    assert r.admission_end is None
    assert r.scored_start is None
    assert r.scored_end is None
    assert r.block_days is None
    assert r.cv_train_days is None
    assert r.cv_test_days is None
    assert r.lift_threshold is None
    assert r.lift_se_mult is None
    assert r.baseline_hash is None
    assert r.profile_name is None
    assert r.frequency is None
    # round-trip 含新键且为 None
    d = r.to_dict()
    for k in (
        "admission_start", "admission_end", "scored_start", "scored_end",
        "block_days", "cv_train_days", "cv_test_days",
        "lift_threshold", "lift_se_mult", "baseline_hash",
        "profile_name", "frequency",
    ):
        assert k in d
        assert d[k] is None


def test_upsert_row_provenance_beats_meta(tmp_path):
    """row 级 admission provenance 优先于 upsert meta 同名字段。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    row = _lift_row(
        "rank(vol)",
        lift=0.02,
        lift_se=0.0,
        lift_second_half=0.03,
        admission_start="20230101",
        admission_end="20230601",
        scored_start="20230105",
        scored_end="20230530",
        block_days=15,
        cv_train_days=90,
        cv_test_days=15,
        baseline_hash="deadbeefcafe0001",
        profile_name="row_profile",
        frequency="daily",
        threshold=0.002,
        lift_se_mult=1.5,
    )
    upsert_lift_admissions(
        [row],
        market="ashare",
        root=str(tmp_path),
        meta=_meta(
            admission_start="19990101",  # 应被 row 压过
            admission_end="19991231",
            block_days=99,
            profile_name="meta_profile",
            baseline_hash="should_not_win",
            now="2026-07-14",
        ),
        # 裁决用低门槛保证写入；落盘 lift_threshold 仍取 row.threshold=0.002
        threshold=0.001,
        se_mult=9.0,  # row.lift_se_mult=1.5 优先
        allow_active=True,
    )
    rec = load_library("ashare", root=str(tmp_path))[0]
    assert rec.admission_start == "20230101"
    assert rec.admission_end == "20230601"
    assert rec.scored_start == "20230105"
    assert rec.scored_end == "20230530"
    assert rec.block_days == 15
    assert rec.cv_train_days == 90
    assert rec.cv_test_days == 15
    assert rec.baseline_hash == "deadbeefcafe0001"
    assert rec.profile_name == "row_profile"
    assert rec.frequency == "daily"
    assert rec.lift_threshold == 0.002
    assert rec.lift_se_mult == 1.5


# ── P1-①: 裸 IC 同号门 ────────────────────────────────────────────────────────

def test_naked_ic_sign_gate_rejects_negative_admission_ic():
    """裸 IC 为负的候选必拒——即使残差 lift 远超门槛。

    **P1-① 口径错配**：准入判据用**残差** IC（`residual_ic_v1`，对库正交化后），
    而部署 `combine_from_library` 是**裸值等权**（z-score 后等权相加，无正交化）。
    等权无法表达负贡献，故裸 IC 为负的因子在部署时是纯拖累。

    实证（2026-07-19，csi300 2020-2026 全窗，同 CV）：库内 85 条 active 有 23 条
    统一窗口裸 IC 为负；剔除后等权组合 rank_ic **0.05601 → 0.06048**、
    ICIR **0.2282 → 0.2416**，**配对 t = +5.338（n=1454）显著**。
    （两组共享 62 因子高度相关，必须配对——独立样本 SE 是配对 SE 的 11 倍，
    用它得 t=+0.486「不显著」，结论相反。）

    代价已知并接受：W4 已证符号反向属**抑制变量效应**（设计行为，非 bug），
    本门会误杀部分真抑制变量；但实测整体净收益为正。结论绑定「等权部署」前提——
    若将来部署改用能表达负贡献的权重，此门需重新评估。
    """
    from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
    from factorzen.discovery.lift_test import lift_admission

    thr = DEFAULT_LIFT_THRESHOLD
    strong = {"lift": thr * 10, "lift_se": 0.0, "lift_second_half": 0.01}

    # 裸 IC 为负 → 拒（lift 再高也拒）
    assert lift_admission({**strong, "admission_ic": -0.0001}, threshold=thr) == "reject"
    assert lift_admission({**strong, "admission_ic": -0.02}, threshold=thr) == "reject"
    # 裸 IC 非负 → 正常裁决
    assert lift_admission({**strong, "admission_ic": 0.0}, threshold=thr) == "active"
    assert lift_admission({**strong, "admission_ic": 0.03}, threshold=thr) == "active"


def test_naked_ic_sign_gate_skips_when_field_absent():
    """`admission_ic` 缺失 → **跳过**该门，不因缺失而拒。

    与 `lift_se` 缺失的处理**有意不同**：SE 缺失意味着算不出 bar（区间证据不完整，
    必拒）；裸 IC 缺失只是少一道额外门——历史库记录、`lift_null` 零假设校准等
    路径本就不带该字段，按缺失即拒会把它们全部误杀。
    """
    from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
    from factorzen.discovery.lift_test import lift_admission

    thr = DEFAULT_LIFT_THRESHOLD
    base = {"lift": thr * 10, "lift_se": 0.0, "lift_second_half": 0.01}
    assert lift_admission(base, threshold=thr) == "active"                    # 无该键
    assert lift_admission({**base, "admission_ic": None}, threshold=thr) == "active"
    # 非有限值同样跳过（不可判，不等于负）
    assert lift_admission({**base, "admission_ic": float("nan")}, threshold=thr) == "active"


def test_naked_ic_sign_gate_has_escape_hatch():
    """`require_positive_naked_ic=False` 关闭该门——留对照/复检逃生口。

    与 CLAUDE.md「护栏咬合：passed 默认参与筛选，留 --all 逃生口」同款。
    """
    from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD
    from factorzen.discovery.lift_test import lift_admission

    thr = DEFAULT_LIFT_THRESHOLD
    row = {"lift": thr * 10, "lift_se": 0.0, "lift_second_half": 0.01,
           "admission_ic": -0.02}
    assert lift_admission(row, threshold=thr) == "reject"
    assert lift_admission(
        row, threshold=thr, require_positive_naked_ic=False,
    ) == "active"
