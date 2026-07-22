"""合并自: test_library_targeted_rebuild.py, test_combine_library_cli.py
目标: test_library_cli.py

--- 来源 test_library_targeted_rebuild.py ---
定向重估：``rebuild(only=[...])`` 只重估指定子集，不触发全局贪心去相关级联。

语义（见 factor_library.rebuild docstring）：
- 绝不清库（`fresh` 被强制 False）；
- 只评估 `only` 子集；lift 轨复审也只覆盖子集；
- 去相关 **只降不升**（`preserve_status=True`）：可下调 correlated，绝不上调 active。

--- 来源 test_combine_library_cli.py ---
test_combine_from_library.py：combine from-library：因子库选品 → 物化 → 四方法 OOS。
test_combine_cli_smoke.py：fz combine run CLI 冒烟。
test_library_provider.py：registry library provider：load_library_factors 注入 expression 型（Batch 2）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.cli.main import main
from factorzen.discovery.library_provider import load_library_factors
from factorzen.pipelines import factor_combine


# ==== 来自 test_library_targeted_rebuild.py ====
def _panel(vals_per_stock, n_days=8, n_stocks=None):
    """[trade_date, ts_code, factor_value]：每股取 vals_per_stock[i]，每日相同。"""
    n_stocks = n_stocks if n_stocks is not None else len(vals_per_stock)
    rows = []
    for d in range(n_days):
        dt = date(2024, 1, 2) + timedelta(days=d)
        for i in range(n_stocks):
            rows.append({"trade_date": dt, "ts_code": f"{i:06d}.SH",
                         "factor_value": float(vals_per_stock[i])})
    return pl.DataFrame(rows)


def _cand(expr, *, ic_train=0.05, holdout_ic=0.04, dsr_pvalue=0.2, ir_train=0.4,
          n_train=200, **extra):
    d = {"expression": expr, "ic_train": ic_train, "holdout_ic": holdout_ic,
         "dsr_pvalue": dsr_pvalue, "ir_train": ir_train, "n_train": n_train}
    d.update(extra)
    return d


def _rec(expr, **kw):
    from factorzen.discovery.factor_library import FactorRecord
    base = {"expression": expr, "market": "ashare", "ic_train": 0.05,
            "ir_train": 0.4, "holdout_ic": 0.04, "n_train": 200,
            "status": "active", "added_at": "2026-07-01", "updated_at": "2026-07-01"}
    base.update(kw)
    return FactorRecord(**base)


def _by_expr(root):
    from factorzen.discovery.factor_library import load_library
    return {r.expression: r for r in load_library("ashare", root=str(root))}


# ── 1. 只评估子集 + 子集外记录零改动 ────────────────────────────────────────

def test_targeted_rebuild_subset_suite(tmp_path):
    """only 子集之外的库记录一个字节都不能动（含 updated_at / 指标 / status）。；lift 轨复审是最贵的一步：定向模式下只能对子集跑 add-one lift。；定向 + fresh=True：库文件不得被清空（否则子集外记录全丢）。；only 里不在库的表达式：不静默吞，记 manifest.targeted_missing。；only 走与库同一套规范形：写法带多余空格也能命中。"""
    # -- 原 test_targeted_rebuild_evaluates_only_subset_and_leaves_others_untouched --
    def _section_0_test_targeted_rebuild_evaluates_only_subset_and_leaves_others_untouched(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [
            _rec("rank(close)", ic_train=0.05, ir_train=0.4),
            _rec("rank(open)", ic_train=0.06, ir_train=0.5),
            _rec("rank(high)", ic_train=0.07, ir_train=0.6, status="correlated",
                 correlated_with="rank(open)", max_corr_in_lib=0.91),
        ], root=str(tmp_path))

        seen: dict = {}

        def evaluate(exprs):
            seen["exprs"] = list(exprs)
            # 只该拿到定向目标；返回刷新后的指标
            return [_cand(e, ic_train=0.11, ir_train=0.9, holdout_ic=0.12) for e in exprs]

        res = rebuild("ashare", sources=["rank(close)", "rank(open)", "rank(high)"],
                      eval_window=("20200101", "20260101"), universe="csi300", horizon=1,
                      evaluate=evaluate, git_sha="x", now="2026-07-19",
                      only=["rank(close)"], root=str(tmp_path))

        # evaluate 只看到定向目标（不是全部 3 个源）
        assert seen["exprs"] == ["rank(close)"], seen

        lib = _by_expr(tmp_path)
        assert set(lib) == {"rank(close)", "rank(open)", "rank(high)"}   # 一条都没丢
        # 目标：指标已刷新
        assert lib["rank(close)"].ic_train == 0.11
        assert lib["rank(close)"].updated_at == "2026-07-19"
        # 非目标：完全不动
        assert lib["rank(open)"].ic_train == 0.06
        assert lib["rank(open)"].updated_at == "2026-07-01"
        assert lib["rank(high)"].status == "correlated"
        assert lib["rank(high)"].correlated_with == "rank(open)"
        assert lib["rank(high)"].updated_at == "2026-07-01"
        assert res.updated == 1 and res.added == 0

        man = __import__("json").loads(
            (Path(tmp_path) / "rebuild_ashare_manifest.json").read_text(encoding="utf-8"))
        assert man["targeted"] is True
        assert man["n_targeted"] == 1
        assert man["fresh"] is False          # 定向绝不清库

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_targeted_rebuild_evaluates_only_subset_and_leaves_others_untouched(_tp0)

    # -- 原 test_targeted_rebuild_limits_lift_review_to_subset --
    def _section_1_test_targeted_rebuild_limits_lift_review_to_subset(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [
            _rec("rank(close)", admission_track="lift", status="active", lift=0.004),
            _rec("rank(open)", admission_track="lift", status="active", lift=0.005),
        ], root=str(tmp_path))

        called: list[str] = []

        def lift_runner(cands, *, active_factor_dfs=None, **kw):
            called.extend(c["expression"] for c in cands)
            return [{"expression": c["expression"], "lift": 0.0001,   # < 阈值 → reject
                     "lift_se": 0.0, "lift_metric": "residual_ic_v1"} for c in cands]

        rebuild("ashare", sources=[], eval_window=("20200101", "20260101"),
                universe=None, horizon=1, evaluate=lambda e: [], git_sha="x",
                now="2026-07-19", lift_runner=lift_runner,
                only=["rank(close)"], root=str(tmp_path))

        assert called == ["rank(close)"], f"lift 复审外溢到子集之外: {called}"
        lib = _by_expr(tmp_path)
        assert lib["rank(close)"].status == "no_lift"        # 目标被复审并降级
        assert lib["rank(open)"].status == "active"          # 非目标原样
        assert lib["rank(open)"].updated_at == "2026-07-01"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_targeted_rebuild_limits_lift_review_to_subset(_tp1)

    # -- 原 test_targeted_rebuild_forces_non_fresh --
    def _section_2_test_targeted_rebuild_forces_non_fresh(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [
            _rec("rank(close)"), _rec("rank(open)"), _rec("rank(high)"),
        ], root=str(tmp_path))

        rebuild("ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
                universe=None, horizon=1,
                evaluate=lambda exprs: [_cand(e) for e in exprs],
                git_sha="x", now="2026-07-19", fresh=True,
                only=["rank(close)"], root=str(tmp_path))

        assert set(_by_expr(tmp_path)) == {"rank(close)", "rank(open)", "rank(high)"}

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_targeted_rebuild_forces_non_fresh(_tp2)

    # -- 原 test_targeted_rebuild_records_missing_targets --
    def _section_3_test_targeted_rebuild_records_missing_targets(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [_rec("rank(close)")], root=str(tmp_path))

        rebuild("ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
                universe=None, horizon=1,
                evaluate=lambda exprs: [_cand(e) for e in exprs],
                git_sha="x", now="2026-07-19",
                only=["rank(close)", "rank(nonexistent_leaf)"], root=str(tmp_path))

        man = __import__("json").loads(
            (Path(tmp_path) / "rebuild_ashare_manifest.json").read_text(encoding="utf-8"))
        assert man["targeted_missing"] == ["rank(nonexistent_leaf)"]

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_targeted_rebuild_records_missing_targets(_tp3)

    # -- 原 test_targeted_rebuild_normalizes_only_expressions --
    def _section_4_test_targeted_rebuild_normalizes_only_expressions(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [_rec("ts_mean(close, 5)")], root=str(tmp_path))

        seen: dict = {}

        def evaluate(exprs):
            seen["exprs"] = list(exprs)
            return [_cand(e, ic_train=0.33) for e in exprs]

        rebuild("ashare", sources=["ts_mean(close, 5)"], eval_window=("20200101", "20260101"),
                universe=None, horizon=1, evaluate=evaluate, git_sha="x", now="2026-07-19",
                only=["ts_mean( close ,5 )"], root=str(tmp_path))

        assert seen["exprs"] == ["ts_mean(close, 5)"]
        assert _by_expr(tmp_path)["ts_mean(close, 5)"].ic_train == 0.33

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_targeted_rebuild_normalizes_only_expressions(_tp4)


# ── 2. 只降不升：核心去相关语义 ──────────────────────────────────────────────

def test_targeted_rebuild_decorr_suite(tmp_path):
    """已判 correlated 的目标，即使重估后与库内 active 全不相关，也**不得**升回 active。；只降不升 ≠ 什么都不做：与库内未重估 active 超阈的目标仍被下调 correlated。"""
    # -- 原 test_targeted_rebuild_never_promotes_correlated_to_active --
    def _section_0_test_targeted_rebuild_never_promotes_correlated_to_active(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [
            _rec("rank(close)", ir_train=0.9),                     # 库内 active（不重估）
            _rec("rank(open)", ir_train=0.2, status="correlated",  # 目标：曾被判重复
                 correlated_with="rank(close)", max_corr_in_lib=0.95),
        ], root=str(tmp_path))

        # 两条面板互相独立 → 去相关不会给出任何 correlated 裁决
        # （逐日截面相关要求当日有效股 ≥30，故 40 只）
        base = [float((i * 37) % 40) for i in range(40)]
        panels = {"rank(close)": _panel(base),
                  "rank(open)": _panel([float((i * 11) % 40) for i in range(40)])}

        def evaluate(exprs):
            return [_cand(e, ic_train=0.09, ir_train=0.8, holdout_ic=0.08) for e in exprs]

        rebuild("ashare", sources=["rank(close)", "rank(open)"],
                eval_window=("20200101", "20260101"), universe=None, horizon=1,
                evaluate=evaluate, git_sha="x", now="2026-07-19",
                materialize=lambda e: panels.get(e),
                only=["rank(open)"], root=str(tmp_path))

        lib = _by_expr(tmp_path)
        # 指标刷新了（证明确实重估过，不是整条跳过）
        assert lib["rank(open)"].ic_train == 0.09
        # 但 status 保持 correlated —— 只降不升
        assert lib["rank(open)"].status == "correlated", "定向重估把 correlated 上调成了 active"

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_targeted_rebuild_never_promotes_correlated_to_active(_tp0)

    # -- 原 test_targeted_rebuild_still_demotes_when_now_correlated --
    def _section_1_test_targeted_rebuild_still_demotes_when_now_correlated(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [
            _rec("rank(close)", ir_train=0.9),                  # 库内 active（不重估，在池中）
            _rec("rank(open)", ir_train=0.2, status="active"),  # 目标：现与上者高度相关
        ], root=str(tmp_path))

        same = [float((i * 37) % 40) for i in range(40)]   # 逐日截面相关要求当日 ≥30 只
        panels = {"rank(close)": _panel(same),
                  "rank(open)": _panel([x * 2.0 + 1.0 for x in same])}   # corr = 1

        def evaluate(exprs):
            return [_cand(e, ic_train=0.09, ir_train=0.3, holdout_ic=0.08) for e in exprs]

        res = rebuild("ashare", sources=["rank(close)", "rank(open)"],
                      eval_window=("20200101", "20260101"), universe=None, horizon=1,
                      evaluate=evaluate, git_sha="x", now="2026-07-19",
                      materialize=lambda e: panels.get(e),
                      only=["rank(open)"], root=str(tmp_path))

        lib = _by_expr(tmp_path)
        assert lib["rank(open)"].status == "correlated"
        assert lib["rank(open)"].correlated_with == "rank(close)"
        assert res.correlated == 1
        # 未重估的那条不受影响
        assert lib["rank(close)"].status == "active"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_targeted_rebuild_still_demotes_when_now_correlated(_tp1)


# ── 3. 状态 / 轨道 / provenance 不被 _record_from_candidate 抹掉 ──────────────

def test_targeted_rebuild_provenance_gate_suite(tmp_path):
    """定向重估只刷新指标：admission_track / hypothesis / lift* / name 不得被抹成 None。；补算场景：候选给了新的 admission_ic → 落盘新值（不是被 prev 的 None 顶回去）。；已在库记录重估后不再满足 library gate：仍写真实指标，但计入 gate_failed。"""
    # -- 原 test_targeted_rebuild_preserves_track_and_provenance_fields --
    def _section_0_test_targeted_rebuild_preserves_track_and_provenance_fields(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [
            _rec("rank(close)", status="probation", admission_track="single",
                 hypothesis="反转", admission_decision="probation",
                 lift=0.0031, lift_baseline=0.02, lift_metric="residual_ic_v1",
                 admission_ic=0.021, name="rev_close", evidence_tier="legacy"),
        ], root=str(tmp_path))

        def evaluate(exprs):
            # 刷新后的候选 dict 不含 hypothesis / lift / name（rebuild 评估器就是这样）
            return [_cand(e, ic_train=0.12, ir_train=1.1, holdout_ic=0.13) for e in exprs]

        rebuild("ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
                universe=None, horizon=1, evaluate=evaluate, git_sha="x", now="2026-07-19",
                only=["rank(close)"], root=str(tmp_path))

        r = _by_expr(tmp_path)["rank(close)"]
        assert r.ic_train == 0.12 and r.holdout_ic == 0.13     # 指标刷新
        assert r.status == "probation"                         # 状态保留
        assert r.admission_track == "single"
        assert r.admission_decision == "probation"
        assert r.hypothesis == "反转"                           # provenance 未被抹
        assert r.lift == 0.0031 and r.lift_metric == "residual_ic_v1"
        assert r.admission_ic == 0.021
        assert r.name == "rev_close"
        assert r.added_at == "2026-07-01"                      # 入库日保留

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_targeted_rebuild_preserves_track_and_provenance_fields(_tp0)

    # -- 原 test_targeted_rebuild_refreshes_admission_ic_when_provided --
    def _section_1_test_targeted_rebuild_refreshes_admission_ic_when_provided(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [_rec("rank(close)", admission_ic=None)], root=str(tmp_path))

        def evaluate(exprs):
            return [_cand(e, admission_ic=0.0177) for e in exprs]

        rebuild("ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
                universe=None, horizon=1, evaluate=evaluate, git_sha="x", now="2026-07-19",
                only=["rank(close)"], root=str(tmp_path))
        assert _by_expr(tmp_path)["rank(close)"].admission_ic == 0.0177

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_targeted_rebuild_refreshes_admission_ic_when_provided(_tp1)

    # -- 原 test_targeted_rebuild_reports_gate_failure_but_refreshes_metrics --
    def _section_2_test_targeted_rebuild_reports_gate_failure_but_refreshes_metrics(tmp_path):
        from factorzen.discovery.factor_library import _save_library, rebuild
        _save_library("ashare", [_rec("rank(close)", ic_train=0.05, holdout_ic=0.04)],
                      root=str(tmp_path))

        def evaluate(exprs):
            # holdout 反号 → library gate 不过
            return [_cand(e, ic_train=0.05, holdout_ic=-0.04) for e in exprs]

        res = rebuild("ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
                      universe=None, horizon=1, evaluate=evaluate, git_sha="x",
                      now="2026-07-19", only=["rank(close)"], root=str(tmp_path))

        r = _by_expr(tmp_path)["rank(close)"]
        assert r.holdout_ic == -0.04, "gate 失败的目标没有刷新指标，库里留了陈旧值"
        assert r.status == "active"                      # status 不由本路径裁决
        assert res.gate_failed == ["rank(close)"]
        man = __import__("json").loads(
            (Path(tmp_path) / "rebuild_ashare_manifest.json").read_text(encoding="utf-8"))
        assert man["targeted_gate_failed"] == ["rank(close)"]

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_targeted_rebuild_reports_gate_failure_but_refreshes_metrics(_tp2)


# ── 4. lift 轨复审只覆盖子集 ────────────────────────────────────────────────


# ── 5. 绝不清库（哪怕调用方传 fresh=True）──────────────────────────────────


# ── 6. gate 失败：刷新指标 + 大声记账，不静默留陈旧值 ──────────────────────


def test_non_targeted_rebuild_still_enforces_gate_for_new_expressions(tmp_path):
    """零回归：非定向 rebuild 的 gate 行为不变（反号候选仍被挡在库外）。"""
    from factorzen.discovery.factor_library import load_library, rebuild

    def evaluate(exprs):
        return [_cand("rank(close)", holdout_ic=0.04),
                _cand("rank(open)", holdout_ic=-0.04)]

    res = rebuild("ashare", sources=["rank(close)", "rank(open)"],
                  eval_window=("20200101", "20260101"), universe=None, horizon=1,
                  evaluate=evaluate, git_sha="x", now="2026-07-19", root=str(tmp_path))
    lib = {r.expression for r in load_library("ashare", root=str(tmp_path))}
    assert lib == {"rank(close)"} and res.skipped == 1


# ── 7. only 目标不在库 / python 型目标 ──────────────────────────────────────


# ── 8. CLI 接线（能力层↔接线层漂移：必须从最外层 parse_args 出发）─────────────

def _patch_cli_for_rebuild(monkeypatch, tmp_path, seen: dict):
    """把 CLI rebuild 的数据装配/源收集/评估器全部打桩，只留 only 透传这条线。"""
    import factorzen.cli.main as cli_main
    from factorzen.discovery import factor_library as fl

    monkeypatch.setattr(cli_main, "_prepare_agent_mining_data",
                        lambda args: (pl.DataFrame({"trade_date": [date(2024, 1, 2)]}),
                                      None, {}))
    monkeypatch.setattr(fl, "collect_source_expressions",
                        lambda market: ["rank(close)", "rank(open)"])

    def _evaluate(exprs):
        seen["exprs"] = list(exprs)
        return [_cand(e, ic_train=0.31) for e in exprs]

    monkeypatch.setattr(fl, "build_library_evaluator",
                        lambda *a, **k: (_evaluate, None))
    monkeypatch.setattr(cli_main, "_lift_admission_str", lambda v: None)
    monkeypatch.setattr(cli_main, "split_holdout", lambda *a, **k: (None, None, None),
                        raising=False)

    orig_rebuild = fl.rebuild

    def rebuild_to_tmp(*a, **kw):
        kw.setdefault("root", str(tmp_path))
        seen["only"] = kw.get("only")
        return orig_rebuild(*a, **kw)

    monkeypatch.setattr(fl, "rebuild", rebuild_to_tmp)


def test_cli_rebuild_only_wiring_suite(tmp_path, capsys):
    """`fz factor-library rebuild --only <expr>`：定向目标真的传到引擎且只评估它。；`--only-file`：一行一条、'#' 注释与空行跳过（上百条批量补账的入口）。；定向旗标给了却解析出空集 → exit 1，绝不静默降级成会重排全库的全量 rebuild。"""
    # -- 原 test_cli_rebuild_only_flag_reaches_engine --
    def _section_0_test_cli_rebuild_only_flag_reaches_engine(mp, tmp_path):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser
        from factorzen.discovery.factor_library import _save_library

        _save_library("ashare", [_rec("rank(close)"), _rec("rank(open)")], root=str(tmp_path))
        seen: dict = {}
        _patch_cli_for_rebuild(mp, tmp_path, seen)

        args = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi300", "--start", "20200101", "--end", "20201231",
            "--only", "rank(close)",
        ])
        assert cli_main._cmd_factor_library_rebuild(args) == 0
        assert seen["only"] == ["rank(close)"]
        assert seen["exprs"] == ["rank(close)"], "CLI --only 没有裁剪评估集"
        lib = _by_expr(tmp_path)
        assert lib["rank(close)"].ic_train == 0.31
        assert lib["rank(open)"].ic_train == 0.05      # 非目标未动

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cli_rebuild_only_flag_reaches_engine(mp, _tp0)

    # -- 原 test_cli_rebuild_only_file --
    def _section_1_test_cli_rebuild_only_file(mp, tmp_path):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser
        from factorzen.discovery.factor_library import _save_library

        _save_library("ashare", [_rec("rank(close)"), _rec("rank(open)")], root=str(tmp_path))
        listing = tmp_path / "targets.txt"
        listing.write_text("# 本批目标\nrank(close)\n\nrank(open)\n", encoding="utf-8")
        seen: dict = {}
        _patch_cli_for_rebuild(mp, tmp_path, seen)

        args = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi300", "--start", "20200101", "--end", "20201231",
            "--only-file", str(listing),
        ])
        assert cli_main._cmd_factor_library_rebuild(args) == 0
        assert seen["only"] == ["rank(close)", "rank(open)"]

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cli_rebuild_only_file(mp, _tp1)

    # -- 原 test_cli_rebuild_empty_only_fails_loudly --
    def _section_2_test_cli_rebuild_empty_only_fails_loudly(mp, tmp_path, capsys):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser

        empty = tmp_path / "empty.txt"
        empty.write_text("# 全是注释\n\n", encoding="utf-8")
        seen: dict = {}
        _patch_cli_for_rebuild(mp, tmp_path, seen)

        args = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi300", "--start", "20200101", "--end", "20201231",
            "--only-file", str(empty),
        ])
        assert cli_main._cmd_factor_library_rebuild(args) == 1
        assert "空目标集" in capsys.readouterr().err
        assert "only" not in seen, "空目标集不该走到 rebuild"

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cli_rebuild_empty_only_fails_loudly(mp, _tp2, capsys)


def test_cli_rebuild_intraday_suite(tmp_path, capsys):
    """`--intraday-leaves` 必须真到达数据装配层。；库里有含 i_* 叶子的记录 → **不给旗标也自动装日内面板**。；引擎报了求值失败 → CLI 非零退出 + stderr 点名（禁止「表面成功」）。"""
    # -- 原 test_cli_rebuild_intraday_flags_reach_data_assembly --
    def _section_0_test_cli_rebuild_intraday_flags_reach_data_assembly(mp, tmp_path):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser
        from factorzen.discovery.factor_library import _save_library

        _save_library("ashare", [_rec("rank(close)")], root=str(tmp_path))
        seen: dict = {}
        _patch_cli_for_rebuild(mp, tmp_path, seen)
        # 自动检测会读库：必须指到 tmp，否则读的是真实工作区的库（真库含 i_* 记录，
        # 会让「不给旗标 → False」这条断言假失败，更糟的是测试依赖本机数据）
        mp.setattr("factorzen.discovery.factor_library.DEFAULT_ROOT",
                            str(tmp_path), raising=False)

        def _spy_prepare(args):
            seen["intraday_leaves"] = getattr(args, "intraday_leaves", None)
            seen["intraday_freq"] = getattr(args, "intraday_freq", None)
            return pl.DataFrame({"trade_date": [date(2024, 1, 2)]}), None, {}

        mp.setattr(cli_main, "_prepare_agent_mining_data", _spy_prepare)

        args = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi800", "--start", "20200101", "--end", "20201231",
            "--only", "rank(close)", "--intraday-leaves", "--intraday-freq", "5min",
        ])
        assert cli_main._cmd_factor_library_rebuild(args) == 0
        assert seen["intraday_leaves"] is True
        assert seen["intraday_freq"] == "5min"

        # 不给旗标、库里也没有 i_* 记录 → False（默认关，零回归）
        args2 = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi800", "--start", "20200101", "--end", "20201231",
            "--only", "rank(close)",
        ])
        assert cli_main._cmd_factor_library_rebuild(args2) == 0
        assert seen["intraday_leaves"] is False

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_cli_rebuild_intraday_flags_reach_data_assembly(mp, _tp0)

    # -- 原 test_cli_rebuild_auto_enables_intraday_from_library --
    def _section_1_test_cli_rebuild_auto_enables_intraday_from_library(mp, tmp_path):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser
        from factorzen.discovery.factor_library import FactorRecord, _save_library

        seen: dict = {}
        _patch_cli_for_rebuild(mp, tmp_path, seen)

        def _spy_prepare(args):
            seen["intraday_leaves"] = getattr(args, "intraday_leaves", None)
            return pl.DataFrame({"trade_date": [date(2024, 1, 2)]}), None, {}

        mp.setattr(cli_main, "_prepare_agent_mining_data", _spy_prepare)
        # CLI 用默认 root 读库来做检测 → 把默认 root 指到 tmp
        mp.setattr(
            "factorzen.discovery.factor_library.DEFAULT_ROOT", str(tmp_path), raising=False,
        )
        _save_library("ashare", [
            FactorRecord(
                expression="ts_mean(neg(abs(i_ret_open30)), 20)", market="ashare",
                status="probation", admission_track="lift", ic_train=0.05,
                added_at="2026-07-17", updated_at="2026-07-17",
            ),
        ], root=str(tmp_path))

        args = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi800", "--start", "20200101", "--end", "20201231",
        ])
        cli_main._cmd_factor_library_rebuild(args)
        assert seen["intraday_leaves"] is True, "库内 i_* 记录未触发日内面板自动装配"

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_cli_rebuild_auto_enables_intraday_from_library(mp, _tp1)

    # -- 原 test_cli_rebuild_exits_nonzero_on_lift_eval_failure --
    def _section_2_test_cli_rebuild_exits_nonzero_on_lift_eval_failure(mp, tmp_path, capsys):
        import factorzen.cli.main as cli_main
        from factorzen.cli.main import build_parser
        from factorzen.discovery import factor_library as fl

        seen: dict = {}
        _patch_cli_for_rebuild(mp, tmp_path, seen)
        mp.setattr(fl, "collect_source_expressions", lambda market: [])
        mp.setattr(fl, "rebuild", lambda *a, **kw: fl.UpsertResult(
            lift_eval_failed=["ts_mean(neg(abs(i_ret_open30)), 20)"],
        ))

        args = build_parser().parse_args([
            "factor-library", "rebuild", "--market", "ashare",
            "--universe", "csi800", "--start", "20200101", "--end", "20201231",
        ])
        assert cli_main._cmd_factor_library_rebuild(args) == 1
        err = capsys.readouterr().err
        assert "求值失败" in err
        assert "i_ret_open30" in err
        # 报错引用的旗标必须真实存在（help 承诺与 parser 定义不许漂移）
        assert "--intraday-leaves" in err
        flagged = build_parser().parse_args(
            ["factor-library", "rebuild", "--intraday-leaves"]
        )
        assert flagged.intraday_leaves is True

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_cli_rebuild_exits_nonzero_on_lift_eval_failure(mp, _tp2, capsys)


# ── 9. 双路径分工：lift 轨目标不得同时走 upsert 和 lift 复审 ────────────────

def test_targeted_lift_track_goes_only_through_lift_review(tmp_path):
    """lift 轨目标只走 lift 复审，不喂 evaluate/upsert。

    全量 rebuild 里 lift 轨记录被抽进 preserved_lift、不进 upsert；定向必须同分工，
    否则同一条记录被两条路径各写一遍（真实库 smoke 抓到的漏）。
    """
    from factorzen.discovery.factor_library import _save_library, rebuild
    _save_library("ashare", [
        _rec("rank(close)", admission_track="single"),
        _rec("rank(open)", admission_track="lift", status="active", lift=0.006),
    ], root=str(tmp_path))

    evaluated: list[str] = []
    lift_called: list[str] = []

    def evaluate(exprs):
        evaluated.extend(exprs)
        return [_cand(e, ic_train=0.22) for e in exprs]

    def lift_runner(cands, *, active_factor_dfs=None, **kw):
        lift_called.extend(c["expression"] for c in cands)
        return [{"expression": c["expression"], "lift": 0.006, "lift_se": 0.001,
                 "lift_second_half": 0.004} for c in cands]

    rebuild("ashare", sources=["rank(close)", "rank(open)"],
            eval_window=("20200101", "20260101"), universe=None, horizon=1,
            evaluate=evaluate, git_sha="x", now="2026-07-19",
            lift_runner=lift_runner,
            only=["rank(close)", "rank(open)"], root=str(tmp_path))

    assert evaluated == ["rank(close)"], f"lift 轨目标被喂进了表达式评估器: {evaluated}"
    assert lift_called == ["rank(open)"]
    lib = _by_expr(tmp_path)
    assert lib["rank(open)"].admission_track == "lift"   # 没被 upsert 打回 single
    assert lib["rank(close)"].ic_train == 0.22

# ==== 来自 test_combine_library_cli.py ====
# ==== 来自 test_combine_from_library.py ====

def _daily(n_stocks=40, n_days=200, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2023, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c, px = f"{i:06d}.SZ", 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "close_adj": px,
                "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
            })
    return pl.DataFrame(rows)


def _write_lib__combine_from_library(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _expr_rec(
    expression: str,
    *,
    name: str | None = None,
    status: str = "active",
    ic_train: float = 0.05,
    **extra,
) -> dict:
    d = {
        "expression": expression,
        "market": "ashare",
        "status": status,
        "kind": "expression",
        "ic_train": ic_train,
        "name": name,
    }
    d.update(extra)
    return d


def test_combine_from_library_behavior_suite(tmp_path):
    """3 条 expression active → 跑通；factors_used 是 name；manifest/返回字段齐全。；probation 默认不入选；statuses 含 probation 则入选。；python 面板注入 + expression 同台进组合；universe=None + python → ValueError。；坏表达式跳过并记入 skipped_materialize；剩余 ≥2 仍跑。；<2 记录 → ValueError；top_n 截断记 truncated_from。；manifest 必须记全窗口/票池/选品参数——否则事后无法判断一次 run 覆盖了什么。"""
    # -- 原 test_combine_from_library_end_to_end --
    def _section_0_test_combine_from_library_end_to_end(tmp_path, mp):
        import factorzen.pipelines.factor_mine as fm

        mp.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

        lib = tmp_path / "lib"
        _write_lib__combine_from_library(lib, "ashare", [
            _expr_rec("rank(close)", name="f_close", ic_train=0.08),
            _expr_rec("ts_mean(vol,5)", name="f_vol", ic_train=0.06),
            _expr_rec("neg(rank(ts_std(close,10)))", name="f_vol_neg", ic_train=0.04),
        ])
        res = factor_combine.combine_from_library(
            market="ashare",
            library_root=str(lib),
            start="20230103",
            end="20231231",
            universe=None,
            horizon=5,
            train_days=60,
            test_days=15,
            decorr_threshold=1.0,
            out_dir=str(tmp_path / "out"),
        )
        comp = res["comparison"]
        methods = set(comp["method"].to_list())
        assert {"equal_weight", "ic_weighted", "max_ir"} <= methods
        assert comp.height >= 3
        # 可读 name，不是 factor_{i} / 表达式原文
        assert set(res["factors_used"]) == {"f_close", "f_vol", "f_vol_neg"}
        assert res["factors_status"] == {
            "f_close": "active", "f_vol": "active", "f_vol_neg": "active",
        }
        assert res["skipped_materialize"] == []
        assert res["dropped_correlated"] == []
        assert res["market"] == "ashare"
        assert res["statuses"] == ["active"]
        assert res["library_hash"] is not None
        assert "run_dir" in res
        assert res.get("truncated_from") is None

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_combine_from_library_end_to_end(_tp0, mp)

    # -- 原 test_combine_from_library_statuses_filter --
    def _section_1_test_combine_from_library_statuses_filter(tmp_path, mp):
        import factorzen.pipelines.factor_mine as fm

        mp.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

        lib = tmp_path / "lib"
        _write_lib__combine_from_library(lib, "ashare", [
            _expr_rec("rank(close)", name="a_active", status="active", ic_train=0.08),
            _expr_rec("rank(vol)", name="a_prob", status="probation", ic_train=0.07),
            _expr_rec("rank(high)", name="b_active", status="active", ic_train=0.05),
        ])
        # 默认 statuses=active → 只有 2 条 active；应能跑
        res = factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            start="20230103", end="20231231",
            train_days=60, test_days=15, decorr_threshold=1.0,
            out_dir=str(tmp_path / "o1"),
        )
        assert "a_prob" not in res["factors_used"]
        assert set(res["factors_used"]) == {"a_active", "b_active"}

        # 含 probation → 3 条入选
        res2 = factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            statuses=("active", "probation"),
            start="20230103", end="20231231",
            train_days=60, test_days=15, decorr_threshold=1.0,
            out_dir=str(tmp_path / "o2"),
        )
        assert "a_prob" in res2["factors_used"]
        assert len(res2["factors_used"]) == 3

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_combine_from_library_statuses_filter(_tp1, mp)

    # -- 原 test_combine_from_library_python_with_expression --
    def _section_2_test_combine_from_library_python_with_expression(tmp_path, mp):
        import factorzen.discovery.factor_library as fl
        import factorzen.pipelines.factor_mine as fm

        mp.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

        daily = _daily()
        # 与网格同口径的假 python 面板
        py_panel = daily.select([
            "trade_date", "ts_code",
            (pl.col("close_adj") * 0.01).alias("factor_value"),
        ])

        def _fake_mat(r, df, *, market, universe, python_materializer, start, end):
            # 只服务我们的 fake_py；其它走原路径（本测无）
            if (r.name or "") == "fake_py" or fl.is_python_identity(r.expression):
                return (
                    df.select(["trade_date", "ts_code"])
                    .join(py_panel, on=["trade_date", "ts_code"], how="inner")
                )
            return None

        mp.setattr(fl, "_materialize_python_on_grid", _fake_mat)

        lib = tmp_path / "lib"
        py_key = fl.python_identity("fake_py")
        _write_lib__combine_from_library(lib, "ashare", [
            _expr_rec("rank(close)", name="e1", ic_train=0.08),
            {
                "expression": py_key, "market": "ashare", "status": "active",
                "kind": "python", "name": "fake_py", "impl": "fake_py",
                "ic_train": 0.06,
            },
            _expr_rec("ts_mean(vol,5)", name="e2", ic_train=0.04),
        ])

        res = factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            start="20230103", end="20231231", universe="csi300",
            train_days=60, test_days=15, decorr_threshold=1.0,
            out_dir=str(tmp_path / "o"),
        )
        assert "fake_py" in res["factors_used"]
        assert "e1" in res["factors_used"] and "e2" in res["factors_used"]

        with pytest.raises(ValueError, match=r"universe|python"):
            factor_combine.combine_from_library(
                market="ashare", library_root=str(lib),
                start="20230103", end="20231231", universe=None,
                train_days=60, test_days=15, out_dir=str(tmp_path / "o2"),
            )

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_combine_from_library_python_with_expression(_tp2, mp)

    # -- 原 test_combine_from_library_skip_bad_expression --
    def _section_3_test_combine_from_library_skip_bad_expression(tmp_path, mp):
        import factorzen.pipelines.factor_mine as fm

        mp.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

        lib = tmp_path / "lib"
        bad = "this_is_not_a_valid_expr_zzz()"
        _write_lib__combine_from_library(lib, "ashare", [
            _expr_rec("rank(close)", name="ok1", ic_train=0.08),
            _expr_rec(bad, name="bad_one", ic_train=0.07),
            _expr_rec("ts_mean(vol,5)", name="ok2", ic_train=0.05),
        ])
        res = factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            start="20230103", end="20231231",
            train_days=60, test_days=15, decorr_threshold=1.0,
            out_dir=str(tmp_path / "o"),
        )
        assert bad in res["skipped_materialize"]
        assert "bad_one" not in res["factors_used"]
        assert set(res["factors_used"]) == {"ok1", "ok2"}

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_3_test_combine_from_library_skip_bad_expression(_tp3, mp)

    # -- 原 test_combine_from_library_needs_two_and_top_n --
    def _section_4_test_combine_from_library_needs_two_and_top_n(tmp_path, mp):
        import factorzen.pipelines.factor_mine as fm

        mp.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

        lib = tmp_path / "lib"
        _write_lib__combine_from_library(lib, "ashare", [
            _expr_rec("rank(close)", name="only_one", ic_train=0.08),
        ])
        with pytest.raises(ValueError, match="不足 2 个"):
            factor_combine.combine_from_library(
                market="ashare", library_root=str(lib),
                start="20230103", end="20231231",
                out_dir=str(tmp_path / "o"),
            )

        _write_lib__combine_from_library(lib, "ashare", [
            _expr_rec("rank(close)", name="n1", ic_train=0.09),
            _expr_rec("rank(vol)", name="n2", ic_train=0.07),
            _expr_rec("rank(high)", name="n3", ic_train=0.05),
            _expr_rec("ts_mean(vol,5)", name="n4", ic_train=0.03),
        ])
        res = factor_combine.combine_from_library(
            market="ashare", library_root=str(lib),
            start="20230103", end="20231231",
            top_n=2, train_days=60, test_days=15, decorr_threshold=1.0,
            out_dir=str(tmp_path / "o2"),
        )
        assert res["truncated_from"] == 4
        assert len(res["factors_used"]) == 2
        # |ic_train| 降序：n1, n2
        assert res["factors_used"] == ["n1", "n2"]

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_4_test_combine_from_library_needs_two_and_top_n(_tp4, mp)

    # -- 原 test_manifest_records_full_provenance --
    def _section_5_test_manifest_records_full_provenance(tmp_path, mp):
        import factorzen.pipelines.factor_mine as fm

        mp.setattr(fm, "prepare_mining_daily", lambda *a, **k: _daily())

        lib = tmp_path / "lib"
        _write_lib__combine_from_library(lib, "ashare", [
            _expr_rec("rank(close)", name="f_close", ic_train=0.08),
            _expr_rec("ts_mean(vol,5)", name="f_vol", ic_train=0.06),
        ])
        res = factor_combine.combine_from_library(
            market="ashare",
            library_root=str(lib),
            start="20230103",
            end="20231231",
            universe="csi300",
            horizon=5,
            train_days=60,
            test_days=15,
            decorr_threshold=1.0,
            methods=["equal_weight"],
            out_dir=str(tmp_path / "out"),
        )
        manifest = json.loads((Path(res["run_dir"]) / "manifest.json").read_text())
        cfg = manifest.get("config") or {}

        # 窗口与票池：判断「这次 run 覆盖了哪段数据」的最小充分集
        assert cfg.get("start") == "20230103", cfg
        assert cfg.get("end") == "20231231", cfg
        assert cfg.get("universe") == "csi300", cfg
        assert cfg.get("market") == "ashare", cfg
        assert cfg.get("horizon") == 5, cfg
        # 选品参数：决定纳入哪些因子
        assert cfg.get("statuses") == ["active"], cfg
        assert cfg.get("decorr_threshold") == 1.0, cfg
        assert cfg.get("seed") is not None, cfg
        # 库指纹：同窗口不同库版本结果不同
        assert cfg.get("library_hash") is not None, cfg

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_manifest_records_full_provenance(_tp5, mp)

    # -- store 命中 vs --no-store 重算：组合输入数值一致 --
    def _section_6_test_store_hit_matches_recompute(tmp_path, mp):
        """store 命中路径与 no_store 重算路径对同一微型数据数值一致。"""
        import factorzen.pipelines.factor_mine as fm
        from factorzen.discovery.evaluation import (
            _factor_df_from_prepped,
            _preprocess_daily,
        )
        from factorzen.discovery.expression import parse_expr
        from factorzen.discovery.factor_library import FactorRecord
        from factorzen.discovery.factor_store import load_materialized_factor

        # 面板从 2023-01-03 起；请求窗后移，滚动算子预热后 parquet min 仍 ≤ start
        daily = _daily(n_stocks=20, n_days=120, seed=7)
        mp.setattr(fm, "prepare_mining_daily", lambda *a, **k: daily)

        lib = tmp_path / "lib"
        store = tmp_path / "store"
        # 截面算子不丢头部日；保证 min(trade_date) 覆盖请求 start
        exprs = [
            ("f_close", "rank(close)", 0.08),
            ("f_vol", "rank(vol)", 0.06),
        ]
        _write_lib__combine_from_library(lib, "ashare", [
            _expr_rec(e, name=n, ic_train=ic) for n, e, ic in exprs
        ])

        prepped = _preprocess_daily(daily)
        req_start, req_end = "20230201", "20230630"
        start_d = dt.datetime.strptime(req_start, "%Y%m%d").date()
        end_d = dt.datetime.strptime(req_end, "%Y%m%d").date()
        for n, e, _ic in exprs:
            full = _factor_df_from_prepped(parse_expr(e), prepped).select(
                ["trade_date", "ts_code", "factor_value"]
            )
            adir = store / "ashare" / n
            adir.mkdir(parents=True, exist_ok=True)
            full.write_parquet(adir / "factor.parquet")
            meta = {
                "name": n,
                "kind": "expression",
                "expression": e,
                "frequency": "daily",
                "description": "",
                "materialization": {
                    "start": "2016-01-01",
                    "end": "2023-12-31",
                    "universe": "all_a",
                    "git_sha": "abc123",
                    "n_rows": full.height,
                    "generated_at": "2026-07-01T00:00:00+00:00",
                    "expression": e,
                },
            }
            (adir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            (adir / "factor.py").write_text("# stub\n", encoding="utf-8")

        common = dict(
            market="ashare",
            library_root=str(lib),
            store_root=str(store),
            start=req_start,
            end=req_end,
            universe="all_a",
            horizon=5,
            train_days=40,
            test_days=10,
            decorr_threshold=1.0,
            methods=["equal_weight"],
        )
        res_hit = factor_combine.combine_from_library(
            **common, no_store=False, out_dir=str(tmp_path / "out_hit"),
        )
        res_re = factor_combine.combine_from_library(
            **common, no_store=True, out_dir=str(tmp_path / "out_re"),
        )
        su = res_hit.get("store_usage") or {}
        assert su.get("no_store") is False
        assert set(su.get("hits") or {}) == {"f_close", "f_vol"}, su
        assert (res_re.get("store_usage") or {}).get("no_store") is True

        ch = res_hit["comparison"].sort("method")
        cr = res_re["comparison"].sort("method")
        assert ch["method"].to_list() == cr["method"].to_list()
        for col in ch.columns:
            if col == "method":
                continue
            if ch[col].dtype in (pl.Float32, pl.Float64):
                a = ch[col].to_numpy()
                b = cr[col].to_numpy()
                assert np.allclose(a, b, equal_nan=True), f"col={col} {a} vs {b}"

        # 直接对账：store 切片 vs 重算 fdf（防「读了但读错」）
        for n, e, _ic in exprs:
            rec = FactorRecord(
                expression=e, market="ashare", kind="expression", name=n, status="active",
            )
            f_store, reason, _m = load_materialized_factor(
                rec,
                market="ashare",
                root=str(store),
                start=req_start,
                end=req_end,
                universe="all_a",
            )
            assert reason is None and f_store is not None
            f_re = _factor_df_from_prepped(
                parse_expr(e), prepped, eval_start=start_d, eval_end=end_d,
            ).select(["trade_date", "ts_code", "factor_value"])
            f_store = f_store.filter(
                (pl.col("trade_date") >= start_d) & (pl.col("trade_date") <= end_d)
            )
            a = f_store.sort(["trade_date", "ts_code"])
            b = f_re.sort(["trade_date", "ts_code"])
            assert a.height == b.height and a.height > 0
            from polars.testing import assert_frame_equal

            assert_frame_equal(a, b, check_dtypes=False)

    _tp6 = tmp_path / "_s6"
    _tp6.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_6_test_store_hit_matches_recompute(_tp6, mp)


# ==== 来自 test_combine_cli_smoke.py ====

def _write_inputs(tmp_path, n_days=120, n_stocks=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.8 * fa - 0.4 * fb + rng.standard_normal(n_stocks) * 0.3
        for s in range(n_stocks):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    fa_p = tmp_path / "fa.parquet"
    fb_p = tmp_path / "fb.parquet"
    ret_p = tmp_path / "ret.parquet"
    pl.DataFrame(ra).write_parquet(fa_p)
    pl.DataFrame(rb).write_parquet(fb_p)
    pl.DataFrame(rr).write_parquet(ret_p)
    return fa_p, fb_p, ret_p


def test_fz_combine_run_smoke(tmp_path):
    fa_p, fb_p, ret_p = _write_inputs(tmp_path)
    out = tmp_path / "out"
    rc = main(
        [
            "combine", "run",
            "--factor", str(fa_p),
            "--factor", str(fb_p),
            "--ret", str(ret_p),
            "--train-days", "60",
            "--test-days", "20",
            "--purge-days", "5",
            "--methods", "equal_weight,lgbm",
            "--seed", "0",
            "--run-id", "cli1",
            "--out-dir", str(out),
        ]
    )
    assert rc == 0
    run_dir = out / "cli1"
    assert (run_dir / "comparison.csv").exists()
    assert (run_dir / "report.md").exists()
    comp = pl.read_csv(run_dir / "comparison.csv")
    assert set(comp["method"].to_list()) == {"equal_weight", "lgbm"}


# ==== 来自 test_library_provider.py ====

def _write_lib__library_provider(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _default_name(expr: str) -> str:
    return f"mined_{hashlib.sha1(expr.encode()).hexdigest()[:8]}"


@pytest.fixture
def reg_mod():
    """daily registry 模块；teardown reset 全局单例——LibFactor 注入若滞留会
    污染同进程后续测试文件（全量跑实锤过 test_daily_factors 次序失败）。"""
    import factorzen.daily.factors.registry as reg

    yield reg
    reg._registry.reset()


# ── 1. 基本注入 ──────────────────────────────────────────────────────────────


def test_load_library_factors_suite(tmp_path, reg_mod, caplog, capsys):
    """test_load_library_factors_registers_expression_records；test_load_library_factors_yields_to_existing；test_load_library_factors_skips_python_kind；test_load_library_factors_idempotent；test_load_library_factors_tolerates_corrupt_jsonl；test_cmd_factor_list_includes_library_factor"""
    # -- 原 test_load_library_factors_registers_expression_records --
    def _section_0_test_load_library_factors_registers_expression_records(tmp_path, reg_mod):
        named = "lib_prov_named_alpha"
        expr_named = "rank(close)"
        expr_anon = "neg(rank(close))"
        anon_name = _default_name(expr_anon)
        _write_lib__library_provider(
            tmp_path,
            "ashare",
            [
                {
                    "expression": expr_named,
                    "market": "ashare",
                    "kind": "expression",
                    "name": named,
                    "status": "active",
                },
                {
                    "expression": expr_anon,
                    "market": "ashare",
                    "kind": "expression",
                    "status": "probation",
                    # name 缺省 → default_name_for_expression
                },
            ],
        )
        n = load_library_factors(market="ashare", root=str(tmp_path))
        assert n == 2

        cls_named = reg_mod.get_factor(named)
        inst = cls_named()
        assert inst.name == named
        assert inst.expression == expr_named
        assert inst.lookback_days >= 60
        assert "[active]" in inst.description

        cls_anon = reg_mod.get_factor(anon_name)
        inst_anon = cls_anon()
        assert inst_anon.expression == expr_anon
        assert "[probation]" in inst_anon.description

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    _section_0_test_load_library_factors_registers_expression_records(_tp0, reg_mod)

    # -- 原 test_load_library_factors_yields_to_existing --
    def _section_1_test_load_library_factors_yields_to_existing(tmp_path, reg_mod, caplog):
        from factorzen.daily.factors.base import DailyFactor

        conflict = "lib_prov_conflict_builtin"
        # 先注册同名假因子（模拟 workspace/builtin 占用）
        fake = type(
            "FakeConflict",
            (DailyFactor,),
            {
                "name": conflict,
                "frequency": "daily",
                "description": "fake occupant",
                "lookback_days": 20,
                "compute": lambda self, ctx: None,
            },
        )
        assert reg_mod._registry.register(fake, override=True) is True

        _write_lib__library_provider(
            tmp_path,
            "ashare",
            [
                {
                    "expression": "rank(vol)",
                    "market": "ashare",
                    "kind": "expression",
                    "name": conflict,
                    "status": "active",
                }
            ],
        )
        with caplog.at_level("WARNING"):
            n = load_library_factors(market="ashare", root=str(tmp_path))
        assert n == 0
        assert any("让位" in r.message or conflict in r.message for r in caplog.records)
        # 仍是假因子，非 LibFactor
        assert reg_mod.get_factor(conflict) is fake

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    _section_1_test_load_library_factors_yields_to_existing(_tp1, reg_mod, caplog)

    # -- 原 test_load_library_factors_skips_python_kind --
    def _section_2_test_load_library_factors_skips_python_kind(tmp_path, reg_mod):
        py_name = "lib_prov_python_skip_xyz"
        _write_lib__library_provider(
            tmp_path,
            "ashare",
            [
                {
                    "expression": f"py::{py_name}",
                    "market": "ashare",
                    "kind": "python",
                    "name": py_name,
                    "impl": py_name,
                    "status": "active",
                },
                {
                    "expression": "rank(amount)",
                    "market": "ashare",
                    "kind": "expression",
                    "name": "lib_prov_expr_ok",
                    "status": "active",
                },
            ],
        )
        n = load_library_factors(market="ashare", root=str(tmp_path))
        assert n == 1
        reg_mod.get_factor("lib_prov_expr_ok")
        with pytest.raises(KeyError):
            reg_mod.get_factor(py_name)

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    _section_2_test_load_library_factors_skips_python_kind(_tp2, reg_mod)

    # -- 原 test_load_library_factors_idempotent --
    def _section_3_test_load_library_factors_idempotent(tmp_path, reg_mod, caplog):
        name = "lib_prov_idempotent_once"
        _write_lib__library_provider(
            tmp_path,
            "ashare",
            [
                {
                    "expression": "rank(high)",
                    "market": "ashare",
                    "kind": "expression",
                    "name": name,
                    "status": "active",
                }
            ],
        )
        n1 = load_library_factors(market="ashare", root=str(tmp_path))
        assert n1 == 1
        names_after_1 = reg_mod.list_factors()
        assert names_after_1.count(name) == 1

        with caplog.at_level("WARNING"):
            n2 = load_library_factors(market="ashare", root=str(tmp_path))
        assert n2 == 0
        names_after_2 = reg_mod.list_factors()
        assert names_after_2.count(name) == 1
        # 二次 load 不因自身已注入再刷「让位」warning
        assert not any("让位" in r.message and name in r.message for r in caplog.records)

    _tp3 = tmp_path / "_s3"
    _tp3.mkdir(exist_ok=True)
    _section_3_test_load_library_factors_idempotent(_tp3, reg_mod, caplog)

    # -- 原 test_load_library_factors_tolerates_corrupt_jsonl --
    def _section_4_test_load_library_factors_tolerates_corrupt_jsonl(tmp_path, reg_mod):
        path = tmp_path / "ashare.jsonl"
        path.write_text(
            '{"expression":"rank(low)","market":"ashare","kind":"expression","name":"lib_prov_ok_corrupt","status":"active"}\n'
            "NOT_JSON_LINE\n"
            '{"expression":"neg(rank(low))","market":"ashare","kind":"expression","name":"lib_prov_ok2_corrupt","status":"correlated"}\n',
            encoding="utf-8",
        )
        n = load_library_factors(market="ashare", root=str(tmp_path))
        assert n == 2
        reg_mod.get_factor("lib_prov_ok_corrupt")
        reg_mod.get_factor("lib_prov_ok2_corrupt")

    _tp4 = tmp_path / "_s4"
    _tp4.mkdir(exist_ok=True)
    _section_4_test_load_library_factors_tolerates_corrupt_jsonl(_tp4, reg_mod)

    # -- 原 test_cmd_factor_list_includes_library_factor --
    def _section_5_test_cmd_factor_list_includes_library_factor(tmp_path, reg_mod, mp, capsys):
        import argparse

        from factorzen.cli import main as cli

        name = "lib_prov_cli_list_visible"
        _write_lib__library_provider(
            tmp_path,
            "ashare",
            [
                {
                    "expression": "rank(open)",
                    "market": "ashare",
                    "kind": "expression",
                    "name": name,
                    "status": "active",
                }
            ],
        )
        # 把默认库根指到 tmp（load_library_factors 无 root 参数时用 DEFAULT_ROOT）
        mp.setattr(
            "factorzen.discovery.factor_library.DEFAULT_ROOT",
            str(tmp_path),
        )
        # 同时 patch daily registry 里 load 用的默认（若已 import DEFAULT_ROOT 为值则走函数内再 import）
        args = argparse.Namespace(freq="daily")
        rc = cli._cmd_factor_list(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert name in out

    _tp5 = tmp_path / "_s5"
    _tp5.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_5_test_cmd_factor_list_includes_library_factor(_tp5, reg_mod, mp, capsys)


# ── 2. 冲突让位 ──────────────────────────────────────────────────────────────


# ── 3. python 型跳过 ─────────────────────────────────────────────────────────


# ── 4. 幂等 ──────────────────────────────────────────────────────────────────


# ── 5. 损坏库文件 ────────────────────────────────────────────────────────────


# ── 6. CLI 冒烟 ──────────────────────────────────────────────────────────────


