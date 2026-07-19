"""定向重估：``rebuild(only=[...])`` 只重估指定子集，不触发全局贪心去相关级联。

语义（见 factor_library.rebuild docstring）：
- 绝不清库（`fresh` 被强制 False）；
- 只评估 `only` 子集；lift 轨复审也只覆盖子集；
- 去相关 **只降不升**（`preserve_status=True`）：可下调 correlated，绝不上调 active。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl


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

def test_targeted_rebuild_evaluates_only_subset_and_leaves_others_untouched(tmp_path):
    """only 子集之外的库记录一个字节都不能动（含 updated_at / 指标 / status）。"""
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


# ── 2. 只降不升：核心去相关语义 ──────────────────────────────────────────────

def test_targeted_rebuild_never_promotes_correlated_to_active(tmp_path):
    """已判 correlated 的目标，即使重估后与库内 active 全不相关，也**不得**升回 active。

    升 active = 往去相关池里加成员 → 可能让某条未重估的 active 实际变重复而库里仍
    标 active。上调不安全，必须跑全量 rebuild。
    """
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


def test_targeted_rebuild_still_demotes_when_now_correlated(tmp_path):
    """只降不升 ≠ 什么都不做：与库内未重估 active 超阈的目标仍被下调 correlated。"""
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


# ── 3. 状态 / 轨道 / provenance 不被 _record_from_candidate 抹掉 ──────────────

def test_targeted_rebuild_preserves_track_and_provenance_fields(tmp_path):
    """定向重估只刷新指标：admission_track / hypothesis / lift* / name 不得被抹成 None。"""
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


def test_targeted_rebuild_refreshes_admission_ic_when_provided(tmp_path):
    """补算场景：候选给了新的 admission_ic → 落盘新值（不是被 prev 的 None 顶回去）。"""
    from factorzen.discovery.factor_library import _save_library, rebuild
    _save_library("ashare", [_rec("rank(close)", admission_ic=None)], root=str(tmp_path))

    def evaluate(exprs):
        return [_cand(e, admission_ic=0.0177) for e in exprs]

    rebuild("ashare", sources=["rank(close)"], eval_window=("20200101", "20260101"),
            universe=None, horizon=1, evaluate=evaluate, git_sha="x", now="2026-07-19",
            only=["rank(close)"], root=str(tmp_path))
    assert _by_expr(tmp_path)["rank(close)"].admission_ic == 0.0177


# ── 4. lift 轨复审只覆盖子集 ────────────────────────────────────────────────

def test_targeted_rebuild_limits_lift_review_to_subset(tmp_path):
    """lift 轨复审是最贵的一步：定向模式下只能对子集跑 add-one lift。"""
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


# ── 5. 绝不清库（哪怕调用方传 fresh=True）──────────────────────────────────

def test_targeted_rebuild_forces_non_fresh(tmp_path):
    """定向 + fresh=True：库文件不得被清空（否则子集外记录全丢）。"""
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


# ── 6. gate 失败：刷新指标 + 大声记账，不静默留陈旧值 ──────────────────────

def test_targeted_rebuild_reports_gate_failure_but_refreshes_metrics(tmp_path):
    """已在库记录重估后不再满足 library gate：仍写真实指标，但计入 gate_failed。

    gate 是**准入**门不是**留任**门；把真值挡在库外只会留下「看起来合法的陈旧值」。
    """
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

def test_targeted_rebuild_records_missing_targets(tmp_path):
    """only 里不在库的表达式：不静默吞，记 manifest.targeted_missing。"""
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


def test_targeted_rebuild_normalizes_only_expressions(tmp_path):
    """only 走与库同一套规范形：写法带多余空格也能命中。"""
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


def test_cli_rebuild_only_flag_reaches_engine(monkeypatch, tmp_path):
    """`fz factor-library rebuild --only <expr>`：定向目标真的传到引擎且只评估它。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import _save_library

    _save_library("ashare", [_rec("rank(close)"), _rec("rank(open)")], root=str(tmp_path))
    seen: dict = {}
    _patch_cli_for_rebuild(monkeypatch, tmp_path, seen)

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


def test_cli_rebuild_intraday_flags_reach_data_assembly(monkeypatch, tmp_path):
    """`--intraday-leaves` 必须真到达数据装配层。

    接线层漂移实锤：该旗标此前只在 mine search/agent/team 上有，rebuild 没有 →
    `_prepare_agent_mining_data` 的 `getattr(args, "intraday_leaves", False)` 恒 False
    → 含 i_* 叶子的 lift 记录复审必物化失败。help 里承诺的旗标必须真的通到底。
    """
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import _save_library

    _save_library("ashare", [_rec("rank(close)")], root=str(tmp_path))
    seen: dict = {}
    _patch_cli_for_rebuild(monkeypatch, tmp_path, seen)
    # 自动检测会读库：必须指到 tmp，否则读的是真实工作区的库（真库含 i_* 记录，
    # 会让「不给旗标 → False」这条断言假失败，更糟的是测试依赖本机数据）
    monkeypatch.setattr("factorzen.discovery.factor_library.DEFAULT_ROOT",
                        str(tmp_path), raising=False)

    def _spy_prepare(args):
        seen["intraday_leaves"] = getattr(args, "intraday_leaves", None)
        seen["intraday_freq"] = getattr(args, "intraday_freq", None)
        return pl.DataFrame({"trade_date": [date(2024, 1, 2)]}), None, {}

    monkeypatch.setattr(cli_main, "_prepare_agent_mining_data", _spy_prepare)

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


def test_cli_rebuild_auto_enables_intraday_from_library(monkeypatch, tmp_path):
    """库里有含 i_* 叶子的记录 → **不给旗标也自动装日内面板**。

    只靠旗标不够：lift 复审覆盖库内全部 lift 轨记录，操作者忘了加 --intraday-leaves
    就会让它们物化失败。与 `factor-library lift-test` 的自动置位同款。
    """
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    seen: dict = {}
    _patch_cli_for_rebuild(monkeypatch, tmp_path, seen)

    def _spy_prepare(args):
        seen["intraday_leaves"] = getattr(args, "intraday_leaves", None)
        return pl.DataFrame({"trade_date": [date(2024, 1, 2)]}), None, {}

    monkeypatch.setattr(cli_main, "_prepare_agent_mining_data", _spy_prepare)
    # CLI 用默认 root 读库来做检测 → 把默认 root 指到 tmp
    monkeypatch.setattr(
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


def test_cli_rebuild_exits_nonzero_on_lift_eval_failure(monkeypatch, tmp_path, capsys):
    """引擎报了求值失败 → CLI 非零退出 + stderr 点名（禁止「表面成功」）。

    引擎侧「不降级 + 记账」的行为锚在 test_lift_admissions.py；这里只锁接线层：
    `UpsertResult.lift_eval_failed` 非空必须变成 exit 1，且报错点名表达式与
    `--intraday-leaves` 这条真实可用的补救旗标。
    """
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery import factor_library as fl

    seen: dict = {}
    _patch_cli_for_rebuild(monkeypatch, tmp_path, seen)
    monkeypatch.setattr(fl, "collect_source_expressions", lambda market: [])
    monkeypatch.setattr(fl, "rebuild", lambda *a, **kw: fl.UpsertResult(
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


def test_cli_rebuild_only_file(monkeypatch, tmp_path):
    """`--only-file`：一行一条、'#' 注释与空行跳过（上百条批量补账的入口）。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from factorzen.discovery.factor_library import _save_library

    _save_library("ashare", [_rec("rank(close)"), _rec("rank(open)")], root=str(tmp_path))
    listing = tmp_path / "targets.txt"
    listing.write_text("# 本批目标\nrank(close)\n\nrank(open)\n", encoding="utf-8")
    seen: dict = {}
    _patch_cli_for_rebuild(monkeypatch, tmp_path, seen)

    args = build_parser().parse_args([
        "factor-library", "rebuild", "--market", "ashare",
        "--universe", "csi300", "--start", "20200101", "--end", "20201231",
        "--only-file", str(listing),
    ])
    assert cli_main._cmd_factor_library_rebuild(args) == 0
    assert seen["only"] == ["rank(close)", "rank(open)"]


def test_cli_rebuild_empty_only_fails_loudly(monkeypatch, tmp_path, capsys):
    """定向旗标给了却解析出空集 → exit 1，绝不静默降级成会重排全库的全量 rebuild。"""
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser

    empty = tmp_path / "empty.txt"
    empty.write_text("# 全是注释\n\n", encoding="utf-8")
    seen: dict = {}
    _patch_cli_for_rebuild(monkeypatch, tmp_path, seen)

    args = build_parser().parse_args([
        "factor-library", "rebuild", "--market", "ashare",
        "--universe", "csi300", "--start", "20200101", "--end", "20201231",
        "--only-file", str(empty),
    ])
    assert cli_main._cmd_factor_library_rebuild(args) == 1
    assert "空目标集" in capsys.readouterr().err
    assert "only" not in seen, "空目标集不该走到 rebuild"


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
