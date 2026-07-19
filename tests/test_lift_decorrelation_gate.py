"""W1：lift 准入轨接相关性门。

背景：`upsert_lift_admissions` 体内从无 `_decorrelate` 调用（全仓唯一调用点是
`upsert`，`rebuild` 是委托它），所以整条 lift 准入轨从未去重。实测 150 条候选按
threshold=0.005 准入 12 条，两两 |rho| 最大 0.965、>=0.7 有 8/66 对，聚成的独立
信号仅 4-5 个——「准入数 != 信息量」。

机理：残差 lift 是对**冻结基线**算的，同批候选共用同一基线快照，所以两条互为
重复的候选拿到相同的正 lift，双双准入。语义正确的解法是「每准入一条就并入基线
再测下一条」（贪心 re-lift），0.7 相关门是它的廉价代理。

本文件的第一优先级是 P0 锚：`_decorrelate` 会**无条件覆写 status**，
而 lift 轨默认 `allow_active=False` 把 decision=active 压成 probation。
直接接线会把全部 capped probation 冲成 active，运营护栏整体失效。
"""
from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

_DATES = [f"2024-01-{d:02d}" for d in range(1, 21)]
_CODES = [f"{i:06d}.SZ" for i in range(40)]


def _panel(values: np.ndarray) -> pl.DataFrame:
    """values: (n_date, n_stock) → 长表面板。"""
    return pl.DataFrame({
        "trade_date": np.repeat(_DATES, len(_CODES)),
        "ts_code": np.tile(_CODES, len(_DATES)),
        "factor_value": values.reshape(-1).astype(float),
    })


def _make_panels(seed: int = 0) -> dict[str, pl.DataFrame]:
    """构造一组已知相关结构的面板：

    - ``dup_a`` / ``dup_b``：完全相同（rho = 1.0）
    - ``indep``：独立
    - ``lib_x``：库内既有 active 因子，与上述都不相关
    """
    rng = np.random.default_rng(seed)
    shape = (len(_DATES), len(_CODES))
    base = rng.standard_normal(shape)
    return {
        "dup_a": _panel(base),
        "dup_b": _panel(base.copy()),
        "indep": _panel(rng.standard_normal(shape)),
        "lib_x": _panel(rng.standard_normal(shape)),
    }


def _materializer(panels: dict[str, pl.DataFrame]):
    def _mat(expr: str) -> pl.DataFrame | None:
        return panels.get(expr)
    return _mat


def _row(expr: str, lift: float, *, se: float = 0.0005) -> dict:
    """一行 lift 结果。lift 远超阈值 → decision=active。

    注意真实 lift 结果行**没有 ir_train 字段**（实证：replay_150 parquet 仅 12 列），
    这正是贪心排序键必须改用 lift 的原因。此处也刻意不给 ir_train。
    """
    return {
        "expression": expr,
        "lift": lift,
        "lift_se": se,
        "lift_first_half": lift,
        "lift_second_half": lift,
        "n_blocks": 20,
        "lift_metric": "residual_ic_v1",
    }


def _seed_library(tmp_path, panels) -> str:
    """库里先放一条 single 轨 active 因子（lib_x）。"""
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    rec = FactorRecord(
        expression="lib_x", market="ashare", status="active",
        admission_track="single", ir_train=0.5,
    )
    _save_library("ashare", [rec], root=str(tmp_path))
    return str(tmp_path)


# ── P0 锚：cap 不得被去相关冲掉 ──────────────────────────────────────────────

def test_capped_probation_survives_decorrelation(tmp_path):
    """【W1 第一验收锚】allow_active=False 下走完去相关，capped probation 仍是 probation。

    `_decorrelate` 原实现在不超阈分支无条件写 `rec.status = "active"`，
    直接接线会把 lift 轨的运营护栏（§14.1 cap）整体冲掉——比不加门更糟。
    """
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    panels = _make_panels()
    root = _seed_library(tmp_path, panels)
    out = upsert_lift_admissions(
        [_row("indep", 0.02)],
        market="ashare", root=root,
        materialize=_materializer(panels),
        allow_active=False,
    )
    assert out.get("capped_active") == 1, "前提不成立：这一行本应触发 cap"
    rec = {r.expression: r for r in load_library("ashare", root=root)}["indep"]
    assert rec.status == "probation", f"cap 被去相关冲成 {rec.status}"
    assert rec.admission_decision == "active", "统计裁决原文丢失"


def test_missing_materializer_skips_gate_without_touching_status(tmp_path):
    """物化器缺失 → 跳过去相关 + 显式告警，status 一律不动。

    `_decorrelate(compact_of=None)` 会把记录全部置成 active；若参数没接通就调用，
    等于静默把 probation 冲成 active。必须 fail-loudly 而非静默降级。
    """
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    panels = _make_panels()
    root = _seed_library(tmp_path, panels)
    with pytest.warns(UserWarning, match="去相关"):
        out = upsert_lift_admissions(
            [_row("indep", 0.02)],
            market="ashare", root=root,
            allow_active=False,   # 不传 materialize
        )
    assert out.get("decorrelation_skipped") is True
    rec = {r.expression: r for r in load_library("ashare", root=root)}["indep"]
    assert rec.status == "probation"


def test_noop_skip_sets_flag_without_warning_noise(tmp_path):
    """门无事可做时（单条准入 + 库内无 active）只落标志、不告警。

    无差别告警会造成告警疲劳——真漏接线时反而没人看见。但 `decorrelation_skipped`
    仍须无条件落盘，provenance 不能因为降噪而缺失。
    """
    from factorzen.discovery.factor_library import upsert_lift_admissions

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # 任何 warning 都会变成异常
        out = upsert_lift_admissions(
            [_row("indep", 0.02)],
            market="ashare", root=str(tmp_path),   # 空库
            allow_active=False,
        )
    assert out.get("decorrelation_skipped") is True


# ── 去重本体 ────────────────────────────────────────────────────────────────

def test_duplicate_pair_one_marked_correlated(tmp_path):
    """rho=1.0 的一对同批进入 → 一条留、一条标 correlated + max_corr_in_lib 回填。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    panels = _make_panels()
    root = _seed_library(tmp_path, panels)
    upsert_lift_admissions(
        [_row("dup_a", 0.02), _row("dup_b", 0.01)],
        market="ashare", root=root,
        materialize=_materializer(panels),
        allow_active=False,
    )
    by = {r.expression: r for r in load_library("ashare", root=root)}
    statuses = {by["dup_a"].status, by["dup_b"].status}
    assert statuses == {"probation", "correlated"}, f"得到 {statuses}"
    loser = by["dup_a"] if by["dup_a"].status == "correlated" else by["dup_b"]
    assert loser.max_corr_in_lib is not None
    assert loser.max_corr_in_lib > 0.99, f"rho 应≈1.0，得 {loser.max_corr_in_lib}"
    assert loser.correlated_with in ("dup_a", "dup_b")
    assert loser.admission_decision == "active", "被判 correlated 不应抹掉裁决原文"


def test_greedy_order_follows_lift_not_alphabet(tmp_path):
    """【陷阱 3 反例锚】贪心顺序由 lift 决定，不是表达式字母序。

    lift 结果行没有 ir_train，`_decorrelate` 默认排序键 `-abs(ir_train or 0)`
    对整批恒为 0 → 退化成字母序，谁占 active 位完全随机。
    构造：lift 高者字母序靠后（dup_b > dup_a），断言 lift 高的那条留下。
    """
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    panels = _make_panels()
    root = _seed_library(tmp_path, panels)
    upsert_lift_admissions(
        [_row("dup_a", 0.010), _row("dup_b", 0.050)],   # b 的 lift 高 5 倍，但字母序靠后
        market="ashare", root=root,
        materialize=_materializer(panels),
        allow_active=False,
    )
    by = {r.expression: r for r in load_library("ashare", root=root)}
    assert by["dup_b"].status == "probation", "lift 高者应占住位"
    assert by["dup_a"].status == "correlated", "lift 低者应被标 correlated"


def test_independent_candidates_all_admitted(tmp_path):
    """不相关的候选不该被误杀（门的特异性）。"""
    from factorzen.discovery.factor_library import load_library, upsert_lift_admissions

    panels = _make_panels()
    root = _seed_library(tmp_path, panels)
    upsert_lift_admissions(
        [_row("indep", 0.02), _row("dup_a", 0.03)],
        market="ashare", root=root,
        materialize=_materializer(panels),
        allow_active=False,
    )
    by = {r.expression: r for r in load_library("ashare", root=root)}
    assert by["indep"].status == "probation"
    assert by["dup_a"].status == "probation"


def test_library_active_blocks_correlated_candidate(tmp_path):
    """与库内既有 active 因子高相关的候选 → correlated（跨批去重，不只批内）。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    panels = _make_panels()
    # 库内 active 因子就是 dup_a；候选 dup_b 与它 rho=1.0
    _save_library("ashare", [FactorRecord(
        expression="dup_a", market="ashare", status="active",
        admission_track="single", ir_train=0.5,
    )], root=str(tmp_path))
    upsert_lift_admissions(
        [_row("dup_b", 0.02)],
        market="ashare", root=str(tmp_path),
        materialize=_materializer(panels),
        allow_active=False,
    )
    by = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert by["dup_b"].status == "correlated"
    assert by["dup_b"].correlated_with == "dup_a"


def test_library_probation_does_not_block(tmp_path):
    """D2：库内既有 probation 不进比较池（试用因子不该挡新候选）。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    panels = _make_panels()
    _save_library("ashare", [FactorRecord(
        expression="dup_a", market="ashare", status="probation",
        admission_track="lift", ir_train=0.5,
    )], root=str(tmp_path))
    upsert_lift_admissions(
        [_row("dup_b", 0.02)],
        market="ashare", root=str(tmp_path),
        materialize=_materializer(panels),
        allow_active=False,
    )
    by = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert by["dup_b"].status == "probation", "库内 probation 不应挡住新候选"


def test_forward_confirmed_active_not_downgraded_by_new_candidate(tmp_path):
    """状态机单调性：已 forward-confirmed 的 active 不得被同批高 lift 新候选挤成 correlated。

    既有代码专门保证「幂等重跑不得撤销已确认状态」。若把已确认记录也丢进去相关的
    affected 集，一条 lift 更高的重复候选就能撤销它——门反而破坏了更强的不变式。
    正确语义：已确认者留在**比较池**里挡住新候选，但自身不参与下调。
    """
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert_lift_admissions,
    )

    panels = _make_panels()
    _save_library("ashare", [FactorRecord(
        expression="dup_a", market="ashare", status="active",
        admission_track="lift", forward_confirmed_at="2026-06-01",
    )], root=str(tmp_path))

    upsert_lift_admissions(
        # dup_b 的 lift 远高于 dup_a，且两者 rho=1.0
        [_row("dup_a", 0.010), _row("dup_b", 0.900)],
        market="ashare", root=str(tmp_path),
        materialize=_materializer(panels), allow_active=False,
    )
    by = {r.expression: r for r in load_library("ashare", root=str(tmp_path))}
    assert by["dup_a"].status == "active", "已确认状态被撤销了"
    assert by["dup_a"].forward_confirmed_at == "2026-06-01"
    assert by["dup_b"].status == "correlated", "新候选应被已确认者挡住"


def test_correlated_records_removed_from_admission_counts(tmp_path):
    """计数口径：被改判 correlated 的不应继续计进 added_active/added_probation。"""
    from factorzen.discovery.factor_library import upsert_lift_admissions

    panels = _make_panels()
    root = _seed_library(tmp_path, panels)
    out = upsert_lift_admissions(
        [_row("dup_a", 0.02), _row("dup_b", 0.01), _row("indep", 0.03)],
        market="ashare", root=root,
        materialize=_materializer(panels), allow_active=False,
    )
    assert out.get("correlated") == 1
    # 3 条准入、1 条被判重复 → 只剩 2 条真进库
    assert out["added_active"] + out["added_probation"] == 2, out


# ── 双路径一致性（登记簿要求）────────────────────────────────────────────────

def test_same_verdict_as_single_track(tmp_path):
    """两条轨对同一对因子的去相关裁决必须一致——复用 `_decorrelate` 单点的意义。"""
    from factorzen.discovery.factor_library import (
        FactorRecord,
        _save_library,
        load_library,
        upsert,
        upsert_lift_admissions,
    )

    panels = _make_panels()
    lib_seed = [FactorRecord(
        expression="dup_a", market="ashare", status="active",
        admission_track="single", ir_train=0.5,
    )]

    # 单因子轨
    single_root = tmp_path / "single"
    single_root.mkdir()
    _save_library("ashare", list(lib_seed), root=str(single_root))
    upsert(
        "ashare",
        [{"expression": "dup_b", "ic_train": 0.05, "ir_train": 0.3,
          "holdout_ic": 0.03, "dsr_pvalue": 0.01, "n_holdout_days": 300}],
        eval_window=("2024-01-01", "2024-01-20"), universe=None, horizon=1,
        run_id=None, session_dir=None, git_sha=None, now="2026-07-18",
        materialize=_materializer(panels), root=str(single_root),
    )
    single_status = {
        r.expression: r for r in load_library("ashare", root=str(single_root))
    }["dup_b"].status

    # lift 轨
    lift_root = tmp_path / "lift"
    lift_root.mkdir()
    _save_library("ashare", list(lib_seed), root=str(lift_root))
    upsert_lift_admissions(
        [_row("dup_b", 0.02)],
        market="ashare", root=str(lift_root),
        materialize=_materializer(panels), allow_active=False,
    )
    lift_status = {
        r.expression: r for r in load_library("ashare", root=str(lift_root))
    }["dup_b"].status

    assert single_status == "correlated"
    assert lift_status == "correlated", (
        f"双路径漂移：单因子轨判 {single_status}、lift 轨判 {lift_status}"
    )


# ── 接线锚：能力做完 ≠ 调用方传了（从最外层出发，不用 inspect.signature）──────

def test_cli_lift_apply_wires_materializer(tmp_path, monkeypatch):
    """`fz factor-library lift-test --apply` 必须把物化器传进去。

    不传 = 门在生产上静默失效（`decorrelation_skipped`），单测全绿也照样漏。
    故断言必须从 CLI 最外层出发，而不是 `inspect.signature` 那种零判别力检查。
    """
    import factorzen.cli.main as cli_main
    from factorzen.cli.main import build_parser
    from tests.test_cli_lift_apply import _patch_lift_deps, _write_gray_session

    run_dir = _write_gray_session(tmp_path)
    upsert_calls: list = []
    _patch_lift_deps(monkeypatch, upsert_calls=upsert_calls)

    args = build_parser().parse_args([
        "factor-library", "lift-test", "--session", str(run_dir),
        "--market", "ashare", "--start", "20200101", "--end", "20201231",
        "--library-root", str(tmp_path / "lib"), "--apply",
    ])
    assert cli_main._cmd_factor_library_lift_test(args) == 0
    assert len(upsert_calls) == 1
    got = upsert_calls[0]
    mat = got.get("materialize") or got.get("compact_materialize")
    assert mat is not None, "CLI 未把物化器传给 upsert_lift_admissions，相关性门静默失效"
    assert callable(mat)


def test_team_lift_hook_wires_materializer(monkeypatch):
    """team session 末 lift 钩子同样必须传物化器（第二个调用方，不能只修一侧）。"""
    import factorzen.discovery.factor_library as fl

    calls: list = []

    def fake_upsert(rows, **kw):
        calls.append(kw)
        return {"added_active": 0, "added_probation": 0, "rejected": 0}

    monkeypatch.setattr(fl, "upsert_lift_admissions", fake_upsert)

    import inspect

    import factorzen.agents.team_orchestrator as to

    src = inspect.getsource(to)
    idx = src.find("adm = upsert_lift_admissions(")
    assert idx > 0, "调用点不存在了——本测试需跟着重写"
    call_src = src[idx:idx + src[idx:].find("\n        )")]
    assert "materialize=" in call_src, (
        "team lift 钩子未传物化器，相关性门在 session 自动路径上静默失效"
    )


# ── 零回归锚：单因子轨默认行为逐位不变 ───────────────────────────────────────

def test_decorrelate_default_behavior_unchanged():
    """`_decorrelate` 新增参数必须带默认值且默认行为不变（A股单因子轨零回归底线）。"""
    from factorzen.discovery.factor_library import FactorRecord, _decorrelate

    panels = _make_panels()

    def compact_of(expr):
        p = panels.get(expr)
        if p is None:
            return None
        return p["factor_value"].to_numpy().reshape(len(_DATES), len(_CODES))

    affected = [
        FactorRecord(expression="dup_a", market="ashare", ir_train=0.2),
        FactorRecord(expression="dup_b", market="ashare", ir_train=0.9),
        FactorRecord(expression="indep", market="ashare", ir_train=0.5),
    ]
    n = _decorrelate(affected, [], compact_of, 0.7)
    by = {r.expression: r for r in affected}
    assert n == 1
    # 默认排序键仍是 |ir_train| 降序 → dup_b(0.9) 先占位，dup_a(0.2) 被标 correlated
    assert by["dup_b"].status == "active"
    assert by["dup_a"].status == "correlated"
    assert by["indep"].status == "active"


def test_decorrelate_none_compact_still_forces_active_by_default():
    """默认 `preserve_status=False` 时 `compact_of=None` 仍全置 active（现语义不变）。"""
    from factorzen.discovery.factor_library import FactorRecord, _decorrelate

    affected = [FactorRecord(expression="e1", market="ashare", status="probation")]
    assert _decorrelate(affected, [], None, 0.7) == 0
    assert affected[0].status == "active"


def test_decorrelate_preserve_status_blocks_upgrade():
    """`preserve_status=True` 只允许下调到 correlated，绝不上调。"""
    from factorzen.discovery.factor_library import FactorRecord, _decorrelate

    affected = [FactorRecord(expression="e1", market="ashare", status="probation")]
    assert _decorrelate(affected, [], None, 0.7, preserve_status=True) == 0
    assert affected[0].status == "probation"
