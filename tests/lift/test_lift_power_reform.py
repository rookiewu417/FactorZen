"""R1 lift 准入功效改造回归套（P0a 拆组门 / P0b 裸 IC 降级 / P1 se_mult=1.645）。

依据 ``.artifacts/r1_lift_power/REPORT.md``（2026-07-28）：

- **组门一票否决占 88% 拒绝且数学上零功效增益**：t_group = mean_i(δ_i)·√T/σ_u，
  √K_eff 精确抵消——检验的是批内平均，150 条批 2 条真阳 → t=0.15 永不过。
  拆组门后 δ=.01 功效 0.26 → 0.99。
- **裸 IC 同号门对 ρ≤−0.15 且 δ≤0.02 的抑制变量 48–100% 全屏蔽**：历史仅有的
  2 条过 bar 真阳（i_limit_up_* 涨停封板族）正是被它杀的 → 降级为强制 probation。
- **单候选门 1×SE 反而太松**（df=12 单侧 p≈0.169，H0 放行 15–18%）→ 调用方默认
  改 1.645（生产 lift_null p_active 0.155 → 0.064）。

判别力约定：门槛类断言一律**行为反算**（对 ``lift_admission`` 二分求临界值），
不复述 ``max(threshold, se_mult*se)`` 实现；1.645 的来历用 scipy 正态分位点做
外部 ground truth。
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD

# ── 共用夹具 ─────────────────────────────────────────────────────────────────

def _fake_daily() -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [dt.date(2020, 1, 2)],
        "ts_code": ["000001.SZ"],
        "close": [10.0],
        "close_adj": [10.0],
    })


def _write_session(
    tmp_path: Path, *, n: int, index_path: str | None = None, name: str = "run1",
) -> Path:
    """n 条 lift_queue 候选的假 session（residual_ic_train 均在噪声地板之上）。"""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "start": "20200101", "end": "20201231",
        "universe": "csi300", "market": "ashare",
    }
    if index_path is not None:
        params["index_path"] = index_path
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "attempts": [
                {
                    "expression": f"cand_{i}",
                    "reject_category": "lift_queue",
                    "residual_ic_train": 0.02,
                    "ic_train": 0.02,
                    "n_residual_holdout_days": 100,
                }
                for i in range(n)
            ],
            "candidates": [],
            "params": params,
        }),
        encoding="utf-8",
    )
    return run_dir


def _cli_args(run_dir: Path, lib_root: Path, extra: list[str] | None = None):
    from factorzen.cli.main import build_parser

    argv = [
        "factor-library", "lift-test",
        "--session", str(run_dir),
        "--market", "ashare",
        "--start", "20200101", "--end", "20201231",
        "--universe", "csi300",
        "--set", "top_m=0",
    ]
    if extra:
        argv.extend(extra)
    args = build_parser().parse_args(argv)
    args.library_root = str(lib_root)
    return args


def _patch_cli_group(mp, *, group: dict, lift_rows_of):
    """mock 数据准备 / 覆盖门 / 组门 / 逐候选；返回 run_lift_tests 收到的队列列表。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.lift_test as lt_mod

    seen: list[list[str]] = []
    mp.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda a: (_fake_daily(), None, {}),
    )
    mp.setattr(
        lt_mod, "filter_candidates_by_coverage",
        lambda cands, **k: (list(cands), []),
    )
    mp.setattr(lt_mod, "run_group_lift", lambda queue, **k: dict(group))
    mp.setattr(lt_mod, "resolve_lift_workers", lambda w: 2)

    def fake_lift(queue, **k):
        seen.append([c.get("expression") for c in queue])
        return [lift_rows_of(c.get("expression")) for c in queue]

    mp.setattr(lt_mod, "run_lift_tests", fake_lift)
    return seen


def _read_manifest(run_dir: Path) -> dict:
    return json.loads(
        (run_dir / "lift_test_manifest.json").read_text(encoding="utf-8")
    )


# ── 1. 组门不过时单强候选仍被测（1 强 + 59 弱） ────────────────────────────────

def test_group_gate_weak_still_tests_individual_candidates(tmp_path):
    """R1 核心：组 t=0.15 的 150 条批里 2 条真阳不该陪葬。

    1 强 + 59 弱、组门不过 → 逐条仍跑，强候选 decision≠reject。
    反例：全弱批同样逐条跑，但**全部** reject（证明门没被拆穿）。
    """
    import factorzen.cli.main as cli_main
    from factorzen.discovery.lift_test import lift_admission

    thr = DEFAULT_LIFT_THRESHOLD
    # 组值：lift 远低于 bar → 组门必不过；t = -0.7 < 0 但 K=60 > 20 → 不许短路
    group = {"lift": -0.00007, "lift_se": 0.0001, "error": None,
             "lift_metric": "residual_ic_v1"}

    def rows_one_strong(expr):
        if expr == "cand_0":
            return {"expression": expr, "lift": thr * 10, "lift_se": 0.0001,
                    "lift_second_half": 0.01, "baseline": 0.02, "passed": True}
        return {"expression": expr, "lift": thr * 0.01, "lift_se": 0.0001,
                "lift_second_half": -0.01, "baseline": 0.02, "passed": False}

    with pytest.MonkeyPatch.context() as mp:
        run_dir = _write_session(tmp_path, n=60, name="strong")
        lib_root = tmp_path / "lib_strong"
        lib_root.mkdir()
        seen = _patch_cli_group(mp, group=group, lift_rows_of=rows_one_strong)
        assert cli_main._cmd_factor_library_lift_test(
            _cli_args(run_dir, lib_root)
        ) == 0

    assert seen and len(seen[0]) == 60, "组门不过仍须逐条跑全部候选"
    man = _read_manifest(run_dir)
    assert man["lift_group"]["group_gate_weak"] is True
    assert man["lift_group"]["group_gate_ok"] is False
    assert len(man["results"]) == 60
    assert not any(
        str(r.get("error") or "").startswith("group_gate") for r in man["results"]
    ), "拆组门后不得再有连坐 group_gate_fail 行"
    decisions = {
        r["expression"]: lift_admission(r, threshold=thr, se_mult=1.645)
        for r in man["results"]
    }
    assert decisions["cand_0"] != "reject", decisions["cand_0"]
    assert all(d == "reject" for e, d in decisions.items() if e != "cand_0")

    # 反例：全弱批 → 逐条跑但全 reject
    def rows_all_weak(expr):
        return {"expression": expr, "lift": thr * 0.01, "lift_se": 0.0001,
                "lift_second_half": -0.01, "baseline": 0.02, "passed": False}

    with pytest.MonkeyPatch.context() as mp:
        run_dir2 = _write_session(tmp_path, n=60, name="weak")
        lib_root2 = tmp_path / "lib_weak"
        lib_root2.mkdir()
        seen2 = _patch_cli_group(mp, group=group, lift_rows_of=rows_all_weak)
        assert cli_main._cmd_factor_library_lift_test(
            _cli_args(run_dir2, lib_root2)
        ) == 0
    assert seen2 and len(seen2[0]) == 60
    man2 = _read_manifest(run_dir2)
    assert all(
        lift_admission(r, threshold=thr, se_mult=1.645) == "reject"
        for r in man2["results"]
    )


# ── 2. CLI 与 orchestrator 双路径同判（漂移守卫） ─────────────────────────────

def _cli_ran_per_candidate(mp, tmp_path, *, group: dict, k: int, tag: str) -> bool:
    import factorzen.cli.main as cli_main

    run_dir = _write_session(tmp_path, n=k, name=f"cli_{tag}")
    lib_root = tmp_path / f"lib_cli_{tag}"
    lib_root.mkdir()
    seen = _patch_cli_group(
        mp, group=group,
        lift_rows_of=lambda e: {
            "expression": e, "lift": 0.0, "lift_se": 0.001,
            "lift_second_half": 0.0, "baseline": 0.02, "passed": False,
        },
    )
    assert cli_main._cmd_factor_library_lift_test(_cli_args(run_dir, lib_root)) == 0
    return bool(seen)


def _orch_ran_per_candidate(mp, tmp_path, *, group: dict, k: int, tag: str) -> bool:
    import datetime as _dt

    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.agents.team_orchestrator import _session_end_auto_lift
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_QUEUE

    days, d = [], _dt.date(2022, 1, 3)
    while len(days) < 200:
        if d.weekday() < 5:
            days.append(d)
        d += _dt.timedelta(days=1)
    panel = pl.DataFrame([
        {"trade_date": dd, "ts_code": "000001.SZ", "factor_value": 0.5}
        for dd in days
    ])
    daily = panel.rename({"factor_value": "close"})
    holdout = daily.filter(pl.col("trade_date") >= _dt.date(2022, 8, 1))

    state = AgentState(seed=1)
    for i in range(k):
        state.attempts.append(AttemptRecord(
            0, "H", f"cand_{i}", True, 0.02, False, "keep", None,
            ir_train=0.1, reject_category=REJECT_CATEGORY_LIFT_QUEUE,
            residual_ic_train=0.02,
        ))

    seen: list[int] = []
    mp.setattr("factorzen.discovery.lift_test.run_group_lift",
               lambda *a, **kw: dict(group))
    mp.setattr(
        "factorzen.discovery.lift_test.filter_candidates_by_coverage",
        lambda cands, **kw: (list(cands), []),
    )
    mp.setattr(
        "factorzen.discovery.lift_test.make_lift_context",
        lambda *a, **kw: SimpleNamespace(
            horizon=1, prepped=daily,
            admission_start="2022-08-01", admission_end="2022-12-31",
        ),
    )

    def fake_lift(queue, **kw):
        seen.append(len(list(queue)))
        return [
            {"expression": c.get("expression"), "lift": 0.0, "lift_se": 0.001,
             "lift_second_half": 0.0, "baseline": 0.02, "passed": False}
            for c in queue
        ]

    mp.setattr("factorzen.discovery.lift_test.run_lift_tests", fake_lift)
    mp.setattr(
        "factorzen.discovery.factor_library.upsert_lift_admissions",
        lambda *a, **kw: {"added_active": 0, "added_probation": 0, "rejected": 0},
        raising=False,
    )

    meta = _session_end_auto_lift(
        state, daily=daily, holdout_df=holdout, profile=None,
        ctx=SimpleNamespace(leaf_map=None), market="ashare",
        library_root=str(tmp_path / f"orch_lib_{tag}"), seed=1,
        auto_lift=True, lift_se_mult=1.645, horizon=1,
        materialize_candidate=lambda expr: panel,
        data_window={"start": "20220101", "end": "20221231",
                     "universe": "csi300", "market": "ashare"},
    )
    assert meta.get("lift_error") is None, meta.get("lift_error")
    return bool(seen)


@pytest.mark.parametrize(
    ("tag", "group", "k", "expected_per_candidate"),
    [
        # K≤20 且组 t<0 → 保留旧短路（唯一豁免）
        ("small_neg", {"lift": -0.0007, "lift_se": 0.0001, "error": None}, 5, False),
        # K>20（同一组值）→ 必逐条
        ("big_neg", {"lift": -0.0007, "lift_se": 0.0001, "error": None}, 25, True),
        # 组 t≥0 但没过 bar → 必逐条
        ("small_pos", {"lift": 0.0001, "lift_se": 0.01, "error": None}, 5, True),
        # 组门过 → 本来就逐条
        ("pass", {"lift": 0.05, "lift_se": 0.001, "error": None}, 5, True),
    ],
)
def test_cli_and_orchestrator_same_group_dict_same_verdict(
    tmp_path, tag, group, k, expected_per_candidate,
):
    """双路径漂移守卫：同一 group dict + 同一 K，两条路径逐字段同判。"""
    with pytest.MonkeyPatch.context() as mp:
        cli = _cli_ran_per_candidate(mp, tmp_path, group=group, k=k, tag=tag)
    with pytest.MonkeyPatch.context() as mp:
        orch = _orch_ran_per_candidate(mp, tmp_path, group=group, k=k, tag=tag)
    assert cli == orch == expected_per_candidate, (tag, cli, orch)


# ── 3. 短路豁免分支单测 ───────────────────────────────────────────────────────

def test_group_gate_short_circuit_unit():
    """K≤20 且组 t=lift/lift_se<0 才短路；其余一律逐条。t 不可算 → 逐条。"""
    from factorzen.discovery.lift_test import (
        GROUP_GATE_SHORTCIRCUIT_MAX_K,
        group_gate_short_circuit,
    )

    assert GROUP_GATE_SHORTCIRCUIT_MAX_K == 20
    neg = {"lift": -0.0007, "lift_se": 0.0001}     # t = -7
    assert group_gate_short_circuit(neg, n_candidates=20) is True   # 边界含等号
    assert group_gate_short_circuit(neg, n_candidates=21) is False  # 大批不许短路
    assert group_gate_short_circuit(neg, n_candidates=1) is True

    # t ≥ 0 一律逐条（组门可能只是没过 bar）
    assert group_gate_short_circuit({"lift": 0.0001, "lift_se": 0.01},
                                    n_candidates=3) is False
    assert group_gate_short_circuit({"lift": 0.0, "lift_se": 0.01},
                                    n_candidates=3) is False

    # t 不可算（SE 缺失 / 非有限 / 0 / lift 缺失）→ 逐条，不搭便车短路
    for bad in (
        {"lift": -0.01, "lift_se": None},
        {"lift": -0.01, "lift_se": float("nan")},
        {"lift": -0.01, "lift_se": float("inf")},
        {"lift": -0.01, "lift_se": 0.0},
        {"lift": None, "lift_se": 0.001},
        {"lift": float("nan"), "lift_se": 0.001},
        {},
    ):
        assert group_gate_short_circuit(bad, n_candidates=2) is False, bad

    # 判据是 t 的符号，不是 lift 的符号：独立反算逐位比对
    for lift, se in ((-1e-9, 1e-3), (-5.0, 0.1), (1e-9, 1e-3), (3.0, 0.2)):
        assert group_gate_short_circuit(
            {"lift": lift, "lift_se": se}, n_candidates=5,
        ) is bool(lift / se < 0)


# ── 4. P0b：裸 IC 负 → 最高只能 probation ────────────────────────────────────

def test_naked_negative_ic_caps_at_probation():
    """R1：抑制变量 48–100% 被同号门屏蔽 → 降级为强制 probation（进 forward-track）。

    三分支：过 bar + second_half>0 → probation；过 bar + second_half≤0 → probation；
    admission_ic 缺失/非有限 → 原路径（可 active）。没过 bar 照旧 reject。
    """
    from factorzen.discovery.lift_test import lift_admission

    thr = DEFAULT_LIFT_THRESHOLD
    strong = {"lift": thr * 10, "lift_se": 0.0}

    # ① 过 bar + second_half > 0 + 裸 IC 负 → probation（不再 reject，也不 active）
    assert lift_admission(
        {**strong, "lift_second_half": 0.01, "admission_ic": -0.0001}, threshold=thr,
    ) == "probation"
    assert lift_admission(
        {**strong, "lift_second_half": 0.01, "admission_ic": -0.02}, threshold=thr,
    ) == "probation"
    # ② 过 bar + second_half ≤ 0 + 裸 IC 负 → probation
    assert lift_admission(
        {**strong, "lift_second_half": -0.01, "admission_ic": -0.02}, threshold=thr,
    ) == "probation"
    assert lift_admission(
        {**strong, "lift_second_half": None, "admission_ic": -0.02}, threshold=thr,
    ) == "probation"
    # ③ 缺失 / 非有限 → 原路径不变（可 active）
    assert lift_admission({**strong, "lift_second_half": 0.01}, threshold=thr) == "active"
    assert lift_admission(
        {**strong, "lift_second_half": 0.01, "admission_ic": None}, threshold=thr,
    ) == "active"
    assert lift_admission(
        {**strong, "lift_second_half": 0.01, "admission_ic": float("nan")},
        threshold=thr,
    ) == "active"
    # 裸 IC 非负仍走原路径
    assert lift_admission(
        {**strong, "lift_second_half": 0.01, "admission_ic": 0.0}, threshold=thr,
    ) == "active"

    # 没过 bar → 照旧 reject（降级只作用于"已过 bar"的候选）
    assert lift_admission(
        {"lift": thr * 0.1, "lift_se": 0.0, "lift_second_half": 0.01,
         "admission_ic": -0.02},
        threshold=thr,
    ) == "reject"
    # SE 缺失仍 reject（区间证据不完整，与裸 IC 缺失有意不同）
    assert lift_admission(
        {"lift": thr * 10, "lift_se": None, "lift_second_half": 0.01,
         "admission_ic": -0.02},
        threshold=thr,
    ) == "reject"


def test_require_positive_naked_ic_false_escape_hatch():
    """逃生口语义不变：False = 完全不看裸 IC，负裸 IC 也能拿 active。"""
    from factorzen.discovery.lift_test import lift_admission

    thr = DEFAULT_LIFT_THRESHOLD
    row = {"lift": thr * 10, "lift_se": 0.0, "lift_second_half": 0.01,
           "admission_ic": -0.02}
    assert lift_admission(row, threshold=thr) == "probation"
    assert lift_admission(
        row, threshold=thr, require_positive_naked_ic=False,
    ) == "active"
    # 关门后 second_half≤0 仍是 probation（另一条独立分支）
    assert lift_admission(
        {**row, "lift_second_half": -0.01}, threshold=thr,
        require_positive_naked_ic=False,
    ) == "probation"


# ── 5. bar = max(threshold, se_mult×se)：行为反算 ─────────────────────────────

def _critical_lift(*, se: float, threshold: float, se_mult: float) -> float:
    """二分探测 ``lift_admission`` 的放行临界 lift（不复述 bar 公式）。"""
    from factorzen.discovery.lift_test import lift_admission

    def passes(x: float) -> bool:
        return lift_admission(
            {"lift": x, "lift_se": se, "lift_second_half": 0.01},
            threshold=threshold, se_mult=se_mult,
        ) != "reject"

    lo, hi = 0.0, 1.0
    assert not passes(lo) and passes(hi)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if passes(mid):
            hi = mid
        else:
            lo = mid
    return hi


def test_bar_is_max_threshold_and_se_mult_times_se():
    """独立公式反算：SE 主导段临界值 = se_mult×se；threshold 主导段 = threshold。"""
    from scipy.stats import norm

    from factorzen.discovery.lift_test import lift_admission

    # 1.645 的外部 ground truth：标准正态单侧 95% 分位点
    assert abs(float(norm.ppf(0.95)) - 1.645) < 5e-4

    thr = 0.001
    # SE 主导：1.645×0.01 = 0.01645 ≫ thr
    crit = _critical_lift(se=0.01, threshold=thr, se_mult=1.645)
    assert abs(crit / 0.01 - 1.645) < 1e-6, crit
    # threshold 主导：1.645×1e-6 ≪ thr
    crit2 = _critical_lift(se=1e-6, threshold=thr, se_mult=1.645)
    assert abs(crit2 - thr) < 1e-9, crit2
    # 1.0 与 1.645 的临界值之比就是 1.645（证明乘数真的接在 SE 上）
    c1 = _critical_lift(se=0.01, threshold=thr, se_mult=1.0)
    assert abs(crit / c1 - 1.645) < 1e-6

    # 边界含等号：恰好等于 bar 放行
    assert lift_admission(
        {"lift": 1.645 * 0.01, "lift_se": 0.01, "lift_second_half": 0.01},
        threshold=thr, se_mult=1.645,
    ) == "active"


# ── 6. P1：调用方实参默认 1.645，lift_admission 形参默认保持 1.0 ─────────────

def test_se_mult_caller_defaults_moved_to_1645():
    """调用方默认全线 1.645；``lift_admission`` 形参默认不动（向后兼容）。"""
    from factorzen.agents.team_orchestrator import run_team_agent
    from factorzen.cli.main import _LIFT_TEST_SET, _MINE_TEAM_SET, build_parser
    from factorzen.discovery.lift_test import lift_admission
    from factorzen.pipelines.factor_mine_team import run_team_mine

    assert _LIFT_TEST_SET["se_mult"][1] == 1.645
    assert _MINE_TEAM_SET["lift_se_mult"][1] == 1.645

    args = build_parser().parse_args([
        "factor-library", "lift-test", "--session", "ws/x",
        "--start", "20200101", "--end", "20201231",
    ])
    assert args.se_mult == 1.645
    team_args = build_parser().parse_args([
        "mine", "team", "--start", "20200101", "--end", "20201231",
    ])
    assert team_args.lift_se_mult == 1.645

    sig = inspect.signature(run_team_agent).parameters
    assert sig["lift_se_mult"].default == 1.645
    assert inspect.signature(run_team_mine).parameters["lift_se_mult"].default == 1.645

    # 单一裁决函数的形参默认是兼容契约，不随调用方漂移
    assert inspect.signature(lift_admission).parameters["se_mult"].default == 1.0


# ── 7. 写回 experiment_index 的是个体 lift 而非组值 ──────────────────────────

def test_index_writeback_uses_individual_lift_not_group_value(tmp_path):
    """R1 脏账根因：组门不过时 150 行写的是同一个组值。拆组门后必须是逐条个体值。"""
    import factorzen.cli.main as cli_main
    import factorzen.discovery.factor_library as fl
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_REJECTED

    index_path = tmp_path / "experiment_index.jsonl"
    run_dir = _write_session(tmp_path, n=25, index_path=str(index_path), name="idx")
    lib_root = tmp_path / "lib_idx"
    lib_root.mkdir()
    group_lift = -0.0007
    group = {"lift": group_lift, "lift_se": 0.0001, "error": None,
             "lift_metric": "residual_ic_v1"}

    def rows(expr):
        i = int(str(expr).split("_")[1])
        return {
            "expression": expr,
            "lift": -0.001 + i * 1e-5,   # 逐条互不相同，且都进不了 bar
            "lift_se": 0.001,
            "lift_second_half": -0.001,
            "baseline": 0.02,
            "passed": False,
        }

    with pytest.MonkeyPatch.context() as mp:
        seen = _patch_cli_group(mp, group=group, lift_rows_of=rows)
        mp.setattr(
            fl, "upsert_lift_admissions",
            lambda *a, **k: {"added_active": 0, "added_probation": 0, "rejected": 25},
        )
        rc = cli_main._cmd_factor_library_lift_test(
            _cli_args(run_dir, lib_root, extra=["--apply"])
        )
    assert rc == 0
    assert seen and len(seen[0]) == 25

    recs = [
        r for r in ExperimentIndex(str(index_path)).load()
        if r.get("reject_category") == REJECT_CATEGORY_LIFT_REJECTED
    ]
    assert len(recs) == 25
    lifts = [r["lift"] for r in recs]
    assert len(set(lifts)) == 25, "每行必须是自己的 lift，不是组值复制"
    assert all(abs(v - group_lift) > 1e-9 for v in lifts)
    assert {r["lift_reason"] for r in recs} == {"below_bar"}


def test_orchestrator_weak_group_writeback_uses_individual_lift(tmp_path):
    """双路径配对：session 末钩子在组门弱时同样写个体 lift（不是组值）。"""
    import datetime as _dt

    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.state import AgentState, AttemptRecord
    from factorzen.agents.team_orchestrator import _session_end_auto_lift
    from factorzen.discovery.guardrails import (
        REJECT_CATEGORY_LIFT_QUEUE,
        REJECT_CATEGORY_LIFT_REJECTED,
    )

    days, d = [], _dt.date(2022, 1, 3)
    while len(days) < 200:
        if d.weekday() < 5:
            days.append(d)
        d += _dt.timedelta(days=1)
    panel = pl.DataFrame([
        {"trade_date": dd, "ts_code": "000001.SZ", "factor_value": 0.5}
        for dd in days
    ])
    daily = panel.rename({"factor_value": "close"})
    holdout = daily.filter(pl.col("trade_date") >= _dt.date(2022, 8, 1))

    k = 25  # > GROUP_GATE_SHORTCIRCUIT_MAX_K → 组门弱也必逐条
    state = AgentState(seed=1)
    for i in range(k):
        state.attempts.append(AttemptRecord(
            0, "H", f"cand_{i}", True, 0.02, False, "keep", None,
            ir_train=0.1, reject_category=REJECT_CATEGORY_LIFT_QUEUE,
            residual_ic_train=0.02,
        ))

    group_lift = -0.0007
    index = ExperimentIndex(str(tmp_path / "experiment_index.jsonl"))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "factorzen.discovery.lift_test.run_group_lift",
            lambda *a, **kw: {"lift": group_lift, "lift_se": 0.0001, "error": None},
        )
        mp.setattr(
            "factorzen.discovery.lift_test.filter_candidates_by_coverage",
            lambda cands, **kw: (list(cands), []),
        )
        mp.setattr(
            "factorzen.discovery.lift_test.make_lift_context",
            lambda *a, **kw: SimpleNamespace(
                horizon=1, prepped=daily,
                admission_start="2022-08-01", admission_end="2022-12-31",
            ),
        )
        mp.setattr(
            "factorzen.discovery.lift_test.run_lift_tests",
            lambda queue, **kw: [
                {
                    "expression": c.get("expression"),
                    "lift": -0.001 + int(str(c.get("expression")).split("_")[1]) * 1e-5,
                    "lift_se": 0.001, "lift_second_half": -0.001,
                    "baseline": 0.02, "passed": False,
                }
                for c in queue
            ],
        )
        mp.setattr(
            "factorzen.discovery.factor_library.upsert_lift_admissions",
            lambda *a, **kw: {"added_active": 0, "added_probation": 0, "rejected": k},
            raising=False,
        )
        meta = _session_end_auto_lift(
            state, daily=daily, holdout_df=holdout, profile=None,
            ctx=SimpleNamespace(leaf_map=None), market="ashare",
            library_root=str(tmp_path / "lib"), seed=1,
            auto_lift=True, lift_se_mult=1.645, horizon=1,
            materialize_candidate=lambda expr: panel,
            index=index,
            data_window={"start": "20220101", "end": "20221231",
                         "universe": "csi300", "market": "ashare"},
        )

    assert meta.get("lift_error") is None, meta.get("lift_error")
    assert meta["lift_group"]["group_gate_weak"] is True
    assert meta["lift_group"]["group_gate_short_circuit"] is False
    assert len(meta["lift_results"]) == k

    recs = [
        r for r in index.load()
        if r.get("reject_category") == REJECT_CATEGORY_LIFT_REJECTED
    ]
    assert len(recs) == k
    lifts = [r["lift"] for r in recs]
    assert len(set(lifts)) == k, "组门弱时也必须逐条写自己的 lift"
    assert all(abs(v - group_lift) > 1e-9 for v in lifts)
    assert {r["lift_reason"] for r in recs} == {"below_bar"}


# ── 8. 库 schema：naked_ic_negative 可选布尔 + 旧记录向前兼容 ────────────────

def test_naked_ic_negative_flag_written_and_old_records_load(tmp_path):
    from factorzen.discovery.factor_library import (
        FactorRecord,
        load_library,
        upsert_lift_admissions,
    )

    thr = DEFAULT_LIFT_THRESHOLD
    root = tmp_path / "lib"
    rows = [
        {"expression": "rank(close)", "lift": thr * 10, "lift_se": 0.0,
         "lift_second_half": 0.01, "admission_ic": -0.02, "baseline": 0.04},
        {"expression": "rank(vol)", "lift": thr * 10, "lift_se": 0.0,
         "lift_second_half": 0.01, "admission_ic": 0.02, "baseline": 0.04},
    ]
    out = upsert_lift_admissions(
        rows, market="ashare", root=str(root), threshold=thr, se_mult=1.645,
        allow_active=True,
        compact_materialize=lambda e: None,
    )
    assert out["errors"] == [], out["errors"]
    lib = {r.expression: r for r in load_library("ashare", root=str(root))}
    neg = lib["rank(close)"]
    pos = lib["rank(vol)"]
    assert neg.naked_ic_negative is True
    assert neg.status == "probation" and neg.admission_decision == "probation"
    assert not pos.naked_ic_negative
    assert pos.status == "active" and pos.admission_decision == "active"
    # 落盘可 JSON 序列化且字段存在
    assert json.loads(json.dumps(neg.to_dict()))["naked_ic_negative"] is True

    # 旧记录（无该字段）读回不炸，值为 None
    old = FactorRecord.from_dict({
        "expression": "ts_mean(close, 5)", "market": "ashare",
        "status": "probation", "admission_track": "lift", "lift": 0.002,
    })
    assert old.naked_ic_negative is None
