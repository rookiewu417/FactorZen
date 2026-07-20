"""
test_agent_eval_parity.py：Merged discovery tests: test_agent_eval_parity.py
test_base_panel_share.py：任务 G：base_panel 共享 + 自适应 lift_workers。
"""

from __future__ import annotations

import ast
import datetime as dt
import json
from datetime import (
    date,
    timedelta,
)
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.discovery.evaluation import evaluate_expressions
from factorzen.discovery.expression import (
    parse_expr,
    to_expr_string,
)
from factorzen.discovery.guardrails import (
    DeflationBasis,
    deflated_pvalue,
)
from factorzen.discovery.scoring import DataBundle
from factorzen.pipelines.factor_mine_agent import run_agent_mine
from factorzen.pipelines.factor_mine_team import run_team_mine
from factorzen.validation.deflated_sharpe import deflated_sharpe

# ==== 来自 test_agent_eval_parity.py ====
# ==== 来自 test_agent_pipeline.py ====
# tests/test_agent_pipeline.py

def _mock_daily__agent_pipeline(n_stocks=40, n_days=180, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
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
            rows.append({"trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                         "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)

def _scripted_llm():
    prop = json.dumps({"hypothesis": "动量", "expressions": ["ts_mean(close,5)"], "rationale": "r"})
    sem = json.dumps({"consistent": True, "reason": "ok"})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [prop, sem, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn

def _scripted_team(expr: str):
    """team 路径固定表达式 stub：Hypothesis→Coder→Critic(keep) 循环。"""
    hyp = json.dumps({"hypotheses": ["动量"]})
    code = json.dumps({"expressions": [expr]})
    crit = json.dumps({"verdict": "keep", "reason": "ok"})
    seq = [hyp, code, crit] * 50
    i = {"k": 0}
    def fn(messages):
        v = seq[i["k"] % len(seq)]
        i["k"] += 1
        return v
    return fn

def _n_train_from_manifest(run_dir: str, expr: str) -> int:
    """从落盘 manifest 的 attempts 里取某表达式的 train 段有效 IC 天数。"""
    m = json.loads((Path(run_dir) / "manifest.json").read_text())
    norm = to_expr_string(parse_expr(expr))
    xs = [a["n_train"] for a in m["attempts"] if a["expression"] == norm and a["n_train"]]
    assert xs, f"{norm} 未被评估或 n_train 为空: {[a['expression'] for a in m['attempts']]}"
    return xs[0]

def test_run_agent_mine_writes_manifest(tmp_path: Path):
    daily = _mock_daily__agent_pipeline()
    res = run_agent_mine(daily, n_rounds=2, seed=42, out_dir=str(tmp_path),
                         llm_fn=_scripted_llm(), run_id="t1", export=False)
    run_dir = Path(res["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "candidates.csv").exists()   # 兼容 fz mine leaderboard
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["n_trials"] >= 1
    assert res["n_trials"] == m["n_trials"]

def test_run_agent_mine_forwards_eval_start_clipping_warmup(tmp_path: Path):
    """接线漂移回归：pipeline `run_agent_mine` 必须把 eval_start 透传给 `run_llm_agent`。

    否则生产 `fz mine agent` 里 warmup-parity 修复完全失效——`daily` 由
    `prepare_mining_daily` 带 60 天预热前缀，不透传 eval_start 时整帧（含预热段）
    随 `split_holdout` 进 mining_df/DataBundle，预热噪声照旧灌进 train IC。

    判别力（纯行为，读落盘 manifest 的 attempts）：同一含预热前缀的完整帧，
    透传 eval_start 后 train 段裁到 eval_start，有效 IC 天数严格少于不透传
    （不透传时 train 段从帧首起、覆盖预热段）。签名断言无判别力，故从 pipeline
    最外层出发、以 n_train 的可观测差异为准。
    """
    daily = _mock_daily__agent_pipeline(n_days=180, seed=7)
    dates = sorted(set(daily["trade_date"].to_list()))
    eval_start = dates[40]                          # 前 40 个交易日作预热前缀
    expr = "ts_mean(close,5)"

    clip = run_agent_mine(daily, n_rounds=1, seed=42, out_dir=str(tmp_path / "clip"),
                          llm_fn=_scripted_llm(), run_id="clip", export=False,
                          eval_start=eval_start.strftime("%Y%m%d"))
    noclip = run_agent_mine(daily, n_rounds=1, seed=42, out_dir=str(tmp_path / "noclip"),
                            llm_fn=_scripted_llm(), run_id="noclip", export=False)

    n_clip = _n_train_from_manifest(clip["run_dir"], expr)
    n_noclip = _n_train_from_manifest(noclip["run_dir"], expr)
    assert n_clip < n_noclip, (
        f"eval_start 未透传/未裁预热段：clip n_train={n_clip} 应 < noclip n_train={n_noclip}")

def test_run_team_mine_forwards_eval_start_clipping_warmup(tmp_path: Path):
    """接线漂移回归：pipeline `run_team_mine` 必须把 eval_start 透传给 `run_team_agent`。

    与 agent 单路径同理：不透传时生产 `fz mine team` 的 warmup-parity 修复失效。
    判别力同上：透传后 train 段有效 IC 天数严格少于不透传。
    """
    daily = _mock_daily__agent_pipeline(n_days=180, seed=7)
    dates = sorted(set(daily["trade_date"].to_list()))
    eval_start = dates[40]
    expr = "ts_mean(close, 5)"

    clip = run_team_mine(daily, n_rounds=1, seed=42, index_path=str(tmp_path / "i1.jsonl"),
                         out_dir=str(tmp_path / "clip"), llm_fn=_scripted_team(expr),
                         run_id="clip", export=False, heal_rounds=0,
                         eval_start=eval_start.strftime("%Y%m%d"))
    noclip = run_team_mine(daily, n_rounds=1, seed=42, index_path=str(tmp_path / "i2.jsonl"),
                           out_dir=str(tmp_path / "noclip"), llm_fn=_scripted_team(expr),
                           run_id="noclip", export=False, heal_rounds=0)

    n_clip = _n_train_from_manifest(clip["run_dir"], expr)
    n_noclip = _n_train_from_manifest(noclip["run_dir"], expr)
    assert n_clip < n_noclip, (
        f"eval_start 未透传/未裁预热段：clip n_train={n_clip} 应 < noclip n_train={n_noclip}")

# ==== 来自 test_agent_evaluation.py ====
# tests/test_agent_evaluation.py

def _mock_daily__agent_evaluation(n_stocks=40, n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2022, 1, 3)
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
            rows.append({"trade_date": dd, "ts_code": c, "close": px,
                         "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)

def test_evaluate_valid_expressions():
    daily = _mock_daily__agent_evaluation()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["ts_mean(close,5)", "rank(vol)"], daily, bundle)
    assert len(out) == 2
    for r in out:
        assert r["compile_ok"] is True
        assert r["ic_train"] is not None        # 真算出了 IC（非 None）
        assert isinstance(r["ic_train"], float)

def test_evaluate_rejects_illegal_expression():
    daily = _mock_daily__agent_evaluation()
    bundle = DataBundle.build(daily)
    out = evaluate_expressions(["this_is_not_an_operator(close)", "ts_mean(close,5)"], daily, bundle)
    assert out[0]["compile_ok"] is False and out[0]["error"]   # 非法被拒，记错误
    assert out[0]["ic_train"] is None
    assert out[1]["compile_ok"] is True                         # 合法的照常评估

# ==== 来自 test_agent_eval_real_adj_and_leaves.py ====
def _daily_with_adj_and_basic(n_stocks=40, n_days=120, seed=1):
    rng = np.random.default_rng(seed)
    days, d = [], date(2022, 1, 3)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for i in range(n_stocks):
        c = f"{i:06d}.SZ"
        px = 10.0
        for dd in days:
            prev = px
            px *= 1 + rng.standard_normal() * 0.02
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px, "open": px * 0.99,
                "high": px * 1.01, "low": px * 0.98, "pre_close": prev,
                # close_adj 明显区别于 close（模拟复权：×2），验证不被 close 冒充
                "close_adj": px * 2.0, "open_adj": px * 0.99 * 2.0,
                "high_adj": px * 1.01 * 2.0, "low_adj": px * 0.98 * 2.0,
                "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                "total_mv": 5e5 + i * 1e4, "pb": 1.0 + i * 0.1,
            })
    return pl.DataFrame(rows)

def test_derived_and_basic_leaves_evaluable():
    daily = _daily_with_adj_and_basic()
    bundle = DataBundle.build(daily)
    # ret_1d(派生)、total_mv(基本面)、amplitude(派生) —— 修复前评估帧缺这些列 → 报错
    out = evaluate_expressions(["rank(ret_1d)", "rank(total_mv)", "rank(amplitude)"], daily, bundle)
    for r in out:
        assert r["compile_ok"], f"{r['expression']} 应可编译"
        assert r["error"] is None, f"{r['expression']} 不应报错，实得 {r['error']}"
        assert r["ic_train"] is not None, f"{r['expression']} 应算出 IC"

def test_uses_real_close_adj_not_faked_from_close():
    """close_adj 明显≠close 时，ret_1d 须用 close_adj 计算，而非被 close 冒充。"""
    from factorzen.discovery.evaluation import _preprocess_daily

    daily = _daily_with_adj_and_basic(n_stocks=2, n_days=10)
    prepped = _preprocess_daily(daily)
    # ret_1d 由 close_adj 算；close_adj=close×2 是等比缩放，比率与 close 算的相同，
    # 但关键是 prep 未把 close 覆盖成 close_adj —— close_adj 仍是 close 的 2 倍。
    a = prepped.filter(pl.col("close_adj").is_not_null()).select(
        (pl.col("close_adj") / pl.col("close")).alias("r"))["r"]
    assert all(abs(v - 2.0) < 1e-9 for v in a.to_list()), "close_adj 不应被未复权 close 覆盖"

# ==== 来自 test_agent_candidates_csv.py ====
def test_agent_candidates_csv_df_has_rank_passed():
    from factorzen.discovery.export import agent_candidates_csv_df

    df = agent_candidates_csv_df([{"expression": "rank(close)", "holdout_ic": 0.1, "dsr": 0.6}])
    assert "rank" in df.columns and "passed" in df.columns and "expression" in df.columns
    assert df["rank"].to_list() == [1]
    assert df["passed"].to_list() == [True]

def test_export_alpha_reads_agent_candidates(tmp_path: Path):
    """read_candidate_expression（export-alpha 用）能读 Agent candidates.csv，不再报缺 rank。"""
    from factorzen.discovery.export import agent_candidates_csv_df, read_candidate_expression

    cands = [{"expression": "rank(close)", "holdout_ic": 0.1, "dsr": 0.6},
             {"expression": "ts_mean(vol, 5)", "holdout_ic": 0.05, "dsr": 0.4}]
    agent_candidates_csv_df(cands).write_csv(tmp_path / "candidates.csv")

    assert read_candidate_expression(str(tmp_path), rank=1, require_passed=True) == "rank(close)"
    assert read_candidate_expression(str(tmp_path), rank=2, require_passed=True) == "ts_mean(vol, 5)"

# ==== 来自 test_deflation_recipe_parity.py ====
# tests/test_deflation_recipe_parity.py

_SRC = Path(__file__).resolve().parents[2] / "src" / "factorzen"

# ── 共享配方的语义 ──────────────────────────────────────────────────────────

def test_basis_uses_population_variance_and_pool_size():
    """N 与 sharpe_variance 必须同源（R8）：都来自同一批 trial 的 IR 池。"""
    pool = [0.20, 0.10, -0.13, 0.05]
    basis = DeflationBasis.from_ir_pool(pool)

    assert basis.n_trials == 4
    assert basis.sharpe_variance == pytest.approx(float(np.var(np.asarray(pool))))

def test_basis_degenerates_to_unit_variance_for_single_trial():
    """池大小 < 2 时经验方差无意义，退化为 1.0（与 M1 既有行为一致）。"""
    assert DeflationBasis.from_ir_pool([0.3]).sharpe_variance == 1.0
    assert DeflationBasis.from_ir_pool([]).sharpe_variance == 1.0
    assert DeflationBasis.from_ir_pool([]).n_trials == 0

def test_basis_drops_none_and_nonfinite():
    """死表达式(None)与 nan/inf 不得进池——它们会同时污染方差与计数。"""
    basis = DeflationBasis.from_ir_pool([0.2, None, float("nan"), 0.1, float("inf")])

    assert basis.n_trials == 2
    assert basis.sharpe_variance == pytest.approx(float(np.var(np.asarray([0.2, 0.1]))))

def test_nan_in_pool_does_not_poison_every_candidate():
    """一个畸形 IR 不得静默废掉整个 session 的护栏。

    旧的 M1 写法 `np.array([...]).var()` 遇 nan → `sharpe_variance=nan`
    → `expected_max_sharpe` 的 `sharpe_variance <= 0` 判否（nan 比较恒 False）→ `sqrt(nan)`
    → `sr0=nan` → 所有候选的 `dsr_pvalue=nan` → `guardrail_passed` 因 nan 检查一律判否。
    **整批候选被静默拒绝，且看不出原因。** `from_ir_pool` 剔除非有限值后不再如此。
    """
    basis = DeflationBasis.from_ir_pool([0.42, float("nan"), 0.18, -0.13])

    assert basis.sharpe_variance == basis.sharpe_variance, "sharpe_variance 不得是 nan"
    assert basis.n_trials == 3
    _dsr, p = deflated_pvalue(0.42, basis, 305)
    assert p == p and 0.0 <= p <= 1.0, "p 值必须可用，而非被 nan 传染"

def test_deflated_pvalue_delegates_with_basis():
    basis = DeflationBasis.from_ir_pool([0.2, 0.1, -0.13])
    got = deflated_pvalue(0.2, basis, n_obs=300)
    want = deflated_sharpe(0.2, basis.n_trials, 300, sharpe_variance=basis.sharpe_variance)
    assert got == want

# ── 架构守卫：任何一条挖掘路径都不得绕过共享配方 ──────────────────────────────

@pytest.mark.parametrize("rel", ["discovery/mining_session.py", "agents/nodes.py"])
def test_mining_paths_never_call_deflated_sharpe_directly(rel):
    """两条挖掘路径必须经 `deflated_pvalue`。直接调 `deflated_sharpe` 就能自选
    `sharpe_variance`/`n_trials`，口径会再次漂移——那正是 P0 的成因。

    抓两种形式：`deflated_sharpe(...)`（Name）与 `ds.deflated_sharpe(...)`（Attribute）。
    """
    tree = ast.parse((_SRC / rel).read_text(encoding="utf-8"))
    direct = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "deflated_sharpe")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "deflated_sharpe")
        )
    ]
    assert not direct, (
        f"{rel} 直接调用了 deflated_sharpe（第 {[n.lineno for n in direct]} 行），"
        f"绕过共享的 deflated_pvalue → 两路 deflation 口径会再次漂移"
    )

def test_deflated_sharpe_is_imported_only_by_guardrails():
    """把守卫从「绊线」升级成「墙」：`deflated_sharpe` 只许 `guardrails.py` 导入。

    仅禁止调用形式挡不住 `import factorzen.validation.deflated_sharpe as ds` 之后的花式引用。
    源头收口——拿不到这个符号，就没法绕过 `deflated_pvalue` 自选 deflation 参数。
    （`validation/` 内部与测试不受限；本断言只约束 src/factorzen 下的生产代码。）
    """
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        rel = path.relative_to(_SRC).as_posix()
        if rel.startswith("validation/") or rel == "discovery/guardrails.py":
            continue
        # utf-8-sig：仓库里有文件带 BOM，ast.parse 遇 U+FEFF 会抛 SyntaxError
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for n in ast.walk(tree):
            if (isinstance(n, ast.ImportFrom) and n.module
                    and "deflated_sharpe" in n.module
                    and any(a.name == "deflated_sharpe" for a in n.names)):
                offenders.append(f"{rel}:{n.lineno}")          # from ... import deflated_sharpe
            elif isinstance(n, ast.Import) and any("deflated_sharpe" in a.name for a in n.names):
                offenders.append(f"{rel}:{n.lineno}")          # import ...deflated_sharpe [as ds]

    assert not offenders, (
        "只有 discovery/guardrails.py 可以导入 deflated_sharpe（其余须经 deflated_pvalue）；"
        f"违规：{offenders}"
    )

# ── 真正的 cross-path decision-parity（驱动两条真实路径）──────────────────────

def _mk_daily(n_stocks: int = 40, n_days: int = 260, seed: int = 5) -> pl.DataFrame:
    """M1 的 run_session 需要复权价列（add_derived_columns 用 close_adj 算 ret_1d）。

    股票数 ≥ 40：`_MIN_CROSS_SAMPLES = 30` 会把截面股票数不足 30 的日期全部过滤，
    IC 序列为空时 `quick_fitness` 落回 sentinel 0.0，测试就跑在 IC≡0 的垃圾数据上了。
    """
    rng = np.random.default_rng(seed)
    days, d = [], dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        px = rng.uniform(8, 15)
        for dd in days:
            px = float(max(px * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": dd, "ts_code": c,
                         "close": px, "open": px * 0.99, "high": px * 1.01, "low": px * 0.98,
                         "close_adj": px, "open_adj": px * 0.99,
                         "high_adj": px * 1.01, "low_adj": px * 0.98, "pre_close": px,
                         "vol": float(abs(rng.standard_normal()) * 1e6 + 1e5),
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6)})
    return pl.DataFrame(rows)

def test_m1_dsr_pvalue_is_produced_by_the_shared_recipe(tmp_path):
    """驱动真实 `run_session`，用它自报的 basis 复算每个候选的 p 值，必须逐位吻合。

    这一步把「配方 == M1 真实所做」从一次性反解升格为 CI 断言。
    """
    from factorzen.discovery.mining_session import run_session

    res = run_session(_mk_daily(), n_trials=25, top_k=5, seed=3, method="random",
                      out_dir=str(tmp_path))
    assert res["candidates"], "本测试需要 M1 至少产出一个候选"

    basis = DeflationBasis(n_trials=res["n_trials"], sharpe_variance=res["sharpe_variance"])
    for c in res["candidates"]:
        _dsr, want = deflated_pvalue(c["ir_train"], basis, c["n_train"])
        assert c["dsr_pvalue"] == pytest.approx(round(float(want), 4), abs=1e-9), (
            f"M1 的 dsr_pvalue 与共享配方不符：{c['expression']}"
        )

def test_m1_reports_basis_for_reproducibility(tmp_path):
    """`sharpe_variance` 决定 deflation 门槛，属于「事后能重跑出同样结果」的必要信息。"""
    import json

    from factorzen.discovery.mining_session import run_session

    res = run_session(_mk_daily(), n_trials=20, top_k=3, seed=4, method="random",
                      out_dir=str(tmp_path))
    m = json.loads((Path(res["session_dir"]) / "manifest.json").read_text())

    assert m["sharpe_variance"] == pytest.approx(res["sharpe_variance"])
    assert m["n_trials"] == res["n_trials"]



# ==== 来自 test_base_panel_share.py ====
def _dates(n_days: int):
    days, d = [], date(2024, 1, 2)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return days


def _long_factor(
    dates: list[str],
    codes: list[str],
    rng: np.random.Generator,
    *,
    scale: float = 1.0,
    extra_rows: list[tuple[str, str, float]] | None = None,
) -> pl.DataFrame:
    rows = []
    for d in dates:
        vals = rng.standard_normal(len(codes)) * scale
        for s, code in enumerate(codes):
            rows.append({
                "trade_date": d,
                "ts_code": code,
                "factor_value": float(vals[s]),
            })
    if extra_rows:
        for d, code, v in extra_rows:
            rows.append({
                "trade_date": d,
                "ts_code": code,
                "factor_value": float(v),
            })
    return pl.DataFrame(rows)


def _synth_five_plus_one(
    *,
    n_days: int = 80,
    n_stocks: int = 24,
    seed: int = 7,
    candidate_extra: bool = True,
):
    """5 基线因子 + 1 候选；可选候选独有行（基线 5 列全缺）。"""
    rng = np.random.default_rng(seed)
    dates = _dates(n_days)
    codes = [f"{i:04d}.SZ" for i in range(n_stocks)]
    baseline = {
        f"b{i}": _long_factor(dates, codes, rng, scale=1.0 + 0.1 * i)
        for i in range(5)
    }
    extra = None
    if candidate_extra:
        # 候选覆盖超出基线：新日期 + 新股票（基线全缺）
        extra_date = _dates(n_days + 5)[-1]
        extra = [
            (extra_date, "9999.SZ", 1.5),
            (extra_date, codes[0], -0.8),
            (dates[10], "8888.SZ", 2.0),
        ]
        # ret 覆盖这些边角行
        ret_extra = list(extra)
    else:
        ret_extra = []

    cand = _long_factor(dates, codes, rng, scale=1.3, extra_rows=extra)
    ret_rows = []
    for d in dates:
        rets = 0.05 * rng.standard_normal(len(codes))
        for s, code in enumerate(codes):
            ret_rows.append({"trade_date": d, "ts_code": code, "ret": float(rets[s])})
    for d, code, _v in ret_extra:
        ret_rows.append({
            "trade_date": d,
            "ts_code": code,
            "ret": float(0.01 * rng.standard_normal()),
        })
    ret = pl.DataFrame(ret_rows)
    full = {**baseline, "cand": cand}
    return baseline, cand, ret, full


def _assert_panel_equal(a: pl.DataFrame, b: pl.DataFrame, *, atol: float = 1e-12):
    """宽表面板逐值一致（按键排序后比列）。"""
    keys = ["trade_date", "ts_code"]
    a_s = a.sort(keys)
    b_s = b.sort(keys)
    assert a_s.columns == b_s.columns, f"列序/列名不一致: {a_s.columns} vs {b_s.columns}"
    assert a_s.height == b_s.height
    assert a_s.select(keys).equals(b_s.select(keys))
    for col in a_s.columns:
        if col in keys:
            continue
        av = a_s[col].to_numpy().astype(float)
        bv = b_s[col].to_numpy().astype(float)
        # null 对齐
        a_null = a_s[col].is_null().to_numpy()
        b_null = b_s[col].is_null().to_numpy()
        assert np.array_equal(a_null, b_null), f"null 掩码不一致: {col}"
        both = ~a_null
        if both.any():
            np.testing.assert_allclose(av[both], bv[both], atol=atol, rtol=0)


def test_build_panel_base_share_parity_with_extra_rows():
    """build_panel 全量 vs base+候选：含候选超出基线行集边角。"""
    from factorzen.research.combination.models import build_panel

    baseline, cand, ret, full = _synth_five_plus_one(candidate_extra=True)
    full_panel = build_panel(full, ret)
    base = build_panel(baseline, ret)
    incr = build_panel({"cand": cand}, ret, base_panel=base)

    # 列：基线序 + cand 在最后（与 full dict 插入序一致）
    assert _feature_tail(full_panel) == ["b0", "b1", "b2", "b3", "b4", "cand"]
    assert _feature_tail(incr) == ["b0", "b1", "b2", "b3", "b4", "cand"]
    _assert_panel_equal(full_panel, incr)

    # 边角：存在基线全缺、候选有值、且有 ret 的行
    feat_cols = ["b0", "b1", "b2", "b3", "b4"]
    edge = incr.filter(
        pl.all_horizontal([pl.col(c).is_null() for c in feat_cols])
        & pl.col("cand").is_not_null()
    )
    assert edge.height >= 1, "应含候选超出基线行集的边角行"


def _feature_tail(panel: pl.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in ("trade_date", "ts_code", "ret")]


def test_combine_lgbm_base_panel_parity_golden():
    """核心 golden：combine_lgbm(全量) vs combine_lgbm(候选, base_panel=)。

    两条独立路径互证；禁止恒真。atol=1e-12。
    """
    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import build_panel, combine_lgbm

    baseline, cand, ret, full = _synth_five_plus_one(
        n_days=90, n_stocks=20, seed=11, candidate_extra=True,
    )
    cv = PurgedWalkForwardCV(
        train_days=40, test_days=15, purge_days=5, embargo_days=0, expanding=False,
    )
    kw = dict(seed=3, n_estimators=30, min_child_samples=10, num_leaves=15)

    out_full = combine_lgbm(full, ret, cv, **kw)
    base = build_panel(baseline, ret)
    out_incr = combine_lgbm({"cand": cand}, ret, cv, base_panel=base, **kw)

    assert out_full.height > 0 and out_incr.height > 0
    # 两条路径都必须有真实预测，且非全零（防恒真）
    assert float(out_full["factor_value"].std()) > 1e-6
    assert float(out_incr["factor_value"].std()) > 1e-6

    a = out_full.sort(["trade_date", "ts_code"])
    b = out_incr.sort(["trade_date", "ts_code"])
    assert a.height == b.height
    assert a.select(["trade_date", "ts_code"]).equals(b.select(["trade_date", "ts_code"]))
    np.testing.assert_allclose(
        a["factor_value"].to_numpy().astype(float),
        b["factor_value"].to_numpy().astype(float),
        atol=1e-12,
        rtol=0,
    )

    # 全量 dict + base_panel 同样一致
    out_full_bp = combine_lgbm(full, ret, cv, base_panel=base, **kw)
    c = out_full_bp.sort(["trade_date", "ts_code"])
    np.testing.assert_allclose(
        a["factor_value"].to_numpy().astype(float),
        c["factor_value"].to_numpy().astype(float),
        atol=1e-12,
        rtol=0,
    )


# ── G1 自适应 workers ──────────────────────────────────────────────────────


def test_adaptive_lift_workers_from_sysconf(monkeypatch):
    """可用内存 → workers = max(2, min(4, gb//5))。"""
    from factorzen.discovery import lift_test as lt

    # 23GB → 4；12GB → 2；4GB → 2（下限）；0 → 2；cap 100GB → 4
    cases = [
        (23 * 1024**3, 4),
        (12 * 1024**3, 2),
        (4 * 1024**3, 2),
        (0, 2),
        (100 * 1024**3, 4),  # cap
    ]
    page = 4096

    for avail_bytes, expected in cases:
        pages = avail_bytes // page

        def _sysconf(name, _pages=pages, _page=page):
            if name == "SC_AVPHYS_PAGES":
                return _pages
            if name == "SC_PAGE_SIZE":
                return _page
            raise ValueError(name)

        monkeypatch.setattr(lt.os, "sysconf", _sysconf)
        assert lt.adaptive_lift_workers() == expected


def test_adaptive_lift_workers_sysconf_error_fallback(monkeypatch):
    from factorzen.discovery import lift_test as lt

    def boom(_name):
        raise OSError("no sysconf")

    monkeypatch.setattr(lt.os, "sysconf", boom)
    assert lt.adaptive_lift_workers() == 2
    assert lt.resolve_lift_workers(None) == 2


def test_resolve_lift_workers_explicit_not_overridden(monkeypatch):
    from factorzen.discovery import lift_test as lt

    def _sysconf(name):
        # 假装只有 4GB → 自适应下限 2；显式 1 仍串行
        if name == "SC_AVPHYS_PAGES":
            return (4 * 1024**3) // 4096
        if name == "SC_PAGE_SIZE":
            return 4096
        raise ValueError(name)

    monkeypatch.setattr(lt.os, "sysconf", _sysconf)
    assert lt.adaptive_lift_workers() == 2
    assert lt.resolve_lift_workers(6) == 6
    assert lt.resolve_lift_workers(1) == 1
    assert lt.resolve_lift_workers(0) == 0


def test_run_lift_tests_default_workers_adaptive(monkeypatch):
    """lift_workers=None（默认）走自适应；低内存仍 ≥2 建池；显式 1 不建池。"""
    from factorzen.discovery import lift_test as lt

    def _sysconf(name):
        if name == "SC_AVPHYS_PAGES":
            return (8 * 1024**3) // 4096  # 8GB → max(2, 1)=2
        if name == "SC_PAGE_SIZE":
            return 4096
        raise ValueError(name)

    monkeypatch.setattr(lt.os, "sysconf", _sysconf)

    created = {"n": 0, "max_workers": None}
    real = lt.ThreadPoolExecutor

    class SpyPool:
        def __init__(self, *a, **k):
            created["n"] += 1
            created["max_workers"] = k.get("max_workers", a[0] if a else None)
            self._inner = real(*a, **k)

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *a):
            return self._inner.__exit__(*a)

    monkeypatch.setattr(lt, "ThreadPoolExecutor", SpyPool)

    dates = _dates(30)
    # residual_ic_v1 日守卫 max(30, k+10)：股票数须 ≥30，否则整日被跳过
    codes = [f"{i:04d}.SZ" for i in range(40)]
    rng_r = np.random.default_rng(2)
    active = {"lib": _long_factor(dates, codes, np.random.default_rng(0))}
    cand = _long_factor(dates, codes, np.random.default_rng(1))
    ret = pl.DataFrame([
        {"trade_date": d, "ts_code": c, "ret": float(rng_r.standard_normal())}
        for d in dates for c in codes
    ])

    common = dict(
        market="ashare",
        daily=pl.DataFrame(),
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: cand,
        top_m=None,
    )
    rows = lt.run_lift_tests(
        [{"expression": "c0", "residual_ic_train": 0.01}],
        lift_workers=None,  # 显式 None = 自适应
        **common,
    )
    assert len(rows) == 1
    # 8GB → workers=2 → 建池
    assert created["n"] == 1
    assert created["max_workers"] == 2

    created["n"] = 0
    created["max_workers"] = None
    rows1 = lt.run_lift_tests(
        [{"expression": "c0", "residual_ic_train": 0.01}],
        lift_workers=1,  # 显式串行
        **common,
    )
    assert len(rows1) == 1
    assert created["n"] == 0


# ── H2 全零行：退化候选 / 安全名一致性 ──────────────────────────────────────


def test_degenerate_candidate_shared_path_is_error_not_zero_lift():
    """契约：物化后全 null 候选 → 显式 error，禁止 lift=0 假结论。

    residual_ic_v1 下该契约更吃紧：共线候选**合法**产出 lift=0.0（SE=None →
    reject），因此「全 null」必须与之区分为 ``degenerate_candidate`` + lift=None，
    否则两种成因在库里长得一样、无法诊断。正常候选同批不受影响。
    """
    import factorzen.discovery.lift_test as lt

    baseline, cand_ok, ret, _full = _synth_five_plus_one(
        n_days=80, n_stocks=40, seed=9, candidate_extra=False,
    )
    active = {"lib_a": baseline["b0"], "lib_b": baseline["b1"]}
    dates = _dates(80)
    codes = [f"{i:04d}.SZ" for i in range(40)]
    # 全 null 候选（与 drop_degenerate 同口径）
    cand_null = pl.DataFrame({
        "trade_date": [d for d in dates for _ in codes],
        "ts_code": codes * len(dates),
        "factor_value": [None] * (len(dates) * len(codes)),
    }).with_columns(pl.col("factor_value").cast(pl.Float64))

    mats = {"null_cand": cand_null, "ok_cand": cand_ok}
    grays = [
        {"expression": "null_cand", "residual_ic_train": 0.03},
        {"expression": "ok_cand", "residual_ic_train": 0.02},
    ]
    daily = pl.DataFrame({"trade_date": [], "ts_code": [], "close": []})

    rows = lt.run_lift_tests(
        grays,
        market="ashare",
        daily=daily,
        active_factor_dfs=active,
        ret_df=ret,
        materialize_candidate=lambda e: mats[e],
        lift_workers=1,
        seed=0,
        top_m=None,
        threshold=-1.0,  # 阈值放到最低，确保拒因来自 error 而非未过门
    )

    assert len(rows) == 2
    by = {r["expression"]: r for r in rows}
    bad = by["null_cand"]
    good = by["ok_cand"]
    # 退化：显式 error，不是 lift=0 / se=0 的假结论
    assert bad["error"] == "degenerate_candidate"
    assert bad["lift"] is None
    assert bad["passed"] is False
    # 正常候选不受影响
    assert good["error"] is None
    assert good["lift"] is not None


def test_combine_lgbm_rejects_all_degenerate_new_factors():
    """combine 层防御：意图新增的因子全被 drop → 显式 ValueError。"""
    import pytest

    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import build_panel, combine_lgbm

    baseline, _cand, ret, _ = _synth_five_plus_one(
        n_days=50, n_stocks=10, seed=1, candidate_extra=False,
    )
    active = {"b0": baseline["b0"], "b1": baseline["b1"]}
    base = build_panel(active, ret)
    dates = _dates(50)
    codes = [f"{i:04d}.SZ" for i in range(10)]
    null_cand = pl.DataFrame({
        "trade_date": [d for d in dates for _ in codes],
        "ts_code": codes * len(dates),
        "factor_value": [None] * (len(dates) * len(codes)),
    }).with_columns(pl.col("factor_value").cast(pl.Float64))
    cv = PurgedWalkForwardCV(
        train_days=25, test_days=10, purge_days=2, expanding=False,
    )
    with pytest.raises(ValueError, match="degenerate_new_factors"):
        combine_lgbm(
            {"new_null": null_cand},
            ret,
            cv,
            base_panel=base,
            seed=0,
            n_estimators=10,
            min_child_samples=5,
        )


def test_paired_lift_stats_all_zero_diff_se_is_none():
    """diff 全零：lift=0 但 lift_se=None（不许当 SE=0 强结论）。"""
    from factorzen.discovery.lift_test import paired_lift_stats

    dates = [f"202401{d:02d}" for d in range(1, 41)]
    ics = [0.01 + 0.001 * (i % 5) for i in range(40)]
    daily = pl.DataFrame(
        {"trade_date": dates, "ic": ics},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )
    stats = paired_lift_stats(daily, daily, block_days=10)
    assert stats["lift"] == 0.0
    assert stats["n_days"] == 40
    assert stats["lift_se"] is None
    assert stats["n_blocks"] == 4


def test_fold_test_dates_invariant_to_factor_dict_order():
    """判别测试：fold test 日期不得依赖因子 dict 插入序（旧实现取首因子,是潜伏 bug）。

    构造覆盖异质的两因子（f_wide 覆盖全部日期,f_narrow 只覆盖前半）,两种插入序的
    combine_lgbm 输出必须一致——旧 next(iter(...)) 实现下 narrow 在前会丢后半 test 行。
    """
    import datetime as dt

    import polars as pl

    from factorzen.research.combination.cv import PurgedWalkForwardCV
    from factorzen.research.combination.models import combine_lgbm

    days = []
    d = dt.date(2024, 1, 2)
    while len(days) < 90:
        if d.weekday() < 5:
            days.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(12)]

    def _panel(dates):
        return pl.DataFrame({
            "trade_date": [dd for dd in dates for _ in codes],
            "ts_code": codes * len(dates),
            "factor_value": [
                (hash((dd, c)) % 1000) / 1000.0 for dd in dates for c in codes
            ],
        })

    f_wide = _panel(days)
    f_narrow = _panel(days[: len(days) // 2])
    ret = pl.DataFrame({
        "trade_date": [dd for dd in days for _ in codes],
        "ts_code": codes * len(days),
        "ret": [(hash((c, dd)) % 200 - 100) / 5000.0 for dd in days for c in codes],
    })
    cv = PurgedWalkForwardCV(train_days=30, test_days=10, purge_days=2,
                             embargo_days=0, expanding=False)
    out_a = combine_lgbm({"w": f_wide, "n": f_narrow}, ret, cv, seed=7)
    out_b = combine_lgbm({"n": f_narrow, "w": f_wide}, ret, cv, seed=7)
    a = out_a.sort(["trade_date", "ts_code"])
    b = out_b.sort(["trade_date", "ts_code"])
    # 判别点=行集(fold test 日期):旧「取首因子」实现 narrow 在前会丢后半 test 行。
    # 不断言预测值:lgbm 对特征列序有平手裁决差异,值级不变性不成立(已知特性)。
    assert a.height == b.height and a.height > 0
    assert (a["trade_date"] == b["trade_date"]).all()
    assert (a["ts_code"] == b["ts_code"]).all()
    wide_only_dates = set(d for d in a["trade_date"].unique().to_list())
    # 后半日期(narrow 无覆盖)必须出现在 test 行里——旧实现会整段缺失
    assert any(d >= "20240401" for d in wide_only_dates)
