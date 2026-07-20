"""
test_export_integration.py：Merged discovery tests: test_export_integration.py
test_session.py：Merged discovery tests: test_session.py
"""

from __future__ import annotations

import json
import math
from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    date,
    timedelta,
)
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.discovery.export import (
    export_alpha_cross_section,
    read_candidate_expression,
)
from factorzen.discovery.expression import (
    Feature,
    compile_expr,
    feature_names,
    parse_expr,
)
from factorzen.discovery.mining_session import run_session
from factorzen.discovery.operators import LEAF_FEATURES
from factorzen.discovery.scoring import ic_overfit_report
from factorzen.discovery.search.genetic import GeneticSearcher
from factorzen.discovery.search.random_search import (
    RandomSearcher,
    random_expression,
)
from factorzen.markets.crypto.mining import (
    build_crypto_daily,
    validate_crypto_expression,
)
from factorzen.markets.crypto.profile import build_crypto_profile
from factorzen.pipelines.portfolio_build import run_portfolio
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModelResult
from tests.markets.test_crypto_mining_lake import FakeCCXTBulk

# ==== 来自 test_export_integration.py ====
# ==== 来自 test_mine_export_alpha.py ====
# tests/test_mine_export_alpha.py

def _make_daily_lf(n_stocks=8, n_days=60, seed=42) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        price = 10.0
        for day in days:
            price = float(max(price * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": price,
                         "open": price, "high": price, "low": price, "pre_close": price,
                         "close_adj": price, "open_adj": price, "high_adj": price,
                         "low_adj": price,
                         "amount": float(abs(rng.standard_normal()) * 1e7 + 1e6),
                         "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows).lazy()

@dataclass
class MockCtx:
    start: str = "20240301"
    end: str = "20240301"
    required_data: list = field(default_factory=lambda: ["daily", "daily_basic"])
    lookback_days: int = 40
    universe: list | None = None
    snapshot_mode: str = "daily"
    _daily: pl.LazyFrame | None = None
    _basic: pl.LazyFrame | None = None

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return self._basic if self._basic is not None else pl.DataFrame(
            {"trade_date": [], "ts_code": []}).lazy()

def test_export_alpha_writes_two_column_parquet(tmp_path):
    """挖掘候选 + 指定日期 → 落 [ts_code, alpha] 两列截面 parquet：非空且值有限。"""
    from factorzen.discovery.export import export_alpha_cross_section

    ctx = MockCtx(_daily=_make_daily_lf())
    out = tmp_path / "alpha.parquet"
    p = export_alpha_cross_section("pct_change(close, 20)", ctx, "20240301", str(out))

    assert p.exists()
    df = pl.read_parquet(p)
    # 恰有 ts_code + alpha 两列
    assert df.columns == ["ts_code", "alpha"]
    assert df.height > 0
    # 值全部有限且非空
    assert df["alpha"].is_finite().all()
    assert df["alpha"].null_count() == 0
    # 单日截面：每只股票至多一行
    assert df["ts_code"].n_unique() == df.height

def test_read_candidate_expression_by_rank(tmp_path):
    """candidates.csv 按 rank 取表达式。"""
    from factorzen.discovery.export import read_candidate_expression

    # 用 polars 写，忠实复现 mining_session 产出的 candidates.csv（含逗号的表达式会被加引号）
    pl.DataFrame({
        "rank": [1, 2],
        "n_trials": [100, 100],
        "expression": ["rank(close)", "ts_mean(close, 5)"],
    }).write_csv(tmp_path / "candidates.csv")
    assert read_candidate_expression(str(tmp_path), 1) == "rank(close)"
    assert read_candidate_expression(str(tmp_path), 2) == "ts_mean(close, 5)"

# ==== 来自 test_discovery_export.py ====
def _write_candidates_csv(tmp_path, *, with_passed=True):
    import polars as pl
    rows = [
        {"rank": 1, "expression": "close", "passed": True},
        {"rank": 2, "expression": "neg(close)", "passed": False},
    ]
    if not with_passed:
        rows = [{k: v for k, v in r.items() if k != "passed"} for r in rows]
    d = tmp_path / "sess"
    d.mkdir()
    pl.DataFrame(rows).write_csv(d / "candidates.csv")
    return str(d)

def test_read_candidate_require_passed_rejects_unpassed(tmp_path: Path):
    """R1：require_passed=True 时，请求未过护栏的 rank 报错并提示 --all；过的正常返回。"""
    import pytest

    from factorzen.discovery.export import read_candidate_expression
    sess = _write_candidates_csv(tmp_path)
    assert read_candidate_expression(sess, rank=1, require_passed=True) == "close"       # 过
    with pytest.raises(ValueError, match="--all"):
        read_candidate_expression(sess, rank=2, require_passed=True)                     # 未过 → 拒
    assert read_candidate_expression(sess, rank=2, require_passed=False) == "neg(close)"  # 逃生口

def test_read_candidate_backward_compat_no_passed_column(tmp_path: Path):
    """老 session 无 passed 列时 require_passed 不生效（不破坏向后兼容）。"""
    from factorzen.discovery.export import read_candidate_expression
    sess = _write_candidates_csv(tmp_path, with_passed=False)
    assert read_candidate_expression(sess, rank=2, require_passed=True) == "neg(close)"

# render_factor_file / export_candidate / exported/*.py 桥已废除（Batch 2）；
# lookback 契约见 test_export_lookback.py → lookback_for_expression。

# ==== 来自 test_integration_mine_export_validate.py ====
_N_STOCKS = 30

def _stock_codes(n_stocks: int = _N_STOCKS) -> list[str]:
    return [f"{i:06d}.SH" for i in range(n_stocks)]

def _mining_daily(n_stocks: int = _N_STOCKS, n_days: int = 150, seed: int = 42) -> pl.DataFrame:
    """价量合成日线（不含基本面列），与 tests/test_discovery_session.py::_daily 同构：
    只用价格/成交量派生的 leaf 特征，保证 run_session() 内部评分不会因缺列而跳过候选。
    """
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in _stock_codes(n_stocks):
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "open_adj": p, "high_adj": p, "low_adj": p, "open": p, "high": p, "low": p,
                         "pre_close": p,
                         "amount": 1e7, "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)

@dataclass
class _MockFactorContext:
    """duck-type 满足 ExpressionFactor.compute(ctx) 的最小契约：daily/daily_basic/start。

    真实 CLI 路径（_cmd_mine_export_alpha）用的是打了 Tushare 数据源的 FactorDataContext；
    离线测试用同样字段形状的合成数据替代数据源本身（不是替代 mine→export 这条产物链路）。
    daily_basic 留空：run_session() 用的挖掘数据本就不含基本面列，凡引用了基本面 leaf
    （total_mv/pb/pe_ttm 等）的候选在 run_session() 内部评分阶段就已经因缺列跳过，
    不可能进入 candidates.csv，因此此处选中的候选一定只引用价量 leaf。
    """
    start: str
    _daily: pl.LazyFrame

    @property
    def daily(self) -> pl.LazyFrame:
        return self._daily

    @property
    def daily_basic(self) -> pl.LazyFrame:
        return pl.DataFrame({"trade_date": [], "ts_code": []}).lazy()

def _risk_result_for_universe(codes: list[str], seed: int = 3) -> RiskModelResult:
    n = len(codes)
    rng = np.random.default_rng(seed)
    names = ["size", "ind_A", "ind_B"]
    mat = rng.standard_normal((n, 3))
    half = n // 2
    mat[:, 1] = [1.0] * half + [0.0] * (n - half)
    mat[:, 2] = [0.0] * half + [1.0] * (n - half)
    F = rng.standard_normal((3, 3))
    F = F @ F.T * 0.01
    return RiskModelResult(
        factor_exposures=ExposureMatrix(codes, names, mat),
        factor_covariance=F, specific_risk=np.full(n, 0.1), factor_names=names,
    )

def _run_mining_session(tmp_path: Path) -> tuple[pl.DataFrame, dict]:
    daily = _mining_daily()
    res = run_session(daily, n_trials=25, top_k=3, seed=42, method="random",
                      out_dir=str(tmp_path / "sessions"))
    assert len(res["candidates"]) > 0, "合成数据下 run_session 应至少产出 1 个候选"
    return daily, res

def test_mine_session_to_export_alpha_cross_section(tmp_path: Path):
    """run_session() 真实产出 candidates.csv → read_candidate_expression() 真实解析
    → export_alpha_cross_section() 真实计算，产出恰好 [ts_code, alpha] 两列、非空、
    值全部有限的截面——这正是 run_portfolio() 的 alpha 参数所期望的格式。
    """
    daily, res = _run_mining_session(tmp_path)
    session_dir = res["session_dir"]
    assert (Path(session_dir) / "candidates.csv").exists()
    assert (Path(session_dir) / "manifest.json").exists()

    # 真实从落盘的 candidates.csv 解析表达式（而非直接读内存里的 res["candidates"]），
    # 这样才是"读真实产物文件"，而不是绕开文件 I/O 这一段契约。
    expr = read_candidate_expression(session_dir, rank=1)
    assert expr
    assert expr == res["candidates"][0]["expression"], "candidates.csv 落盘/读回的表达式应与内存一致"

    last_date = daily["trade_date"].max()
    assert last_date is not None
    date_str = last_date.strftime("%Y%m%d")
    ctx = _MockFactorContext(start=date_str, _daily=daily.lazy())

    out_path = tmp_path / "alpha.parquet"
    out = export_alpha_cross_section(expr, ctx, date_str, str(out_path))

    assert out.exists()
    alpha_df = pl.read_parquet(out)
    # cli/main.py::_cmd_portfolio_build 读 --alpha-file 时期望恰好 [ts_code, alpha] 两列。
    assert alpha_df.columns == ["ts_code", "alpha"]
    assert alpha_df.height > 0
    assert alpha_df["alpha"].null_count() == 0
    assert alpha_df["alpha"].is_finite().all()
    assert alpha_df["ts_code"].n_unique() == alpha_df.height, "单日截面每只股票至多一行"

def test_mine_export_alpha_consumed_by_portfolio_build(tmp_path: Path):
    """再往前一步：export-alpha 产出的截面真实喂给 run_portfolio()，不报错、正常落盘。

    对齐方式复刻 cli/main.py::_cmd_portfolio_build 的真实逻辑（ts_code→alpha 字典，
    codes 中缺失的股票 alpha 填 0），而不是绕开对齐直接手造一个形状凑巧对的 ndarray。
    """
    daily, res = _run_mining_session(tmp_path)
    session_dir = res["session_dir"]
    expr = read_candidate_expression(session_dir, rank=1)

    last_date = daily["trade_date"].max()
    assert last_date is not None
    date_str = last_date.strftime("%Y%m%d")
    ctx = _MockFactorContext(start=date_str, _daily=daily.lazy())
    out_path = tmp_path / "alpha.parquet"
    export_alpha_cross_section(expr, ctx, date_str, str(out_path))

    codes = _stock_codes()
    rr = _risk_result_for_universe(codes)

    # 与 cli/main.py::_cmd_portfolio_build 相同的对齐方式。
    adf = pl.read_parquet(out_path)
    amap = dict(zip(adf["ts_code"].to_list(), adf["alpha"].to_list(), strict=False))
    alpha = np.array([float(amap.get(c, 0.0)) for c in codes])
    assert np.isfinite(alpha).all()

    half = len(codes) // 2
    build_res = run_portfolio(
        alpha, rr, codes=codes,
        stock_returns=np.zeros(len(codes)),
        sectors=(["A"] * half) + (["B"] * (len(codes) - half)),
        factor_returns_latest={},
        risk_aversion=1.0, w_max=0.1,
        out_dir=str(tmp_path / "portfolio_from_mined"), run_id="mined_alpha",
    )

    run_dir = Path(build_res["run_dir"])
    for f in ["weights.parquet", "attribution.csv", "risk_summary.csv", "manifest.json"]:
        assert (run_dir / f).exists(), f"missing: {f}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    # w_max=0.1 × 30 只股票 = 3.0 >= 1，长仓 + Σw=1 应始终可行，不因 α 数值来源
    # （挖掘表达式的原始量纲，未做 zscore 归一化）而变得不可行——可行性只取决于约束。
    assert manifest["status"] in ("optimal", "optimal_inaccurate"), (
        f"真实挖掘出的 α 截面喂给 run_portfolio 应可行求解，实际 status={manifest['status']!r}"
    )
    assert manifest["n_holdings"] == build_res["n_holdings"]

    weights_df = pl.read_parquet(run_dir / "weights.parquet")
    assert weights_df.height == len(codes)
    # optimal_inaccurate 时求解器可能有轻微数值残差，容差放宽到 1e-2；核心诉求是
    # 排除"其实是 infeasible 兜底全零仓位"（那种情况下 sum 应为 0，而不是接近 1）。
    assert abs(float(weights_df["target_weight"].sum()) - 1.0) < 1e-2, "Σw 应接近 1（budget 约束）"

# ==== 来自 test_markets_crypto_validation.py ====
def _profile_syms():
    fake = FakeCCXTBulk()
    return build_crypto_profile(client=fake), fake.symbols

def test_ic_overfit_report_market_agnostic():
    """ic_overfit_report 吃 factor_df+daily(任意市场)，产出 IC/IR/DSR/CI。"""
    profile, syms = _profile_syms()
    daily = build_crypto_daily(profile.provider, syms, "20240101", "20240224")
    daily = profile.factors.derived_columns(daily)
    factor_df = daily.select(["trade_date", "ts_code"]).with_columns(
        daily["ret_1d"].alias("factor_value")
    )
    rep = ic_overfit_report(factor_df, daily)
    assert set(rep) >= {"ic_mean", "ir", "dsr_p", "ci_lo", "ci_hi", "n"}
    assert rep["n"] > 0
    assert all(math.isfinite(rep[k]) for k in ("ic_mean", "ir", "dsr_p"))
    assert rep["ci_lo"] <= rep["ci_hi"]

def test_validate_crypto_expression():
    """crypto 单表达式防过拟合验证：bootstrap CI + DSR 在 crypto 上跑通。"""
    profile, syms = _profile_syms()
    rep = validate_crypto_expression(
        profile, "ts_mean(ret_1d, 5)", syms, "20240101", "20240224"
    )
    assert rep["n"] > 0
    assert math.isfinite(rep["ir"])
    assert math.isfinite(rep["dsr_p"])
    assert 0.0 <= rep["dsr_p"] <= 1.0


# ==== 来自 test_session.py ====
# ==== 来自 test_discovery_session.py ====
def _daily(seed=3, n_stocks=40, n_days=120):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 2)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows = []
    for s in [f"{i:06d}.SH" for i in range(n_stocks)]:
        p = 10.0
        for day in days:
            p = float(max(p * (1 + rng.standard_normal() * 0.02), 0.1))
            rows.append({"trade_date": day, "ts_code": s, "close": p, "close_adj": p,
                         "open_adj": p, "high_adj": p, "low_adj": p, "open": p, "high": p, "low": p,
                         "pre_close": p,
                         "amount": 1e7, "vol": float(abs(rng.standard_normal()) * 1e5 + 1e4)})
    return pl.DataFrame(rows)

def _mk_factor(vals_per_stock, n_days=5):
    """构造 [trade_date, ts_code, factor_value]：每只股票取 vals_per_stock[i]，每日相同。"""
    rows = []
    for d in range(n_days):
        dt = date(2024, 1, 2) + timedelta(days=d)
        for i, v in enumerate(vals_per_stock):
            rows.append({"trade_date": dt, "ts_code": f"{i:06d}.SH", "factor_value": float(v)})
    return pl.DataFrame(rows)

def test_rank_fingerprint_merges_monotone_equivalents():
    """R5：截面 rank 指纹对单调(同向)变换一致 → 数学等价簇同指纹；反向/不同向不同指纹。"""
    from factorzen.discovery.mining_session import _rank_fingerprint
    base = [((i * 37) % 40) + 0.5 for i in range(40)]  # 40 个互异值
    f_inc = _mk_factor(base)
    f_inc2 = _mk_factor([x * 3.0 + 7.0 for x in base])   # 单调递增变换 → rank 序不变
    f_dec = _mk_factor([-x for x in base])               # neg → 递减
    f_dec2 = _mk_factor([100.0 - x for x in base])       # 2-x 型 → 同样递减，与 f_dec 同序
    f_other = _mk_factor([((i * 11) % 40) + 0.5 for i in range(40)])  # 不同排序
    assert _rank_fingerprint(f_inc) == _rank_fingerprint(f_inc2)      # 递增簇合并
    assert _rank_fingerprint(f_dec) == _rank_fingerprint(f_dec2)      # 递减簇合并
    assert _rank_fingerprint(f_inc) != _rank_fingerprint(f_dec)       # 方向不同 → 区分
    assert _rank_fingerprint(f_inc) != _rank_fingerprint(f_other)     # 不同因子 → 区分

def test_cross_section_variability_flags_degenerate():
    """R7：近常数因子截面变异占比≈0（被过滤）；有变异因子≈1（保留）。"""
    from factorzen.discovery.mining_session import _cross_section_variability
    const = _mk_factor([1.0] * 40)
    varying = _mk_factor([((i * 37) % 40) + 0.5 for i in range(40)])
    assert _cross_section_variability(const) < 0.5
    assert _cross_section_variability(varying) > 0.5

def test_oos_adjusted_fitness_demotes_valid_reversal():
    """R6：valid t-stat 与 train 反号时按 |valid_tstat| 扣分（同尺度），把 train 高/valid 反号降权。"""
    from factorzen.discovery.mining_session import _oos_adjusted_fitness
    assert _oos_adjusted_fitness(3.0, 3.0, 1.5) == 3.0     # 同号一致 → 不调整
    assert _oos_adjusted_fitness(3.0, 3.0, -2.0) == 1.0    # 反号 → 扣 |valid_tstat|
    assert _oos_adjusted_fitness(3.0, 3.0, 0.0) == 3.0     # valid 样本不足(tstat=0) → 不调整
    # 反号候选(train fitness 3.0→1.0) 应排到一致候选(2.0)之后
    assert _oos_adjusted_fitness(3.0, 3.0, -2.0) < _oos_adjusted_fitness(2.0, 2.0, 1.0)

def test_run_session_respects_config_knobs(tmp_path):
    """cfg：去相关阈值不再写死——decorr_threshold=0.0 时 mc<0.0 恒 False → top-K 一个都选不进。"""
    from factorzen.discovery.mining_session import run_session
    daily = _daily(n_stocks=40, n_days=150)
    base = dict(n_trials=30, top_k=5, seed=42, method="random", holdout_ratio=0.2)
    r_default = run_session(daily, out_dir=str(tmp_path / "d"), **base)
    r_strict = run_session(daily, decorr_threshold=0.0, out_dir=str(tmp_path / "s"), **base)
    assert len(r_default["candidates"]) > 0        # 默认阈值 0.7 → 有候选
    assert len(r_strict["candidates"]) == 0        # 阈值 0.0 → 全被去相关门槛挡下

def test_guard_passed_respects_dsr_alpha():
    """cfg：strict 口径下 DSR 阈值可配——收紧 dsr_alpha 让边界候选从 passed 变 not passed。
    （library 默认口径不含 DSR，dsr_alpha 不影响，故此测显式 gate="strict"。）"""
    from factorzen.discovery.mining_session import _guard_passed
    c = {"dsr_pvalue": 0.03, "holdout_ic": 0.05, "ic_ci_low": 0.02, "ic_train": 0.06}
    assert _guard_passed(c, dsr_alpha=0.05, gate="strict") is True     # 0.03 < 0.05 → 过
    assert _guard_passed(c, dsr_alpha=0.01, gate="strict") is False    # 0.03 ≥ 0.01 → 收紧后不过

def test_guard_passed_criteria():
    """护栏软标记（2026-07 因子库化, 默认 library）= 真(holdout 与 train 同号) + 有信号
    (|train_IC|≥0.015)；**不含 DSR**（显著性挪到组合层）。gate="strict" 回到 DSR 显著+同号。
    """
    from factorzen.discovery.mining_session import _guard_passed
    ok = {"dsr_pvalue": 0.01, "holdout_ic": 0.05, "ic_ci_low": 0.02, "ic_train": 0.06}
    assert _guard_passed(ok) is True                                   # 真+有信号(0.06≥0.015)
    assert _guard_passed({**ok, "ic_train": 0.006}) is False           # |IC| 太弱=纯噪声
    assert _guard_passed({**ok, "holdout_ic": -0.05}) is False         # 反号=过拟合
    assert _guard_passed({**ok, "holdout_ic": float("nan")}) is False  # NaN 保守判否
    assert _guard_passed({"ic_train": 0.06}) is False                  # 缺 holdout 保守判否
    assert _guard_passed({**ok, "dsr_pvalue": 0.9}) is True            # library 不看 DSR
    # strict 口径仍按 DSR：0.2 ≥ 0.10 不过
    assert _guard_passed({**ok, "dsr_pvalue": 0.2}, gate="strict") is False

def test_session_writes_passed_flag(tmp_path: Path):
    """R1 集成：每个候选带 bool passed，candidates.csv 有 passed 列；passed=True 者确满足护栏。"""
    import polars as pl

    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(n_stocks=40, n_days=150), n_trials=30, top_k=5, seed=42,
                      method="random", holdout_ratio=0.2, out_dir=str(tmp_path))
    for c in res["candidates"]:
        assert isinstance(c["passed"], bool)
        if c["passed"]:  # 标记为过的候选，独立复核确满足因子库口径(真+有信号)
            assert abs(c["ic_train"]) >= 0.015                         # 有信号(非纯噪声)
            assert (c["holdout_ic"] > 0) == (c["ic_train"] > 0)        # holdout 同号(不崩)
    df = pl.read_csv(Path(res["session_dir"]) / "candidates.csv")
    assert "passed" in df.columns

def test_factor_values_eval_start_trims():
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.mining_session import _factor_values
    daily = _daily()
    dates = sorted(daily["trade_date"].unique().to_list())
    cutoff = dates[len(dates) // 2]
    es = cutoff.strftime("%Y%m%d")
    out = _factor_values(parse_expr("close"), daily, eval_start=es)
    assert out["trade_date"].min() >= cutoff

def test_session_runs_and_writes_artifacts(tmp_path: Path):
    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(), n_trials=20, top_k=5, seed=42,
                      method="random", out_dir=str(tmp_path))
    session_dir = Path(res["session_dir"])
    assert (session_dir / "candidates.csv").exists()
    assert (session_dir / "manifest.json").exists()
    assert 0 < len(res["candidates"]) <= 5
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["cli_n_trials"] == 20
    assert manifest["seed"] == 42
    for c in res["candidates"]:
        assert c["max_corr"] < 0.7  # 贪心去相关保证：top-K 互不近重复，max_corr 是真实测量

def test_session_reproducible_same_seed(tmp_path: Path):
    from factorzen.discovery.mining_session import run_session
    a = run_session(_daily(), n_trials=20, top_k=5, seed=7, out_dir=str(tmp_path / "a"))
    b = run_session(_daily(), n_trials=20, top_k=5, seed=7, out_dir=str(tmp_path / "b"))
    expr_a = [c["expression"] for c in a["candidates"]]
    expr_b = [c["expression"] for c in b["candidates"]]
    assert expr_a == expr_b

def test_session_has_guard_metrics_and_holdout_isolated(tmp_path):
    from factorzen.discovery.mining_session import run_session
    res = run_session(_daily(n_stocks=40, n_days=150), n_trials=30, top_k=5, seed=42,
                      method="random", holdout_ratio=0.2, out_dir=str(tmp_path))
    assert 0 < len(res["candidates"]) <= 5
    for c in res["candidates"]:
        # 护栏指标齐全
        for key in ("n_trials", "pbo", "holdout_ic", "dsr_pvalue", "ic_ci_low"):
            assert key in c
        assert c["n_trials"] > 0          # 真实评估数（非 CLI n_trials 摆设）
        assert 0.0 <= c["pbo"] <= 1.0 or c["pbo"] != c["pbo"]  # [0,1] 或 nan
    # holdout 永久隔离：挖掘期数据严格早于 holdout（删除 daily=mining_df 会让此断言失败）
    assert res["mining_end"] < res["holdout_start"]

def test_dsr_n_trials_same_source_as_sharpe_variance(tmp_path, monkeypatch):
    """R8：DSR 的 n_trials 必须与 sharpe_variance 同源（都来自存活集 scored），
    而非取被 height/n_train/退化/去重跳过者膨胀的 seen/eval_cache 计数。"""
    import factorzen.discovery.mining_session as ms
    captured: list = []
    real = ms.deflated_pvalue

    def spy(sharpe, basis, n_obs):
        captured.append(basis)
        return real(sharpe, basis, n_obs)

    monkeypatch.setattr(ms, "deflated_pvalue", spy)
    res = ms.run_session(_daily(n_stocks=40, n_days=150), n_trials=40, top_k=5, seed=42,
                         method="random", holdout_ratio=0.2, out_dir=str(tmp_path))
    assert captured, "deflated_pvalue 应至少被调用一次"
    # 抽出 DeflationBasis 后「同源」成了结构性保证：一个对象同时携带 n_trials 与
    # sharpe_variance，由 from_ir_pool 一次算出，不可能各取各的池。
    assert len({id(b) for b in captured}) == 1          # 所有候选共用同一个 basis 对象
    basis = captured[0]
    assert basis.n_trials == res["n_scored"]            # N == 存活集大小
    assert basis.sharpe_variance == res["sharpe_variance"]
    assert res["n_scored"] >= len(res["candidates"])    # 存活集 ⊇ top-K
    assert res["n_scored"] > 0

def test_deflated_sharpe_train_n_vs_mining_window_n_flips_significance():
    """数值对照：DSR 显著性检验必须用候选自己的 train 段样本数(n_train)，不能用
    mining 全段交易日数(n_obs_mining)——后者系统性偏大（约 1.43x：500/350），且放大
    方向是让候选看起来比实际更显著（危险方向）。固定 sharpe/n_trials/sharpe_variance，
    分别用「正确的 n_train=350」和「错误的 n_obs_mining=500」算 DSR，断言两者的
    显著性结论（p<0.05 与否）相反。"""
    from factorzen.validation.deflated_sharpe import deflated_sharpe
    sharpe, n_trials, sharpe_var = 0.14, 30, 0.001
    _dsr_correct, p_correct = deflated_sharpe(sharpe, n_trials, 350, sharpe_variance=sharpe_var)
    _dsr_wrong, p_wrong = deflated_sharpe(sharpe, n_trials, 500, sharpe_variance=sharpe_var)
    assert p_correct > 0.05  # 正确：用 train 段真实样本数 → 不显著
    assert p_wrong < 0.05  # 错误：用放大的 mining 全段样本数 → 假显著（危险方向）

def test_session_dsr_uses_candidate_own_train_n(tmp_path, monkeypatch):
    """集成测试：run_session 内对每个候选调用 deflated_pvalue() 时，传入的样本数
    必须是该候选自己在 train 段的真实样本数(c["n_train"])，而不是退化为所有候选共用
    的全局 mining 段交易日数。用 monkeypatch 拦截 deflated_pvalue 的调用参数核对。"""
    import factorzen.discovery.mining_session as ms_mod
    from factorzen.validation.holdout import split_holdout

    daily = _daily(n_stocks=40, n_days=150)
    # 独立重算旧 bug 会传入的「mining 段全局交易日数」，不依赖 mining_session 内部实现
    sorted_daily = daily.sort(["ts_code", "trade_date"])
    mining_df, _holdout_df, _holdout_start = split_holdout(sorted_daily, holdout_ratio=0.2)
    legacy_n_obs_mining = mining_df["trade_date"].n_unique()

    calls: list[int] = []
    real_dsr = ms_mod.deflated_pvalue

    def _spy_deflated_pvalue(sharpe, basis, n_obs):
        calls.append(n_obs)
        return real_dsr(sharpe, basis, n_obs)

    monkeypatch.setattr(ms_mod, "deflated_pvalue", _spy_deflated_pvalue)

    res = ms_mod.run_session(daily, n_trials=30, top_k=5, seed=42,
                             method="random", holdout_ratio=0.2, out_dir=str(tmp_path))

    assert calls, "deflated_pvalue 应至少被调用一次"
    assert len(calls) == len(res["candidates"])
    for n_obs_used, c in zip(calls, res["candidates"], strict=True):
        assert "n_train" in c
        assert n_obs_used == c["n_train"]          # 用的是候选自己的 train 段样本数
        assert n_obs_used < legacy_n_obs_mining     # 不是放大过的 mining 全段样本数（旧 bug）

def test_genetic_dsr_n_spans_generations_not_just_survivors(tmp_path):
    """F6：genetic 的 DSR N 应反映跨代真实评估过的唯一表达式数（eval_cache），而非
    仅最终代存活集 len(scored)≈pop_size。elitism 使最终代最优即全程 argmax，选择实际
    发生在整个搜索上；N 低估会系统性放松 DSR（passed 偏松，危险方向）。"""
    from factorzen.discovery.mining_session import run_session

    res = run_session(_daily(n_stocks=40, n_days=150), n_trials=120, top_k=5, seed=42,
                      method="genetic", holdout_ratio=0.2, out_dir=str(tmp_path))
    assert res["n_scored"] > 0
    assert res["n_trials"] > res["n_scored"], (
        f"genetic 的 DSR N({res['n_trials']}) 应大于最终代存活集 n_scored({res['n_scored']})"
        "——反映跨代评估广度，而非只数最终代 pop_size"
    )

def test_random_dsr_n_still_equals_scored(tmp_path, monkeypatch):
    """回归：random 路径 N 仍等于存活集大小（与 sharpe_variance 同源，R8 不变）。"""
    import factorzen.discovery.mining_session as ms
    captured: list[int] = []
    real = ms.deflated_pvalue
    monkeypatch.setattr(ms, "deflated_pvalue",
                        lambda s, b, o: (captured.append(b.n_trials), real(s, b, o))[1])
    res = ms.run_session(_daily(n_stocks=40, n_days=150), n_trials=40, top_k=5, seed=42,
                         method="random", holdout_ratio=0.2, out_dir=str(tmp_path))
    assert captured and len(set(captured)) == 1
    assert captured[0] == res["n_scored"]

def test_run_session_and_agent_agree_reject_underwarmed(monkeypatch, tmp_path):
    """双路径一致：预热不足的表达式在 M1(run_session) 与 agent(evaluate_expressions) 都被拒。

    消除双路径漂移——M1 此前对超预热表达式不拒绝，让首段截断窗口噪声进 train IC。
    两侧共用 warmup_bars vs required_lookback 判据（双路径登记簿：新增第二路径必加一致性测试）。

    M1 端（端到端，n_scored 判别）：注入 rank(close)(rl=0) 与 ts_mean(close,20)(rl=20)，
    短预热帧（eval_start 前仅 3 交易日）下后者被门拒 → n_scored 比足预热（前 30 交易日）少 1。
    agent 端：同表达式、同短预热，evaluate_expressions 记 ic_train=None + 预热不足。
    """
    import datetime as _dt

    from factorzen.discovery.evaluation import evaluate_expressions
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.mining_session import run_session
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import split_holdout

    daily = _daily(n_days=120)
    dates = sorted(set(daily["trade_date"].to_list()))
    short_s = dates[3].strftime("%Y%m%d")   # 前 3 交易日预热 → ts_mean(,20) 欠预热
    full_s = dates[30].strftime("%Y%m%d")   # 前 30 交易日预热 → 两者都评估

    both = [parse_expr("ts_mean(close, 20)"), parse_expr("rank(close)")]
    cnt = {"i": 0}

    def _fixed(*a, **k):
        node = both[cnt["i"] % 2]
        cnt["i"] += 1
        return node

    monkeypatch.setattr(
        "factorzen.discovery.search.random_search.random_expression", _fixed)

    def _n_scored(eval_start_s):
        cnt["i"] = 0
        return run_session(daily, n_trials=4, top_k=5, seed=1, method="random",
                           eval_start=eval_start_s, out_dir=str(tmp_path / eval_start_s))["n_scored"]

    n_short = _n_scored(short_s)
    n_full = _n_scored(full_s)
    assert n_full == n_short + 1, f"M1 门应恰拒 1 个超预热表达式: short={n_short} full={n_full}"

    # agent 端一致：同表达式、同短预热帧 → 也被拒
    mining_df, _, _ = split_holdout(daily.sort(["ts_code", "trade_date"]), holdout_ratio=0.2)
    bundle = DataBundle.build(mining_df)
    train_end = _dt.datetime.strptime(bundle.train_end, "%Y%m%d").date()
    ares = evaluate_expressions(["ts_mean(close, 20)"], daily, bundle,
                                eval_start=dates[3], eval_end=train_end)[0]
    assert ares["ic_train"] is None and "预热不足" in (ares["error"] or "")

# ==== 来自 test_discovery_crypto_session.py ====
def _synthetic_crypto_daily(n_sym: int = 40, n_days: int = 55, seed: int = 7) -> pl.DataFrame:
    # 截面样本数需 ≥ MIN_IC_SAMPLES(30) 否则 compute_rank_ic 跳过该日 → IC 序列空
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2024, 1, 1)
    for s in range(n_sym):
        code = f"SYM{s:02d}USDT"
        price = 100.0 + s
        for d in range(n_days):
            ret = rng.normal(0, 0.02)
            price = max(1.0, price * (1 + ret))
            vol = float(rng.uniform(50, 500))
            rows.append({
                "ts_code": code,
                "trade_date": start + timedelta(days=d),
                "open": price * (1 + rng.normal(0, 0.001)),
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "vol": vol,
                "amount": price * vol,
                "funding_rate": float(rng.normal(0.0001, 0.0002)),
                "open_interest": float(rng.uniform(1000, 5000)),
            })
    return pl.DataFrame(rows)

def test_run_session_crypto_profile(tmp_path):
    """crypto daily(无 close_adj) + crypto profile → 挖出带 holdout/DSR/PBO 的 candidates。"""
    daily = _synthetic_crypto_daily()
    profile = build_crypto_profile()
    result = run_session(
        daily,
        n_trials=40,
        top_k=5,
        seed=1,
        method="random",
        out_dir=str(tmp_path),
        profile=profile,
    )
    assert result["candidates"], "crypto 挖掘应产出至少一个候选"
    sess = tmp_path / "session_1_random"
    assert (sess / "candidates.csv").exists()
    cand = pl.read_csv(sess / "candidates.csv")
    # OOS + 防过拟合列齐全
    for col in ["holdout_ic", "dsr_pvalue", "pbo", "ic_train", "ir_train"]:
        assert col in cand.columns
    # 候选表达式只用 crypto 叶子(含 funding_rate/open_interest 可能出现)
    from factorzen.discovery.expression import feature_names, parse_expr
    crypto_leaves = set(profile.factors.leaf_features().keys())
    for expr in cand["expression"].to_list():
        assert feature_names(parse_expr(expr, crypto_leaves)) <= crypto_leaves

def test_run_session_ashare_default_unchanged(tmp_path):
    """A 股默认路径(profile=None)：真产候选、只用 A 股叶子、护栏列齐全。

    旧版只断言 `"candidates" in result`（字典必然有这个 key）与 CSV 文件存在，
    并跑在 **10 只股票**上——`_MIN_CROSS_SAMPLES=30` 把每个截面整天丢弃，IC 恒空。
    这条守卫是 CLAUDE.md 的「A 股零回归是底线」，判别力必须真实存在。
    """
    rng = np.random.default_rng(3)
    rows = []
    start_d = date(2024, 1, 1)
    for s_i in range(40):        # ≥ _MIN_CROSS_SAMPLES(=30)，否则 IC 序列为空
        code = f"{600000 + s_i:06d}.SH"
        price = 10.0 + s_i
        for d in range(120):
            prev_price = price
            price = max(1.0, price * (1 + rng.normal(0, 0.02)))
            vol = float(rng.uniform(1e5, 1e6))
            rows.append({
                "ts_code": code, "trade_date": start_d + timedelta(days=d),
                "pre_close": prev_price,
                "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
                "open_adj": price, "high_adj": price * 1.01, "low_adj": price * 0.99,
                "close_adj": price, "vol": vol, "amount": price * vol,
                "total_mv": price * 1e6, "circ_mv": price * 8e5, "pb": 1.5,
                "pe_ttm": 15.0, "ps_ttm": 3.0, "dv_ttm": 2.0,
            })
    daily = pl.DataFrame(rows)
    from factorzen.discovery.operators import BASIC_FEATURES

    daily = daily.with_columns([
        pl.lit(1.0).alias(c) for c in sorted(BASIC_FEATURES) if c not in daily.columns
    ])
    result = run_session(daily, n_trials=30, top_k=3, seed=2, out_dir=str(tmp_path))

    # 与 crypto profile 那条测试对称：断言**非空**，而非 `"candidates" in result`
    # ——后者是字典必然有的 key，恒真。
    assert result["candidates"], "A 股默认路径应产出至少一个候选"

    sess = tmp_path / "session_2_random"
    assert (sess / "candidates.csv").exists()
    cand = pl.read_csv(sess / "candidates.csv")
    assert cand.height > 0, "A 股默认路径应产出候选（IC 全空时这里会是 0 行）"

    # 护栏列齐全（与 crypto profile 那条测试对称）
    for col in ["holdout_ic", "dsr_pvalue", "pbo", "ic_train", "ir_train", "passed"]:
        assert col in cand.columns

    # IC 真的被算出来了——全 NaN 说明截面被整天丢弃
    ic = cand["ic_train"].to_list()
    assert any(v == v and v != 0.0 for v in ic), f"ic_train 全为 nan/0：{ic}"

    # 表达式只用 A 股叶子（profile=None 不该混入 crypto 叶子）
    from factorzen.discovery.expression import feature_names, parse_expr
    from factorzen.discovery.operators import LEAF_FEATURES

    ashare_leaves = set(LEAF_FEATURES.keys())
    for expr in cand["expression"].to_list():
        assert feature_names(parse_expr(expr)) <= ashare_leaves
    crypto_only = {"funding_rate", "open_interest"}
    assert not (set().union(*(feature_names(parse_expr(e))
                              for e in cand["expression"].to_list())) & crypto_only)

def test_ashare_default_derives_ret_1d_from_close_adj():
    """docstring 承诺的「仍用 close_adj 派生」必须真的被验证。

    造一个 close 与 close_adj 显著不同的帧（模拟除权）：`ret_1d` 必须由 close_adj 算出。
    旧测试通篇没碰过这件事。
    """
    from factorzen.discovery.derived import add_derived_columns

    rows = []
    for d in range(4):
        rows.append({
            "ts_code": "600000.SH", "trade_date": date(2024, 1, 1) + timedelta(days=d),
            "pre_close": 10.0, "open": 10.0, "high": 10.1, "low": 9.9,
            "close": 100.0 * (d + 1),          # 未复权价：乱跳
            "close_adj": 10.0 * (1.10 ** d),   # 复权价：每日 +10%
            "open_adj": 10.0, "high_adj": 10.1, "low_adj": 9.9,
            "vol": 1e5, "amount": 1e6,
        })
    out = add_derived_columns(pl.DataFrame(rows))
    ret = out["ret_1d"].to_list()

    assert ret[0] is None
    for v in ret[1:]:
        assert v == pytest.approx(0.10, abs=1e-9), (
            f"ret_1d={v}，应为 close_adj 的 10% 日涨幅；若由 close 派生会得到 1.0/0.5/…"
        )

# ==== 来自 test_discovery_leaf_injection.py ====
_CRYPTO_LEAF_MAP = {
    "close": "close", "vol": "vol", "funding_rate": "funding_rate",
    "open_interest": "open_interest",
}

# ── T1: expression 注入 ───────────────────────────────────────────────────────
def test_compile_default_ashare_unchanged():
    """默认走 A 股 LEAF_FEATURES：close→close_adj。"""
    expr = compile_expr(Feature("close"))
    df = pl.DataFrame({"close_adj": [1.0, 2.0], "close": [9.0, 9.0]})
    assert df.with_columns(expr.alias("x"))["x"].to_list() == [1.0, 2.0]

def test_compile_with_crypto_leaf_map():
    """注入 crypto leaf_map：close→close(无复权)，funding_rate 可编译。"""
    df = pl.DataFrame({"close": [1.0, 2.0], "funding_rate": [0.01, 0.02]})
    close_expr = compile_expr(Feature("close"), leaf_map=_CRYPTO_LEAF_MAP)
    assert df.with_columns(close_expr.alias("x"))["x"].to_list() == [1.0, 2.0]
    fr_expr = compile_expr(Feature("funding_rate"), leaf_map=_CRYPTO_LEAF_MAP)
    assert df.with_columns(fr_expr.alias("x"))["x"].to_list() == [0.01, 0.02]

def test_parse_with_crypto_leaves():
    """注入 crypto leaves：funding_rate 合法解析；默认 A 股拒绝。"""
    node = parse_expr("ts_mean(funding_rate, 3)", leaves=_CRYPTO_LEAF_MAP)
    assert "funding_rate" in feature_names(node)
    # 默认 A 股叶子集不含 funding_rate
    import pytest
    with pytest.raises(ValueError, match="未知叶子"):
        parse_expr("funding_rate")

# ── T2: 搜索注入 ──────────────────────────────────────────────────────────────
def test_random_expression_uses_injected_leaves():
    rng = np.random.default_rng(0)
    crypto_leaves = list(_CRYPTO_LEAF_MAP.keys())
    for _ in range(50):
        node = random_expression(rng, max_depth=3, leaves=crypto_leaves)
        assert feature_names(node) <= set(crypto_leaves)

def test_random_expression_default_ashare():
    rng = np.random.default_rng(0)
    for _ in range(30):
        node = random_expression(rng, max_depth=3)
        assert feature_names(node) <= set(LEAF_FEATURES.keys())

def test_random_searcher_leaves():
    rng = np.random.default_rng(1)
    s = RandomSearcher(rng, max_depth=3, leaves=list(_CRYPTO_LEAF_MAP.keys()))
    for _ in range(30):
        assert feature_names(s.propose()) <= set(_CRYPTO_LEAF_MAP.keys())

def test_genetic_searcher_leaves():
    rng = np.random.default_rng(2)
    gs = GeneticSearcher(rng, max_depth=3, leaves=list(_CRYPTO_LEAF_MAP.keys()))
    pop = gs.evolve(lambda n: -float(len(feature_names(n))), pop_size=12, generations=2)
    for node in pop:
        assert feature_names(node) <= set(_CRYPTO_LEAF_MAP.keys())

