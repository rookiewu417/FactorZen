"""Merged discovery tests: test_export_integration.py

test_mine_export_alpha.py：export-alpha 写两列 parquet，并按 rank 读候选表达式
test_discovery_export.py：read_candidate_expression 的 require_passed 门禁与无 passed 列向后兼容
test_integration_mine_export_validate.py：集成：run_session → export-alpha → run_portfolio 产物契约贯通
test_markets_crypto_validation.py：crypto 上 bootstrap IC CI + DSR 防过拟合护栏有效（MC2）
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

from factorzen.discovery.export import (
    export_alpha_cross_section,
    read_candidate_expression,
)
from factorzen.discovery.mining_session import run_session
from factorzen.discovery.scoring import ic_overfit_report
from factorzen.markets.crypto.mining import (
    build_crypto_daily,
    validate_crypto_expression,
)
from factorzen.markets.crypto.profile import build_crypto_profile
from factorzen.pipelines.portfolio_build import run_portfolio
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModelResult
from tests.markets.test_crypto_mining import FakeCCXTBulk

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

