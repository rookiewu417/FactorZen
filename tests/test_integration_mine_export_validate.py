"""集成测试：M1 挖掘（run_session）→ export-alpha → M4 组合构建（run_portfolio）产物契约贯通。

真实调用链路（不手写 candidates.csv / alpha 截面去顶替真实产物）：

    discovery.mining_session.run_session()          → 真实落盘 candidates.csv + session_dir
    discovery.export.read_candidate_expression()    → 真实解析 candidates.csv 取表达式（1-based rank）
    discovery.export.export_alpha_cross_section()   → 真实计算某日截面 α，落 [ts_code, alpha] parquet
    pipelines.portfolio_build.run_portfolio()        → 真实消费该 α 截面（对齐 codes 顺序）

项目文档已记录过"mine search 产出的 candidates 不是 alpha 截面，需要 export-alpha 转换"
这个已知的集成注意点；本测试验证这一步真实转换出的产物，字段名/shape 是否真的符合
下游 run_portfolio() 的 alpha 参数期望（而不是靠人工检查代码"看起来应该对得上"）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.discovery.export import export_alpha_cross_section, read_candidate_expression
from factorzen.discovery.mining_session import run_session
from factorzen.pipelines.portfolio_build import run_portfolio
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModelResult

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
