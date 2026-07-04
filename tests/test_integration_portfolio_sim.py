"""集成测试：M4 组合构建（run_portfolio）→ M7 模拟交易（run_portfolio_simulation）贯通。

真实调用链路（不手写 manifest/weights 去顶替真实产物）：

    pipelines.portfolio_build.run_portfolio()        → 真实落盘 weights.parquet + manifest.json
    sim.engine.run_portfolio_simulation()             → 真实读取上面的 run_dir 消费

tests/test_portfolio_pipeline.py 与 tests/test_sim_engine.py 分别只单独测过两侧；
后者用的是手写 ``json.dumps({"run_id":..., "signal_date":...})`` 式的 manifest fixture，
从未验证过 run_portfolio() 真实产出的 manifest 字段（如 signal_date/status）能否被
run_portfolio_simulation() 正确消费——这正是 manifest 曾经漏过 signal_date 字段、
导致 sim 崩溃的那类回归的覆盖盲区。
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.pipelines.portfolio_build import run_portfolio
from factorzen.risk.exposures import ExposureMatrix
from factorzen.risk.model import RiskModelResult
from factorzen.sim.engine import run_portfolio_simulation


def _risk_result(n: int = 6, k: int = 3) -> RiskModelResult:
    """与 tests/test_portfolio_pipeline.py::_risk_result 同构（n=6 时行业各占一半）。"""
    rng = np.random.default_rng(2)
    names = ["size", "ind_A", "ind_B"]
    mat = rng.standard_normal((n, k))
    mat[:, 1] = [1, 1, 1, 0, 0, 0]
    mat[:, 2] = [0, 0, 0, 1, 1, 1]
    F = rng.standard_normal((k, k))
    F = F @ F.T * 0.01
    return RiskModelResult(
        factor_exposures=ExposureMatrix([f"{i:06d}.SZ" for i in range(n)], names, mat),
        factor_covariance=F, specific_risk=np.full(n, 0.1), factor_names=names)


def _fake_daily(codes: list[str], start: str = "20230101", end: str = "20230228") -> pl.DataFrame:
    """构造 mock 日线数据（不连接真实数据源），真正使用 start/end（覆盖整段回测窗口）。"""
    start_d = datetime.strptime(start, "%Y%m%d").date()
    end_d = datetime.strptime(end, "%Y%m%d").date()
    dates = pl.date_range(start_d, end_d, "1d", eager=True)
    rng = np.random.default_rng(0)
    rows = []
    for c in codes:
        for dt in dates:
            rows.append({
                "trade_date": dt, "ts_code": c,
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
                "pre_close": 10.0, "change": 0.0, "pct_chg": float(rng.normal(0, 1)),
                # amount 需远大于 BacktestConfig 默认 initial_capital(1e8) ×
                # max_participation_rate(0.05) 隐含的 ADV 门槛，否则 fast path 会按
                # 20 日 ADV 参与率把单日调仓幅度限制到远小于目标权重（真实的流动性约束
                # 行为，非 bug），导致建仓后长期停留在接近全现金的状态，测不出本测试要
                # 验证的"跳过 infeasible 信号"效果。
                "vol": 1e6, "amount": 1e10,
            })
    return pl.DataFrame(rows)


def _build_portfolio(rr: RiskModelResult, *, w_max: float, out_dir: str, run_id: str,
                     signal_date: str) -> dict:
    codes = rr.factor_exposures.codes
    alpha = np.array([0.1, 0.05, 0.02, 0.08, 0.03, 0.01])
    return run_portfolio(
        alpha, rr, codes=codes,
        stock_returns=np.array([0.03, 0.01, -0.02, 0.04, 0.0, 0.01]),
        sectors=["A", "A", "A", "B", "B", "B"],
        factor_returns_latest={"size": 0.02, "ind_A": 0.0, "ind_B": 0.0},
        risk_aversion=1.0, w_max=w_max, out_dir=out_dir, run_id=run_id,
        signal_date=signal_date,
    )


def test_portfolio_build_to_sim_happy_path(tmp_path: Path):
    """run_portfolio() 真实落盘的 run_dir 直接喂给 run_portfolio_simulation()：
    不抛异常，产出非空 nav 与完整 metrics 字段（含 total_cost/ann_turnover）。
    """
    rr = _risk_result()
    codes = rr.factor_exposures.codes
    build_res = _build_portfolio(
        rr, w_max=0.4, out_dir=str(tmp_path / "portfolios"), run_id="link_happy",
        signal_date="2023-01-05",
    )
    assert build_res["status"] == "optimal"

    daily = _fake_daily(codes, start="20230101", end="20230228")
    sim_res = run_portfolio_simulation(
        [build_res["run_dir"]], daily, out_dir=str(tmp_path / "sim"), run_id="sim_happy",
    )

    run_dir = Path(sim_res["run_dir"])
    for f in ["nav.parquet", "metrics.json", "manifest.json"]:
        assert (run_dir / f).exists(), f"missing: {f}"

    nav_df = pl.read_parquet(run_dir / "nav.parquet")
    assert not nav_df.is_empty(), "串联后 nav 不应为空"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    for k in ("ann_ret", "ann_vol", "sharpe", "max_dd", "avg_turnover", "total_cost", "ann_turnover"):
        assert k in metrics, f"metrics.json 缺少字段: {k}"

    for k in ("run_dir", "sharpe", "max_dd", "ann_ret"):
        assert k in sim_res, f"run_portfolio_simulation 返回值缺少字段: {k}"

    sim_manifest = json.loads((run_dir / "manifest.json").read_text())
    assert sim_manifest["n_signals"] == 1


def test_portfolio_build_infeasible_status_not_treated_as_valid_signal(tmp_path: Path):
    """一个 optimal + 一个 infeasible（w_max 过小导致约束不可行）的 run_dir 一起喂给 sim：
    infeasible 那次 run_portfolio() 会把全零持仓兜底写盘（status != optimal），
    sim 必须跳过它、不能当成"清仓"信号执行——否则有效仓位会被这个假信号错误抹平。
    """
    rr = _risk_result()  # n=6
    codes = rr.factor_exposures.codes

    valid = _build_portfolio(
        rr, w_max=0.4, out_dir=str(tmp_path / "portfolios"), run_id="valid1",
        signal_date="2023-01-05",
    )
    assert valid["status"] == "optimal"

    # w_max=0.1 时 6 只股票的最大可行仓位 = 0.6 < 1，Σw=1 无法满足 → infeasible。
    infeasible = _build_portfolio(
        rr, w_max=0.1, out_dir=str(tmp_path / "portfolios"), run_id="infeasible1",
        signal_date="2023-01-20",
    )
    assert infeasible["status"] != "optimal"
    # infeasible 的兜底权重必须全零（否则下面的行为验证会失去意义）。
    infeasible_w = pl.read_parquet(Path(infeasible["run_dir"]) / "weights.parquet")
    assert (infeasible_w["target_weight"] == 0.0).all()

    daily = _fake_daily(codes, start="20230101", end="20230228")
    sim_res = run_portfolio_simulation(
        [valid["run_dir"], infeasible["run_dir"]], daily,
        out_dir=str(tmp_path / "sim_skip"), run_id="sim_skip",
    )

    nav_df = pl.read_parquet(Path(sim_res["run_dir"]) / "nav.parquet")
    assert not nav_df.is_empty(), "有效信号（valid1）应正常执行，nav 不应整体为空"

    # 若 infeasible 的全零兜底权重被误当有效清仓信号执行，2023-01-21 起 cash_weight
    # 会跳升到接近 1.0（全部清仓为现金）；跳过后应继续持有 valid1 建立的仓位，
    # cash_weight 应保持在低位（组合优化约束 Σw=1，建仓后接近满仓）。
    after = nav_df.filter(pl.col("trade_date") >= date(2023, 1, 21))
    assert after.height > 0
    assert (after["cash_weight"] < 0.5).all(), (
        "infeasible run 的全零兜底持仓被当成了有效清仓信号执行（cash_weight 跳升到接近 1.0）"
    )


def test_portfolio_build_only_infeasible_run_raises(tmp_path: Path):
    """所有 run_dir 都是 infeasible 兜底时，sim 应彻底找不到有效信号并明确报错，
    而不是静默产出一份"看似正常但其实全零仓位"的净值曲线。
    """
    rr = _risk_result()
    codes = rr.factor_exposures.codes

    infeasible = _build_portfolio(
        rr, w_max=0.1, out_dir=str(tmp_path / "portfolios"), run_id="only_bad",
        signal_date="2023-01-05",
    )
    assert infeasible["status"] != "optimal"

    daily = _fake_daily(codes, start="20230101", end="20230228")
    with pytest.raises(ValueError, match="no portfolio weights"):
        run_portfolio_simulation(
            [infeasible["run_dir"]], daily,
            out_dir=str(tmp_path / "sim_only_bad"), run_id="only_bad_sim",
        )
