"""择时 vs 基线端到端离线 smoke：run_trend_timing_experiment 跑通两套实验，
且 risk-off 段策略确实降仓/空仓（而基线始终满仓），非恒真——独立构造场景，
用 SessionStore.load_state() 的持仓做跨函数验证，不依赖生成产物本身的字段。
"""
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from factorzen.execution.store import SessionStore
from factorzen.strategies.runner import run_trend_timing_experiment


def _dates(n: int, start: date = date(2026, 1, 5)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _idx(dates: list[date], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"trade_date": dates, "close": closes})


def _price(dates: list[date], codes: list[str], amount: float = 1e9) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "trade_date": d,
                "ts_code": c,
                "open": 10.0,
                "pre_close": 10.0,
                "close": 10.0,
                "vol": 1e8,
                "amount": amount,
            }
            for d in dates
            for c in codes
        ]
    )


def _fake_members(code: str, date_str: str) -> list[str]:  # 注入,避免网络
    return ["A.SZ", "B.SZ", "C.SZ"]


def _held(state: dict | None) -> dict:
    if not state:
        return {}
    pos = state.get("pos", state.get("positions", {}))
    return {
        c: p
        for c, p in pos.items()
        if (p.get("volume", 0) if isinstance(p, dict) else 0) > 0
    }


def test_strategy_vs_baseline_experiment(tmp_path: Path):
    ma_window = 3
    dates = _dates(10)
    codes = ["A.SZ", "B.SZ", "C.SZ"]
    price = _price(dates, codes)

    # 指数：前段单调上行(站上 MA, risk-on)，最后 3 天骤跌到远低于近期 MA(risk-off)。
    closes = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 5.0, 5.0, 5.0]
    idx = _idx(dates, closes)

    t_on = dates[4]  # close=18, MA(tail3<=t_on)=mean(14,16,18)=16 → 18>16 risk-on
    # t_off 放在 dates[8]（而非最后一天 dates[9]）：signal 次一交易日才执行（s<d，与 sim
    # 对齐），放最后一天则其清仓信号无执行日、永不生效。dates[8] 的清仓于 dates[9] 执行。
    t_off = dates[8]  # close=5, MA(tail3<=t_off)=mean(22,5,5)=10.67 → 5<10.67 risk-off
    rebalance_dates = [t_on, t_off]

    out = run_trend_timing_experiment(
        str(tmp_path / "exp"),
        idx,
        price,
        rebalance_dates,
        initial_cash=1_000_000.0,
        from_date=dates[0],
        to_date=dates[-1],
        members_fn=_fake_members,
        ma_window=ma_window,
        top_n=3,
    )

    assert set(out) == {"strategy", "baseline"}
    for label in ("strategy", "baseline"):
        session_dir = Path(out[label]["session_dir"])
        assert (session_dir / "nav.parquet").exists(), f"{label} 应产 nav.parquet"
        metrics = out[label]["metrics"]
        assert {"ann_ret", "sharpe", "max_dd"} <= set(metrics)

    # risk-off 段(t_off 当天清仓)：策略应清空持仓；基线(timing=False)始终满仓，
    # 不受均线信号影响 —— 两者用同一份行情/成分股/资金,唯一差异是 timing 开关,
    # 若引擎/生成器未正确接入择时信号,这条断言会失败(非恒真)。
    strat_held = _held(SessionStore(out["strategy"]["session_dir"]).load_state())
    base_held = _held(SessionStore(out["baseline"]["session_dir"]).load_state())
    assert strat_held == {}, f"择时 risk-off 段末应空仓, 实际 {strat_held}"
    assert base_held != {}, "基线应始终满仓(不受择时信号影响)"
