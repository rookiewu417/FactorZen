import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.execution.drivers import run_replay


def _write_portfolio_run(dir_: Path, sig: date, code: str, w: float) -> str:
    dir_.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"ts_code": [code], "target_weight": [w]}).write_parquet(dir_ / "weights.parquet")
    (dir_ / "manifest.json").write_text(json.dumps(
        {"signal_date": sig.isoformat(), "status": "optimal"}))
    return str(dir_)


def _daily(codes, dates):
    rows = []
    for d in dates:
        for c in codes:
            rows.append({"trade_date": d, "ts_code": c, "open": 10.0,
                         "pre_close": 10.0, "close": 10.0, "vol": 1e6})
    return pl.DataFrame(rows)


def test_replay_produces_session_artifacts(tmp_path: Path):
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    daily = _daily(["A.SZ"], dates)
    rd = _write_portfolio_run(tmp_path / "pf", date(2026, 1, 5), "A.SZ", 0.5)
    out = run_replay(
        session_dir=tmp_path / "sess", portfolio_run_dirs=[rd],
        daily=daily, initial_cash=1_000_000.0,
        from_date=dates[0], to_date=dates[-1], seed=0,
    )
    assert out["n_steps"] >= 1
    assert (tmp_path / "sess" / "nav.parquet").exists()
    assert (tmp_path / "sess" / "ledger.parquet").exists()
    assert (tmp_path / "sess" / "manifest.json").exists()
    nav = pl.read_parquet(tmp_path / "sess" / "nav.parquet")
    assert nav.height == out["n_steps"]


def test_replay_binds_adv_capacity_constraint(tmp_path: Path):
    """容量约束须真的在 replay 路径生效：day1 提供成交额历史用于算 day2 的
    trailing ADV（_precompute_adv_20d_by_date 对当日 shift(1)，故 day1 自身无
    ADV，day2 才有）。day2 信号从 0 → 满仓 1.0，若 ADV 未接入，delta=1.0 不会
    被 capacity 截断，买入股数将接近 100000 股（满仓）；若 ADV 正确接入，
    max_participation_rate=0.05 * adv=200000 / portfolio_value=1e6 = 0.01，
    截断后买入应恰好是 1000 股（独立手算 ground truth，非恒真）。
    """
    code = "A.SZ"
    dates = [date(2026, 1, 5), date(2026, 1, 6)]
    rows = [
        {"trade_date": d, "ts_code": code, "open": 10.0, "pre_close": 10.0,
         "close": 10.0, "vol": 1_000_000.0, "amount": 200_000.0}
        for d in dates
    ]
    daily = pl.DataFrame(rows)
    # day1 发满仓信号，次日 day2 执行(s<d)；day1 参与行情驱动为 day2 的 ADV 提供
    # 历史 amount。day1 自身无更早信号→不下单。
    rd = _write_portfolio_run(tmp_path / "pf", dates[0], code, 1.0)
    run_replay(
        session_dir=tmp_path / "sess", portfolio_run_dirs=[rd],
        daily=daily, initial_cash=1_000_000.0,
        from_date=dates[0], to_date=dates[-1], seed=0,
    )
    ledger = pl.read_parquet(tmp_path / "sess" / "ledger.parquet")
    day2 = ledger.filter(pl.col("as_of_date") == dates[1].isoformat())
    assert day2.height == 1
    payload = json.loads(day2["payload"][0])
    fills = payload["fills"]
    assert len(fills) == 1, fills
    assert fills[0]["filled_volume"] == 1000, fills  # ground truth: 手算见上


def test_replay_is_idempotent_on_rerun(tmp_path: Path):
    dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    daily = _daily(["A.SZ"], dates)
    rd = _write_portfolio_run(tmp_path / "pf", date(2026, 1, 5), "A.SZ", 0.5)
    kwargs = dict(
        session_dir=tmp_path / "sess", portfolio_run_dirs=[rd],
        daily=daily, initial_cash=1_000_000.0,
        from_date=dates[0], to_date=dates[-1], seed=0,
    )
    out1 = run_replay(**kwargs)
    nav1 = pl.read_parquet(tmp_path / "sess" / "nav.parquet")
    out2 = run_replay(**kwargs)  # 重跑同一 session_dir
    nav2 = pl.read_parquet(tmp_path / "sess" / "nav.parquet")
    assert nav2.height == nav1.height  # 不翻倍/不追加重复日期行
    assert out2["n_steps"] == 0  # 第二次全部日期已记录，跳过不重复下单
    assert out1["n_steps"] >= 1
