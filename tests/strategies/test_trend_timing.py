from datetime import date
from pathlib import Path

import polars as pl

from factorzen.strategies.trend_timing import generate_trend_timing_products


def _idx(rows):  # rows: (date, close)
    return pl.DataFrame([{"trade_date": d, "close": c} for d, c in rows])


def _price(dates, codes, amount=1e9):
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


def _fake_members(code, date_str):  # 注入,避免网络
    return ["A.SZ", "B.SZ", "C.SZ"]


def test_pit_no_lookahead_ma(tmp_path: Path):
    # T 处 MA 只用 ≤T；T 之后暴涨不能改变 T 的 risk 判定
    dates = [date(2026, 1, d) for d in range(5, 12)]
    idx = _idx([(dates[i], 10.0) for i in range(6)] + [(dates[6], 100.0)])  # 最后一天暴涨
    price = _price(dates, ["A.SZ", "B.SZ", "C.SZ"])
    # 在 T=dates[5] 调仓, ma_window=3: MA=mean(close[≤T].tail(3))=10, close(T)=10 → not >MA → risk-off
    dirs = generate_trend_timing_products(
        str(tmp_path / "s"),
        idx,
        price,
        [dates[5]],
        members_fn=_fake_members,
        ma_window=3,
        top_n=3,
        timing=True,
    )
    w = pl.read_parquet(Path(dirs[0]) / "weights.parquet")
    assert w.height == 0, "T 处 close==MA 未站上 → 应 risk-off 空仓; 若受 T 之后暴涨影响则泄漏"


def test_risk_on_equal_weight_topn(tmp_path: Path):
    dates = [date(2026, 1, d) for d in range(5, 12)]
    # 让 close 明显站上 MA
    idx = _idx([(dates[i], 10.0) for i in range(6)] + [(dates[6], 10.0)])
    idx = idx.with_columns(
        pl.when(pl.col("trade_date") == dates[5]).then(20.0).otherwise(pl.col("close")).alias("close")
    )
    price = _price(dates, ["A.SZ", "B.SZ", "C.SZ"])
    dirs = generate_trend_timing_products(
        str(tmp_path / "s"),
        idx,
        price,
        [dates[5]],
        members_fn=_fake_members,
        ma_window=3,
        top_n=2,
        timing=True,
    )
    w = pl.read_parquet(Path(dirs[0]) / "weights.parquet")
    assert w.height == 2 and abs(w["target_weight"][0] - 0.5) < 1e-9  # top2 各 1/2


def test_baseline_always_full(tmp_path: Path):
    dates = [date(2026, 1, d) for d in range(5, 9)]
    idx = _idx([(d, 5.0) for d in dates])  # 一直低于任何 MA
    price = _price(dates, ["A.SZ", "B.SZ", "C.SZ"])
    dirs = generate_trend_timing_products(
        str(tmp_path / "s"),
        idx,
        price,
        [dates[2]],
        members_fn=_fake_members,
        ma_window=2,
        top_n=3,
        timing=False,
    )  # 基线
    w = pl.read_parquet(Path(dirs[0]) / "weights.parquet")
    assert w.height == 3, "基线 timing=False 应始终满仓,无视信号"
