"""引擎日期键 dtype 分派:Datetime 帧过 DataBundle/quick_fitness/_factor_values 不炸。"""
from datetime import date, datetime

import polars as pl

from factorzen.discovery.scoring import DataBundle, _cut_literal, quick_fitness


def _intraday_daily(n_bars: int = 48, n_syms: int = 40) -> pl.DataFrame:
    # ≥MIN_IC_SAMPLES(30) 个标的,否则 compute_rank_ic 跳过全部横截面 → IC 序列空
    ts = [datetime(2026, 5, 1 + i // 24, i % 24, 0) for i in range(n_bars)]
    rows = []
    for si in range(n_syms):
        base = 100.0 + si * 10
        for i, t in enumerate(ts):
            rows.append({"ts_code": f"C{si:02d}USDT", "trade_date": t,
                         "close": base + i * 0.5, "vol": 1.0, "amount": 100.0})
    return pl.DataFrame(rows).with_columns(pl.col("trade_date").cast(pl.Datetime("us")))


def test_cut_literal_dispatch():
    intraday = _intraday_daily()
    daily = intraday.with_columns(pl.col("trade_date").cast(pl.Date))
    assert _cut_literal(intraday, "20260501") == datetime(2026, 5, 1)
    assert _cut_literal(daily, "20260501") == date(2026, 5, 1)


def test_databundle_and_fitness_on_datetime_frame():
    df = _intraday_daily()
    bundle = DataBundle.build(df, train_ratio=0.7)
    factor = df.select("trade_date", "ts_code",
                       pl.col("close").alias("factor_value"))
    res = quick_fitness(factor, bundle, "train")
    assert res["n"] > 0  # 切分/过滤在 Datetime 键上正常工作


def test_factor_values_eval_start_on_datetime_frame():
    from factorzen.discovery.expression import parse_expr
    from factorzen.discovery.mining_session import _factor_values
    df = _intraday_daily()
    leaf_map = {"close": "close", "vol": "vol", "amount": "amount"}
    out = _factor_values(parse_expr("close", leaf_map), df, eval_start="20260502",
                         leaf_map=leaf_map)
    assert out["trade_date"].min() >= datetime(2026, 5, 2)
