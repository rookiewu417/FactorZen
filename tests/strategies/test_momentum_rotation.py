from datetime import date, timedelta
from pathlib import Path

import polars as pl

from factorzen.strategies.momentum_rotation import generate_momentum_rotation_products


def _idx(closes: list[float], start=date(2026, 1, 1)):
    return pl.DataFrame(
        [{"trade_date": start + timedelta(days=i), "close": c} for i, c in enumerate(closes)]
    )


def _price(dates, codes, amount=1e9):
    return pl.DataFrame(
        [{"trade_date": d, "ts_code": c, "open": 10.0, "pre_close": 10.0,
          "close": 10.0, "vol": 1e8, "amount": amount} for d in dates for c in codes]
    )


def _members(code, date_str):  # 注入避网络：两指数各自成分
    return {"IDXA": ["A1.SZ", "A2.SZ"], "IDXB": ["B1.SZ", "B2.SZ"]}[code]


def test_rotation_picks_stronger_index():
    # IDXA 涨 20%, IDXB 涨 5% → 选 IDXA 的成分
    closes_a = [10.0] * 5 + [12.0]   # lookback=5: 12/10-1=+20%
    closes_b = [10.0] * 5 + [10.5]   # +5%
    T = date(2026, 1, 6)
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(6)]
    price = _price(dates, ["A1.SZ", "A2.SZ", "B1.SZ", "B2.SZ"])
    dirs = generate_momentum_rotation_products(
        "/tmp/_mr_a", {"IDXA": _idx(closes_a), "IDXB": _idx(closes_b)}, price, [T],
        members_fn=_members, lookback=5, top_n=2)
    held = set(pl.read_parquet(Path(dirs[0]) / "weights.parquet")["ts_code"].to_list())
    assert held == {"A1.SZ", "A2.SZ"}, f"应持强者 IDXA 成分, 实际 {held}"


def test_all_negative_momentum_goes_cash():
    closes_a = [10.0] * 5 + [9.0]    # -10%
    closes_b = [10.0] * 5 + [9.5]    # -5%
    T = date(2026, 1, 6)
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(6)]
    price = _price(dates, ["A1.SZ", "A2.SZ", "B1.SZ", "B2.SZ"])
    dirs = generate_momentum_rotation_products(
        "/tmp/_mr_b", {"IDXA": _idx(closes_a), "IDXB": _idx(closes_b)}, price, [T],
        members_fn=_members, lookback=5, top_n=2)
    assert pl.read_parquet(Path(dirs[0]) / "weights.parquet").height == 0, "全负动量应空仓"


def test_pit_no_lookahead():
    # T 处动量只用 ≤T；T 之后 IDXB 暴涨不能让 T 处改选 IDXB
    T = date(2026, 1, 6)
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(8)]
    closes_a = [10.0] * 5 + [12.0, 12.0, 12.0]  # 到 T(idx5) +20%
    closes_b = [10.0] * 5 + [10.5, 100.0, 200.0]  # T 处 +5%, 但 T 之后暴涨
    price = _price(dates, ["A1.SZ", "A2.SZ", "B1.SZ", "B2.SZ"])
    dirs = generate_momentum_rotation_products(
        "/tmp/_mr_c", {"IDXA": _idx(closes_a), "IDXB": _idx(closes_b)}, price, [T],
        members_fn=_members, lookback=5, top_n=2)
    held = set(pl.read_parquet(Path(dirs[0]) / "weights.parquet")["ts_code"].to_list())
    assert held == {"A1.SZ", "A2.SZ"}, "T 处应仍选 IDXA; 若受 T 之后 IDXB 暴涨影响=泄漏未来"
