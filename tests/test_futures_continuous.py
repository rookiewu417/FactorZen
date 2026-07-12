"""主力连续合约后复权 ground-truth 测试（不写恒真断言，逐值对手工计算）。

契约（见 markets/futures/continuous.py 与计划 2.1）：
- 乘法后复权，earliest-anchored：首段 adj_factor=1.0；
- 展期日 roll_step = 旧主力前日收盘 / 新主力前日收盘，adj_factor 沿品种 cum_prod；
- 复权后展期日 ret = 新主力自身当日收益，ts_* 跨展期无跳变。
"""
from __future__ import annotations

from datetime import date

import polars as pl

from factorzen.markets.futures.continuous import build_continuous


def _daily(rows: list[tuple]) -> pl.DataFrame:
    # (ts_code, trade_date, open, high, low, close, vol, amount, oi)
    return pl.DataFrame(
        rows,
        schema=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "oi"],
        orient="row",
    )


def _mapping(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=["ts_code", "trade_date", "mapping_ts_code"], orient="row"
    )


def test_single_roll_ground_truth() -> None:
    """两合约一次展期，逐值对手工计算的后复权 close。

    品种 CU.SHF：d1-d3 主力 A(CU2401)，d4-d5 主力 B(CU2402)。
    A close: 100,102,101（d1-d3）。B close: d3=110, d4=112, d5=111（B 在 d3 已上市交易）。
    展期日 d4：roll_step = A_{d3}/B_{d3} = 101/110。adj_factor: d1-3=1，d4-5=101/110=0.9181818…
    手工后复权 close: [100, 102, 101, 112*0.9181818=102.8363636, 111*0.9181818=101.9181818]
    """
    d = [date(2024, 1, i) for i in range(1, 6)]
    daily = _daily([
        ("CU2401.SHF", d[0], 100, 100, 100, 100.0, 10, 1.0, 5),
        ("CU2401.SHF", d[1], 102, 102, 102, 102.0, 10, 1.0, 5),
        ("CU2401.SHF", d[2], 101, 101, 101, 101.0, 10, 1.0, 5),
        # B 在 d3 已交易（非主力），提供 new_prev_close
        ("CU2402.SHF", d[2], 110, 110, 110, 110.0, 20, 2.0, 8),
        ("CU2402.SHF", d[3], 112, 112, 112, 112.0, 20, 2.0, 8),
        ("CU2402.SHF", d[4], 111, 111, 111, 111.0, 20, 2.0, 8),
    ])
    mapping = _mapping([
        ("CU.SHF", d[0], "CU2401.SHF"),
        ("CU.SHF", d[1], "CU2401.SHF"),
        ("CU.SHF", d[2], "CU2401.SHF"),
        ("CU.SHF", d[3], "CU2402.SHF"),
        ("CU.SHF", d[4], "CU2402.SHF"),
    ])
    out = build_continuous(mapping, daily, fut_codes={"CU"}).sort("trade_date")
    assert out["ts_code"].unique().to_list() == ["CU.SHF"]

    f = 101.0 / 110.0
    expected_close = [100.0, 102.0, 101.0, 112.0 * f, 111.0 * f]
    got = out["close"].to_list()
    for g, e in zip(got, expected_close, strict=True):
        assert abs(g - e) < 1e-9, f"close {g} != {e}"

    expected_adj = [1.0, 1.0, 1.0, f, f]
    for g, e in zip(out["adj_factor"].to_list(), expected_adj, strict=True):
        assert abs(g - e) < 1e-12

    # 展期日 ret = 新主力自身收益（非跨合约跳变）
    ret = (out["close"] / out["close"].shift(1) - 1.0).to_list()
    assert abs(ret[3] - (112.0 / 110.0 - 1.0)) < 1e-9  # roll day = B 自身 d4 收益
    assert abs(ret[4] - (111.0 / 112.0 - 1.0)) < 1e-9
    assert abs(ret[1] - (102.0 / 100.0 - 1.0)) < 1e-9  # 非展期 = A 自身收益

    # 量列不复权（原始）
    assert out.sort("trade_date")["vol"].to_list() == [10, 10, 10, 20, 20]
    assert out.sort("trade_date")["oi"].to_list() == [5, 5, 5, 8, 8]


def test_two_rolls_cumulative_ground_truth() -> None:
    """三合约两次展期，累乘 adj_factor 逐值校验（防单展期恰好巧合）。"""
    d = [date(2024, 1, i) for i in range(1, 7)]
    daily = _daily([
        ("A.C1", d[0], 100, 100, 100, 100.0, 10, 1.0, 5),
        ("A.C1", d[1], 101, 101, 101, 101.0, 10, 1.0, 5),
        ("A.C2", d[1], 200, 200, 200, 200.0, 10, 1.0, 5),  # B d2 (new_prev for roll@d3)
        ("A.C2", d[2], 202, 202, 202, 202.0, 10, 1.0, 5),
        ("A.C2", d[3], 203, 203, 203, 203.0, 10, 1.0, 5),
        ("A.C3", d[3], 50, 50, 50, 50.0, 10, 1.0, 5),      # C d4 (new_prev for roll@d5)
        ("A.C3", d[4], 51, 51, 51, 51.0, 10, 1.0, 5),
        ("A.C3", d[5], 52, 52, 52, 52.0, 10, 1.0, 5),
    ])
    mapping = _mapping([
        ("A.DCE", d[0], "A.C1"), ("A.DCE", d[1], "A.C1"),
        ("A.DCE", d[2], "A.C2"), ("A.DCE", d[3], "A.C2"),
        ("A.DCE", d[4], "A.C3"), ("A.DCE", d[5], "A.C3"),
    ])
    out = build_continuous(mapping, daily, fut_codes={"A"}).sort("trade_date")

    step3 = 101.0 / 200.0   # A_{d2}/B_{d2}
    step5 = 203.0 / 50.0    # B_{d4}/C_{d4}
    c = [1.0, 1.0, step3, step3, step3 * step5, step3 * step5]
    raw_close = [100.0, 101.0, 202.0, 203.0, 51.0, 52.0]
    expected = [r * cc for r, cc in zip(raw_close, c, strict=True)]
    for g, e in zip(out["close"].to_list(), expected, strict=True):
        assert abs(g - e) < 1e-9, f"{g} != {e}"

    ret = (out["close"] / out["close"].shift(1) - 1.0).to_list()
    assert abs(ret[2] - (202.0 / 200.0 - 1.0)) < 1e-9  # roll@d3 = B own
    assert abs(ret[4] - (51.0 / 50.0 - 1.0)) < 1e-9    # roll@d5 = C own
    assert abs(ret[3] - (203.0 / 202.0 - 1.0)) < 1e-9
    assert abs(ret[5] - (52.0 / 51.0 - 1.0)) < 1e-9


def test_secondary_L_code_filtered() -> None:
    """次主力连续（L 后缀）经 fut_codes 过滤，只留主力连续。"""
    d = [date(2024, 1, 1), date(2024, 1, 2)]
    daily = _daily([
        ("CU2401.SHF", d[0], 100, 100, 100, 100.0, 10, 1.0, 5),
        ("CU2401.SHF", d[1], 101, 101, 101, 101.0, 10, 1.0, 5),
        ("CU2402.SHF", d[0], 110, 110, 110, 110.0, 20, 2.0, 8),
        ("CU2402.SHF", d[1], 111, 111, 111, 111.0, 20, 2.0, 8),
    ])
    mapping = _mapping([
        ("CU.SHF", d[0], "CU2401.SHF"), ("CU.SHF", d[1], "CU2401.SHF"),
        ("CUL.SHF", d[0], "CU2402.SHF"), ("CUL.SHF", d[1], "CU2402.SHF"),
    ])
    out = build_continuous(mapping, daily, fut_codes={"CU"})
    assert out["ts_code"].unique().to_list() == ["CU.SHF"]  # CUL.SHF 被过滤


def test_missing_new_prev_close_no_adjust() -> None:
    """新主力在展期前一日无报价 → 该展期不复权（roll_step=1），诚实退化不崩。"""
    d = [date(2024, 1, i) for i in range(1, 4)]
    daily = _daily([
        ("X2401.SHF", d[0], 100, 100, 100, 100.0, 10, 1.0, 5),
        ("X2401.SHF", d[1], 101, 101, 101, 101.0, 10, 1.0, 5),
        # 新主力 X2402 在 d2（前一日）无行情，只在 d3 出现
        ("X2402.SHF", d[2], 111, 111, 111, 111.0, 20, 2.0, 8),
    ])
    mapping = _mapping([
        ("X.SHF", d[0], "X2401.SHF"),
        ("X.SHF", d[1], "X2401.SHF"),
        ("X.SHF", d[2], "X2402.SHF"),  # 展期日，但 new_prev 缺失
    ])
    out = build_continuous(mapping, daily, fut_codes={"X"}).sort("trade_date")
    # 无法算 roll_step → 保持 1.0，close 原样（接受原始跳变，不 NaN 不崩）
    assert out["adj_factor"].to_list() == [1.0, 1.0, 1.0]
    assert out["close"].to_list() == [100.0, 101.0, 111.0]


def test_empty_inputs() -> None:
    empty_m = _mapping([]).clear()
    empty_d = _daily([]).clear()
    out = build_continuous(empty_m, empty_d, fut_codes={"CU"})
    assert out.is_empty()
    assert set(["ts_code", "trade_date", "close", "adj_factor"]).issubset(out.columns)
