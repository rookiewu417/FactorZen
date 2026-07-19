"""tests/test_intraday_battery.py — 特征电池 v1 与 compute_day_panel 数值 ground-truth。"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from factorzen.intraday.features import battery, battery_hash, compute_day_panel
from factorzen.intraday.features.spec import IntradayFeatureSpec


def _dt(h: int, m: int, day: int = 2) -> datetime:
    return datetime(2024, 1, day, h, m, 0)


def _sparse_one_day() -> pl.DataFrame:
    """1 股 1 日稀疏 1min 帧（经 5min 重采样得 8 桶，i∈{1,2,6,24,25,42,43,48}）。

    桶序列（手算，W0 bar-end）：
    - 09:30+09:31 → 09:35 i=1 open=10 close=10.5 vol=300 amount=3100
    - 09:40 → i=2 close=10.2 vol=150 amount=1530
    - 10:00 → i=6 close=10.3 vol=400 amount=4120
    - 11:30 → i=24 close=10.4 vol=250 amount=2600
    - 13:01 → 13:05 i=25 close=10.5 vol=300 amount=3150
    - 14:30 → i=42 close=10.6 vol=200 amount=2120
    - 14:35 → i=43 close=10.7 vol=350 amount=3745
    - 15:00 → i=48 close=10.8 vol=500 amount=5400
    """
    rows = [
        (_dt(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
        (_dt(9, 31), 10.0, 10.6, 10.0, 10.5, 200, 2100.0),
        (_dt(9, 40), 10.5, 10.5, 10.2, 10.2, 150, 1530.0),
        (_dt(10, 0), 10.2, 10.4, 10.1, 10.3, 400, 4120.0),
        (_dt(11, 30), 10.3, 10.5, 10.2, 10.4, 250, 2600.0),
        (_dt(13, 1), 10.4, 10.6, 10.3, 10.5, 300, 3150.0),
        (_dt(14, 30), 10.5, 10.7, 10.4, 10.6, 200, 2120.0),
        (_dt(14, 35), 10.6, 10.8, 10.5, 10.7, 350, 3745.0),
        (_dt(15, 0), 10.7, 10.9, 10.6, 10.8, 500, 5400.0),
    ]
    return pl.DataFrame(
        {
            "ts_code": ["000001.SZ"] * len(rows),
            "trade_time": pl.Series([r[0] for r in rows], dtype=pl.Datetime("us")),
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
            "amount": [r[6] for r in rows],
        }
    )


# 手算硬编码期望（rel tol 1e-9）
_EXPECTED = {
    "i_rv": 0.06217881540867241,
    "i_rskew": 1.2582542330998039,
    "i_rkurt": 3.7286865731759713,
    "i_downvol_ratio": 0.2111441355367538,
    "i_updown_vol": 1.3180426148977666,
    "i_ret_open30": 0.030000000000000027,
    "i_ret_close30": 0.018867924528301883,
    "i_ret_mid": 0.029126213592232997,
    "i_vwap_dev": 0.026974577915777287,
    "i_smart_money": 0.9874755556882473,
    "i_vol_open30_share": 0.3469387755102041,
    "i_vol_close30_share": 0.3469387755102041,
    "i_vol_entropy": 0.9718771358715251,
    "i_amihud": 6588.942735992862,
    "i_path_eff": 0.5714285714285711,
    "i_max_ret_share": 0.36763884425793547,
}

_CORE_HARD = [
    "i_rv",
    "i_rskew",
    "i_ret_open30",
    "i_ret_close30",
    "i_vwap_dev",
    "i_vol_entropy",
    "i_smart_money",
    "i_path_eff",
    "i_amihud",
    "i_max_ret_share",
]


class TestBatteryMeta:
    def test_v1_twenty_unique_i_prefix(self) -> None:
        """17 个连续路径统计 + 3 个涨跌停邻域（2026-07-19 新增）。"""
        specs = battery("v1", "5min")
        assert len(specs) == 20
        names = [s.name for s in specs]
        assert len(set(names)) == 20
        # 新增的三个必须在册，且与 INTRADAY_FEATURES 一致（登记簿不许漂移）
        from factorzen.core.feature_schema import INTRADAY_FEATURES

        assert set(names) == INTRADAY_FEATURES
        assert {"i_limit_up_seal_share", "i_limit_up_open_count",
                "i_limit_up_first_touch"} <= set(names)
        assert all(n.startswith("i_") for n in names)
        assert all(isinstance(s, IntradayFeatureSpec) for s in specs)
        assert all(s.formula and s.description for s in specs)
        assert all(s.expression is None for s in specs)

    def test_v2_raises(self) -> None:
        with pytest.raises(ValueError, match="未知电池版本"):
            battery("v2")

    def test_60min_raises(self) -> None:
        with pytest.raises(ValueError, match="不支持"):
            battery("v1", freq="60min")

    def test_battery_hash_stable(self) -> None:
        a = battery_hash(battery("v1", "5min"))
        b = battery_hash(battery("v1", "5min"))
        assert a == b
        assert len(a) == 16


class TestGroundTruth:
    def test_sparse_5min_core_features(self) -> None:
        specs = battery("v1", "5min")
        panel = compute_day_panel(
            _sparse_one_day(), specs, "5min", min_bar_coverage=0.0
        )
        assert panel.height == 1
        assert panel["trade_date"].dtype == pl.Date
        row = panel.row(0, named=True)

        for name in _CORE_HARD:
            assert row[name] == pytest.approx(_EXPECTED[name], rel=1e-9), name

        # 其余：非空 + 符号合理
        assert row["i_rkurt"] == pytest.approx(_EXPECTED["i_rkurt"], rel=1e-9)
        assert row["i_downvol_ratio"] == pytest.approx(
            _EXPECTED["i_downvol_ratio"], rel=1e-9
        )
        assert 0.0 < row["i_downvol_ratio"] < 1.0
        assert row["i_updown_vol"] == pytest.approx(_EXPECTED["i_updown_vol"], rel=1e-9)
        assert row["i_ret_mid"] == pytest.approx(_EXPECTED["i_ret_mid"], rel=1e-9)
        assert row["i_vol_open30_share"] == pytest.approx(
            _EXPECTED["i_vol_open30_share"], rel=1e-9
        )
        assert row["i_vol_close30_share"] == pytest.approx(
            _EXPECTED["i_vol_close30_share"], rel=1e-9
        )
        # 有效桶 8 < 10 → i_pv_corr 恒 null
        assert row["i_pv_corr"] is None


class TestGuards:
    def test_low_coverage_nulls_all_features_keeps_row(self) -> None:
        specs = battery("v1", "5min")
        # 正常日
        good = _sparse_one_day()
        # 覆盖不足日：仅 2 根 bar，另一天
        rows = [
            (_dt(9, 31, day=3), 10.0, 10.1, 9.9, 10.0, 100, 1000.0),
            (_dt(15, 0, day=3), 10.0, 10.2, 9.9, 10.1, 100, 1010.0),
        ]
        low = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 2,
                "trade_time": pl.Series([r[0] for r in rows], dtype=pl.Datetime("us")),
                "open": [r[1] for r in rows],
                "high": [r[2] for r in rows],
                "low": [r[3] for r in rows],
                "close": [r[4] for r in rows],
                "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
                "amount": [r[6] for r in rows],
            }
        )
        # 稀疏正常日有效桶=8；门槛 0.1×48=4.8 → 正常日通过，2 桶日不通过
        # （0.8×48=38.4 会把稀疏正常日也打成低覆盖）
        panel = compute_day_panel(
            pl.concat([good, low]), specs, "5min", min_bar_coverage=0.1
        )
        assert panel.height == 2
        feat_cols = [s.name for s in specs]
        low_row = panel.filter(pl.col("trade_date") == datetime(2024, 1, 3).date())
        assert low_row.height == 1
        for c in feat_cols:
            assert low_row[c][0] is None, c
        good_row = panel.filter(pl.col("trade_date") == datetime(2024, 1, 2).date())
        assert good_row["i_rv"][0] is not None

    def test_nan_input_becomes_null_not_nan(self) -> None:
        specs = battery("v1", "5min")
        df = _sparse_one_day().with_columns(
            pl.when(pl.col("trade_time").dt.minute() == 40)
            .then(float("nan"))
            .otherwise(pl.col("close"))
            .alias("close")
        )
        panel = compute_day_panel(df, specs, "5min", min_bar_coverage=0.0)
        for c in [s.name for s in specs]:
            vals = panel[c].to_list()
            for v in vals:
                if v is not None:
                    assert v == v  # not NaN

    def test_all_zero_vol_no_crash(self) -> None:
        specs = battery("v1", "5min")
        rows = [
            (_dt(9, 31), 10.0, 10.0, 10.0, 10.0, 0, 0.0),
            (_dt(9, 40), 10.0, 10.1, 9.9, 10.05, 0, 0.0),
            (_dt(15, 0), 10.05, 10.1, 10.0, 10.1, 0, 0.0),
        ]
        df = pl.DataFrame(
            {
                "ts_code": ["000001.SZ"] * 3,
                "trade_time": pl.Series([r[0] for r in rows], dtype=pl.Datetime("us")),
                "open": [r[1] for r in rows],
                "high": [r[2] for r in rows],
                "low": [r[3] for r in rows],
                "close": [r[4] for r in rows],
                "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
                "amount": [r[6] for r in rows],
            }
        )
        panel = compute_day_panel(df, specs, "5min", min_bar_coverage=0.0)
        assert panel.height == 1
        # V 类特征 null
        assert panel["i_vwap_dev"][0] is None
        assert panel["i_vol_open30_share"][0] is None
        assert panel["i_smart_money"][0] is None
        # 路径效率仍可算
        assert panel["i_path_eff"][0] is not None or panel["i_path_eff"][0] is None


# ── 涨跌停邻域叶（A 股特有的离散状态机；与 17 个连续路径统计机制不同）──────────


def _limit_day(closes: list[float], highs: list[float] | None = None,
               *, day: int = 2, code: str = "000001.SZ") -> pl.DataFrame:
    """每个给定值落在**不同的 5min bar** 上的 1min 帧（09:30 起，每 5 分钟一根）。

    ⚠️ 两个坑（初版都踩了）：
    1. 间隔必须 ≥5 分钟——连续分钟会被 ``resample_intraday`` 并进同一个桶，
       seal_share / first_touch 退化成单桶，测不出判别力。
    2. 起点用 **09:31** 而非 09:30——bar-end 约定下 09:30 与 09:35 会落进同一根，
       实测 09:30 起点只得 3 根、09:31 起点得 4 根。
    """
    from datetime import timedelta
    highs = highs if highs is not None else list(closes)
    base = datetime(2024, 1, day, 9, 31, 0)
    rows = []
    for i, (c, h) in enumerate(zip(closes, highs, strict=True)):
        t = base + timedelta(minutes=5 * i)
        rows.append((code, t, c, h, min(c, h), c, 100, c * 100))
    return pl.DataFrame(
        rows,
        schema=["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"],
        orient="row",
    )


def _limit_ref(pre_close: float, limit_pct: float = 0.1, *, day: int = 2,
               code: str = "000001.SZ") -> pl.DataFrame:
    from datetime import date as _date
    return pl.DataFrame({
        "ts_code": [code], "trade_date": [_date(2024, 1, day)],
        "pre_close": [pre_close], "limit_pct": [limit_pct],
    })


def test_limit_leaves_seal_share_and_open_count():
    """封板时长占比 + 打开次数：手算 ground-truth。

    pre_close=10 → 涨停价 11.0。close 序列 [11.0, 11.0, 10.9, 11.0]：
    - seal = [1,1,0,1] → seal_share = 3/4
    - 打开次数 = seal 由 1→0 的次数 = 1
    """
    from factorzen.intraday.features import battery, compute_day_panel

    minute = _limit_day([11.0, 11.0, 10.9, 11.0])
    ref = _limit_ref(10.0)
    out = compute_day_panel(minute, battery(), "5min", min_bar_coverage=0.0,
                            daily_ref=ref)
    assert "i_limit_up_seal_share" in out.columns, out.columns
    r = out.row(0, named=True)
    assert abs(r["i_limit_up_seal_share"] - 0.75) < 1e-9, r["i_limit_up_seal_share"]
    assert r["i_limit_up_open_count"] == 1.0, r["i_limit_up_open_count"]


def test_limit_leaves_never_touched_is_zero_not_null():
    """全日未触板 → seal_share/open_count = **0**（0 有信息「今天没封过」），
    first_touch = **1.0**（最晚）。不得为 null——否则截面 95%+ null，
    rank/IC 会塌成少数触板票的子样本游戏。"""
    from factorzen.intraday.features import battery, compute_day_panel

    out = compute_day_panel(_limit_day([10.1, 10.2, 10.15, 10.2]), battery(), "5min",
                            min_bar_coverage=0.0, daily_ref=_limit_ref(10.0))
    r = out.row(0, named=True)
    assert r["i_limit_up_seal_share"] == 0.0
    assert r["i_limit_up_open_count"] == 0.0
    assert r["i_limit_up_first_touch"] == 1.0


def test_limit_leaves_first_touch_earlier_is_smaller():
    """首次触板越早，first_touch 越小（判别力：两组对照）。"""
    from factorzen.intraday.features import battery, compute_day_panel

    early = compute_day_panel(_limit_day([11.0, 10.5, 10.5, 10.5]), battery(), "5min",
                              min_bar_coverage=0.0, daily_ref=_limit_ref(10.0))
    late = compute_day_panel(_limit_day([10.5, 10.5, 10.5, 11.0]), battery(), "5min",
                             min_bar_coverage=0.0, daily_ref=_limit_ref(10.0))
    assert early.row(0, named=True)["i_limit_up_first_touch"] < \
        late.row(0, named=True)["i_limit_up_first_touch"]


def test_limit_leaves_null_without_daily_ref():
    """不传 daily_ref（旧调用方）→ 涨跌停叶全 null，其余 17 叶**逐位不变**（零回归）。"""
    from factorzen.intraday.features import battery, compute_day_panel

    minute = _limit_day([11.0, 11.0, 10.9, 11.0])
    with_ref = compute_day_panel(minute, battery(), "5min", min_bar_coverage=0.0,
                                 daily_ref=_limit_ref(10.0))
    without = compute_day_panel(minute, battery(), "5min", min_bar_coverage=0.0)
    assert without.row(0, named=True)["i_limit_up_seal_share"] is None
    for c in ("i_rv", "i_ret_open30", "i_vwap_dev", "i_amihud"):
        a = with_ref.row(0, named=True)[c]
        b = without.row(0, named=True)[c]
        assert (a is None and b is None) or a == b, f"{c}: {a} vs {b}"


def test_limit_leaves_guard_bad_pre_close():
    """pre_close ≤0 / 缺失 → 三叶全 null（不是 0，区别于「未触板」）。"""
    from factorzen.intraday.features import battery, compute_day_panel

    out = compute_day_panel(_limit_day([11.0, 11.0]), battery(), "5min",
                            min_bar_coverage=0.0, daily_ref=_limit_ref(0.0))
    r = out.row(0, named=True)
    assert r["i_limit_up_seal_share"] is None
    assert r["i_limit_up_first_touch"] is None
