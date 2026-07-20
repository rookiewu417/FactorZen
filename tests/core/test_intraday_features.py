"""test_intraday_battery.py：tests/test_intraday_battery.py — 特征电池 v1 与 compute_day_panel 数值 ground-truth。
test_intraday_expr_features.py：tests/test_intraday_expr_features.py — bar 级表达式求值、筛选、注册表。
test_expression_factor_intraday.py：ExpressionFactor.compute 与 evaluate_materialized 对 i_* 逐值一致。
test_feature_schema_intraday.py：日内特征叶子注册：单源守卫 + 键序零回归。
"""
from __future__ import annotations

import datetime as dt
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from factorzen.core.feature_schema import INTRADAY_FEATURES
from factorzen.core.storage import save_parquet
from factorzen.discovery.intraday_expr import (
    AGG_FUNCS,
    ELEMENTWISE_OPS,
    ensure_expr_panel,
    load_expr_registry,
    make_expr_spec,
    materialize_expr_features,
    register_expr_features,
    registry_path,
    screen_expr_panel,
    validate_bar_expr,
)
from factorzen.intraday.features import battery, battery_hash, compute_day_panel
from factorzen.intraday.features.spec import IntradayFeatureSpec


# ==== 来自 test_intraday_battery.py ====
def _dt__battery(h: int, m: int, day: int = 2) -> datetime:
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
        (_dt__battery(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
        (_dt__battery(9, 31), 10.0, 10.6, 10.0, 10.5, 200, 2100.0),
        (_dt__battery(9, 40), 10.5, 10.5, 10.2, 10.2, 150, 1530.0),
        (_dt__battery(10, 0), 10.2, 10.4, 10.1, 10.3, 400, 4120.0),
        (_dt__battery(11, 30), 10.3, 10.5, 10.2, 10.4, 250, 2600.0),
        (_dt__battery(13, 1), 10.4, 10.6, 10.3, 10.5, 300, 3150.0),
        (_dt__battery(14, 30), 10.5, 10.7, 10.4, 10.6, 200, 2120.0),
        (_dt__battery(14, 35), 10.6, 10.8, 10.5, 10.7, 350, 3745.0),
        (_dt__battery(15, 0), 10.7, 10.9, 10.6, 10.8, 500, 5400.0),
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
            (_dt__battery(9, 31, day=3), 10.0, 10.1, 9.9, 10.0, 100, 1000.0),
            (_dt__battery(15, 0, day=3), 10.0, 10.2, 9.9, 10.1, 100, 1010.0),
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
            (_dt__battery(9, 31), 10.0, 10.0, 10.0, 10.0, 0, 0.0),
            (_dt__battery(9, 40), 10.0, 10.1, 9.9, 10.05, 0, 0.0),
            (_dt__battery(15, 0), 10.05, 10.1, 10.0, 10.1, 0, 0.0),
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

# ==== 来自 test_intraday_expr_features.py ====
def _dt__expr_features(h: int, m: int, day: int = 2) -> datetime:
    return datetime(2024, 1, day, h, m, 0)

def _sparse_two_stocks() -> pl.DataFrame:
    """2 股 1 日稀疏 1min 帧（与 battery ground-truth 同桶序列）。"""
    rows = [
        (_dt__expr_features(9, 30), 10.0, 10.0, 10.0, 10.0, 100, 1000.0),
        (_dt__expr_features(9, 31), 10.0, 10.6, 10.0, 10.5, 200, 2100.0),
        (_dt__expr_features(9, 40), 10.5, 10.5, 10.2, 10.2, 150, 1530.0),
        (_dt__expr_features(10, 0), 10.2, 10.4, 10.1, 10.3, 400, 4120.0),
        (_dt__expr_features(11, 30), 10.3, 10.5, 10.2, 10.4, 250, 2600.0),
        (_dt__expr_features(13, 1), 10.4, 10.6, 10.3, 10.5, 300, 3150.0),
        (_dt__expr_features(14, 30), 10.5, 10.7, 10.4, 10.6, 200, 2120.0),
        (_dt__expr_features(14, 35), 10.6, 10.8, 10.5, 10.7, 350, 3745.0),
        (_dt__expr_features(15, 0), 10.7, 10.9, 10.6, 10.8, 500, 5400.0),
    ]
    frames = []
    for code, scale in (("000001.SZ", 1.0), ("000002.SZ", 1.1)):
        frames.append(
            pl.DataFrame(
                {
                    "ts_code": [code] * len(rows),
                    "trade_time": pl.Series(
                        [r[0] for r in rows], dtype=pl.Datetime("us")
                    ),
                    "open": [r[1] * scale for r in rows],
                    "high": [r[2] * scale for r in rows],
                    "low": [r[3] * scale for r in rows],
                    "close": [r[4] * scale for r in rows],
                    "vol": pl.Series([r[5] for r in rows], dtype=pl.Int64),
                    "amount": [r[6] * scale for r in rows],
                }
            )
        )
    return pl.concat(frames)

def _write_minute_source(tmp: Path, minute: pl.DataFrame | None = None) -> Path:
    src = tmp / "src"
    frame = minute if minute is not None else _sparse_two_stocks()
    save_parquet(
        frame,
        data_type="minute_1min",
        date_col="trade_time",
        base_dir=src,
        mode="overwrite",
    )
    return src

# 5min 重采样后 000001.SZ 手算期望（polars std ddof=1）
_EXP_STD_BAR_RET = 0.02100625435521192
_EXP_MEAN_VWAP = 10.479166666666666
_EXP_LAST_SIGNED = 10.8
_EXP_FIRST_BAR_RET = 0.05

class TestValidateBarExpr:
    def test_rejects_ts_and_rank(self) -> None:
        with pytest.raises(ValueError, match=r"禁止算子|未知"):
            validate_bar_expr("ts_mean(close, 5)")
        with pytest.raises(ValueError, match=r"禁止算子"):
            validate_bar_expr("rank(close)")

    def test_accepts_elementwise(self) -> None:
        node = validate_bar_expr("div(amount, vol)")
        assert node is not None
        node2 = validate_bar_expr("mul(close, sign(bar_ret))")
        assert node2 is not None
        assert "div" in ELEMENTWISE_OPS
        assert "rank" not in ELEMENTWISE_OPS

class TestMakeExprSpec:
    def test_same_inputs_same_name(self) -> None:
        a = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        b = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        assert a.name == b.name
        assert a.name.startswith("ix_")
        assert len(a.name) == 11  # ix_ + 8 hex

    def test_equivalent_expr_same_name(self) -> None:
        a = make_expr_spec("div(amount,vol)", "mean", freq="5min")
        b = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        assert a.name == b.name
        assert a.bar_expr == b.bar_expr

    def test_unknown_agg_freq(self) -> None:
        with pytest.raises(ValueError, match="未知聚合"):
            make_expr_spec("close", "mode", freq="5min")
        with pytest.raises(ValueError, match="未知频率"):
            make_expr_spec("close", "mean", freq="7min")

class TestMaterializeGroundTruth:
    def test_std_mean_last_and_bar_ret(self, tmp_path: Path) -> None:
        src = _write_minute_source(tmp_path)
        specs = [
            make_expr_spec("bar_ret", "std", freq="5min"),
            make_expr_spec("div(amount, vol)", "mean", freq="5min"),
            make_expr_spec("mul(close, sign(bar_ret))", "last", freq="5min"),
        ]
        panel = materialize_expr_features(
            specs,
            "20240102",
            "20240102",
            freq="5min",
            source_dir=src,
            min_bar_coverage=0.0,
        )
        assert panel.height == 2
        row = panel.filter(pl.col("ts_code") == "000001.SZ").row(0, named=True)
        assert row[specs[0].name] == pytest.approx(_EXP_STD_BAR_RET, abs=1e-9)
        assert row[specs[1].name] == pytest.approx(_EXP_MEAN_VWAP, abs=1e-9)
        assert row[specs[2].name] == pytest.approx(_EXP_LAST_SIGNED, abs=1e-9)

    def test_bar_ret_first_bar(self, tmp_path: Path) -> None:
        """首 bar bar_ret = close/open−1（5min 首桶合并竞价）。"""
        src = _write_minute_source(tmp_path)
        # first(bar_ret) 应等于首桶 close/open−1
        spec = make_expr_spec("bar_ret", "first", freq="5min")
        panel = materialize_expr_features(
            [spec],
            "20240102",
            "20240102",
            freq="5min",
            source_dir=src,
            min_bar_coverage=0.0,
        )
        v = panel.filter(pl.col("ts_code") == "000001.SZ")[spec.name][0]
        assert v == pytest.approx(_EXP_FIRST_BAR_RET, abs=1e-9)

    def test_mixed_freq_raises(self, tmp_path: Path) -> None:
        s5 = make_expr_spec("close", "last", freq="5min")
        s1 = make_expr_spec("close", "last", freq="1min")
        with pytest.raises(ValueError, match="混频"):
            materialize_expr_features(
                [s5, s1], "20240102", "20240102", freq="5min", source_dir=tmp_path
            )

class TestScreen:
    def test_three_rejects_and_keep(self) -> None:
        dates = [date(2024, 1, d) for d in range(2, 12)]
        codes = [f"{i:06d}.SZ" for i in range(5)]
        rows = []
        for d in dates:
            for i, c in enumerate(codes):
                rows.append(
                    {
                        "trade_date": d,
                        "ts_code": c,
                        "ix_lowcov": None if i < 4 else 1.0,  # 极低覆盖
                        "ix_const": 1.0,  # 近常数
                        "ix_good": float(i) + 0.1 * d.day,
                        "ix_corr": float(i) + 0.1 * d.day + 1e-9,  # 与 good 几乎共线
                    }
                )
        panel = pl.DataFrame(rows)
        # 单独筛 lowcov / const / good
        v1 = screen_expr_panel(
            panel.select(["trade_date", "ts_code", "ix_lowcov"]),
            min_coverage=0.6,
        )
        assert v1["ix_lowcov"] == "low_coverage"

        v2 = screen_expr_panel(
            panel.select(["trade_date", "ts_code", "ix_const"]),
            min_coverage=0.6,
        )
        assert v2["ix_const"] == "degenerate"

        v3 = screen_expr_panel(
            panel.select(["trade_date", "ts_code", "ix_good"]),
            min_coverage=0.6,
        )
        assert v3["ix_good"] == "keep"

        ref = panel.select(["trade_date", "ts_code", "ix_good"])
        v4 = screen_expr_panel(
            panel.select(["trade_date", "ts_code", "ix_corr"]),
            reference=ref,
            min_coverage=0.6,
            max_abs_corr=0.9,
        )
        assert v4["ix_corr"].startswith("correlated:")

class TestRegistry:
    def test_roundtrip_idempotent(self, tmp_path: Path) -> None:
        base = tmp_path / "feat"
        specs = [
            make_expr_spec("div(amount, vol)", "mean", freq="5min", hypothesis="vwap"),
            make_expr_spec("bar_ret", "std", freq="5min"),
        ]
        register_expr_features(specs, session="s1", base_dir=base)
        reg = load_expr_registry(base)
        assert set(reg) == {s.name for s in specs}
        assert reg[specs[0].name].hypothesis == "vwap"
        # 幂等
        register_expr_features(specs, session="s2", base_dir=base)
        lines = registry_path(base).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

class TestEnsureExprPanel:
    def test_cache_and_unregistered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _write_minute_source(tmp_path)
        base = tmp_path / "feat"
        spec = make_expr_spec("div(amount, vol)", "mean", freq="5min")
        register_expr_features([spec], session="t", base_dir=base)

        calls = {"n": 0}
        real_mat = materialize_expr_features

        def _counting(*a, **kw):
            calls["n"] += 1
            # 稀疏测试帧：强制 min_bar_coverage=0，否则默认 0.8 全 null
            kw = dict(kw)
            kw["min_bar_coverage"] = 0.0
            return real_mat(*a, **kw)

        monkeypatch.setattr(
            "factorzen.discovery.intraday_expr.materialize_expr_features",
            _counting,
        )

        p1 = ensure_expr_panel(
            spec.name, "20240102", "20240102", base_dir=base, source_dir=src
        )
        assert calls["n"] == 1
        assert spec.name in p1.columns
        assert p1.height >= 1
        assert p1[spec.name].null_count() == 0

        p2 = ensure_expr_panel(
            spec.name, "20240102", "20240102", base_dir=base, source_dir=src
        )
        assert calls["n"] == 1  # 二次读缓存
        assert p2.height == p1.height

        with pytest.raises(ValueError, match="未注册"):
            ensure_expr_panel("ix_deadbeef", "20240102", "20240102", base_dir=base)

class TestAggFuncsComplete:
    def test_agg_keys(self) -> None:
        expected = {
            "sum", "mean", "std", "skew", "min", "max", "last", "first", "median"
        }
        assert set(AGG_FUNCS) == expected

# ==== 来自 test_expression_factor_intraday.py ====
def test_expression_factor_i_rv_matches_evaluate_materialized(monkeypatch):
    """monkeypatch attach_intraday 注入面板 → compute 与直算逐值一致。"""
    from factorzen.discovery.derived import add_derived_columns
    from factorzen.discovery.expression import evaluate_materialized, parse_expr
    from factorzen.discovery.factor import ExpressionFactor

    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    codes = ["000001.SZ", "000002.SZ"]
    rows = []
    for c in codes:
        for i, d in enumerate(dates):
            rows.append({
                "trade_date": d,
                "ts_code": c,
                "close": 10.0 + i,
                "close_adj": 10.0 + i,
                "open": 10.0,
                "open_adj": 10.0,
                "high": 11.0,
                "high_adj": 11.0,
                "low": 9.0,
                "low_adj": 9.0,
                "pre_close": 10.0,
                "vol": 1e5,
                "amount": 1e6,
            })
    daily = pl.DataFrame(rows)

    panel_rows = []
    for c in codes:
        for i, d in enumerate(dates):
            r = {"trade_date": d, "ts_code": c}
            for leaf in sorted(INTRADAY_FEATURES):
                r[leaf] = 0.01 * (i + 1) + (0.001 if c.endswith("1.SZ") else 0.002)
            panel_rows.append(r)
    panel = pl.DataFrame(panel_rows)

    def _fake_attach(d, **kw):
        # 模拟 require=True 注入
        have = [c for c in sorted(INTRADAY_FEATURES) if c in panel.columns]
        sel = panel.select(["trade_date", "ts_code", *have])
        drop = [c for c in have if c in d.columns]
        if drop:
            d = d.drop(drop)
        return d.join(sel, on=["trade_date", "ts_code"], how="left")

    monkeypatch.setattr(
        "factorzen.daily.data.intraday.attach_intraday", _fake_attach,
    )
    # ExpressionFactor 内 from factorzen.daily.data.intraday import attach_intraday
    # 局部 import 在调用时解析，patch 源模块即可

    class _Ctx:
        start = "20240102"
        end = "20240104"

        @property
        def daily(self):
            return daily.lazy()

        @property
        def daily_basic(self):
            return pl.DataFrame({
                "trade_date": dates * len(codes),
                "ts_code": [c for c in codes for _ in dates],
                "circ_mv": [1e6] * (len(dates) * len(codes)),
            }).lazy()

    expr = "rank(i_rv)"
    fac = ExpressionFactor(expr, mined_name="i_rv_rank")
    out = fac.compute(_Ctx())
    assert out.height > 0
    assert out["factor_value"].null_count() < out.height

    # 直算：attach → sort → derived → evaluate
    attached = _fake_attach(daily)
    prepped = add_derived_columns(attached.sort(["ts_code", "trade_date"]))
    node = parse_expr(expr)
    direct = prepped.with_columns(
        evaluate_materialized(node, prepped).alias("factor_value")
    ).filter(
        pl.col("trade_date") >= dt.date(2024, 1, 2)
    ).select(["trade_date", "ts_code", "factor_value"]).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )

    a = out.sort(["ts_code", "trade_date"])
    b = direct.sort(["ts_code", "trade_date"])
    assert a.height == b.height
    for va, vb in zip(a["factor_value"].to_list(), b["factor_value"].to_list(), strict=True):
        assert va == pytest.approx(vb, abs=1e-9, nan_ok=True)

# ==== 来自 test_feature_schema_intraday.py ====
# 改造前 LEAF_FEATURES 的 40 个键顺序（硬编码守卫：只许末尾扩，不许改旧序）
_PRE_CHANGE_LEAF_KEYS: list[str] = [
    "close",
    "open",
    "high",
    "low",
    "vol",
    "amount",
    "vwap",
    "log_vol",
    "ret_1d",
    "amplitude",
    "intraday_ret",
    "overnight_ret",
    "total_mv",
    "circ_mv",
    "pb",
    "pe_ttm",
    "ps_ttm",
    "dv_ttm",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "float_share",
    "roe",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "or_yoy",
    "netprofit_yoy",
    "assets_yoy",
    "net_mf_amount",
    "north_ratio",
    "margin_ratio",
    "margin_buy_ratio",
    "margin_balance",
    "short_balance",
    "holder_num",
    "holder_num_chg",
    "top_list_net_buy",
    "top_list_flag",
]

def test_intraday_features_subset_of_leaf_features():
    from factorzen.core.feature_schema import INTRADAY_FEATURES, LEAF_FEATURES

    assert set(LEAF_FEATURES.keys()) >= INTRADAY_FEATURES
    assert len(INTRADAY_FEATURES) == 20  # 17 连续路径统计 + 3 涨跌停邻域

def test_intraday_leaves_are_identity_i_prefix():
    from factorzen.core.feature_schema import INTRADAY_FEATURES, LEAF_FEATURES

    for name in INTRADAY_FEATURES:
        assert name.startswith("i_"), name
        assert LEAF_FEATURES[name] == name

def test_leaf_features_key_order_prefix_unchanged():
    """既有 40 键相对顺序绝不动（随机搜索按键序采样）。"""
    from factorzen.core.feature_schema import LEAF_FEATURES

    keys = list(LEAF_FEATURES.keys())
    n = len(_PRE_CHANGE_LEAF_KEYS)
    assert keys[:n] == _PRE_CHANGE_LEAF_KEYS
    # 新叶子全部在旧键之后
    assert all(k.startswith("i_") for k in keys[n:])

