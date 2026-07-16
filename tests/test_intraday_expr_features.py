"""tests/test_intraday_expr_features.py — bar 级表达式求值、筛选、注册表。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

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


def _dt(h: int, m: int, day: int = 2) -> datetime:
    return datetime(2024, 1, day, h, m, 0)


def _sparse_two_stocks() -> pl.DataFrame:
    """2 股 1 日稀疏 1min 帧（与 battery ground-truth 同桶序列）。"""
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
