"""tests/test_intraday_engine.py — build_intraday_features 端到端与 manifest 守卫。"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from factorzen.core.storage import partition_exists, save_parquet
from factorzen.intraday.features import (
    battery,
    battery_hash,
    build_intraday_features,
    read_manifest,
)


def _make_day_bars(
    code: str,
    day: datetime,
    *,
    n: int = 20,
    base_px: float = 10.0,
) -> pl.DataFrame:
    """合成约 n 根 1min bar（分散在 session 内）。"""
    # 选取 canonical 分钟：09:30 起连续 + 下午
    slots: list[tuple[int, int]] = [(9, 30)]
    for i in range(1, n):
        # 09:31.. 向前铺，跳过午休
        idx = i  # 近似 session index
        if idx <= 120:
            tod = 570 + idx  # 09:30 + idx
            h, m = divmod(tod, 60)
        else:
            tod = 780 + (idx - 120)
            h, m = divmod(tod, 60)
        if h > 15 or (h == 15 and m > 0):
            break
        if h == 12 or (h == 11 and m > 30):
            continue
        slots.append((h, m))
    # 保证有 15:00
    if (15, 0) not in slots:
        slots.append((15, 0))
    slots = slots[:n]
    if (15, 0) not in slots:
        slots[-1] = (15, 0)

    rows_t = [day.replace(hour=h, minute=m, second=0, microsecond=0) for h, m in slots]
    px = [base_px + 0.01 * i for i in range(len(rows_t))]
    return pl.DataFrame(
        {
            "ts_code": [code] * len(rows_t),
            "trade_time": pl.Series(rows_t, dtype=pl.Datetime("us")),
            "open": px,
            "high": [p + 0.05 for p in px],
            "low": [p - 0.05 for p in px],
            "close": [p + 0.02 for p in px],
            "vol": pl.Series([100 + i * 10 for i in range(len(rows_t))], dtype=pl.Int64),
            "amount": [1000.0 + i * 100 for i in range(len(rows_t))],
        }
    )


def _build_mini_source(tmp: Path) -> None:
    """2 个月、2 股、每月 2~3 日、每日 ~20 根 1min bar。"""
    frames: list[pl.DataFrame] = []
    # 2024-01: 2 日
    for d in (2, 3):
        for code, px in (("000001.SZ", 10.0), ("000002.SZ", 20.0)):
            frames.append(
                _make_day_bars(code, datetime(2024, 1, d), n=20, base_px=px)
            )
    # 2024-02: 3 日
    for d in (1, 2, 5):
        for code, px in (("000001.SZ", 10.5), ("000002.SZ", 20.5)):
            frames.append(
                _make_day_bars(code, datetime(2024, 2, d), n=20, base_px=px)
            )
    minute = pl.concat(frames)
    save_parquet(
        minute,
        data_type="minute_1min",
        date_col="trade_time",
        base_dir=tmp,
        mode="overwrite",
    )


class TestBuildIntradayFeatures:
    def test_end_to_end_layout_schema_manifest(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)

        report = build_intraday_features(
            "20240101",
            "20240229",
            freq="5min",
            version="v1",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        assert report.months == ["2024-01", "2024-02"]
        assert report.rows > 0
        assert report.n_stocks == 2

        # 分区布局
        assert (out / "v1" / "5min" / "year=2024" / "month=01" / "data.parquet").exists()
        assert (out / "v1" / "5min" / "year=2024" / "month=02" / "data.parquet").exists()
        assert partition_exists("v1/5min", 2024, 1, base_dir=out)
        assert partition_exists("v1/5min", 2024, 2, base_dir=out)

        # schema
        panel = pl.read_parquet(out / "v1" / "5min" / "year=2024" / "month=01" / "data.parquet")
        assert panel["trade_date"].dtype == pl.Date
        specs = battery("v1", "5min")
        for s in specs:
            assert s.name in panel.columns
            assert panel[s.name].dtype == pl.Float64
        assert panel["ts_code"].dtype == pl.String

        # manifest
        m = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m is not None
        assert m["version"] == "v1"
        assert m["freq"] == "5min"
        assert m["battery_hash"] == battery_hash(specs)
        assert len(m["features"]) == 20  # 17 连续路径统计 + 3 涨跌停邻域
        assert m["source"] == "minute_1min"
        assert m["bar_label_convention"] == "end"
        assert m["session_policy"] == "regular_only_drop_after_1500"
        assert m["units"] == {"vol": "share", "amount": "cny"}
        assert "built_at" in m
        assert set(m["coverage"]["months"]) == {"2024-01", "2024-02"}
        assert m["rows_total"] == report.rows

    def test_idempotent_rerun(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        r1 = build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        r2 = build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        assert r1.rows == r2.rows
        p = pl.read_parquet(out / "v1" / "5min" / "year=2024" / "month=01" / "data.parquet")
        # overwrite 模式，行数不翻倍
        assert p.height == 4  # 2 股 × 2 日

    def test_hash_mismatch_raises_without_overwrite(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        build_intraday_features(
            "20240101",
            "20240131",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        mpath = out / "v1" / "5min" / "manifest.json"
        payload = json.loads(mpath.read_text(encoding="utf-8"))
        payload["battery_hash"] = "deadbeefdeadbeef"
        mpath.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="battery_hash"):
            build_intraday_features(
                "20240101",
                "20240131",
                source_dir=src,
                out_dir=out,
                min_bar_coverage=0.0,
                overwrite=False,
            )

        # overwrite=True 可重写
        r = build_intraday_features(
            "20240101",
            "20240131",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
            overwrite=True,
        )
        assert r.rows > 0
        m2 = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m2 is not None
        assert m2["battery_hash"] == battery_hash(battery("v1", "5min"))

    def test_codes_filter(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        r = build_intraday_features(
            "20240101",
            "20240131",
            source_dir=src,
            out_dir=out,
            codes=["000001.SZ"],
            min_bar_coverage=0.0,
        )
        assert r.n_stocks == 1
        p = pl.read_parquet(out / "v1" / "5min" / "year=2024" / "month=01" / "data.parquet")
        assert set(p["ts_code"].to_list()) == {"000001.SZ"}

    def test_empty_month_skipped(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        # 仅 1 月有数据
        frames = [
            _make_day_bars("000001.SZ", datetime(2024, 1, 2), n=20),
        ]
        save_parquet(
            pl.concat(frames),
            data_type="minute_1min",
            date_col="trade_time",
            base_dir=src,
            mode="overwrite",
        )
        r = build_intraday_features(
            "20240101",
            "20240331",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        assert "2024-01" in r.months
        assert "2024-02" not in r.months
        assert "2024-03" not in r.months

    def test_covered_month_skipped_on_rerun(self, tmp_path: Path) -> None:
        """已覆盖月（分区非空 + coverage + battery_hash）二次 build 跳过重算。"""
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        r1 = build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        assert set(r1.months) == {"2024-01", "2024-02"}
        m1 = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m1 is not None
        built_at_1 = m1["built_at"]
        rows_1 = m1["rows_total"]
        jan_path = out / "v1" / "5min" / "year=2024" / "month=01" / "data.parquet"
        jan_mtime_1 = jan_path.stat().st_mtime_ns
        jan_bytes_1 = jan_path.read_bytes()

        r2 = build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        # 全部跳过：本 run 未处理任何月
        assert r2.months == []
        m2 = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m2 is not None
        # manifest 覆盖与行数不被破坏；built_at 可更新
        assert set(m2["coverage"]["months"]) == {"2024-01", "2024-02"}
        assert m2["rows_total"] == rows_1
        assert m2["battery_hash"] == m1["battery_hash"]
        assert m2["built_at"] != built_at_1 or m2["built_at"] == built_at_1
        # 分区文件未重写
        assert jan_path.stat().st_mtime_ns == jan_mtime_1
        assert jan_path.read_bytes() == jan_bytes_1

    def test_partially_covered_boundary_month_recomputed(self, tmp_path: Path) -> None:
        """上游在边界月内补数后，增量 build 必须重算该月而非跳过。

        回归锚（2026-07-19 实证）：月标签级 coverage 区分不了「整月已算」与
        「算了前几天」。生产上特征面板停在 2026-04-10，而 ``status`` 把
        2026-04 标为已覆盖，增量 build 跳过它，只有 ``--force`` 才补得上——
        用户以为补齐了，实际 17 个分钟叶子在 holdout 段覆盖 0% 被全摘。
        """
        src = tmp_path / "src"
        out = tmp_path / "out"
        # ① 源湖此刻只有 2024-02 的前 2 天（模拟上游数据尚未到月末）
        frames = [
            _make_day_bars(code, datetime(2024, 2, d), n=20, base_px=px)
            for d in (1, 2)
            for code, px in (("000001.SZ", 10.5), ("000002.SZ", 20.5))
        ]
        save_parquet(
            pl.concat(frames), data_type="minute_1min", date_col="trade_time",
            base_dir=src, mode="overwrite",
        )
        r1 = build_intraday_features(
            "20240201", "20240229", source_dir=src, out_dir=out, min_bar_coverage=0.0,
        )
        assert r1.months == ["2024-02"]
        m1 = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m1 is not None
        # 逐月末日被记录下来，这是判定部分覆盖的唯一依据
        assert m1["coverage"]["month_last_date"]["2024-02"] == "2024-02-02"

        # ② 上游补进 02-05（同月，月标签不变）
        frames2 = [
            _make_day_bars(code, datetime(2024, 2, 5), n=20, base_px=px)
            for code, px in (("000001.SZ", 10.5), ("000002.SZ", 20.5))
        ]
        save_parquet(
            pl.concat(frames2), data_type="minute_1min", date_col="trade_time",
            base_dir=src, mode="append",
        )

        # ③ 同样区间再 build：不得跳过（源末日 02-05 > 记录末日 02-02）
        r2 = build_intraday_features(
            "20240201", "20240229", source_dir=src, out_dir=out, min_bar_coverage=0.0,
        )
        assert r2.months == ["2024-02"], "边界月被误判为完整覆盖而跳过"
        m2 = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m2 is not None
        assert m2["coverage"]["month_last_date"]["2024-02"] == "2024-02-05"
        # 分区真的补上了 02-05
        got = pl.read_parquet(
            out / "v1" / "5min" / "year=2024" / "month=02" / "data.parquet"
        )
        assert got["trade_date"].max() == date(2024, 2, 5)

    def test_legacy_manifest_without_month_last_date_recomputes_boundary_only(
        self, tmp_path: Path
    ) -> None:
        """老 manifest 缺逐月末日字段 → 只重算边界月，历史月仍跳过（迁移代价 O(1)）。"""
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        build_intraday_features(
            "20240101", "20240229", source_dir=src, out_dir=out, min_bar_coverage=0.0,
        )
        # 抹掉新字段，伪造升级前的 manifest
        mpath = out / "v1" / "5min" / "manifest.json"
        payload = json.loads(mpath.read_text())
        payload["coverage"].pop("month_last_date", None)
        mpath.write_text(json.dumps(payload))

        r = build_intraday_features(
            "20240101", "20240229", source_dir=src, out_dir=out, min_bar_coverage=0.0,
        )
        # 只有边界月 2024-02 重算，2024-01 仍跳过
        assert r.months == ["2024-02"], r.months

    def test_missing_month_still_computed(self, tmp_path: Path) -> None:
        """coverage 缺月 → 仅补算缺月，已覆盖月跳过。"""
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        build_intraday_features(
            "20240101",
            "20240131",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        jan_path = out / "v1" / "5min" / "year=2024" / "month=01" / "data.parquet"
        jan_mtime = jan_path.stat().st_mtime_ns

        r = build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        assert r.months == ["2024-02"]
        assert jan_path.stat().st_mtime_ns == jan_mtime
        m = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m is not None
        assert set(m["coverage"]["months"]) == {"2024-01", "2024-02"}
        assert partition_exists("v1/5min", 2024, 2, base_dir=out)

    def test_partition_missing_recomputes_even_if_in_coverage(
        self, tmp_path: Path
    ) -> None:
        """coverage 有月但分区文件缺失 → 不跳过，补算。"""
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        build_intraday_features(
            "20240101",
            "20240131",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        jan_path = out / "v1" / "5min" / "year=2024" / "month=01" / "data.parquet"
        jan_path.unlink()

        r = build_intraday_features(
            "20240101",
            "20240131",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        assert r.months == ["2024-01"]
        assert jan_path.exists()

    def test_hash_mismatch_still_fail_loudly_without_overwrite(
        self, tmp_path: Path
    ) -> None:
        """battery_hash 变更：overwrite=False 仍 fail-loudly（与既有语义一致）。"""
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        build_intraday_features(
            "20240101",
            "20240131",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        mpath = out / "v1" / "5min" / "manifest.json"
        payload = json.loads(mpath.read_text(encoding="utf-8"))
        payload["battery_hash"] = "deadbeefdeadbeef"
        mpath.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="battery_hash"):
            build_intraday_features(
                "20240101",
                "20240131",
                source_dir=src,
                out_dir=out,
                min_bar_coverage=0.0,
                force=False,
                overwrite=False,
            )

    def test_force_recomputes_all_covered_months(self, tmp_path: Path) -> None:
        """--force 即使已覆盖也全量重算。"""
        src = tmp_path / "src"
        out = tmp_path / "out"
        _build_mini_source(src)
        build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
        )
        jan_path = out / "v1" / "5min" / "year=2024" / "month=01" / "data.parquet"
        m1 = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m1 is not None
        built_at_before = m1["built_at"]

        r = build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out,
            min_bar_coverage=0.0,
            force=True,
        )
        assert set(r.months) == {"2024-01", "2024-02"}
        assert r.rows > 0
        assert jan_path.exists()
        m2 = read_manifest(version="v1", freq="5min", base_dir=out)
        assert m2 is not None
        assert m2["built_at"] != built_at_before

    def test_workers_two_matches_serial(self, tmp_path: Path) -> None:
        """workers=2 与 workers=1 对同两月输出逐值一致，manifest coverage 一致。"""
        src = tmp_path / "src"
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        _build_mini_source(src)

        r1 = build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out1,
            min_bar_coverage=0.0,
            workers=1,
        )
        r2 = build_intraday_features(
            "20240101",
            "20240229",
            source_dir=src,
            out_dir=out2,
            min_bar_coverage=0.0,
            workers=2,
        )
        assert r1.months == r2.months == ["2024-01", "2024-02"]
        assert r1.rows == r2.rows
        assert r1.n_stocks == r2.n_stocks

        for month in ("01", "02"):
            p1 = pl.read_parquet(
                out1 / "v1" / "5min" / "year=2024" / f"month={month}" / "data.parquet"
            ).sort(["trade_date", "ts_code"])
            p2 = pl.read_parquet(
                out2 / "v1" / "5min" / "year=2024" / f"month={month}" / "data.parquet"
            ).sort(["trade_date", "ts_code"])
            assert p1.columns == p2.columns
            assert p1.shape == p2.shape
            for col in p1.columns:
                if p1[col].dtype == pl.Float64:
                    a = p1[col].to_numpy()
                    b = p2[col].to_numpy()
                    # 允许两边同为 NaN
                    import numpy as np

                    assert np.allclose(a, b, equal_nan=True, atol=1e-12, rtol=1e-12), col
                else:
                    assert p1[col].to_list() == p2[col].to_list(), col

        m1 = read_manifest(version="v1", freq="5min", base_dir=out1)
        m2 = read_manifest(version="v1", freq="5min", base_dir=out2)
        assert m1 is not None and m2 is not None
        assert m1["coverage"]["months"] == m2["coverage"]["months"]
        assert m1["battery_hash"] == m2["battery_hash"]
        assert m1["rows_total"] == m2["rows_total"]


def test_build_passes_limit_ref_to_compute_day_panel(tmp_path, monkeypatch):
    """接线锚：``build_intraday_features`` 必须把 ``daily_ref`` 传给 ``compute_day_panel``。

    能力层↔接线层漂移是本项目头号 bug 源——引擎能算涨跌停叶，但只要月构建不传
    参照，三个 i_limit_up_* 就永远 null，且**不会报错**（设计上参照缺失即 null）。
    """
    import factorzen.intraday.features.engine as eng

    seen: dict = {}
    orig = eng.compute_day_panel

    def _spy(*a, **kw):
        seen["has_ref"] = kw.get("daily_ref") is not None
        return orig(*a, **kw)

    monkeypatch.setattr(eng, "compute_day_panel", _spy)
    monkeypatch.setattr(
        eng, "_load_limit_ref",
        lambda *a, **kw: pl.DataFrame({
            "ts_code": ["000001.SZ"], "trade_date": [date(2024, 1, 2)],
            "pre_close": [10.0], "limit_pct": [0.1],
        }),
    )
    src = tmp_path / "src"
    out = tmp_path / "out"
    _build_mini_source(src)
    eng.build_intraday_features(
        "20240101", "20240131", freq="5min", version="v1",
        source_dir=src, out_dir=out, min_bar_coverage=0.0,
    )
    assert seen.get("has_ref") is True, "月构建没把 daily_ref 传给 compute_day_panel"
