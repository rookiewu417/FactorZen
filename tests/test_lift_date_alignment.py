"""lift 链路 trade_date 形态对齐回归。

根因（2026-07-18 实证）：``_daily_oos_rank_ic`` 对 ``pl.Date`` 走
``strftime("%Y%m%d")``，而 ``_build_ret_panel`` 把收益侧 ``cast(pl.Utf8)``
得到 ISO ``YYYY-MM-DD`` —— 两侧形态不同，join **零命中** → 空 IC 序列。

真实后果（库内仅有的 2 条 lift 轨记录实测 ``admission_ic == 0.0``）：
``_mean_ic`` 对空帧返回 0.0（非 None）→ 写进库 → ``forward_track`` 判
``admission_ic is not None`` 成立、**不回退** ``ic_train`` →
``_sign_from_ic_train(0.0)`` 返回 None → ``missing_sign`` → probation 因子
永远 hold，既不能升 active 也不能降 no_lift。

admission 窗（``cli._lift_admission_str`` 产出 ``YYYY-MM-DD``）与 compact
``YYYYMMDD`` 混比同样静默错行：``"20260405" > "2026-04-10"`` 逐字符比较为真。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def _codes(n: int = 12) -> list[str]:
    return [f"{600000 + i:06d}.SH" for i in range(n)]


def _panels(dates: list, *, col: str = "factor_value", value_col: str = "ret"):
    """构造 (因子面板, 收益面板)；因子值与收益单调同向 → 每日 IC 恒为 +1。"""
    codes = _codes()
    fac = pl.DataFrame([
        {"trade_date": d, "ts_code": c, col: float(j)}
        for d in dates for j, c in enumerate(codes)
    ])
    ret = pl.DataFrame([
        {"trade_date": d, "ts_code": c, value_col: float(j) * 0.5}
        for d in dates for j, c in enumerate(codes)
    ])
    return fac, ret


# ── 1. join 形态对齐：Date 候选面板不得被静默丢空 ────────────────────────────


def test_daily_oos_rank_ic_date_candidate_joins_utf8_returns():
    """候选 pl.Date × 收益 Utf8(ISO)——生产真实组合，必须匹配上。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7)]
    fac, ret = _panels(days)
    # 复刻 _build_ret_panel：收益侧显式 cast Utf8
    ret = ret.with_columns(pl.col("trade_date").cast(pl.Utf8))
    assert fac.schema["trade_date"] == pl.Date

    out = _daily_oos_rank_ic(fac, ret)
    assert out.height == 2, f"Date 候选面板被静默丢空: {out}"
    # 因子与收益严格同序 → 每日 spearman = 1.0
    assert all(abs(v - 1.0) < 1e-12 for v in out["ic"].to_list())


def test_daily_oos_rank_ic_both_date_panels():
    """两侧都 pl.Date 也必须匹配（cast 后形态须一致）。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7)]
    fac, ret = _panels(days)
    out = _daily_oos_rank_ic(fac, ret)
    assert out.height == 2, f"两侧 Date 被静默丢空: {out}"


# ── 2. admission 窗形态：生产窗串必须真的裁到正确日集 ────────────────────────


def test_admission_window_accepts_production_iso_bounds():
    """窗界用 cli._lift_admission_str 的真实产出（YYYY-MM-DD）。"""
    from factorzen.cli.main import _lift_admission_str
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7), dt.date(2026, 4, 15)]
    fac, ret = _panels(days)
    ret = ret.with_columns(pl.col("trade_date").cast(pl.Utf8))

    end = _lift_admission_str(dt.date(2026, 4, 10))
    assert end == "2026-04-10"  # 契约锚定：变了说明上游改了形态

    out = _daily_oos_rank_ic(fac, ret, end=end)
    # 4/5 与 4/7 在窗内，4/15 在窗外
    assert out.height == 2, f"窗内日被误裁: {out}"

    start = _lift_admission_str(dt.date(2026, 4, 6))
    out2 = _daily_oos_rank_ic(fac, ret, start=start, end=end)
    assert out2.height == 1, f"闭区间窗错行: {out2}"


def test_admission_window_accepts_compact_bounds_equivalently():
    """紧凑 YYYYMMDD 与带横杠 ISO 必须裁出同一日集（形态无关）。"""
    from factorzen.discovery.lift_test import _daily_oos_rank_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7), dt.date(2026, 4, 15)]
    fac, ret = _panels(days)
    ret = ret.with_columns(pl.col("trade_date").cast(pl.Utf8))

    iso = _daily_oos_rank_ic(fac, ret, end="2026-04-10")["ic"].to_list()
    compact = _daily_oos_rank_ic(fac, ret, end="20260410")["ic"].to_list()
    assert iso == compact and len(iso) == 2


# ── 3. 残差侧同契约 ──────────────────────────────────────────────────────────


def test_daily_residual_rank_ic_window_format_agnostic():
    """残差日序列的窗过滤对两种日期形态等价，且输出 ISO。"""
    from factorzen.discovery.residual import (
        build_library_panel,
        daily_residual_rank_ic,
    )

    rng = np.random.default_rng(7)
    days = [dt.date(2024, 2, 5), dt.date(2024, 2, 6), dt.date(2024, 2, 7)]
    codes = _codes(45)
    lib_m = rng.normal(0, 1, size=(3, 45))
    cand_m = lib_m + rng.normal(0, 0.8, size=(3, 45))
    fwd_m = cand_m + rng.normal(0, 0.3, size=(3, 45))

    def _long(M, col):
        return pl.DataFrame([
            {"trade_date": d, "ts_code": c, col: float(M[i, j])}
            for i, d in enumerate(days) for j, c in enumerate(codes)
        ])

    panel = build_library_panel({"lib": _long(lib_m, "factor_value")})
    assert panel is not None
    cand = _long(cand_m, "factor_value")
    fwd = _long(fwd_m, "fwd_ret_1d")

    iso = daily_residual_rank_ic(
        cand, panel, fwd, start="2024-02-06", end="2024-02-06",
    )
    compact = daily_residual_rank_ic(
        cand, panel, fwd, start="20240206", end="20240206",
    )
    assert iso.height == 1, f"ISO 窗裁错: {iso}"
    assert compact.height == 1, f"紧凑窗裁错: {compact}"
    assert iso["ic"].to_list() == compact["ic"].to_list()
    # 输出形态锚定 ISO（与 _lift_admission_str / 库内 scored_* 既有形态一致）
    assert iso["trade_date"].to_list() == ["2024-02-06"]


def test_daily_residual_rank_ic_joins_date_candidate_with_utf8_returns():
    """候选 pl.Date × 收益 Utf8——残差引擎的生产真实组合，不得抛 SchemaError。

    候选面板由 ``_materializer_from_prepped`` 产出（prepped 帧原生 pl.Date），
    收益面板由 ``_build_ret_panel`` 显式 ``cast(pl.Utf8)``。旧实现直接 join
    两个不同 dtype 的键 → ``SchemaError``（2026-07-15 apply 全灭事故同款）。
    """
    from factorzen.discovery.residual import (
        build_library_panel,
        daily_residual_rank_ic,
    )

    rng = np.random.default_rng(3)
    days = [dt.date(2024, 2, 5), dt.date(2024, 2, 6), dt.date(2024, 2, 7)]
    codes = _codes(45)

    def _long(M, col):
        return pl.DataFrame([
            {"trade_date": d, "ts_code": c, col: float(M[i, j])}
            for i, d in enumerate(days) for j, c in enumerate(codes)
        ])

    lib_m = rng.normal(0, 1, size=(3, 45))
    cand_m = lib_m + rng.normal(0, 0.8, size=(3, 45))
    fwd_m = cand_m + rng.normal(0, 0.3, size=(3, 45))

    panel = build_library_panel({"lib": _long(lib_m, "factor_value")})
    assert panel is not None
    cand = _long(cand_m, "factor_value")
    fwd_date = _long(fwd_m, "ret")
    # 复刻 _build_ret_panel：收益侧 cast Utf8
    fwd_utf8 = fwd_date.with_columns(pl.col("trade_date").cast(pl.Utf8))
    assert cand.schema["trade_date"] == pl.Date

    same = daily_residual_rank_ic(cand, panel, fwd_date, ret_col="ret")
    mixed = daily_residual_rank_ic(cand, panel, fwd_utf8, ret_col="ret")
    assert same.height == 3
    assert mixed.height == 3, f"Date×Utf8 未对齐: {mixed}"
    # 形态对齐不得改变数值
    assert same["ic"].to_list() == mixed["ic"].to_list()


# ── 4. 真实后果：admission_ic 不得因形态错配退化成 0.0 ───────────────────────


def test_admission_ic_not_silently_zero_for_date_panels():
    """端到端：pl.Date 候选面板下 admission_ic 必须是真实 IC，不是空帧 0.0。

    这是库内 2 条 lift 轨记录 ``admission_ic == 0.0`` 的直接回归锚。
    """
    from factorzen.discovery.lift_test import _daily_oos_rank_ic, _mean_ic

    days = [dt.date(2026, 4, 5), dt.date(2026, 4, 7)]
    fac, ret = _panels(days)
    ret = ret.with_columns(pl.col("trade_date").cast(pl.Utf8))

    admission_ic = _mean_ic(_daily_oos_rank_ic(fac, ret))
    assert admission_ic != 0.0, "admission_ic 退化为空帧哨兵 0.0（方向权威失效）"
    assert abs(admission_ic - 1.0) < 1e-12
