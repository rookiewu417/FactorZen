"""series_lift_stats 单序列内核 + daily_residual_rank_ic 逐日残差 IC 序列。

TDD 夹具：期望值手算或独立公式给出，禁止用生产代码互证恒真。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

# ── helpers ──────────────────────────────────────────────────────────────────


def _ic_df(dates, ics) -> pl.DataFrame:
    return pl.DataFrame(
        {"trade_date": list(dates), "ic": list(ics)},
        schema={"trade_date": pl.Utf8, "ic": pl.Float64},
    )


def _empty_ic() -> pl.DataFrame:
    return pl.DataFrame(schema={"trade_date": pl.Utf8, "ic": pl.Float64})


def _codes(n: int = 45) -> list[str]:
    return [f"{600000 + i:06d}.SH" for i in range(n)]


def _panel_long(
    M: np.ndarray,
    dates: list,
    codes: list,
    *,
    col: str = "factor_value",
) -> pl.DataFrame:
    """M: (n_dates, n_stocks) → long panel；非有限 → null。"""
    rows = []
    for i, d in enumerate(dates):
        for j, c in enumerate(codes):
            v = float(M[i, j])
            rows.append({
                "trade_date": d,
                "ts_code": c,
                col: None if not np.isfinite(v) else v,
            })
    return pl.DataFrame(rows)


def _independent_daily_residual_ic(
    candidate: pl.DataFrame,
    lib_panel,
    fwd_returns: pl.DataFrame,
    *,
    ret_col: str = "fwd_ret_1d",
) -> dict[str, float]:
    """独立逐日残差 IC：lstsq([1|X_zscored], y) + spearman_avg_rank。

    设计矩阵直接取 ``lib_panel.X``（已截面 zscore + null→0），不经生产残差路径。
    """
    from factorzen.core.stats import spearman_avg_rank
    from factorzen.discovery.residual import _day_min_samples
    from factorzen.discovery.scoring import _align_join_key

    cand = candidate.with_columns(pl.col("factor_value").fill_nan(None)).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )
    fwd_sel = fwd_returns.select(["trade_date", "ts_code", ret_col])
    fwd_sel = _align_join_key(fwd_sel, "ts_code", cand)
    joined = cand.join(
        fwd_sel, on=["trade_date", "ts_code"], how="inner",
    ).filter(pl.col(ret_col).is_not_null() & pl.col(ret_col).is_finite())

    min_n = _day_min_samples(lib_panel.k)
    out: dict[str, float] = {}
    for date, day_df in joined.group_by("trade_date", maintain_order=True):
        d = date[0] if isinstance(date, tuple) else date
        di = lib_panel.date_idx.get(d)
        if di is None:
            continue
        codes = day_df["ts_code"].to_list()
        y = day_df["factor_value"].to_numpy().astype(np.float64, copy=False)
        ret = day_df[ret_col].to_numpy().astype(np.float64, copy=False)
        si = np.fromiter(
            (lib_panel.stock_idx.get(c, -1) for c in codes),
            dtype=np.int64,
            count=len(codes),
        )
        valid = si >= 0
        if int(valid.sum()) < min_n:
            continue
        si_v = si[valid]
        y_v = y[valid]
        ret_v = ret[valid]
        if y_v.shape[0] < min_n:
            continue
        X_day = np.asarray(lib_panel.X[di, si_v, :], dtype=np.float64)
        n = y_v.shape[0]
        A = np.column_stack([np.ones(n, dtype=np.float64), X_day])
        beta, *_ = np.linalg.lstsq(A, y_v, rcond=None)
        resid = y_v - A @ beta
        ic = spearman_avg_rank(resid, ret_v)
        if ic is not None:
            # 契约锚定：生产日序列的 trade_date 形态是 ISO YYYY-MM-DD
            # （与 cli._lift_admission_str / 库内 scored_* 一致）。此处**故意**
            # 写字面量而非导入 core.dates，生产若漂移本测试须报警。
            if hasattr(d, "strftime"):
                key = d.strftime("%Y-%m-%d")
            else:
                key = str(d)
            out[key] = float(ic)
    return out


# ── 1. series_lift_stats ground truth ────────────────────────────────────────


def test_series_lift_stats_ground_truth_6day_block2():
    """6 日序列 block_days=2：手算块均值 / lift / SE / 半段。"""
    from factorzen.discovery.lift_test import series_lift_stats

    # 手算：
    # ics = [0.01, 0.03, -0.02, 0.04, 0.00, 0.02]
    # 块均值 (bd=2): [0.02, 0.01, 0.01]；n_blocks=3
    # lift = mean(ics) = 0.08/6 = 0.01333...
    # SE = std([0.02,0.01,0.01], ddof=1) / √3
    # mid=(3+1)//2=2 → 前半 4 日 mean=0.015，后半 2 日 mean=0.01
    ics = [0.01, 0.03, -0.02, 0.04, 0.00, 0.02]
    dates = [f"2024010{i}" for i in range(1, 7)]
    stats = series_lift_stats(_ic_df(dates, ics), block_days=2)

    expected_lift = 0.013333333333333334
    expected_se = 0.0033333333333333335
    expected_first = 0.015
    expected_second = 0.01

    assert stats["n_days"] == 6
    assert stats["n_blocks"] == 3
    assert abs(stats["lift"] - expected_lift) < 1e-12
    assert abs(stats["lift_se"] - expected_se) < 1e-12
    assert abs(stats["lift_first_half"] - expected_first) < 1e-12
    assert abs(stats["lift_second_half"] - expected_second) < 1e-12


# ── 2. paired ≡ series(diff) ─────────────────────────────────────────────────


def test_paired_lift_stats_equals_series_on_diff():
    """随机 cand/base：paired 与测试自构造 diff 帧上的 series 六键相等。"""
    from factorzen.discovery.lift_test import paired_lift_stats, series_lift_stats

    rng = np.random.default_rng(42)
    n = 47
    dates = [f"d{i:04d}" for i in range(n)]
    cand_ics = rng.normal(0.01, 0.02, size=n).tolist()
    base_ics = rng.normal(0.005, 0.015, size=n).tolist()
    # 故意错开几天，验证 inner join 语义
    cand = _ic_df(dates, cand_ics)
    base = _ic_df(dates[2:], base_ics[2:])  # 缺 d0000,d0001

    paired = paired_lift_stats(cand, base, block_days=10)

    # 测试侧独立 join/相减，不经生产代码
    c = cand.select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("ic").alias("cand_ic"),
    )
    b = base.select(
        pl.col("trade_date").cast(pl.Utf8),
        pl.col("ic").alias("base_ic"),
    )
    joined = c.join(b, on="trade_date", how="inner").sort("trade_date")
    diff_df = joined.select(
        pl.col("trade_date"),
        (pl.col("cand_ic") - pl.col("base_ic")).alias("ic"),
    )
    series = series_lift_stats(diff_df, block_days=10)

    for key in (
        "lift",
        "lift_se",
        "n_blocks",
        "n_days",
        "lift_first_half",
        "lift_second_half",
    ):
        a, b_ = paired[key], series[key]
        if a is None or b_ is None:
            assert a is b_, f"{key}: paired={a!r} series={b_!r}"
        else:
            assert abs(float(a) - float(b_)) < 1e-15, f"{key}: {a} vs {b_}"


# ── 3. 边界 ──────────────────────────────────────────────────────────────────


def test_series_lift_stats_empty_frame():
    from factorzen.discovery.lift_test import series_lift_stats

    empty = series_lift_stats(_empty_ic(), block_days=20)
    assert empty == {
        "lift": None,
        "lift_se": None,
        "n_blocks": 0,
        "n_days": 0,
        "lift_first_half": None,
        "lift_second_half": None,
    }
    assert series_lift_stats(None, block_days=20) == empty  # type: ignore[arg-type]


def test_series_lift_stats_single_block_se_none():
    """n_blocks=1 → SE=None；半段：中位归前半 → second=None。"""
    from factorzen.discovery.lift_test import series_lift_stats

    ics = [0.02, 0.04, -0.01]
    dates = ["20240101", "20240102", "20240103"]
    stats = series_lift_stats(_ic_df(dates, ics), block_days=20)
    assert stats["n_blocks"] == 1
    assert stats["n_days"] == 3
    assert stats["lift_se"] is None
    assert abs(stats["lift"] - float(np.mean(ics))) < 1e-12
    assert abs(stats["lift_first_half"] - float(np.mean(ics))) < 1e-12
    assert stats["lift_second_half"] is None


def test_series_lift_stats_all_zero_guard():
    """全零：lift=0.0、SE=None、半段仍按块切分非 None（有两半时）。"""
    from factorzen.discovery.lift_test import series_lift_stats

    ics = [0.0] * 6
    dates = [f"2024010{i}" for i in range(1, 7)]
    stats = series_lift_stats(_ic_df(dates, ics), block_days=2)
    assert stats["lift"] == 0.0
    assert stats["lift_se"] is None
    assert stats["n_blocks"] == 3
    assert stats["n_days"] == 6
    # mid=2 → 前 4 日、后 2 日均值均为 0.0
    assert stats["lift_first_half"] == 0.0
    assert stats["lift_second_half"] == 0.0


def test_series_lift_stats_unsorted_dates_same_as_sorted():
    """乱序输入与排序后结果一致。"""
    from factorzen.discovery.lift_test import series_lift_stats

    ics = [0.01, 0.03, -0.02, 0.04, 0.00, 0.02]
    dates = [f"2024010{i}" for i in range(1, 7)]
    ordered = series_lift_stats(_ic_df(dates, ics), block_days=2)

    perm = [4, 0, 5, 2, 1, 3]
    shuffled_dates = [dates[i] for i in perm]
    shuffled_ics = [ics[i] for i in perm]
    shuffled = series_lift_stats(_ic_df(shuffled_dates, shuffled_ics), block_days=2)

    for key in (
        "lift",
        "lift_se",
        "n_blocks",
        "n_days",
        "lift_first_half",
        "lift_second_half",
    ):
        a, b = ordered[key], shuffled[key]
        if a is None or b is None:
            assert a is b
        else:
            assert abs(float(a) - float(b)) < 1e-15


# ── 4. daily_residual_rank_ic 独立验证 ───────────────────────────────────────


def test_daily_residual_rank_ic_matches_independent_lstsq():
    """k=1、45 股×3 日：候选=库线性组合+正交分量；独立 lstsq+spearman 逐日对齐。"""
    from factorzen.discovery.residual import (
        build_library_panel,
        daily_residual_rank_ic,
    )

    rng = np.random.default_rng(7)
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
    codes = _codes(45)
    n_d, n_s = 3, 45

    # 库因子 + 与之正交的独立 alpha（Gram-Schmidt 近似：减投影后加噪声）
    f1 = rng.normal(0, 1, size=(n_d, n_s))
    alpha_raw = rng.normal(0, 1, size=(n_d, n_s))
    alpha = np.empty_like(alpha_raw)
    for i in range(n_d):
        # 截面去与 f1 的线性相关，保留正交增量
        x = f1[i]
        a = alpha_raw[i]
        coef = float(np.dot(a, x) / np.dot(x, x))
        alpha[i] = a - coef * x

    # 候选 = 0.5*库 + 正交分量
    cand_m = 0.5 * f1 + alpha
    # 收益跟正交分量走 → 残差 IC 应显著非零
    noise = rng.normal(0, 0.3, size=(n_d, n_s))
    fwd_m = alpha + noise

    lib_pool = {"lib_f1": _panel_long(f1, dates, codes)}
    panel = build_library_panel(lib_pool)
    assert panel is not None and panel.k == 1

    cand = _panel_long(cand_m, dates, codes)
    fwd = _panel_long(fwd_m, dates, codes, col="fwd_ret_1d")

    got = daily_residual_rank_ic(cand, panel, fwd)
    assert got.columns == ["trade_date", "ic"]
    assert got["trade_date"].dtype == pl.Utf8
    assert got.height == 3
    # 升序
    assert got["trade_date"].to_list() == sorted(got["trade_date"].to_list())

    expected = _independent_daily_residual_ic(cand, panel, fwd)
    assert set(expected) == set(got["trade_date"].to_list())
    for row in got.iter_rows(named=True):
        exp = expected[row["trade_date"]]
        assert abs(row["ic"] - exp) < 1e-12, (
            f"{row['trade_date']}: got={row['ic']} expected={exp}"
        )


# ── 5. start/end 裁剪 ────────────────────────────────────────────────────────


def test_daily_residual_rank_ic_start_end_window():
    """3 日裁中间 1 日闭区间窗 → 仅剩该日。"""
    from factorzen.discovery.residual import (
        build_library_panel,
        daily_residual_rank_ic,
    )

    rng = np.random.default_rng(11)
    dates = [dt.date(2024, 2, 5), dt.date(2024, 2, 6), dt.date(2024, 2, 7)]
    codes = _codes(45)
    f1 = rng.normal(0, 1, size=(3, 45))
    cand_m = f1 + rng.normal(0, 0.5, size=(3, 45))
    fwd_m = cand_m + rng.normal(0, 0.2, size=(3, 45))

    panel = build_library_panel({"lib_f1": _panel_long(f1, dates, codes)})
    assert panel is not None
    cand = _panel_long(cand_m, dates, codes)
    fwd = _panel_long(fwd_m, dates, codes, col="fwd_ret_1d")

    # 窗界两种形态等价（生产窗串是 ISO，历史调用方可能传紧凑）
    iso = daily_residual_rank_ic(
        cand, panel, fwd, start="2024-02-06", end="2024-02-06",
    )
    compact = daily_residual_rank_ic(
        cand, panel, fwd, start="20240206", end="20240206",
    )
    assert iso.height == 1
    assert compact.height == 1
    assert iso["ic"].to_list() == compact["ic"].to_list()
    # 输出形态锚定 ISO
    assert iso["trade_date"].to_list() == ["2024-02-06"]


# ── 6. compute_residual_ic ≡ mean(daily_residual_rank_ic) ────────────────────


def test_compute_residual_ic_matches_daily_mean():
    """compute_residual_ic 的 (ic_mean, n_days) == daily 帧均值与行数。"""
    from factorzen.discovery.residual import (
        build_library_panel,
        compute_residual_ic,
        daily_residual_rank_ic,
    )

    rng = np.random.default_rng(99)
    dates = [dt.date(2024, 3, i) for i in (1, 4, 5, 6, 7)]  # 跳过周末
    codes = _codes(50)
    n_d, n_s = len(dates), 50
    f1 = rng.normal(0, 1, size=(n_d, n_s))
    alpha = rng.normal(0, 1, size=(n_d, n_s))
    cand_m = 0.4 * f1 + 0.6 * alpha
    fwd_m = alpha + rng.normal(0, 0.25, size=(n_d, n_s))

    panel = build_library_panel({"lib_f1": _panel_long(f1, dates, codes)})
    assert panel is not None
    cand = _panel_long(cand_m, dates, codes)
    fwd = _panel_long(fwd_m, dates, codes, col="fwd_ret_1d")

    daily = daily_residual_rank_ic(cand, panel, fwd)
    res = compute_residual_ic(cand, panel, fwd)

    assert res.n_days == daily.height
    if daily.height == 0:
        assert res.ic_mean != res.ic_mean  # NaN
    else:
        expected_mean = float(daily["ic"].mean())
        assert abs(res.ic_mean - expected_mean) < 1e-12
