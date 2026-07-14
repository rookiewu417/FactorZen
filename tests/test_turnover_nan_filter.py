"""compute_turnover 分组前须过滤 NaN/null，防 polars rank 把 NaN 排最大污染分组。

根因：rank("ordinal").over("trade_date") 前未过滤 → NaN 进最高组，并抬高 max_rank，
改变有效股的分位边界，制造虚假换手 / 污染迁移矩阵。

不变量：因子为 NaN/null 的行应与该行被物理删除等价。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from factorzen.daily.evaluation.turnover import compute_turnover


def _stable_panel(n_days: int = 5, n_stocks: int = 10) -> pl.DataFrame:
    """跨日排名完全稳定的截面（无 NaN）。"""
    rows: list[dict] = []
    d0 = date(2024, 1, 1)
    for di in range(n_days):
        d = d0 + timedelta(days=di)
        for si in range(n_stocks):
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": f"{si:06d}.SZ",
                    "factor_clean": float(si),
                }
            )
    return pl.DataFrame(rows)


def test_nan_does_not_pollute_group_boundaries():
    """次日多出 NaN 股会抬高 max_rank：旧实现挤占最高组、压缩有效股分位 → 虚假换手。

    d0/d1：s0..s4 因子稳定 0..4；d1 额外一只 NaN 股。
    正确：过滤 NaN 后截面与 d0 同 → turnover=0。
    旧：NaN 排最大 → max_rank=6 → 有效股 group 边界漂移 → turnover>0。
    """
    d0, d1 = date(2024, 1, 2), date(2024, 1, 3)
    rows: list[dict] = []
    for d in (d0, d1):
        for si in range(5):
            rows.append(
                {"trade_date": d, "ts_code": f"s{si}", "factor_clean": float(si)}
            )
    # 仅 d1 多一只 NaN（无有效信号，不应参与）
    rows.append({"trade_date": d1, "ts_code": "s_nan", "factor_clean": float("nan")})
    df = pl.DataFrame(rows)

    res = compute_turnover(df, factor_col="factor_clean", n_groups=5)
    assert not res.daily_turnover.is_empty()
    assert res.avg_turnover == pytest.approx(0.0, abs=1e-12), (
        f"NaN 不应抬高 max_rank 污染分组：期望 turnover=0，得 {res.avg_turnover}"
    )


def test_nan_stock_does_not_enter_highest_group_and_jump():
    """中位有效股变 NaN：旧实现跳进最高组并挤开更高有效股 → 虚假迁移。

    d0：s0..s4 因子 0..4，各占一组。
    d1：s2=NaN，其余不变。
    正确：s2 退出；s0/s1/s3/s4 仅因截面变小重分桶（与 drop 路径一致）。
    旧：s2 NaN→最高组，且挤压 s3/s4 的秩。
    """
    d0, d1 = date(2024, 1, 2), date(2024, 1, 3)
    rows: list[dict] = []
    for si in range(5):
        rows.append({"trade_date": d0, "ts_code": f"s{si}", "factor_clean": float(si)})
    for si in range(5):
        val = float("nan") if si == 2 else float(si)
        rows.append({"trade_date": d1, "ts_code": f"s{si}", "factor_clean": val})
    nan_df = pl.DataFrame(rows)
    drop_df = nan_df.filter(pl.col("factor_clean").is_not_nan() & pl.col("factor_clean").is_not_null())

    nan_res = compute_turnover(nan_df, n_groups=5)
    drop_res = compute_turnover(drop_df, n_groups=5)

    assert nan_res.avg_turnover == pytest.approx(drop_res.avg_turnover, abs=1e-12)
    assert nan_res.daily_turnover.equals(drop_res.daily_turnover)
    assert nan_res.migration_matrix.equals(drop_res.migration_matrix)

    # 旧实现：s2 从 group2 跳到最高组，且 s3/s4 被挤压，avg 与 drop 不一致且通常更高
    # 修复后与 drop 一致；此处再断言「不应出现 NaN 跳跃带来的额外换手」
    # drop 路径下 d1 只有 4 只有效股，会有因截面缩减的正常重分桶换手
    assert nan_res.avg_turnover == pytest.approx(drop_res.avg_turnover, abs=1e-12)


def test_nan_factor_rows_equivalent_to_dropped_rows():
    """不变量：每日固定若干 NaN 股 ≡ 物理删除这些行（含迁移矩阵）。

    稳定面板上两者 turnover 都可能为 0，但分位边界不同时 migration 的
    路径计数/概率在修复前后应对齐到「删除路径」。
    用「一半天有 NaN、一半无」制造边界差异。
    """
    d0 = date(2024, 1, 1)
    rows: list[dict] = []
    n_stocks = 10
    for di in range(6):
        d = d0 + timedelta(days=di)
        for si in range(n_stocks):
            # 偶数日：高编号 3 只变 NaN；奇数日：全有效
            is_nan = (di % 2 == 0) and si >= 7
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": f"{si:06d}.SZ",
                    "factor_clean": float("nan") if is_nan else float(si),
                }
            )
    nan_df = pl.DataFrame(rows)
    drop_df = nan_df.filter(
        pl.col("factor_clean").is_not_null() & pl.col("factor_clean").is_not_nan()
    )

    nan_res = compute_turnover(nan_df, factor_col="factor_clean", n_groups=5)
    drop_res = compute_turnover(drop_df, factor_col="factor_clean", n_groups=5)

    assert nan_res.avg_turnover == pytest.approx(drop_res.avg_turnover, abs=1e-12), (
        f"NaN 行应等价于删除：nan_avg={nan_res.avg_turnover} vs drop_avg={drop_res.avg_turnover}"
    )
    assert nan_res.daily_turnover.equals(drop_res.daily_turnover), (
        "daily_turnover 应与删除 NaN 行后一致"
    )
    assert nan_res.migration_matrix.equals(drop_res.migration_matrix), (
        "migration_matrix 应与删除 NaN 行后一致"
    )


def test_no_nan_stable_panel_zero_turnover_regression():
    """零回归：无 NaN 的稳定截面 → 换手率 0，结构完整。"""
    df = _stable_panel(n_days=5, n_stocks=10)
    res = compute_turnover(df, factor_col="factor_clean", n_groups=5)
    assert res.avg_turnover == pytest.approx(0.0, abs=1e-12)
    assert not res.daily_turnover.is_empty()
    assert "trade_date" in res.daily_turnover.columns
    assert "turnover" in res.daily_turnover.columns
    assert not res.migration_matrix.is_empty()
    assert set(res.migration_matrix.columns) == {"prev_group", "group", "prob"}


def test_all_nan_day_does_not_crash_avg_zero():
    """退化：全日 NaN / 过滤后空 → 不崩，avg_turnover=0.0。"""
    df = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 1), date(2024, 1, 1)],
            "ts_code": ["s0", "s1"],
            "factor_clean": [float("nan"), float("nan")],
        }
    )
    res = compute_turnover(df, factor_col="factor_clean", n_groups=2)
    assert res.avg_turnover == 0.0
    assert res.daily_turnover.is_empty()
    assert res.migration_matrix.is_empty() or res.migration_matrix.height == 0


def test_null_factor_equivalent_to_nan_dropped():
    """null 与 NaN 一样不参与分组。"""
    d0 = date(2024, 1, 1)
    rows: list[dict] = []
    for di in range(4):
        d = d0 + timedelta(days=di)
        for si in range(8):
            is_null = (di % 2 == 0) and si >= 6
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": f"{si:06d}.SZ",
                    "factor_clean": None if is_null else float(si),
                }
            )
    null_df = pl.DataFrame(rows).with_columns(pl.col("factor_clean").cast(pl.Float64))
    drop_df = null_df.filter(pl.col("factor_clean").is_not_null())

    null_res = compute_turnover(null_df, n_groups=4)
    drop_res = compute_turnover(drop_df, n_groups=4)
    assert null_res.avg_turnover == pytest.approx(drop_res.avg_turnover, abs=1e-12)
    assert null_res.daily_turnover.equals(drop_res.daily_turnover)
    assert null_res.migration_matrix.equals(drop_res.migration_matrix)
