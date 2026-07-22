"""信号轨报告 generate_signal_report 离线合成数据测试。"""

from __future__ import annotations

import types
from datetime import date, timedelta

import numpy as np
import polars as pl

from factorzen.daily.evaluation.ic_analysis import ICAnalysisResult
from factorzen.reports.signal_report import SIGNAL_BANNER, generate_signal_report


def _dates(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _make_ic(
    n: int = 60,
    *,
    empty: bool = False,
    all_nan: bool = False,
) -> ICAnalysisResult:
    if empty:
        series = pl.DataFrame(
            schema={"trade_date": pl.Date, "ic": pl.Float64}
        )
    else:
        vals = [float("nan")] * n if all_nan else [0.02 + 0.005 * (i % 5) for i in range(n)]
        series = pl.DataFrame({"trade_date": _dates(n), "ic": vals})
    return ICAnalysisResult(
        factor_name="mom",
        ic_mean=0.035,
        ic_std=0.08,
        ir=0.44,
        ic_positive_ratio=0.62,
        n_periods=n if not empty else 0,
        ic_series=series,
        decay={1: 0.035, 5: 0.028, 10: 0.02, 20: 0.012},
        frequency="daily",
        ic_tstat=3.5,
        ic_pvalue=0.0005,
        multi_period={
            1: {"ic_mean": 0.035, "ir": 0.44},
            5: {"ic_mean": 0.028, "ir": 0.31},
        },
    )


def _make_signal(
    *,
    n_days: int = 60,
    n_groups: int = 5,
    empty_frames: bool = False,
    single_day: bool = False,
    all_nan_ic: bool = False,
) -> types.SimpleNamespace:
    if single_day:
        n_days = 1
    dates = _dates(n_days)

    if empty_frames:
        group_returns = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "group": pl.Int32,
                "ret": pl.Float64,
                "n_stocks": pl.UInt32,
            }
        )
        group_nav = pl.DataFrame(
            schema={"trade_date": pl.Date, "group": pl.Int32, "nav": pl.Float64}
        )
        ls_returns = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ls_ret_gross": pl.Float64,
                "ls_ret_net": pl.Float64,
                "ls_turnover": pl.Float64,
            }
        )
        ls_nav = pl.DataFrame(
            schema={"trade_date": pl.Date, "nav_gross": pl.Float64, "nav_net": pl.Float64}
        )
    else:
        gr_rows = []
        gn_rows = []
        for g in range(n_groups):
            nav = 1.0
            for i, d in enumerate(dates):
                ret = 0.001 * (g + 1) * (1 if i % 3 else -0.5)
                nav *= 1.0 + ret
                gr_rows.append(
                    {"trade_date": d, "group": g, "ret": ret, "n_stocks": 50}
                )
                gn_rows.append({"trade_date": d, "group": g, "nav": nav})
        group_returns = pl.DataFrame(gr_rows).with_columns(pl.col("group").cast(pl.Int32))
        group_nav = pl.DataFrame(gn_rows).with_columns(pl.col("group").cast(pl.Int32))

        ls_rows = []
        nav_g = nav_n = 1.0
        for i, d in enumerate(dates):
            rg = 0.002 if i % 4 else -0.001
            rn = rg - 0.0002
            nav_g *= 1.0 + rg
            nav_n *= 1.0 + rn
            ls_rows.append(
                {
                    "trade_date": d,
                    "ls_ret_gross": rg,
                    "ls_ret_net": rn,
                    "ls_turnover": 0.15 + 0.01 * (i % 3),
                }
            )
        ls_returns = pl.DataFrame(ls_rows)
        ls_nav = pl.DataFrame(
            {
                "trade_date": dates,
                "nav_gross": np.cumprod(1 + ls_returns["ls_ret_gross"].to_numpy()),
                "nav_net": np.cumprod(1 + ls_returns["ls_ret_net"].to_numpy()),
            }
        )

    mono = types.SimpleNamespace(
        factor_name="mom",
        monotonicity_score=0.85,
        group_means=[0.001 * (g + 1) for g in range(n_groups)],
        direction="positive",
    )
    turnover = types.SimpleNamespace(avg_turnover=0.18)
    return types.SimpleNamespace(
        factor_name="momentum_20d",
        n_groups=n_groups,
        cost_bps=10.0,
        group_returns=group_returns,
        group_nav=group_nav,
        ls_returns=ls_returns,
        ls_nav=ls_nav,
        ic=_make_ic(
            n=n_days,
            empty=empty_frames,
            all_nan=all_nan_ic,
        ),
        monotonicity=mono,
        turnover=turnover,
        summary_stats={
            "long_short": {
                "ann_ret_gross": 0.12,
                "sharpe_gross": 1.2,
                "max_dd_gross": -0.08,
                "avg_turnover": 0.18,
            },
            "ic": {"ic_mean": 0.035, "ir": 0.44, "tstat": 3.5},
        },
        meta={"exec_lag": 1, "exec_price_col": "open_adj", "dropped_days": 2},
    )


def test_signal_report_suite():
    """信号轨报告：正常输入 / 退化输入 / 毛口径横幅。"""

    # -- 正常输入产出 HTML 非空、含关键区块与口径横幅 --
    def _section_normal():
        sig = _make_signal()
        html = generate_signal_report(
            sig,
            factor_name="momentum_20d",
            date_range="2024-01-01 ~ 2024-06-30",
            universe="csi300",
        )
        assert isinstance(html, str) and len(html) > 500
        assert SIGNAL_BANNER in html
        assert "信号层毛收益" in html
        assert "momentum_20d" in html
        assert "csi300" in html
        assert "核心指标" in html
        for title in (
            "分层累计净值",
            "分层平均收益",
            "IC 时序",
            "IC 累计曲线",
            "IC 衰减曲线",
            "IC 分布",
            "分层收益年度热力图",
            "换手率时序",
        ):
            assert f"<h2>{title}</h2>" in html or f"<h2>{title}" in html, f"缺区块 {title}"
        assert "data:image/png;base64," in html
        assert "0.0350" in html  # IC mean
        assert "exec_lag" in html or "1" in html

    _section_normal()

    # -- 退化输入不炸：空帧 / 单日 / 全 NaN IC --
    def _section_degenerate():
        for kwargs, note in (
            ({"empty_frames": True}, "空帧"),
            ({"single_day": True}, "单日"),
            ({"all_nan_ic": True}, "全NaN IC"),
        ):
            sig = _make_signal(**kwargs)
            html = generate_signal_report(sig, factor_name=f"deg_{note}")
            assert isinstance(html, str) and len(html) > 100, note
            assert SIGNAL_BANNER in html, note
            assert "未计算" in html or "data:image/png;base64," in html or "核心指标" in html

        # signal_result=None
        html = generate_signal_report(None, factor_name="none_sig")
        assert isinstance(html, str)
        assert SIGNAL_BANNER in html
        assert "未计算" in html

    _section_degenerate()

    # -- 毛口径警告文案必须存在 --
    def _section_banner():
        html = generate_signal_report(_make_signal(), factor_name="x")
        assert "不可直接当可实现收益汇报" in html
        assert "fz factor backtest" in html

    _section_banner()

    # -- XSS 转义 --
    def _section_escape():
        sig = _make_signal()
        html = generate_signal_report(
            sig,
            factor_name="<script>alert(1)</script>",
            universe="<img src=x>",
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    _section_escape()
