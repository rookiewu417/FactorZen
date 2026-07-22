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
                "ls_turnover": pl.Float64,
            }
        )
        ls_nav = pl.DataFrame(
            schema={"trade_date": pl.Date, "nav_gross": pl.Float64}
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
                    "ls_turnover": 0.15 + 0.01 * (i % 3),
                }
            )
        ls_returns = pl.DataFrame(ls_rows)
        ls_nav = pl.DataFrame(
            {
                "trade_date": dates,
                "nav_gross": np.cumprod(1 + ls_returns["ls_ret_gross"].to_numpy()),
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
        # 8 张图必须全部渲染成功。只断言「至少一张」时,7 张静默失败也不会红
        # (_safe_chart 逐图吞异常,失败区块照样吐标题 + 「未计算」)。
        assert html.count("data:image/png;base64,") >= 8, (
            f"应渲染 8 张图,实际 {html.count(chr(100)+chr(97)+chr(116)+chr(97))} 处 base64"
        )
        assert "0.0350" in html  # IC mean
        assert "exec_lag" in html

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
            # 外层 try/except 的降级页也满足下面几条断言,必须显式排除,
            # 否则整个 inner 抛异常时本段仍会绿(变异实验实锤)。
            assert "报告生成异常" not in html, f"{note}: 走了降级页,等于没测到真实渲染"
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


def test_ic_unavailable_is_not_disguised_as_zero():
    """n_periods=0 时 IC 类指标必须显示「未计算」,绝不能展示哨兵数值。

    项目 P0 史(admission_ic 恒 0.0):空输入返回 0.0 哨兵被下游当合法值消费。
    信号轨报告的核心就是 IC——若截面样本不足门槛(ic_analysis._MIN_CROSS_SAMPLES=30)
    或 join 零命中,IC 一天都算不出来,此时展示数字会让读者把「算不出来」
    读成「因子无预测力」。

    判别力:fixture 刻意让 ic_mean=0.035(一个看起来完全合法的值)而 n_periods=0。
    实现若不检查 n_periods,报告就会显示 0.0350,本测试即红。
    """
    sig = _make_signal()
    sig.ic = _make_ic(empty=True)  # n_periods=0,但 ic_mean 仍是 0.035
    assert sig.ic.n_periods == 0
    assert sig.ic.ic_mean == 0.035, "fixture 前提:哨兵值非零才有判别力"

    html = generate_signal_report(sig, factor_name="thin_cross_section")

    idx = html.find("IC mean")
    assert idx > 0, "报告应含 IC mean 卡片"
    card = html[idx : idx + 200]
    assert "未计算" in card, "n_periods=0 时 IC mean 卡片应显示未计算"
    assert "0.0350" not in card, "绝不能展示 ic_mean 哨兵值"
    assert "IC 不可用" in html, "应有醒目告警条说明 IC 算不出来"
    assert "请勿把它读成" in html, "告警条应明说不要误读为无预测力"

    # 正常样本不受影响:IC 数值照常展示
    ok = _make_signal()
    html_ok = generate_signal_report(ok, factor_name="normal")
    assert "IC 不可用" not in html_ok
    idx_ok = html_ok.find("IC mean")
    assert "未计算" not in html_ok[idx_ok : idx_ok + 60]


def test_reversed_direction_is_disclosed():
    """信号翻号后,报告必须明说页面数字是翻号口径。

    上游若判定原始因子 IC 显著为负,会先把信号翻号再送进信号轨评估——此时
    页面上所有 IC/分层/多空数字都是翻号后的。不提示的话,读者会把 +0.012 当成
    原始因子的 IC,方向正好读反(实测同一 run 的 ic.parquet 是 −0.012)。
    """
    sig = _make_signal()
    sig.meta = dict(getattr(sig, "meta", {}) or {})
    sig.meta["direction"] = "reversed"
    html = generate_signal_report(sig, factor_name="rev_factor")
    assert "报告生成异常" not in html
    assert "反向信号" in html, "翻号时必须有醒目提示"
    assert "符号相反" in html, "必须说明与原始因子 IC 符号相反"

    # 正向不得误报
    sig2 = _make_signal()
    sig2.meta = dict(getattr(sig2, "meta", {}) or {})
    sig2.meta["direction"] = "normal"
    html2 = generate_signal_report(sig2, factor_name="normal_factor")
    assert "反向信号" not in html2
