"""交易轨报告 generate_trading_report 离线合成数据测试。"""

from __future__ import annotations

import types
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from factorzen.reports.trading_report import (
    TRADING_BANNER,
    compute_avg_gross_exposure,
    compute_avg_net_exposure,
    compute_cost_erosion,
    generate_trading_report,
)


def _dates(n: int, start: date = date(2024, 1, 2)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _make_bt(
    *,
    n_days: int = 80,
    empty_nav: bool = False,
    empty_trades: bool = False,
    single_day: bool = False,
    known_costs: bool = False,
    known_cash: bool = False,
    all_nan: bool = False,
) -> types.SimpleNamespace:
    if single_day:
        n_days = 1
    dates = _dates(n_days)

    if empty_nav:
        nav = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "gross_return": pl.Float64,
                "cost": pl.Float64,
                "borrow_cost": pl.Float64,
                "net_return": pl.Float64,
                "nav": pl.Float64,
                "cash_weight": pl.Float64,
                "turnover": pl.Float64,
            }
        )
        returns = nav
    else:
        rows = []
        nav_v = 1.0
        for i, d in enumerate(dates):
            if known_costs:
                # 固定序列便于手算：gross=[0.01,0.02,-0.005,...] 循环
                pattern_g = [0.01, 0.02, -0.005]
                pattern_c = [0.001, 0.001, 0.001]
                pattern_b = [0.0005, 0.0005, 0.0005]
                g = pattern_g[i % 3]
                c = pattern_c[i % 3]
                b = pattern_b[i % 3]
            elif all_nan:
                g = c = b = float("nan")
            else:
                g = 0.003 if i % 4 else -0.006
                c = 0.0003
                b = 0.0001
            net = (g if np.isfinite(g) else 0.0) - (c if np.isfinite(c) else 0.0) - (
                b if np.isfinite(b) else 0.0
            )
            if np.isfinite(net):
                nav_v *= 1.0 + net
            if known_cash:
                # cash_weight: 0.2, 0.4, 0.6 → util 0.8, 0.6, 0.4 → mean 0.6
                cw = [0.2, 0.4, 0.6][i % 3]
            else:
                cw = 0.15 + 0.05 * (i % 4)
            rows.append(
                {
                    "trade_date": d,
                    "gross_return": g,
                    "cost": c,
                    "borrow_cost": b,
                    "net_return": net if np.isfinite(net) else float("nan"),
                    "nav": nav_v,
                    "cash_weight": cw,
                    "turnover": 0.12 + 0.01 * (i % 3),
                }
            )
        nav = pl.DataFrame(rows)
        returns = nav

    if empty_trades:
        trades = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "prev_weight": pl.Float64,
                "target_weight": pl.Float64,
                "filled_delta_weight": pl.Float64,
                "turnover": pl.Float64,
                "cost": pl.Float64,
                "block_reason": pl.Utf8,
            }
        )
    else:
        reasons = ["", "suspended", "limit_up", "capacity", "limit_down", ""]
        trades = pl.DataFrame(
            {
                "trade_date": [dates[i % len(dates)] for i in range(30)],
                "ts_code": [f"00000{i % 5}.SZ" for i in range(30)],
                "prev_weight": [0.0] * 30,
                "target_weight": [0.05] * 30,
                "filled_delta_weight": [0.05 if reasons[i % len(reasons)] == "" else 0.0 for i in range(30)],
                "turnover": [0.05] * 30,
                "cost": [0.0001] * 30,
                "block_reason": [reasons[i % len(reasons)] for i in range(30)],
            }
        )

    portfolio = {
        "ann_ret": 0.10,
        "ann_vol": 0.16,
        "sharpe": 0.85,
        "max_dd": -0.12,
        "avg_turnover": 0.14,
        "total_cost": 0.02,
        "ann_turnover": 0.14 * 252,
    }
    return types.SimpleNamespace(
        factor_name="momentum_20d",
        strategy_name="top_n",
        n_groups=5,
        returns=returns,
        nav=nav,
        positions=pl.DataFrame(),
        trades=trades,
        summary_stats={"portfolio": portfolio, "long_short": portfolio},
        config={"cost_model": "default"},
        frequency="daily",
    )


def test_trading_report_suite():
    """交易轨报告：正常 / 退化 / 净口径 / 成本侵蚀手算 / 仓位利用率手算。"""

    # -- 正常输入 --
    def _section_normal():
        bt = _make_bt()
        html = generate_trading_report(
            "momentum_20d",
            bt,
            date_range="2024-01-01 ~ 2024-06-30",
            universe="csi300",
            strategy_name="top_n",
            cost_model="AShareCost",
            backtest_direction={"direction": "normal", "reason": "IC 非负"},
        )
        assert isinstance(html, str) and len(html) > 500
        assert TRADING_BANNER in html
        assert "模拟交易净口径" in html
        assert "momentum_20d" in html
        assert "核心指标" in html
        for title in (
            "净值曲线",
            "回撤水下图",
            "成本侵蚀瀑布图",
            "敞口时序",
            "成交受限原因分布",
            "月度收益热力图",
            "换手率与累计成本",
            "滚动 Sharpe",
        ):
            assert title in html, f"缺区块 {title}"
        # 8 张图必须全部渲染成功;只断言「至少一张」则 7 张静默失败仍绿。
        assert html.count("data:image/png;base64,") >= 8, (
            "应渲染 8 张图,base64 处数不足"
        )
        assert "成本占毛收益比" in html
        assert "平均总敞口 Σ|w|" in html
        assert "平均净敞口 Σw" in html

    _section_normal()

    # -- 退化输入不炸 --
    def _section_degenerate():
        cases = [
            (_make_bt(empty_nav=True, empty_trades=True), "空nav+空trades"),
            (_make_bt(single_day=True), "单日"),
            (_make_bt(all_nan=True), "全NaN"),
            (_make_bt(empty_trades=True), "trades全空"),
        ]
        for bt, note in cases:
            html = generate_trading_report(f"deg_{note}", bt, benchmark_result=None)
            assert isinstance(html, str) and len(html) > 100, note
            # 降级页同样满足下面的断言,必须排除(变异实验实锤:inner 恒抛仍绿)。
            assert "报告生成异常" not in html, f"{note}: 走了降级页,等于没测到真实渲染"
            assert TRADING_BANNER in html, note
            assert "未计算" in html or "核心指标" in html

        html = generate_trading_report("none_bt", None)
        assert isinstance(html, str)
        assert TRADING_BANNER in html
        assert "未计算" in html

    _section_degenerate()

    # -- 净口径说明 --
    def _section_banner():
        html = generate_trading_report("x", _make_bt())
        assert "佣金+印花税+滑点+融券成本" in html
        assert "t 日信号 → t+1 开盘" in html

    _section_banner()

    # -- 成本侵蚀比例：手算期望 --
    def _section_cost_erosion():
        # 3 日循环 × 3 轮 = 9 日
        # pattern gross: 0.01, 0.02, -0.005 → sum per cycle = 0.025
        # cost: 0.001 * 3 = 0.003; borrow: 0.0005 * 3 = 0.0015
        # 3 cycles: sum_gross=0.075, sum_cost=0.009, sum_borrow=0.0045
        # erosion = (0.009+0.0045)/0.075 = 0.18
        n_days = 9
        bt = _make_bt(n_days=n_days, known_costs=True)
        stats = compute_cost_erosion(bt.nav)
        assert stats is not None
        expected_gross = 3 * (0.01 + 0.02 - 0.005)
        expected_cost = 9 * 0.001
        expected_borrow = 9 * 0.0005
        expected_ratio = (expected_cost + expected_borrow) / abs(expected_gross)
        assert abs(stats["sum_gross"] - expected_gross) < 1e-12
        assert abs(stats["sum_cost"] - expected_cost) < 1e-12
        assert abs(stats["sum_borrow"] - expected_borrow) < 1e-12
        assert abs(stats["erosion_ratio"] - expected_ratio) < 1e-12
        assert abs(stats["erosion_ratio"] - 0.18) < 1e-12

        html = generate_trading_report("cost_chk", bt)
        # 18.00% 应出现在指标卡
        assert "18.00%" in html

    _section_cost_erosion()

    # -- 平均净敞口：手算 --
    def _section_utilization():
        # cash_weight cycle 0.2, 0.4, 0.6 → 净敞口 0.8, 0.6, 0.4 → mean 0.6
        bt = _make_bt(n_days=9, known_cash=True, known_costs=True)
        net = compute_avg_net_exposure(bt.nav)
        expected = (0.8 + 0.6 + 0.4) / 3.0
        assert net is not None
        assert abs(net - expected) < 1e-12
        assert abs(net - 0.6) < 1e-12

        html = generate_trading_report("util_chk", bt)
        assert "60.00%" in html

    _section_utilization()

    # -- 总敞口：多空场景下净敞口失去判别力，总敞口仍有 --
    def _section_gross_exposure_discriminates_long_short():
        # 多空持仓：两腿各 ±0.5 → Σ|w|=1.0（资金用满），Σw=0（净敞口为零）
        positions = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 1)] * 4 + [date(2024, 1, 2)] * 4,
                "ts_code": ["a", "b", "c", "d"] * 2,
                "weight": [0.25, 0.25, -0.25, -0.25] * 2,
                "market_value": [1.0, 1.0, -1.0, -1.0] * 2,
            }
        )
        nav = pl.DataFrame(
            {
                "trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
                # Σw=0 → cash_weight=1.0
                "cash_weight": [1.0, 1.0],
                "gross_return": [0.01, 0.01],
                "cost": [0.0, 0.0],
                "borrow_cost": [0.0, 0.0],
                "net_return": [0.01, 0.01],
                "nav": [1.01, 1.0201],
            }
        )
        bt = SimpleNamespace(
            nav=nav, returns=nav, positions=positions,
            trades=pl.DataFrame(schema={"block_reason": pl.Utf8}),
            summary_stats={}, config={},
        )
        # 净敞口=0（对多空无判别力），总敞口=1.0（资金确实用满了）
        assert abs(compute_avg_net_exposure(nav) - 0.0) < 1e-12
        assert abs(compute_avg_gross_exposure(positions) - 1.0) < 1e-12

        # 端到端：两个数值都要如实进指标卡
        html = generate_trading_report("ls_exposure", bt)
        # 注意:不能写 `assert "0.00%" in html` —— 它被 "100.00%" 恒真蕴含。
        # 按卡片标签定位各自的值窗口才有判别力。
        gi = html.find("平均总敞口")
        ni = html.find("平均净敞口")
        assert gi > 0 and ni > 0, "两个敞口卡片都应存在"
        assert "100.00%" in html[gi : gi + 120], "总敞口 Σ|w|=1.0 未进指标卡"
        net_win = html[ni : ni + 120]
        assert "0.00%" in net_win and "100.00%" not in net_win, "净敞口 Σw=0 未进指标卡"

    _section_gross_exposure_discriminates_long_short()

    # -- XSS --
    def _section_escape():
        html = generate_trading_report(
            "<script>x</script>",
            _make_bt(n_days=20),
            backtest_direction={
                "direction": "reversed",
                "reason": "<img src=x onerror=alert(1)>",
            },
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "<img src=x" not in html

    _section_escape()


def test_cost_erosion_not_reported_when_gross_is_nonpositive():
    """毛收益 ≤0 或近零时,「成本占毛收益比」必须判为不适用。

    用 abs(sum_gross) 当分母会说谎:毛 −6% / 成本 3.6% 会显示「成本占 60%」,
    读者理解成「毛 alpha 还剩 40%」,而实际毛收益本身就是亏的;毛收益近零时
    比值还会爆成几千个百分点。两种情形都必须显示「未计算/不适用」而非数字。
    """
    import numpy as np

    def _bt(gross_per_day: float, n: int = 60) -> SimpleNamespace:
        d = _dates(n)
        cost, borrow = 0.0005, 0.0001
        nav_v, rows = 1.0, []
        for i in range(n):
            net = gross_per_day - cost - borrow
            nav_v *= 1.0 + net
            rows.append(
                {
                    "trade_date": d[i],
                    "gross_return": gross_per_day,
                    "cost": cost,
                    "borrow_cost": borrow,
                    "net_return": net,
                    "nav": nav_v,
                    "cash_weight": 0.2,
                }
            )
        nav = pl.DataFrame(rows)
        return SimpleNamespace(
            nav=nav, returns=nav, positions=pl.DataFrame(),
            trades=pl.DataFrame(schema={"block_reason": pl.Utf8}),
            summary_stats={}, config={},
        )

    # 毛收益为负:−10bp/日 × 60 日 = −6%,成本 3.6%
    neg = compute_cost_erosion(_bt(-0.001).nav)
    assert neg["sum_gross"] < 0
    assert not np.isfinite(neg["erosion_ratio"]), (
        f"毛收益为负时比例必须不可计,实得 {neg['erosion_ratio']}"
    )
    html_neg = generate_trading_report("neg_gross", _bt(-0.001))
    assert "报告生成异常" not in html_neg
    i = html_neg.find("成本占毛收益比")
    assert i > 0 and "未计算" in html_neg[i : i + 120], "卡片应显示未计算而非编造比例"

    # 毛收益近零:比值会爆炸
    tiny = compute_cost_erosion(_bt(0.00001).nav)
    assert not np.isfinite(tiny["erosion_ratio"]), (
        f"毛收益近零时比例必须不可计,实得 {tiny['erosion_ratio']}"
    )

    # 毛收益显著为正:正常给出数值(手算:60×0.002=0.12 毛,60×0.0006=0.036 成本 → 30%)
    pos = compute_cost_erosion(_bt(0.002).nav)
    assert pos["erosion_ratio"] == pytest.approx(0.036 / 0.12, rel=1e-9)


def test_net_exposure_excludes_synthetic_base_row():
    """引擎前置的合成起点行不得算进敞口均值。

    backtest._build_nav_frame 会在首个信号日前插一条 nav=1.0/cash_weight=1.0/
    收益全 0 的行,它不是真实交易日。算进均值会把结果系统性往 0 拉(偏差约 1/(n+1),
    短窗口下达数个百分点)。
    """
    n = 20
    d = _dates(n + 1)
    # 合成 base 行 + n 个真实日,真实日 cash_weight 恒 0.2 → 真实净敞口恒 0.80
    rows = [{
        "trade_date": d[0], "gross_return": 0.0, "cost": 0.0, "borrow_cost": 0.0,
        "net_return": 0.0, "nav": 1.0, "cash_weight": 1.0,
    }]
    nav_v = 1.0
    for i in range(n):
        nav_v *= 1.001
        rows.append({
            "trade_date": d[i + 1], "gross_return": 0.0012, "cost": 0.0001,
            "borrow_cost": 0.0001, "net_return": 0.001, "nav": nav_v,
            "cash_weight": 0.2,
        })
    nav = pl.DataFrame(rows)

    got = compute_avg_net_exposure(nav)
    assert got == pytest.approx(0.80, abs=1e-12), (
        f"应为真实日的 0.80,实得 {got}"
        "(含 base 行时会被拉到 0.80*n/(n+1))"
    )
    # 确认 base 行确实有拉低效应(否则这个测试没有判别力)
    naive = float(1.0 - np.mean(nav["cash_weight"].to_numpy()))
    assert abs(naive - 0.80) > 1e-3, "构造前提:含 base 行的朴素均值应明显偏离"

    # 没有 base 行的帧不得被误删首行
    plain = nav.slice(1)
    assert compute_avg_net_exposure(plain) == pytest.approx(0.80, abs=1e-12)


def test_dirty_dtype_degrades_per_section_not_whole_page():
    """脏 dtype 只让相关区块「未计算」,不得让整页降级。

    契约是「缺数据只让该区块显示未计算」。dtype 转换若在 try 之外,ValueError 会
    逃到 generate_* 外层的全捕获,返回两百来字节的降级页——8 个区块全丢。
    """
    nav = pl.DataFrame(
        {
            "trade_date": _dates(3),
            "gross_return": ["a", "b", "c"],  # Utf8:np.asarray(dtype=float) 必炸
            "cost": [0.0, 0.0, 0.0],
            "borrow_cost": [0.0, 0.0, 0.0],
            "net_return": [0.001, 0.001, 0.001],
            "nav": [1.001, 1.002, 1.003],
            "cash_weight": [0.2, 0.2, 0.2],
        }
    )
    bt = SimpleNamespace(
        nav=nav, returns=nav, positions=pl.DataFrame(),
        trades=pl.DataFrame(schema={"block_reason": pl.Utf8}),
        summary_stats={}, config={},
    )
    html = generate_trading_report("dirty_dtype", bt)
    assert "报告生成异常" not in html, "整页降级了——契约要求单区块降级"
    assert html.count("<h2") >= 8, f"区块被整体丢弃,只剩 {html.count('<h2')} 个"
    assert TRADING_BANNER in html
    # 受影响的成本区块应显示未计算而非崩溃
    assert compute_cost_erosion(nav) is None
