"""信号层向量化回测（signal_backtest）单测。

离线合成数据；期望值独立手算，禁止用被测 helper 自记期望。
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from factorzen.daily.evaluation.advanced.monotonicity import compute_monotonicity
from factorzen.daily.evaluation.grouping import assign_quantile_groups
from factorzen.daily.evaluation.signal_backtest import run_signal_backtest
from factorzen.daily.evaluation.turnover import compute_turnover

# ---------------------------------------------------------------------------
# 手算金标准：6 股 × 4 日 × 3 组
# ---------------------------------------------------------------------------
# 因子值固定 0..5 → ordinal rank 1..6，n_groups=3：
#   group = (rank-1)*3//6 → g0:{s0,s1}, g1:{s2,s3}, g2:{s4,s5}
# 各日前向收益（手算组均 / ls）：
#
# day0: rets=[0.01,0.02,0.03,0.04,0.05,0.06]
#   g0=(0.01+0.02)/2=0.015; g1=0.035; g2=0.055; ls=0.055-0.015=0.040
# day1: rets=[0.00,0.02,0.04,0.06,0.08,0.10]
#   g0=0.01; g1=0.05; g2=0.09; ls=0.08
# day2: rets=[-0.01,0.01,0.00,0.02,0.03,0.05]
#   g0=0.00; g1=0.01; g2=0.04; ls=0.04
# day3: rets=[0.02,0.00,0.01,0.03,0.04,0.06]
#   g0=0.01; g1=0.02; g2=0.05; ls=0.04
# ---------------------------------------------------------------------------

_GOLD_DATES = [date(2024, 1, 2) + timedelta(days=i) for i in range(4)]
_GOLD_STOCKS = [f"s{i}" for i in range(6)]
_GOLD_FACTOR = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
_GOLD_RETS = [
    [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    [0.00, 0.02, 0.04, 0.06, 0.08, 0.10],
    [-0.01, 0.01, 0.00, 0.02, 0.03, 0.05],
    [0.02, 0.00, 0.01, 0.03, 0.04, 0.06],
]
# 手算期望组均值 [day][group]
_GOLD_GROUP_MEANS = [
    [0.015, 0.035, 0.055],
    [0.01, 0.05, 0.09],
    [0.00, 0.01, 0.04],
    [0.01, 0.02, 0.05],
]
_GOLD_LS = [0.04, 0.08, 0.04, 0.04]


def _gold_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    f_rows: list[dict] = []
    r_rows: list[dict] = []
    for di, d in enumerate(_GOLD_DATES):
        for si, code in enumerate(_GOLD_STOCKS):
            f_rows.append(
                {"trade_date": d, "ts_code": code, "factor_clean": _GOLD_FACTOR[si]}
            )
            r_rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "fwd_ret_1d": _GOLD_RETS[di][si],
                }
            )
    return pl.DataFrame(f_rows), pl.DataFrame(r_rows)


def test_signal_backtest_suite():
    """手算金标准 / 分组公式 / 换手成本 / 退化守卫 / NaN 防线 / 口径透传 / 重构等价。"""

    # -- 手算金标准 --
    def _section_gold_standard():
        factor_df, fwd = _gold_frames()
        res = run_signal_backtest(
            factor_df, fwd, n_groups=3, cost_bps=0.0, factor_name="gold"
        )
        assert res.meta.get("return_basis") == "gross_signal_level"
        assert res.meta.get("dropped_days") == 0
        assert not res.group_returns.is_empty()
        assert not res.ls_returns.is_empty()

        # 组均值逐日逐组
        gr = res.group_returns.sort(["trade_date", "group"])
        for di, d in enumerate(_GOLD_DATES):
            d_iso = d.isoformat()
            day = gr.filter(pl.col("trade_date") == d_iso)
            assert day.height == 3, f"day {d_iso} groups={day.height}"
            for g in range(3):
                got = float(day.filter(pl.col("group") == g)["ret"][0])
                exp = _GOLD_GROUP_MEANS[di][g]
                assert got == pytest.approx(exp, abs=1e-12), (
                    f"group mean d={d_iso} g={g}: got={got} exp={exp}"
                )
                n = int(day.filter(pl.col("group") == g)["n_stocks"][0])
                assert n == 2

        # ls_ret_gross 序列
        ls = res.ls_returns.sort("trade_date")
        got_ls = ls["ls_ret_gross"].to_list()
        assert len(got_ls) == 4
        for i, (g, e) in enumerate(zip(got_ls, _GOLD_LS, strict=True)):
            assert g == pytest.approx(e, abs=1e-12), f"ls[{i}]: got={g} exp={e}"

        # cost_bps=0 → net == gross 逐位
        for g, n in zip(
            ls["ls_ret_gross"].to_list(), ls["ls_ret_net"].to_list(), strict=True
        ):
            assert g == pytest.approx(n, abs=1e-15)

        # NAV 手算：cumprod(1+ret)，首日=1+ret
        # ls: 0.04,0.08,0.04,0.04 → nav: 1.04, 1.04*1.08, ...
        nav_exp = []
        acc = 1.0
        for r in _GOLD_LS:
            acc *= 1.0 + r
            nav_exp.append(acc)
        got_nav = res.ls_nav.sort("trade_date")["nav_gross"].to_list()
        for g, e in zip(got_nav, nav_exp, strict=True):
            assert g == pytest.approx(e, abs=1e-12)

        # summary 末行警告
        text = res.summary()
        assert "信号层毛收益" in text
        assert "不可直接当可实现收益汇报" in text

    _section_gold_standard()

    # -- 分组公式对照（含并列） --
    def _section_group_formula():
        # 截面 7 股，含并列因子值；n_groups=3
        # factor: [1, 1, 2, 3, 3, 4, 5] → ordinal rank 打散并列（稳定序取决于行序）
        codes = [f"t{i}" for i in range(7)]
        factors = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
        df = pl.DataFrame(
            {
                "trade_date": [date(2024, 2, 1)] * 7,
                "ts_code": codes,
                "factor_clean": factors,
            }
        )
        # 独立重写分组公式（不调用 assign_quantile_groups）
        ranked = df.with_columns(
            pl.col("factor_clean")
            .rank("ordinal", descending=False)
            .over("trade_date")
            .alias("_rank")
        )
        expected = ranked.with_columns(
            ((pl.col("_rank") - 1) * 3 // pl.col("_rank").max().over("trade_date"))
            .cast(pl.Int32)
            .alias("group")
        ).select(["ts_code", "group"]).sort("ts_code")

        got = (
            assign_quantile_groups(df, n_groups=3)
            .select(["ts_code", "group"])
            .sort("ts_code")
        )
        assert got.equals(expected)

        # 各组大小差 ≤ 1
        sizes = got.group_by("group").len().sort("group")["len"].to_list()
        assert max(sizes) - min(sizes) <= 1
        assert set(got["group"].to_list()) == {0, 1, 2}

    _section_group_formula()

    # -- 换手与成本 --
    def _section_turnover_cost():
        # 场景 A：两日 top/bottom 完全换仓
        # n_groups=3, 6 股；g0={s0,s1}, g2={s4,s5}
        # day0 factor 0..5；day1 反转 5..0 → top 从 {s4,s5} 换成 {s0,s1}
        d0, d1 = date(2024, 3, 1), date(2024, 3, 2)
        f_rows = []
        r_rows = []
        for d, factors in (
            (d0, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            (d1, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        ):
            for si in range(6):
                f_rows.append(
                    {
                        "trade_date": d,
                        "ts_code": f"s{si}",
                        "factor_clean": factors[si],
                    }
                )
                r_rows.append(
                    {
                        "trade_date": d,
                        "ts_code": f"s{si}",
                        "fwd_ret_1d": 0.01 * (si + 1),
                    }
                )
        factor_df = pl.DataFrame(f_rows)
        fwd = pl.DataFrame(r_rows)
        cost_bps = 10.0  # 10bp
        res = run_signal_backtest(
            factor_df, fwd, n_groups=3, cost_bps=cost_bps, factor_name="to_flip"
        )
        ls = res.ls_returns.sort("trade_date")
        assert ls.height == 2

        # 首日建仓：每腿 turnover=1.0 → ls_turnover=2.0
        to0 = float(ls["ls_turnover"][0])
        assert to0 == pytest.approx(2.0, abs=1e-12)

        # 次日完全换仓：每腿 0.5*Σ|Δw|=1.0 → ls_turnover=2.0
        # 手算：top day0={s4,s5} w=0.5；day1={s0,s1} w=0.5
        # |Δ| 四只各 0.5 → sum=2.0 → *0.5=1.0；bottom 同理
        to1 = float(ls["ls_turnover"][1])
        assert to1 == pytest.approx(2.0, abs=1e-12)

        # net - gross = -ls_turnover * cost_bps / 1e4
        for i in range(2):
            gross = float(ls["ls_ret_gross"][i])
            net = float(ls["ls_ret_net"][i])
            to = float(ls["ls_turnover"][i])
            exp_diff = -to * cost_bps / 10000.0
            assert (net - gross) == pytest.approx(exp_diff, abs=1e-15)

        # 场景 B：完全不换仓
        f_stable = []
        r_stable = []
        factors_s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        for d in (d0, d1):
            for si in range(6):
                f_stable.append(
                    {
                        "trade_date": d,
                        "ts_code": f"s{si}",
                        "factor_clean": factors_s[si],
                    }
                )
                r_stable.append(
                    {
                        "trade_date": d,
                        "ts_code": f"s{si}",
                        "fwd_ret_1d": 0.01,
                    }
                )
        res2 = run_signal_backtest(
            pl.DataFrame(f_stable),
            pl.DataFrame(r_stable),
            n_groups=3,
            cost_bps=cost_bps,
        )
        ls2 = res2.ls_returns.sort("trade_date")
        assert float(ls2["ls_turnover"][0]) == pytest.approx(2.0, abs=1e-12)  # 建仓
        # 次日权重不变：leg_turnover=0 → ls_turnover=0
        assert float(ls2["ls_turnover"][1]) == pytest.approx(0.0, abs=1e-12)

        # cost_bps=0 → net==gross
        res0 = run_signal_backtest(
            factor_df, fwd, n_groups=3, cost_bps=0.0
        )
        for g, n in zip(
            res0.ls_returns["ls_ret_gross"].to_list(),
            res0.ls_returns["ls_ret_net"].to_list(),
            strict=True,
        ):
            assert g == pytest.approx(n, abs=1e-15)

    _section_turnover_cost()

    # -- 退化守卫 --
    def _section_degenerate():
        # 单股票日 < n_groups → 整日剔除
        d0, d1 = date(2024, 4, 1), date(2024, 4, 2)
        # d0: 1 股；d1: 6 股
        f_rows = [{"trade_date": d0, "ts_code": "s0", "factor_clean": 1.0}]
        r_rows = [{"trade_date": d0, "ts_code": "s0", "fwd_ret_1d": 0.01}]
        for si in range(6):
            f_rows.append(
                {
                    "trade_date": d1,
                    "ts_code": f"s{si}",
                    "factor_clean": float(si),
                }
            )
            r_rows.append(
                {
                    "trade_date": d1,
                    "ts_code": f"s{si}",
                    "fwd_ret_1d": 0.01 * si,
                }
            )
        res = run_signal_backtest(
            pl.DataFrame(f_rows), pl.DataFrame(r_rows), n_groups=3
        )
        assert res.meta["dropped_days"] == 1
        assert res.ls_returns.height == 1
        assert res.ls_returns["trade_date"][0] == d1.isoformat()

        # 全 NaN 因子 → 空结构不崩
        nan_f = pl.DataFrame(
            {
                "trade_date": [d0, d0, d1, d1],
                "ts_code": ["a", "b", "a", "b"],
                "factor_clean": [float("nan")] * 4,
            }
        )
        nan_r = pl.DataFrame(
            {
                "trade_date": [d0, d0, d1, d1],
                "ts_code": ["a", "b", "a", "b"],
                "fwd_ret_1d": [0.01, 0.02, 0.01, 0.02],
            }
        )
        res_nan = run_signal_backtest(nan_f, nan_r, n_groups=2)
        assert res_nan.group_returns.is_empty()
        assert res_nan.ls_returns.is_empty()
        assert res_nan.group_nav.is_empty()
        assert res_nan.ls_nav.is_empty()
        assert res_nan.summary_stats["long_short"]["ann_ret_gross"] == 0.0
        assert "group" in res_nan.group_returns.columns
        assert "ls_ret_gross" in res_nan.ls_returns.columns
        # summary 不崩
        assert "信号层毛收益" in res_nan.summary()

        # 所有日 n < n_groups
        tiny_f = pl.DataFrame(
            {
                "trade_date": [d0, d0],
                "ts_code": ["a", "b"],
                "factor_clean": [1.0, 2.0],
            }
        )
        tiny_r = pl.DataFrame(
            {
                "trade_date": [d0, d0],
                "ts_code": ["a", "b"],
                "fwd_ret_1d": [0.01, 0.02],
            }
        )
        res_tiny = run_signal_backtest(tiny_f, tiny_r, n_groups=5)
        assert res_tiny.meta["dropped_days"] == 1
        assert res_tiny.ls_returns.is_empty()

    _section_degenerate()

    # -- NaN 传染防线 --
    def _section_nan_ret():
        # 6 股 1 日 3 组；g2 中一只 fwd_ret 为 float('nan')（非 null）
        # 组均值应等于非 NaN 均值，不被污染
        d = date(2024, 5, 1)
        f = pl.DataFrame(
            {
                "trade_date": [d] * 6,
                "ts_code": [f"s{i}" for i in range(6)],
                "factor_clean": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        # g2 = {s4,s5}；s5 ret = nan → g2 mean 只取 s4=0.10
        rets = [0.01, 0.02, 0.03, 0.04, 0.10, float("nan")]
        r = pl.DataFrame(
            {
                "trade_date": [d] * 6,
                "ts_code": [f"s{i}" for i in range(6)],
                "fwd_ret_1d": rets,
            }
        )
        res = run_signal_backtest(f, r, n_groups=3)
        # s5 因 fill_nan+drop 被剔除后当日剩 5 股 ≥ 3，仍保留；
        # s5 无有效 fwd → 不参与分组。手算（5 有效股, n_groups=3）：
        # rank s0..s4 → groups: 0*3//5=0, 1*3//5=0, 2*3//5=1, 3*3//5=1, 4*3//5=2
        # g0={s0,s1} mean=(0.01+0.02)/2=0.015
        # g1={s2,s3} mean=(0.03+0.04)/2=0.035
        # g2={s4} mean=0.10
        gr = res.group_returns.sort("group")
        g2 = float(gr.filter(pl.col("group") == 2)["ret"][0])
        assert g2 == pytest.approx(0.10, abs=1e-12)
        g0 = float(gr.filter(pl.col("group") == 0)["ret"][0])
        assert g0 == pytest.approx(0.015, abs=1e-12)
        # 组均值非 NaN（NaN 传染防线）：有限数，且与手算一致已由 approx 保证
        assert math.isfinite(g2)

    _section_nan_ret()

    # -- 口径透传接线（数值差异验证） --
    def _section_return_basis_passthrough():
        factor_df, fwd0 = _gold_frames()
        # 第二套：把收益整体 shift 一日（模拟 exec_lag 改变可实现收益）
        # 手写：day0 用原 day1 ret，day1 用 day2，day2 用 day3，day3 用 0
        shifted_rets = [*_GOLD_RETS[1:], [0.0] * 6]
        r_rows = []
        for di, d in enumerate(_GOLD_DATES):
            for si, code in enumerate(_GOLD_STOCKS):
                r_rows.append(
                    {
                        "trade_date": d,
                        "ts_code": code,
                        "fwd_ret_1d": shifted_rets[di][si],
                    }
                )
        fwd1 = pl.DataFrame(r_rows)

        res0 = run_signal_backtest(factor_df, fwd0, n_groups=3)
        res1 = run_signal_backtest(factor_df, fwd1, n_groups=3)
        ls0 = res0.ls_returns.sort("trade_date")["ls_ret_gross"].to_list()
        ls1 = res1.ls_returns.sort("trade_date")["ls_ret_gross"].to_list()
        assert ls0 != ls1, "不同 fwd_returns 口径应产生不同 ls_ret_gross"
        # 手算第一日差异：原 0.04 vs shift 后 day0 用 day1 ret → ls=0.08
        assert ls0[0] == pytest.approx(0.04, abs=1e-12)
        assert ls1[0] == pytest.approx(0.08, abs=1e-12)

    _section_return_basis_passthrough()

    # -- turnover / monotonicity 等价回归（重构后硬编码期望） --
    def _section_refactor_regression():
        # 固定 5 股 × 2 日，n_groups=5；day1 完全反转排名
        d0, d1 = date(2024, 6, 1), date(2024, 6, 2)
        f_rows = []
        for d, factors in (
            (d0, [0.0, 1.0, 2.0, 3.0, 4.0]),
            (d1, [4.0, 3.0, 2.0, 1.0, 0.0]),
        ):
            for si, fv in enumerate(factors):
                f_rows.append(
                    {
                        "trade_date": d,
                        "ts_code": f"x{si}",
                        "factor_clean": fv,
                    }
                )
        fdf = pl.DataFrame(f_rows)

        # 换手：day1 除 x2 外 4 只变更组 → turnover = 4/5 = 0.8
        to = compute_turnover(fdf, n_groups=5)
        assert to.avg_turnover == pytest.approx(0.8, abs=1e-12)
        assert to.daily_turnover.height == 1
        assert float(to.daily_turnover["turnover"][0]) == pytest.approx(0.8, abs=1e-12)

        # 单调性：单日 5 组，ret = factor * 0.01 → 组均严格递增
        mono_df = pl.DataFrame(
            {
                "trade_date": [d0] * 5,
                "ts_code": [f"x{i}" for i in range(5)],
                "factor_clean": [0.0, 1.0, 2.0, 3.0, 4.0],
                "fwd_ret": [0.0, 0.01, 0.02, 0.03, 0.04],
            }
        )
        mono = compute_monotonicity(mono_df, n_groups=5, ret_col="fwd_ret")
        # 每组 1 股，group_means = [0, 0.01, 0.02, 0.03, 0.04]
        exp_means = [0.0, 0.01, 0.02, 0.03, 0.04]
        assert len(mono.group_means) == 5
        for g, e in zip(mono.group_means, exp_means, strict=True):
            assert g == pytest.approx(e, abs=1e-12)
        # 4 个相邻差分全为正，与首尾同向 → score=1.0
        assert mono.monotonicity_score == pytest.approx(1.0, abs=1e-12)
        assert mono.direction == "positive"
        # OLS slope 手算：x=0..4, y=0,0.01,...,0.04 → slope=0.01
        assert mono.ols_slope == pytest.approx(0.01, abs=1e-12)

    _section_refactor_regression()


def test_two_track_consistency_under_controlled_conditions():
    """双路径登记簿:信号轨与交易轨在受控条件下必须收敛。

    两轨是本项目故意分离的两条实现(向量化信号层 vs 日环撮合),按登记簿规矩
    必须有一致性测试。它们日常数字不同是**成本与交易约束**造成的,不该是算法分歧
    ——本测试把成本与约束全部关掉,并令 ``open[t] == close[t-1]``(无隔夜跳空)
    使两轨收益口径等价,此时两轨的多空收益序列必须逐位相同。

    任一侧的分组规则、腿权重、收益归属被改动,此测试即红。
    """
    import numpy as np

    from factorzen.daily.evaluation.backtest import (
        BacktestConfig,
        CostModel,
        QuantileLongShortStrategy,
        run_strategy_backtest,
    )
    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns

    n_stock, n_day, n_group = 12, 30, 3
    rng = np.random.default_rng(20260721)
    codes = [f"{i:06d}.SZ" for i in range(n_stock)]
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_day)]

    closes = np.zeros((n_day, n_stock))
    closes[0] = 10.0
    rets = rng.normal(0.0, 0.02, size=(n_day, n_stock))
    for t in range(1, n_day):
        closes[t] = closes[t - 1] * (1.0 + rets[t])

    price_rows, factor_rows = [], []
    for ti, d in enumerate(dates):
        for si, code in enumerate(codes):
            prev_close = closes[ti - 1, si] if ti > 0 else closes[0, si]
            price_rows.append(
                {
                    "trade_date": d,
                    "ts_code": code,
                    "open": float(prev_close),  # 无隔夜跳空:开盘=昨收
                    "close": float(closes[ti, si]),
                    "pre_close": float(prev_close),
                    "pct_chg": float((closes[ti, si] / prev_close - 1.0) * 100.0),
                    "vol": 1e9,
                    "amount": 1e12,  # ADV 巨大 → participation 不绑定
                }
            )
            factor_rows.append(
                {"trade_date": d, "ts_code": code, "factor_clean": float(rng.normal())}
            )
    price = pl.DataFrame(price_rows)
    factor = pl.DataFrame(factor_rows)

    bt = run_strategy_backtest(
        QuantileLongShortStrategy(n_groups=n_group, factor_col="factor_clean"),
        factor,
        price,
        config=BacktestConfig(
            factor_col="factor_clean",
            initial_capital=1e12,
            max_participation_rate=1.0,
            max_gross_exposure=float("inf"),
            max_abs_weight=float("inf"),
            limit_up_pct=1e9,
            limit_down_pct=-1e9,
        ),
        cost_model=CostModel(commission=0, stamp_tax=0, slippage=0, borrow_annual=0),
        factor_name="consistency",
    )
    fwd = compute_fwd_returns(price, horizons=[1], exec_lag=0, exec_price_col="close")
    sig = run_signal_backtest(
        factor, fwd, factor_col="factor_clean", n_groups=n_group,
        cost_bps=0.0, horizons=[1], factor_name="consistency",
    )

    # 信号轨 t 日 ls 收益(t→t+1) == 交易轨 t+1 日组合收益 → 错一位对齐
    sg = sig.ls_returns.sort("trade_date")["ls_ret_gross"].to_numpy()
    tr = bt.nav.sort("trade_date")["net_return"].to_numpy()
    n = min(sg.size, tr.size - 1)
    assert n >= 20, f"对齐后样本过少 n={n}"
    a, b = sg[:n], tr[1 : n + 1]

    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    max_diff = float(np.max(np.abs(a - b)))
    assert max_diff < 1e-12, (
        f"两轨在零成本零约束下应逐位相同,实测最大差={max_diff:.3e}。"
        "差异非零说明分组规则/腿权重/收益归属在某一侧被改动。"
    )


def test_silent_failure_guards_suite():
    """三类「静默失明」守卫:日期形态哨兵 / inf 污染 / n_groups<2。

    共同点:出问题时结果**看起来完全正常**,不报错也无告警,是最危险的形态。
    """
    import numpy as np
    import pytest as _pytest

    codes = [f"{k:06d}.SZ" for k in range(40)]

    # -- 日期形态不一致不得让 IC 静默变 0 哨兵 --
    def _section_date_format():
        # 因子侧 Date、收益侧 ISO 字符串——生产中两条链路形态不一致是真实场景。
        # 归一化缺失时 join 零命中,compute_rank_ic 返回 ic_mean=0.0 哨兵且无告警,
        # 与「因子真的没有预测力」不可区分(core/dates.py 记录的 live P0 同形态)。
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(40)]
        f_rows, r_rows = [], []
        for d in days:
            for j, c in enumerate(codes):
                f_rows.append({"trade_date": d, "ts_code": c, "factor_clean": float(j)})
                # 因子与收益完全同序 → 真实 Rank IC 必须 ≈ 1
                r_rows.append(
                    {"trade_date": d.isoformat(), "ts_code": c, "fwd_ret_1d": 0.001 * j}
                )
        res = run_signal_backtest(
            pl.DataFrame(f_rows), pl.DataFrame(r_rows), n_groups=5, horizons=[1]
        )
        assert res.ic.n_periods > 0, "日期形态归一后 IC 必须算得出来"
        assert res.ic.ic_mean == pytest.approx(1.0, abs=1e-9), (
            f"完全同序应得 IC≈1,实得 {res.ic.ic_mean}(0.0 说明 join 零命中走了哨兵)"
        )
        assert res.summary_stats["ic"]["n_periods"] == res.ic.n_periods

    _section_date_format()

    # -- inf 不得污染收益序列并伪装成零 alpha --
    def _section_inf_guard():
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        f_rows, r_rows = [], []
        for di, d in enumerate(days):
            for k in range(10):
                code = f"{k:06d}.SZ"
                f_rows.append({"trade_date": d, "ts_code": code, "factor_clean": float(k)})
                # 首日最高分位塞一个 inf(entry 价为 0 时 exit/entry-1 就会产生)
                val = float("inf") if (di == 0 and k == 9) else 0.01 * k
                r_rows.append({"trade_date": d, "ts_code": code, "fwd_ret_1d": val})
        res = run_signal_backtest(
            pl.DataFrame(f_rows), pl.DataFrame(r_rows), n_groups=3, horizons=[1]
        )
        gross = np.asarray(res.ls_returns["ls_ret_gross"].to_numpy(), dtype=float)
        assert np.all(np.isfinite(gross)), f"inf 必须被隔离,实得 {gross.tolist()}"
        nav = np.asarray(res.ls_nav["nav_gross"].to_numpy(), dtype=float)
        assert np.all(np.isfinite(nav)), "NAV 不得被 inf 传染"
        assert np.isfinite(res.summary_stats["long_short"]["max_dd_gross"]), (
            "max_dd 为 nan 说明序列已损坏"
        )
        # n_days 让消费方能区分「没有数据」与「真的是 0」
        assert res.summary_stats["long_short"]["n_days"] == res.ls_returns.height > 0

    _section_inf_guard()

    # -- n_groups<2 必须报错而非产出恒 0 假信号 --
    def _section_n_groups_guard():
        days = [date(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        f_rows, r_rows = [], []
        for d in days:
            for k in range(6):
                code = f"{k:06d}.SZ"
                f_rows.append({"trade_date": d, "ts_code": code, "factor_clean": float(k)})
                r_rows.append({"trade_date": d, "ts_code": code, "fwd_ret_1d": 0.01 * k})
        fdf, rdf = pl.DataFrame(f_rows), pl.DataFrame(r_rows)
        # n_groups=1 时 top 组与 bottom 组是同一组:毛收益恒 0、换手照算,
        # cost_bps>0 会凭空造出「稳定小亏」的曲线,且全程不报错。
        for bad in (1, 0, -3):
            with _pytest.raises(ValueError, match="n_groups"):
                run_signal_backtest(fdf, rdf, n_groups=bad, cost_bps=2.0, horizons=[1])
        # 合法值仍正常
        ok = run_signal_backtest(fdf, rdf, n_groups=2, cost_bps=2.0, horizons=[1])
        assert ok.ls_returns.height > 0

    _section_n_groups_guard()


def test_metric_conventions_match_trading_track():
    """信号轨的 MaxDD / Sharpe / 年化基数必须与交易轨同源,否则两轨数字不可比。"""
    import numpy as np

    from factorzen.config.constants import TRADING_DAYS_PER_YEAR
    from factorzen.daily.evaluation.signal_backtest import (
        _ann_ret_sharpe,
        _max_drawdown,
        periods_per_year,
    )

    # -- MaxDD 必须含首日回撤(交易轨 cum 前置 1.0,信号轨不补就系统性偏乐观) --
    nav = np.array([0.90, 0.95, 1.10])
    # 独立手算:起点 1.0 → 0.90 是 −10%,之后 peak 仍是 1.0,0.95/1.0−1=−5%
    assert _max_drawdown(nav) == pytest.approx(-0.10, abs=1e-12), (
        "首日下跌未计入回撤 → 数字偏乐观"
    )
    # 单调上涨无回撤
    assert _max_drawdown(np.array([1.05, 1.10])) == pytest.approx(0.0, abs=1e-12)

    # -- Sharpe 用 ddof=0(与引擎 np.std 默认一致) --
    rets = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    ar, sh = _ann_ret_sharpe(rets, float(TRADING_DAYS_PER_YEAR))
    # 独立手算,不调用被测 helper
    mean = float(np.mean(rets))
    std0 = float(np.std(rets))  # ddof=0
    exp_ar = mean * TRADING_DAYS_PER_YEAR
    exp_sh = exp_ar / (std0 * np.sqrt(TRADING_DAYS_PER_YEAR))
    assert ar == pytest.approx(exp_ar, rel=1e-12)
    assert sh == pytest.approx(exp_sh, rel=1e-12)
    # ddof=1 会给出不同答案 —— 确认我们没用它
    std1 = float(np.std(rets, ddof=1))
    assert abs(std1 - std0) > 1e-6, "构造前提:两种 ddof 应有可分辨差异"
    assert sh != pytest.approx(exp_ar / (std1 * np.sqrt(TRADING_DAYS_PER_YEAR)), rel=1e-9)

    # -- 年化基数随 frequency,不再硬编码 252 --
    assert periods_per_year("daily") == float(TRADING_DAYS_PER_YEAR)
    assert periods_per_year("weekly") == 52.0
    assert periods_per_year("monthly") == 12.0
    assert periods_per_year("") == float(TRADING_DAYS_PER_YEAR)  # 缺省回落日频


def test_grouping_is_row_order_invariant():
    """并列值的分组不得依赖输入行序,否则同一份数据两次跑出不同结果。

    ordinal rank 按行序打散并列;并列块横跨分组边界时,正序与逆序会得到不同分组
    (实测 monotonicity 的 ols_slope 从 0.200 变 0.100)。离散/事件类因子并列极多。
    """
    from factorzen.daily.evaluation.grouping import assign_quantile_groups

    d = date(2024, 1, 1)
    # 6 只票 3 组,因子值 [0,0,0,0,1,1] —— 并列块横跨 g0/g1 边界
    rows = [
        {"trade_date": d, "ts_code": c, "factor_clean": v}
        for c, v in zip(["a", "b", "c", "d", "e", "f"], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0], strict=True)
    ]
    fwd = pl.DataFrame(rows)
    rev = pl.DataFrame(list(reversed(rows)))

    g1 = assign_quantile_groups(fwd, n_groups=3).sort("ts_code")
    g2 = assign_quantile_groups(rev, n_groups=3).sort("ts_code")
    assert g1["group"].to_list() == g2["group"].to_list(), (
        f"行序不同得到不同分组:{g1['group'].to_list()} vs {g2['group'].to_list()}"
    )


def test_geometric_vs_arithmetic_annualization():
    """几何年化必须与净值曲线自洽;算术年化在高波动下会翻号。

    实测案例:momentum_20d/csi300/2023-2024,日波动 1.7% 的分位组算术年化 +1.37%
    而累计净值 0.96(实际亏损)——柱状图与净值图在同一份报告里结论相反。
    """
    import numpy as np

    from factorzen.daily.evaluation.signal_backtest import (
        _ann_ret_sharpe,
        cum_excluding_top_days,
        geometric_ann_ret,
    )

    # 构造:高波动、算术均值为正但复利后亏损的序列
    # +10% 与 −9% 交替:算术均值 +0.5%/期,几何 (1.1*0.91)^(n/2) = 1.001^(n/2) 略正
    # 用 +20%/−18% 放大方差拖累:算术 +1%,几何 (1.2*0.82)=0.984 → 亏
    rets = np.array([0.20, -0.18] * 60)
    ppy = 252.0
    ar, _ = _ann_ret_sharpe(rets, ppy)
    geo = geometric_ann_ret(rets, ppy)
    cum = float(np.prod(1.0 + rets) - 1.0)

    assert ar > 0, f"构造前提:算术年化应为正,实得 {ar}"
    assert cum < 0, f"构造前提:实际累计应为亏损,实得 {cum}"
    assert geo < 0, f"几何年化必须与累计亏损同号,实得 {geo}(算术 {ar})"
    # 独立手算几何年化
    expected = float(np.prod(1.0 + rets)) ** (ppy / rets.size) - 1.0
    assert geo == pytest.approx(expected, rel=1e-12)

    # -- 极端日剔除:给绝对数字而非占比 --
    base = np.full(50, 0.001)
    spiked = base.copy()
    spiked[10] = 0.25  # 一根不可交易的跳空
    full = float(np.prod(1.0 + spiked) - 1.0)
    ex1 = cum_excluding_top_days(spiked, 1)
    # 手算:剔除后就是 50 个 0.001 里的 49 个
    expected_ex1 = float(1.001**49 - 1.0)
    assert ex1 == pytest.approx(expected_ex1, rel=1e-12)
    assert full - ex1 > 0.2, "剔除极端日应显著改变累计收益,否则告警无意义"
    # k >= 样本数 → NaN 而非崩溃
    assert not np.isfinite(cum_excluding_top_days(np.array([0.01]), 3))
