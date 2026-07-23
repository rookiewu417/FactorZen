"""research/report 路径：逐日 PIT membership 过滤（替代期末快照）。

覆盖：
1. filter_frame_by_membership helper（dtype 对齐 + 调出/调入反例）
2. 稳定成分零回归（membership = 期末快照 × 每日 → 与整窗 filter 等价）
3. daily_single / generate_report / research_run 取用 membership union 而非 end snapshot
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

# ── 合成 membership：A 只在 1 月是成分，2 月调出；C 2 月才调入 ──────────────
_JAN = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
_FEB = [date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 5)]
_ALL_DAYS = _JAN + _FEB


def _membership_with_exit() -> pl.DataFrame:
    """A 1 月在、2 月出；B 全程；C 2 月才入。trade_date 为 Utf8 YYYYMMDD。"""
    rows: list[dict[str, str]] = []
    for d in _JAN:
        ds = d.strftime("%Y%m%d")
        for c in ("A.SZ", "B.SZ"):
            rows.append({"trade_date": ds, "ts_code": c})
    for d in _FEB:
        ds = d.strftime("%Y%m%d")
        for c in ("B.SZ", "C.SZ"):
            rows.append({"trade_date": ds, "ts_code": c})
    return pl.DataFrame(rows)


def _factor_frame_all_codes() -> pl.DataFrame:
    """整窗全股票因子帧（模拟 FactorDataContext 用 union 拉取后的结果）。"""
    rows = []
    for d in _ALL_DAYS:
        for c in ("A.SZ", "B.SZ", "C.SZ"):
            rows.append(
                {
                    "trade_date": d,
                    "ts_code": c,
                    "factor_value": 1.0,
                    "factor_clean": 0.5,
                }
            )
    return pl.DataFrame(rows)


# ── helper 单测 ─────────────────────────────────────────────────────────────
def test_filter_frame_by_membership_suite():
    """调出反例：A 后半段不进评估截面（旧 end-snapshot 若期末不含 A 会整窗删=幸存偏差）。；帧 trade_date 已是 Utf8 时也能 join。；零回归：窗口内无调样时，过滤结果 ≡ 用 union codes 整窗 is_in 过滤。"""
    # -- 原 test_filter_frame_by_membership_excludes_delisted_half --
    def _section_0_test_filter_frame_by_membership_excludes_delisted_half():
        from factorzen.pipelines.daily_single import filter_frame_by_membership

        factor_df = _factor_frame_all_codes()
        mem = _membership_with_exit()
        out = filter_frame_by_membership(factor_df, mem)

        a_jan = out.filter(
            (pl.col("ts_code") == "A.SZ") & pl.col("trade_date").is_in(_JAN)
        )
        a_feb = out.filter(
            (pl.col("ts_code") == "A.SZ") & pl.col("trade_date").is_in(_FEB)
        )
        assert a_jan.height == len(_JAN)
        assert a_feb.height == 0

        c_jan = out.filter(
            (pl.col("ts_code") == "C.SZ") & pl.col("trade_date").is_in(_JAN)
        )
        c_feb = out.filter(
            (pl.col("ts_code") == "C.SZ") & pl.col("trade_date").is_in(_FEB)
        )
        assert c_jan.height == 0  # look-ahead 消除：调入前不进
        assert c_feb.height == len(_FEB)

        # B 全程保留
        assert out.filter(pl.col("ts_code") == "B.SZ").height == len(_ALL_DAYS)

    _section_0_test_filter_frame_by_membership_excludes_delisted_half()

    # -- 原 test_filter_frame_by_membership_utf8_trade_date --
    def _section_1_test_filter_frame_by_membership_utf8_trade_date():
        from factorzen.pipelines.daily_single import filter_frame_by_membership

        df = pl.DataFrame(
            {
                "trade_date": ["20240102", "20240201"],
                "ts_code": ["A.SZ", "A.SZ"],
                "factor_clean": [1.0, 2.0],
            }
        )
        mem = pl.DataFrame(
            {
                "trade_date": ["20240102"],
                "ts_code": ["A.SZ"],
            }
        )
        out = filter_frame_by_membership(df, mem)
        assert out.height == 1
        assert out["trade_date"][0] == "20240102"

    _section_1_test_filter_frame_by_membership_utf8_trade_date()

    # -- 原 test_filter_stable_membership_zero_regression --
    def _section_2_test_filter_stable_membership_zero_regression():
        from factorzen.pipelines.daily_single import filter_frame_by_membership

        codes = ["A.SZ", "B.SZ"]
        rows_f = []
        rows_m = []
        for d in _ALL_DAYS:
            ds = d.strftime("%Y%m%d")
            for c in codes:
                rows_f.append(
                    {"trade_date": d, "ts_code": c, "factor_clean": 1.0}
                )
                rows_m.append({"trade_date": ds, "ts_code": c})
        factor_df = pl.DataFrame(rows_f)
        mem = pl.DataFrame(rows_m)

        pit = filter_frame_by_membership(factor_df, mem)
        legacy = factor_df.filter(pl.col("ts_code").is_in(codes))
        assert pit.sort(["trade_date", "ts_code"]).equals(
            legacy.sort(["trade_date", "ts_code"])
        )

    _section_2_test_filter_stable_membership_zero_regression()


def test_load_pit_membership_suite():
    """load_pit_membership：ts_codes=并集；membership 按日。；test_load_pit_membership_empty_named_index_fails；动态池 get_universe_membership 抛 → 明确报错，不静默回退 end snapshot。"""
    # -- 原 test_load_pit_membership_returns_union_and_membership --
    def _section_0_test_load_pit_membership_returns_union_and_membership(mp):
        from factorzen.pipelines import daily_single as ds

        mem = _membership_with_exit()
        mp.setattr(
            "factorzen.core.universe.get_universe_membership",
            lambda s, e, u: mem,
        )
        mp.setattr(
            ds,
            "get_universe",
            lambda d, u: pl.DataFrame(
                {"ts_code": ["B.SZ", "C.SZ"], "industry": ["银行", "科技"]}
            ),
        )

        membership, ts_codes, universe_meta = ds.load_pit_membership(
            "20240102", "20240205", "csi300"
        )
        assert set(ts_codes) == {"A.SZ", "B.SZ", "C.SZ"}  # 并集含调出 A
        assert membership.height == mem.height
        # industry meta 仍来自期末快照（中性化/归因用）
        assert set(universe_meta["ts_code"].to_list()) == {"B.SZ", "C.SZ"}

    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_load_pit_membership_returns_union_and_membership(mp)

    # -- 原 test_load_pit_membership_empty_named_index_fails --
    def _section_1_test_load_pit_membership_empty_named_index_fails(mp):
        from factorzen.pipelines import daily_single as ds

        mp.setattr(
            "factorzen.core.universe.get_universe_membership",
            lambda *a, **k: pl.DataFrame(
                schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8}
            ),
        )
        with pytest.raises(RuntimeError, match=r"membership|空|PIT"):
            ds.load_pit_membership("20240102", "20240205", "csi300")

    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_load_pit_membership_empty_named_index_fails(mp)

    # -- 原 test_load_pit_membership_raises_on_dynamic_pool --
    def _section_2_test_load_pit_membership_raises_on_dynamic_pool(mp):
        from factorzen.pipelines import daily_single as ds

        def _boom(*a, **k):
            raise ValueError("动态过滤池，不支持逐日 membership")

        mp.setattr(
            "factorzen.core.universe.get_universe_membership", _boom
        )
        with pytest.raises(ValueError, match=r"membership|动态"):
            ds.load_pit_membership("20240102", "20240205", "daily_default")

    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_load_pit_membership_raises_on_dynamic_pool(mp)


# ── daily_single / generate_report / research_run 接线 smoke ────────────────
def test_pipeline_membership_integration_suite(tmp_path):
    """smoke：mock 到 IC 前，断言评估截面无 A 的 2 月行，且落盘 membership。；research_run：面板拉取用 membership 并集，而非 get_universe(end)。；generate_report：IC 截面经 membership 过滤。"""
    # -- 原 test_daily_single_filters_eval_cross_section_by_membership --
    def _section_0_test_daily_single_filters_eval_cross_section_by_membership(mp, tmp_path):
        from factorzen.config.research import RunConfig
        from factorzen.pipelines import daily_single as ds

        mem = _membership_with_exit()
        captured: dict = {}

        class DummyFactor:
            name = "dummy_pit"
            description = "pit test"
            required_data = ["daily"]
            lookback_days = 1
            category = "test"

            def compute(self, ctx):
                return _factor_frame_all_codes().select(
                    ["trade_date", "ts_code", "factor_value"]
                )

            def validate(self, df):
                return {"coverage": 1.0, "n_rows": df.height}

        class FakeCtx:
            def __init__(self, **kw):
                captured["ctx_universe"] = kw.get("universe")

            @property
            def daily(self):
                rows = [
                    {
                        "trade_date": d,
                        "ts_code": c,
                        "close": 10.0,
                        "close_adj": 10.0,
                        "open": 10.0,
                        "open_adj": 10.0,  # 可实现口径默认 exec_price_col=open_adj
                        "high": 10.0,
                        "low": 10.0,
                        "vol": 1e5,
                        "amount": 1e6,
                    }
                    for d in _ALL_DAYS
                    for c in ("A.SZ", "B.SZ", "C.SZ")
                ]
                return pl.DataFrame(rows).lazy()

        def fake_compute_rank_ic(clean_df, ret_df, **kw):
            captured["ic_clean"] = clean_df
            raise RuntimeError("STOP_AFTER_IC")

        mp.setattr(ds, "get_factor", lambda n: DummyFactor)
        mp.setattr(ds, "get_trade_dates", lambda s, e: _ALL_DAYS)
        mp.setattr(ds, "ensure_data_for_daily_run", lambda **kw: None)
        mp.setattr(
            "factorzen.core.universe.get_universe_membership",
            lambda s, e, u: mem,
        )
        mp.setattr(
            ds,
            "get_universe",
            lambda d, u: pl.DataFrame(
                {"ts_code": ["B.SZ", "C.SZ"], "industry": ["银行", "科技"]}
            ),
        )
        mp.setattr(ds, "FactorDataContext", FakeCtx)
        mp.setattr(ds, "compute_rank_ic", fake_compute_rank_ic)
        mp.setattr(
            ds,
            "build_daily_quality_report",
            lambda **kw: {"warnings": [], "status": "ok"},
        )
        mp.setattr(
            ds, "daily_report_output_dir", lambda name: tmp_path / "reports" / name
        )

        # 预处理恒等（保留 factor_value → factor_clean）
        mp.setattr(
            ds,
            "_preprocess_factor",
            lambda factor_df, cfg, **kw: factor_df.with_columns(
                pl.col("factor_value").alias("factor_clean")
            ),
        )

        args = SimpleNamespace(
            factor="dummy_pit",
            start="20240102",
            end="20240205",
            universe="csi300",
            frequency="daily",
            benchmark=None,
            seed=None,
            metrics_out=None,
            # 隔离：不走 ensure_factor_store_panel（会写真实 workspace/factors）
            no_factor_cache=True,
        )
        cfg = RunConfig(factor="dummy_pit", start="20240102", end="20240205")
        run_dir = tmp_path / "run"

        with pytest.raises(RuntimeError, match="STOP_AFTER_IC"):
            ds.run_factor_eval(args, cfg, run_dir)

        assert set(captured["ctx_universe"]) == {"A.SZ", "B.SZ", "C.SZ"}
        clean = captured["ic_clean"]
        assert (
            clean.filter(
                (pl.col("ts_code") == "A.SZ") & pl.col("trade_date").is_in(_FEB)
            ).height
            == 0
        ), "调出股后半段不得进入 IC 评估截面"
        assert (
            clean.filter(
                (pl.col("ts_code") == "A.SZ") & pl.col("trade_date").is_in(_JAN)
            ).height
            == len(_JAN)
        ), "调出前半段应保留"

        # evaluations 不落 universe parquet；membership 仅内存使用
        assert not (run_dir / "universe.parquet").exists()
        assert not list(run_dir.glob("*.parquet"))

    _tp0 = tmp_path / "_s0"
    _tp0.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_0_test_daily_single_filters_eval_cross_section_by_membership(mp, _tp0)

    # -- 原 test_research_run_uses_membership_union --
    def _section_1_test_research_run_uses_membership_union(mp, tmp_path):
        from factorzen.pipelines.research_run import run_research

        mem = _membership_with_exit()
        calls: dict = {"get_membership": 0, "ctx_uni": None}
        tdates = _ALL_DAYS
        codes_union = ["A.SZ", "B.SZ", "C.SZ"]

        def fake_run_mine(**kw):
            return {
                "session_dir": str(tmp_path / "mine"),
                "candidates": [{"expression": "close", "passed": True}],
            }

        def fake_get_universe(d, name):
            # 期末快照不含 A（调出）——若错误地用 end snapshot 作 ctx，会丢 A
            ds = str(d).replace("-", "")[:8]
            if ds >= "20240201":
                return pl.DataFrame(
                    {"ts_code": ["B.SZ", "C.SZ"], "industry": ["银行", "科技"]}
                )
            return pl.DataFrame(
                {"ts_code": ["A.SZ", "B.SZ"], "industry": ["银行", "地产"]}
            )

        def fake_membership(s, e, u):
            calls["get_membership"] += 1
            return mem

        class FakeCtx:
            def __init__(self, **kw):
                calls["ctx_uni"] = kw.get("universe")

        class FakeExpr:
            def __init__(self, expression=None, **kw):
                self.expression = expression

            def compute(self, ctx):
                return pl.DataFrame(
                    [
                        {"trade_date": d, "ts_code": c, "factor_value": 0.1}
                        for d in tdates
                        for c in codes_union
                    ]
                )

        def fake_fetch_daily(start, end):
            return pl.DataFrame(
                [
                    {"trade_date": d, "ts_code": c, "close": 10.0}
                    for d in tdates
                    for c in codes_union
                ]
            )

        class FakeRisk:
            def build(self, daily, daily_basic, stocks, start, end, **_panels):
                codes = stocks["ts_code"].to_list()
                return SimpleNamespace(
                    factor_exposures=SimpleNamespace(codes=codes),
                    factor_names=["size"],
                )

        def fake_portfolio(alpha, risk_result, **kw):
            run_dir = Path(kw["out_dir"]) / kw["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            codes = list(risk_result.factor_exposures.codes)
            n = max(len(codes), 1)
            pl.DataFrame(
                {
                    "ts_code": codes,
                    "target_weight": [1.0 / n] * len(codes),
                    "prev_weight": [0.0] * len(codes),
                }
            ).write_parquet(run_dir / "weights.parquet")
            pl.DataFrame({"type": ["x"], "key": ["y"], "value": [1.0]}).write_csv(
                run_dir / "attribution.csv"
            )
            pl.DataFrame({"metric": ["te"], "value": [0.01]}).write_csv(
                run_dir / "risk_summary.csv"
            )
            (run_dir / "manifest.json").write_text(
                f'{{"signal_date":"{kw["signal_date"]}","status":"optimal"}}'
            )
            return {
                "run_dir": str(run_dir),
                "status": "optimal",
                "n_holdings": len(codes),
            }

        def fake_sim(dirs, daily, *, out_dir, run_id, cost_model=None):
            run_dir = Path(out_dir) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "trade_date": tdates[:3],
                    "net_return": [0.01, -0.02, 0.03],
                    "nav": [1.01, 0.99, 1.02],
                }
            ).write_parquet(run_dir / "nav.parquet")
            (run_dir / "metrics.json").write_text('{"sharpe":1.0,"ann_ret":0.1}')
            return {"run_dir": str(run_dir), "sharpe": 1.0, "ann_ret": 0.1}

        mp.setattr("factorzen.pipelines.factor_mine.run_mine", fake_run_mine)
        mp.setattr(
            "factorzen.core.universe.get_universe_membership", fake_membership
        )
        mp.setattr("factorzen.core.universe.get_universe", fake_get_universe)
        # load_pit_membership 绑定 daily_single.get_universe（import-time）
        mp.setattr(
            "factorzen.pipelines.daily_single.get_universe", fake_get_universe
        )
        mp.setattr("factorzen.daily.data.context.FactorDataContext", FakeCtx)
        mp.setattr("factorzen.discovery.factor.ExpressionFactor", FakeExpr)
        mp.setattr("factorzen.core.loader.fetch_daily", fake_fetch_daily)
        mp.setattr(
            "factorzen.core.loader.fetch_daily_basic",
            lambda s, e: pl.DataFrame(
                {"trade_date": [tdates[0]], "ts_code": ["B.SZ"]}
            ),
        )
        mp.setattr("factorzen.risk.model.RiskModel", FakeRisk)
        mp.setattr(
            "factorzen.pipelines.portfolio_build.run_portfolio", fake_portfolio
        )
        mp.setattr(
            "factorzen.sim.engine.run_portfolio_simulation", fake_sim
        )
        mp.setattr(
            "factorzen.reports.portfolio_report.generate_portfolio_report",
            lambda **kw: "<html/>",
        )

        res = run_research(
            start="20240102",
            end="20240205",
            universe="csi300",
            n_trials=5,
            seed=1,
            rebalance_days=2,
            warmup=1,
            out_root=str(tmp_path),
        )
        assert calls["get_membership"] >= 1
        assert set(calls["ctx_uni"]) == {"A.SZ", "B.SZ", "C.SZ"}
        assert res["n_rebalances"] >= 1

    _tp1 = tmp_path / "_s1"
    _tp1.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_1_test_research_run_uses_membership_union(mp, _tp1)

    # -- 原 test_generate_report_filters_by_membership --
    def _section_2_test_generate_report_filters_by_membership(mp, tmp_path):
        from factorzen.config.research import RunConfig
        from factorzen.pipelines import generate_report as gr

        mem = _membership_with_exit()
        captured: dict = {}

        class DummyFactor:
            name = "dummy_pit"
            description = "x"
            required_data = ["daily"]
            lookback_days = 1

            def compute(self, ctx):
                return _factor_frame_all_codes().select(
                    ["trade_date", "ts_code", "factor_value"]
                )

            def validate(self, df):
                return {"coverage": 1.0}

        class FakeCtx:
            def __init__(self, **kw):
                captured["ctx_uni"] = kw.get("universe")

        def fake_rank_ic(clean_df, ret_df, **kw):
            captured["ic_clean"] = clean_df
            raise RuntimeError("STOP_AFTER_IC")

        mp.setattr(gr, "get_factor", lambda n: DummyFactor)
        mp.setattr(gr, "get_trade_dates", lambda s, e: _ALL_DAYS)
        mp.setattr(gr, "fetch_daily", lambda s, e: None)
        mp.setattr(
            "factorzen.core.universe.get_universe_membership",
            lambda s, e, u: mem,
        )
        _uni_meta = pl.DataFrame(
            {"ts_code": ["B.SZ", "C.SZ"], "industry": ["银行", "科技"]}
        )
        mp.setattr(
            "factorzen.pipelines.daily_single.get_universe", lambda d, u: _uni_meta
        )
        mp.setattr(gr, "FactorDataContext", FakeCtx)
        mp.setattr(
            gr,
            "_preprocess_factor",
            lambda df, cfg, **kw: df.with_columns(
                pl.col("factor_value").alias("factor_clean")
            ),
        )
        mp.setattr(
            gr,
            "_load_daily_with_close_adj",
            lambda s, e: pl.DataFrame(
                [
                    {
                        "trade_date": d,
                        "ts_code": c,
                        "close": 10.0,
                        "close_adj": 10.0,
                    }
                    for d in _ALL_DAYS
                    for c in ("A.SZ", "B.SZ", "C.SZ")
                ]
            ),
        )
        mp.setattr(gr, "compute_rank_ic", fake_rank_ic)
        mp.setattr(
            gr,
            "build_daily_quality_report",
            lambda **kw: {"warnings": []},
        )
        mp.setattr(
            gr, "_save_quality_report", lambda *a, **k: tmp_path / "q.json"
        )

        args = SimpleNamespace(
            factor="dummy_pit",
            start="20240102",
            end="20240205",
            universe="csi300",
            frequency="daily",
            benchmark=None,
            # 隔离：不走 ensure_factor_store_panel（会写真实 workspace/factors）
            no_factor_cache=True,
        )
        cfg = RunConfig(factor="dummy_pit", start="20240102", end="20240205")
        cfg.walk_forward.enabled = False

        with pytest.raises(RuntimeError, match="STOP_AFTER_IC"):
            gr._run(args, cfg, tmp_path / "run")

        assert set(captured["ctx_uni"]) == {"A.SZ", "B.SZ", "C.SZ"}
        clean = captured["ic_clean"]
        assert (
            clean.filter(
                (pl.col("ts_code") == "A.SZ") & pl.col("trade_date").is_in(_FEB)
            ).height
            == 0
        )
        assert (
            clean.filter(
                (pl.col("ts_code") == "A.SZ") & pl.col("trade_date").is_in(_JAN)
            ).height
            == len(_JAN)
        )

    _tp2 = tmp_path / "_s2"
    _tp2.mkdir(exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        _section_2_test_generate_report_filters_by_membership(mp, _tp2)


