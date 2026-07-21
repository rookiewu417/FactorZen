"""组合层直通回测：A 落 OOS 分数面板 + B ``fz combine backtest`` 桥命令。

全部合成数据离线，不依赖 workspace/ 真库。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.experiment import run_combination_experiment

# ── helpers ──────────────────────────────────────────────────────────────


def _synthetic_factor_ret(
    *,
    n_days: int = 100,
    n_stocks: int = 20,
    n_factors: int = 3,
    seed: int = 0,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """合成多因子面板 + 与第一因子相关的收益（保证 OOS 有有效期数）。"""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    date_strs = [x.strftime("%Y%m%d") for x in days]
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]

    factor_dfs: dict[str, pl.DataFrame] = {}
    factor_vals: dict[str, np.ndarray] = {}
    for fi in range(n_factors):
        name = f"f{fi}"
        rows = []
        mat = np.zeros((n_days, n_stocks), dtype=float)
        for di, ds in enumerate(date_strs):
            vals = rng.standard_normal(n_stocks)
            mat[di] = vals
            for si, code in enumerate(codes):
                rows.append(
                    {"trade_date": ds, "ts_code": code, "factor_value": float(vals[si])}
                )
        factor_dfs[name] = pl.DataFrame(rows)
        factor_vals[name] = mat

    ret_rows = []
    for di, ds in enumerate(date_strs):
        noise = rng.standard_normal(n_stocks) * 0.3
        rets = 0.5 * factor_vals["f0"][di] + noise
        for si, code in enumerate(codes):
            ret_rows.append({"trade_date": ds, "ts_code": code, "ret": float(rets[si])})
    return factor_dfs, pl.DataFrame(ret_rows)


def _synthetic_scores_and_prices(
    *,
    n_days: int = 120,
    n_stocks: int = 20,
    seed: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame, str, str]:
    """已知单调分数：高编号股分数更高，且价格漂移更大 → 做多高分有正收益。"""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    days: list[date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    start_s = days[0].strftime("%Y%m%d")
    end_s = days[-1].strftime("%Y%m%d")

    score_rows = []
    price_rows = []
    px = {c: 10.0 + i * 0.1 for i, c in enumerate(codes)}
    for day in days:
        ds = day.strftime("%Y%m%d")
        for i, code in enumerate(codes):
            # 单调分数：编号越大分数越高
            score_rows.append(
                {"trade_date": ds, "ts_code": code, "score": float(i)}
            )
            # 高分股日均漂移略高 → 多空/long-only 有可观毛收益，成本差可检出
            drift = 0.0005 + 0.0003 * (i / max(n_stocks - 1, 1))
            shock = rng.standard_normal() * 0.005
            open_px = px[code]
            close_px = open_px * (1.0 + drift + shock)
            pre_close = open_px
            price_rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "open": open_px,
                    "close": close_px,
                    "pre_close": pre_close,
                    "pct_chg": (close_px / pre_close - 1.0) * 100,
                    "vol": 1_000_000.0,
                    "amount": 1e10,
                    "close_adj": close_px,
                    "open_adj": open_px,
                }
            )
            px[code] = close_px

    scores = pl.DataFrame(score_rows)
    prices = pl.DataFrame(price_rows)
    return scores, prices, start_s, end_s


# ── A: OOS 分数落盘 ──────────────────────────────────────────────────────


def test_oos_scores_persisted_schema_and_no_fold_overlap(tmp_path: Path) -> None:
    """合成 3 因子跑 combine → oos_scores/<method>.parquet 存在、schema 对、
    折间日期零重叠、与 comparison 指标同窗（n_periods 覆盖的日期 ⊆ oos 日期）。"""
    factor_dfs, ret_df = _synthetic_factor_ret(n_days=100, n_stocks=20, n_factors=3)
    cv = PurgedWalkForwardCV(train_days=40, test_days=15, purge_days=5)
    methods = ["equal_weight", "ic_weighted"]
    res = run_combination_experiment(
        factor_dfs,
        ret_df,
        cv=cv,
        methods=methods,
        seed=0,
        out_dir=str(tmp_path / "combinations"),
        run_id="oos_a",
        command=["combine", "run"],
    )

    run_dir = Path(res["run_dir"])
    oos_paths = res.get("oos_scores")
    assert isinstance(oos_paths, dict)
    assert set(oos_paths) == set(methods)

    comparison = res["comparison"]
    for method in methods:
        path = Path(oos_paths[method])
        assert path.exists(), f"missing oos_scores for {method}"
        assert path == run_dir / "oos_scores" / f"{method}.parquet"

        scores = pl.read_parquet(path)
        assert set(scores.columns) == {"trade_date", "ts_code", "score"}
        assert scores.height > 0
        assert scores["score"].null_count() < scores.height

        # 折间日期零重叠：读 combined 带 fold_id 的面板校验
        combined = pl.read_parquet(run_dir / f"combined_{method}.parquet")
        assert "fold_id" in combined.columns
        by_fold = (
            combined.select(["fold_id", "trade_date"])
            .unique()
            .group_by("fold_id")
            .agg(pl.col("trade_date"))
        )
        seen: set[str] = set()
        for row in by_fold.iter_rows(named=True):
            dates = set(row["trade_date"])
            assert not (seen & dates), f"fold date overlap for {method}: {seen & dates}"
            seen |= dates

        # 与 res 指标同窗：oos 日期覆盖 comparison 使用的 test 窗
        oos_dates = set(scores["trade_date"].cast(pl.Utf8).to_list())
        combined_dates = set(combined["trade_date"].cast(pl.Utf8).to_list())
        assert oos_dates == combined_dates
        n_periods = int(
            comparison.filter(pl.col("method") == method)["n_periods"][0]
        )
        assert n_periods > 0
        assert n_periods <= len(oos_dates)

    # manifest 记路径
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "oos_scores" in manifest
    assert set(manifest["oos_scores"]) == set(methods)


def test_oos_scores_overlap_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """人为制造折间日期重叠时必须 raise，不可静默去重。"""
    from factorzen.research.combination import experiment as exp_mod

    factor_dfs, ret_df = _synthetic_factor_ret(n_days=80, n_stocks=10, n_factors=2)
    cv = PurgedWalkForwardCV(train_days=30, test_days=15, purge_days=3)

    real_combine = exp_mod._combine

    def _leaky_combine(method, factor_dfs, ret_df, cv, seed, **kw):
        out = real_combine(method, factor_dfs, ret_df, cv, seed, **kw)
        if out.is_empty() or "fold_id" not in out.columns:
            return out
        # 把 fold 0 的日期也标到 fold 1，制造重叠
        folds = out["fold_id"].unique().sort().to_list()
        if len(folds) < 2:
            return out
        f0_dates = (
            out.filter(pl.col("fold_id") == folds[0])["trade_date"].unique().to_list()
        )
        if not f0_dates:
            return out
        leak_date = f0_dates[0]
        # 在 fold 1 上复制 leak_date 的一行，改 fold_id 已是 1... 改为：把 fold1 某行日期改成 leak
        f1 = out.filter(pl.col("fold_id") == folds[1])
        if f1.is_empty():
            return out
        rest = out.filter(pl.col("fold_id") != folds[1])
        f1_leaked = f1.with_columns(
            pl.when(pl.int_range(0, pl.len()) == 0)
            .then(pl.lit(leak_date))
            .otherwise(pl.col("trade_date"))
            .alias("trade_date")
        )
        return pl.concat([rest, f1_leaked])

    monkeypatch.setattr(exp_mod, "_combine", _leaky_combine)
    with pytest.raises(ValueError, match=r"overlap|重叠"):
        run_combination_experiment(
            factor_dfs,
            ret_df,
            cv=cv,
            methods=["equal_weight"],
            seed=0,
            out_dir=str(tmp_path / "combinations"),
            run_id="overlap_bug",
        )


# ── B: combine backtest 桥 ───────────────────────────────────────────────


def _patch_market_data(
    monkeypatch: pytest.MonkeyPatch,
    prices: pl.DataFrame,
    *,
    module: str = "factorzen.pipelines.combine_backtest",
) -> None:
    """离线注入行情 + 空 ST + 全成分 membership。"""
    codes = prices.select("ts_code").unique()["ts_code"].to_list()
    dates = prices.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    # trade_date 可能是 date
    def _ds(x):
        if isinstance(x, date):
            return x.strftime("%Y%m%d")
        return str(x).replace("-", "")[:8]

    membership = pl.DataFrame(
        [
            {"trade_date": _ds(d), "ts_code": c}
            for d in dates
            for c in codes
        ]
    )

    def fake_load_market_panel(**kwargs):
        return {
            "price_df": prices,
            "membership": membership,
            "is_st_by_date": {},
            "ts_codes": codes,
        }

    monkeypatch.setattr(f"{module}.load_market_panel", fake_load_market_panel)


def test_combine_backtest_cli_cost_transmission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """合成分数面板 → parser→handler 跑通；cost-bps=0 净值严格高于 cost-bps=20。"""
    from factorzen.cli.main import build_parser

    scores, prices, start, end = _synthetic_scores_and_prices()
    scores_path = tmp_path / "scores.parquet"
    scores.write_parquet(scores_path)
    out_root = tmp_path / "combine_backtests"
    _patch_market_data(monkeypatch, prices)

    parser = build_parser()

    def _run(cost_bps: float, run_id: str) -> dict:
        args = parser.parse_args(
            [
                "combine",
                "backtest",
                "--scores",
                str(scores_path),
                "--start",
                start,
                "--end",
                end,
                "--universe",
                "csi300",
                "--strategy",
                "quantile_ls_5",
                "--cost-bps",
                str(cost_bps),
                "--out-dir",
                str(out_root),
                "--run-id",
                run_id,
            ]
        )
        rc = args.func(args)
        assert rc == 0
        run_dir = out_root / run_id
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "nav.parquet").exists()
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        nav = pl.read_parquet(run_dir / "nav.parquet")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        # 铁律字段
        for key in (
            "start",
            "end",
            "universe",
            "command",
            "git_sha",
            "scores_source",
            "cost_bps",
            "strategy",
        ):
            assert key in manifest, f"manifest missing {key}"
        return {"metrics": metrics, "nav": nav, "manifest": manifest}

    zero = _run(0.0, "cost0")
    costly = _run(20.0, "cost20")

    # 终值净值：有成本严格更低
    nav0 = float(zero["nav"].sort("trade_date")["nav"][-1])
    nav20 = float(costly["nav"].sort("trade_date")["nav"][-1])
    assert nav20 < nav0, f"cost-bps 未传导: nav0={nav0}, nav20={nav20}"

    # metrics 含年化/SR/最大回撤/换手/成本
    for m in (zero["metrics"], costly["metrics"]):
        for k in ("ann_ret", "sharpe", "max_dd", "avg_turnover", "total_cost"):
            assert k in m


def test_combine_backtest_run_dir_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--run-dir + --method 读 oos_scores 入口。"""
    from factorzen.cli.main import build_parser

    # 先产一份假 combine 产物
    scores, prices, start, end = _synthetic_scores_and_prices(n_days=80)
    combine_run = tmp_path / "combinations" / "run1"
    oos_dir = combine_run / "oos_scores"
    oos_dir.mkdir(parents=True)
    scores.write_parquet(oos_dir / "equal_weight.parquet")
    (combine_run / "manifest.json").write_text("{}", encoding="utf-8")

    out_root = tmp_path / "combine_backtests"
    _patch_market_data(monkeypatch, prices)
    parser = build_parser()
    args = parser.parse_args(
        [
            "combine",
            "backtest",
            "--run-dir",
            str(combine_run),
            "--method",
            "equal_weight",
            "--start",
            start,
            "--end",
            end,
            "--out-dir",
            str(out_root),
            "--run-id",
            "from_run_dir",
            "--cost-bps",
            "0",
        ]
    )
    assert args.func(args) == 0
    assert (out_root / "from_run_dir" / "metrics.json").exists()
    manifest = json.loads(
        (out_root / "from_run_dir" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scores_source"]["type"] == "run_dir"
    assert manifest["scores_source"]["method"] == "equal_weight"


def test_combine_backtest_scores_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--scores 入口单独跑通。"""
    from factorzen.cli.main import build_parser

    scores, prices, start, end = _synthetic_scores_and_prices(n_days=60, n_stocks=15)
    scores_path = tmp_path / "panel.parquet"
    scores.write_parquet(scores_path)
    out_root = tmp_path / "out"
    _patch_market_data(monkeypatch, prices)
    parser = build_parser()
    args = parser.parse_args(
        [
            "combine",
            "backtest",
            "--scores",
            str(scores_path),
            "--start",
            start,
            "--end",
            end,
            "--strategy",
            "topn_long_only",
            "--out-dir",
            str(out_root),
            "--run-id",
            "scores_only",
            "--cost-bps",
            "0",
        ]
    )
    assert args.func(args) == 0
    manifest = json.loads(
        (out_root / "scores_only" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scores_source"]["type"] == "scores"
    assert manifest["strategy"] == "topn_long_only"


def test_score_col_auto_detect_and_multi_col_error(tmp_path: Path) -> None:
    """单数值列自动识别；多数值列未指定 --score-col → ValueError。"""
    from factorzen.pipelines.combine_backtest import load_scores_panel

    base = pl.DataFrame(
        {
            "trade_date": ["20230103", "20230103"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "score": [1.0, 2.0],
        }
    )
    p1 = tmp_path / "one.parquet"
    base.write_parquet(p1)
    out = load_scores_panel(p1)
    assert "factor_clean" in out.columns
    assert out["factor_clean"].to_list() == [1.0, 2.0]

    multi = base.with_columns(pl.col("score").alias("other"))
    p2 = tmp_path / "multi.parquet"
    multi.write_parquet(p2)
    with pytest.raises(ValueError, match=r"score-col|多列|multiple"):
        load_scores_panel(p2)

    # 显式指定可过
    out2 = load_scores_panel(p2, score_col="score")
    assert out2.height == 2

    # 畸形：缺键列
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(bad)
    with pytest.raises(ValueError):
        load_scores_panel(bad)


def test_combine_backtest_default_strategy_is_quantile_ls_5() -> None:
    """--strategy 默认与 fz factor run 一致：quantile_ls_5；universe 默认 all_a。"""
    from factorzen.cli.main import build_parser

    parser = build_parser()
    # 用 --scores 占位以满足互斥组；不真正执行
    args = parser.parse_args(
        [
            "combine",
            "backtest",
            "--scores",
            "dummy.parquet",
            "--start",
            "20230101",
            "--end",
            "20230601",
        ]
    )
    assert args.strategy == "quantile_ls_5"
    assert args.universe == "all_a"
    assert args.market == "ashare"


# ── rebalance_days 桥层真做 ─────────────────────────────────────────────


def test_apply_rebalance_hold_ffill_keeps_rb_scores() -> None:
    """k=2：非调仓日分数 = 最近调仓日分数；k=1 原样。"""
    from factorzen.pipelines.combine_backtest import apply_rebalance_hold

    df = pl.DataFrame(
        {
            "trade_date": ["d1", "d2", "d3", "d4"] * 2,
            "ts_code": ["A"] * 4 + ["B"] * 4,
            "factor_clean": [1.0, 9.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0],
        }
    )
    out = apply_rebalance_hold(df, 2).sort(["ts_code", "trade_date"])
    a = out.filter(pl.col("ts_code") == "A")["factor_clean"].to_list()
    # d1=1 (rb), d2=1 (ffill), d3=3 (rb), d4=3 (ffill)
    assert a == pytest.approx([1.0, 1.0, 3.0, 3.0])
    assert apply_rebalance_hold(df, 1).equals(df.sort(["ts_code", "trade_date"]))


def _scores_with_alternating_ranks(
    prices: pl.DataFrame,
) -> pl.DataFrame:
    """隔日完全翻转排名：逐日换手极高；k 日 hold 时非调仓日权重冻结，总换手显著更低。"""
    dates = prices.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    codes = sorted(prices.select("ts_code").unique()["ts_code"].to_list())
    n = len(codes)
    rows = []
    for di, day in enumerate(dates):
        ds = day.strftime("%Y%m%d") if isinstance(day, date) else str(day).replace("-", "")[:8]
        for i, code in enumerate(codes):
            # 偶数日升序、奇数日降序 → 每日多空两端对倒
            rank = i if di % 2 == 0 else (n - 1 - i)
            rows.append({"trade_date": ds, "ts_code": code, "score": float(rank)})
    return pl.DataFrame(rows)


def test_rebalance_days_lowers_turnover_and_nav_still_daily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rebalance_days=5 总换手显著低于逐日；非调仓日 nav 仍逐日更新。"""
    from factorzen.pipelines.combine_backtest import run_combine_backtest

    _, prices, start, end = _synthetic_scores_and_prices(n_days=60, n_stocks=20, seed=7)
    scores = _scores_with_alternating_ranks(prices)
    scores_path = tmp_path / "alt_scores.parquet"
    scores.write_parquet(scores_path)
    _patch_market_data(monkeypatch, prices)
    out_root = tmp_path / "bt"

    daily = run_combine_backtest(
        scores=scores_path,
        start=start,
        end=end,
        strategy="quantile_ls_5",
        cost_bps=0.0,
        rebalance_days=None,
        out_dir=out_root,
        run_id="rb_daily",
    )
    every5 = run_combine_backtest(
        scores=scores_path,
        start=start,
        end=end,
        strategy="quantile_ls_5",
        cost_bps=0.0,
        rebalance_days=5,
        out_dir=out_root,
        run_id="rb_5",
    )

    turn_daily = float(daily["result"].returns["turnover"].sum())
    turn_5 = float(every5["result"].returns["turnover"].sum())
    # 隔日翻转排名：逐日总换手应远高于 5 日 hold（数值断言，非签名）
    assert turn_daily > 0.0, "逐日应有正换手（隔日翻转排名）"
    assert turn_5 < turn_daily * 0.5, (
        f"rebalance_days=5 总换手应显著低于逐日: daily={turn_daily:.4f}, k5={turn_5:.4f}"
    )

    nav5 = every5["nav"].sort("trade_date")
    assert nav5.height >= 10, "净值序列应覆盖多个交易日"
    # 非调仓日仍逐日有 nav 行（日环未改）
    nav_dates = nav5["trade_date"].to_list()
    # 相邻 nav 日期不同 → 逐日落点（至少 10 个不同交易日）
    assert len(set(nav_dates)) == len(nav_dates)
    assert len(nav_dates) >= 10

    # 相邻两日 nav 可随行情变化（至少有一些日 net_return != 0）
    rets = every5["result"].returns
    assert float(rets["net_return"].abs().sum()) > 0.0

    # manifest 记 rebalance_days
    assert every5["manifest"]["rebalance_days"] == 5
    assert daily["manifest"]["rebalance_days"] is None
