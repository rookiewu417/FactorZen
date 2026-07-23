# src/factorzen/pipelines/research_run.py
"""fz research run：端到端研究编排器（mine → build → sim → report，同一 run_id 贯穿）。

把原本纯手动的下游链路一条命令串起来，编排器只负责**接产物、补格式桥、循环调仓**，
底层全部复用现有 pipeline（factor_mine / portfolio_build / sim.engine / portfolio_report）。

MVP 边界（诚实标注）：
- **单个头部通过护栏(passed=true)的因子**，不做多因子 combine（combine 需整段面板+收益面板，
  与单日 α 截面格式鸿沟大，作后续增强）。
- **全区间 in-sample 回测**：调仓日取自挖掘同一 [start,end] 区间；严格 OOS 回测窗口留后续扩展。
- 行业中性用 universe 等权基准（与 fz portfolio build 一致，非真实指数基准）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def _rebalance_dates(trade_dates: list, rebalance_days: int, warmup: int) -> list:
    """按 ``rebalance_days`` 步长取调仓日：跳过前 ``warmup`` 个交易日（留 lookback），
    且**不含最后一个交易日**——sim 需调仓日之后还有交易日才能执行该信号。"""
    if rebalance_days < 1:
        raise ValueError("rebalance_days 必须 ≥ 1")
    if len(trade_dates) <= warmup + 1:
        return []
    usable = trade_dates[warmup:-1]
    return usable[::rebalance_days]


def _select_passed_expression(candidates: list[dict]) -> str:
    """从挖掘候选里选**头部**通过防过拟合护栏(passed=true)的表达式；一个都没有则报错。"""
    passed = [c for c in candidates if c.get("passed")]
    if not passed:
        raise RuntimeError(
            "research run: 挖掘未产出任何通过防过拟合护栏(passed=true)的因子。"
            "可加大 --trials，或先用 fz mine search（其 --dsr-alpha/--holdout-ratio 可调护栏）"
            "+ fz mine leaderboard --all 排查候选质量。"
        )
    return str(passed[0]["expression"])


def _to_yyyymmdd(d: Any) -> str:
    """polars Date / str → 'YYYYMMDD'。"""
    if hasattr(d, "strftime"):
        return d.strftime("%Y%m%d")
    return str(d).replace("-", "")[:8]


def _alpha_file_for_date(panel: pl.DataFrame, date_obj: Any, dest: Path) -> Path:
    """从整段因子面板切出某调仓日的 ``[ts_code, alpha]`` 截面并落 parquet（喂 portfolio build）。"""
    cross = (
        panel.filter(pl.col("trade_date") == date_obj)
        .select([pl.col("ts_code"), pl.col("factor_value").alias("alpha")])
        .filter(pl.col("alpha").is_finite())
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    cross.write_parquet(dest)
    return dest


def _write_report(sim_res: dict, build_dirs: list[str], out_root: str, run_id: str) -> Path:
    """复用 generate_portfolio_report：从 sim 产物 + 末次调仓组合产物组装 dashboard HTML 并落盘。"""
    import json
    from types import SimpleNamespace

    from factorzen.reports.portfolio_report import generate_portfolio_report

    sim_dir = Path(sim_res["run_dir"])
    metrics: dict = {}
    mp = sim_dir / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text(encoding="utf-8"))

    sim_result = None
    navp = sim_dir / "nav.parquet"
    if navp.exists():
        nav_df = pl.read_parquet(navp)
        if not nav_df.is_empty():
            sim_result = SimpleNamespace(nav=nav_df, returns=nav_df)

    # 末次调仓的组合产物用于归因 / 风险摘要展示
    attribution_df = risk_summary_df = None
    portfolio_manifest = None
    if build_dirs:
        pdir = Path(build_dirs[-1])
        if (pdir / "attribution.csv").exists():
            attribution_df = pl.read_csv(pdir / "attribution.csv")
        if (pdir / "risk_summary.csv").exists():
            risk_summary_df = pl.read_csv(pdir / "risk_summary.csv")
        if (pdir / "manifest.json").exists():
            portfolio_manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))

    html = generate_portfolio_report(
        sim_result=sim_result, metrics=metrics,
        attribution_df=attribution_df, risk_summary_df=risk_summary_df,
        portfolio_manifest=portfolio_manifest, market="ashare",
    )
    # HTML 报告收口：{out_root}/factors/reports/（与 REPORTS_DIR=workspace/factors/reports 同构）
    out_path = Path(out_root) / "factors" / "reports" / f"portfolio_{run_id}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def run_research(*, start: str, end: str, universe: str | None = None,
                 n_trials: int = 200, method: str = "random", seed: int = 42,
                 top_k: int = 10, rebalance_days: int = 20, warmup: int = 60,
                 risk_aversion: float = 1.0, w_max: float = 0.05,
                 turnover: float | None = None, industry_neutral: bool = False,
                 lookback: int = 60, run_id: str | None = None,
                 out_root: str = "workspace", command: list[str] | None = None,
                 intraday: bool = False, intraday_freq: str = "5min",
                 intraday_expr_leaves: list[str] | None = None,
                 exec_lag: int = 1,
                 exec_price_col: str | None = "open_adj") -> dict:
    """一条命令跑通 mine → 头部 passed 因子 → 按调仓日循环 build → sim → report。

    ``intraday`` / ``intraday_freq`` / ``intraday_expr_leaves`` 透传给 ``run_mine``，
    把 i_*（17 个 builtin）与 ix_*（scout 提案的 bar 级表达式叶）纳入挖掘搜索空间；
    α 面板经 ``ExpressionFactor.compute`` 在表达式含 i_* 时自动 attach 日内面板，
    风险/组合/sim 不直接消费 i_* 叶子。

    ``exec_lag`` / ``exec_price_col``：挖掘段成交口径，与 ``fz mine search`` 默认可实现
    口径一致（1 / open_adj）；旧口径对照传 ``exec_lag=0``。

    返回 ``{run_id, expression, n_rebalances, mining_session_dir, portfolios_root,
    sim_dir, report_html, sharpe, ann_ret}``。所有产物落 ``{out_root}/{stage}/{run_id}...``。
    """
    import json

    import numpy as np

    from factorzen.core import loader
    from factorzen.core.experiment import build_manifest_base
    from factorzen.core.universe import get_universe
    from factorzen.daily.data.context import FactorDataContext
    from factorzen.discovery.factor import ExpressionFactor
    from factorzen.pipelines.daily_single import (
        filter_frame_by_membership,
        load_pit_membership,
    )
    from factorzen.pipelines.factor_mine import run_mine
    from factorzen.pipelines.portfolio_build import run_portfolio
    from factorzen.risk.model import RiskModel
    from factorzen.sim.engine import run_portfolio_simulation

    rid = run_id or f"research_{seed}_{method}"
    uni_name = universe or "all_a"

    # ── 1) 挖掘 → 选头部 passed 因子 ──
    mine_res = run_mine(start=start, end=end, universe=universe, n_trials=n_trials,
                        top_k=top_k, seed=seed, method=method,
                        intraday=intraday, intraday_freq=intraday_freq,
                        intraday_expr_leaves=intraday_expr_leaves,
                        exec_lag=exec_lag, exec_price_col=exec_price_col)
    expr = _select_passed_expression(mine_res["candidates"])

    # ── 2) 整段因子面板：union 拉取（替代期末快照，消除调出股整窗消失）──
    # membership 逐日 PIT；panel 计算后按日过滤，再进调仓 α 截面。
    membership, uni_full, _universe_meta = load_pit_membership(start, end, uni_name)
    ctx = FactorDataContext(
        start=start, end=end, required_data=["daily", "daily_basic"],
        lookback_days=lookback, universe=uni_full if uni_full else None,
    )
    panel = ExpressionFactor(expression=expr).compute(ctx)  # [trade_date, ts_code, factor_value]
    panel = filter_frame_by_membership(panel, membership)

    # ── 3) 全区间日频（sim 用 + 派生调仓日）；拉取用 union，sim 持仓可能含历史成分 ──
    daily_full = loader.fetch_daily(start, end)
    if uni_full:
        daily_full = daily_full.filter(pl.col("ts_code").is_in(uni_full))
    # 风险模型专用：带 lookback 预热的历史（与 fz portfolio build 的 load_risk_inputs
    # 同口径，消除双路径漂移）——否则每个调仓日 RiskModel.build 的窗口首日滚动风格因子
    # 全空、因子集钉死在退化截面、静默退化，且与 portfolio build 产出不同风险模型。
    from factorzen.pipelines.risk_build import load_risk_inputs
    from factorzen.risk.exposures import (
        materialize_industry_panel,
        materialize_style_panel,
        standardize_style_panel,
    )

    risk_daily_full, risk_db_full = load_risk_inputs(loader, start, end, uni_full)
    trade_dates = sorted(daily_full["trade_date"].unique().to_list())
    rb_dates = _rebalance_dates(trade_dates, rebalance_days, warmup)
    if not rb_dates:
        raise RuntimeError(
            f"research run: [{start},{end}] 交易日({len(trade_dates)})不足以产出调仓日"
            f"（warmup={warmup} + rebalance_days={rebalance_days}）。请扩大区间或调小 warmup/步长。"
        )

    # ── 3b) 风格/行业暴露全窗一次物化（W3）──────────────────────────────────
    # 风格 raw：滚动窗只看历史 → PIT 安全；每调仓日按当日 universe 再 CS 标准化。
    # 行业：PIT 按日归属一次物化；每调仓日切片 ≤d + uni_d，行业名只用 ≤d 出现过的
    # （禁止把未来行业列带进早期调仓，否则与 standalone build(start,d) 不等价）。
    # 协方差/特质风险仍在 build 内用 ≤d 暴露重估（NW ~0.03s，近零成本）。
    # 禁止「全窗建一次协方差供所有调仓日」——那会让早期调仓吃到未来数据。
    raw_style_panel = materialize_style_panel(
        risk_daily_full, risk_db_full, standardize=False
    )
    # 行业 fallback 骨架：各调仓日 universe 并集（含 industry），避免早期调入股在期末快照缺失
    _ind_frames = [get_universe(_to_yyyymmdd(d), uni_name) for d in rb_dates]
    stocks_for_ind = pl.concat(_ind_frames, how="diagonal_relaxed").unique(subset=["ts_code"])
    industry_panel_full, industry_names_full = materialize_industry_panel(
        stocks_for_ind, trade_dates
    )

    # ── 4) 按调仓日循环 build（**直调 run_portfolio**：CLI 无法设 run_id、会覆盖同一目录）──
    # 调仓日成分：get_universe(d) 已是 as-of 当日 PIT（含 industry，供 RiskModel）；
    # 与 membership 同口径（指数池 as-of），并补 α 面板已按 membership 过滤。
    portfolios_root = Path(out_root) / "portfolios" / rid
    alpha_tmp = portfolios_root / "_alpha"
    build_dirs: list[str] = []
    prev_w_map: dict[str, float] = {}  # 上期调仓权重（ts_code→w），供换手约束
    for d in rb_dates:
        d_str = _to_yyyymmdd(d)
        iso = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
        stocks_d = get_universe(d_str, uni_name)
        uni_d = stocks_d["ts_code"].to_list()
        # 风险模型用带 lookback 预热的历史（含调仓日之前的滚动窗口数据）
        daily_d = risk_daily_full.filter((pl.col("trade_date") <= d) & pl.col("ts_code").is_in(uni_d))
        db_d = risk_db_full.filter((pl.col("trade_date") <= d) & pl.col("ts_code").is_in(uni_d))
        # 复用 raw 风格面板：≤d + 当日 universe 过滤后 CS 标准化（与 standalone 等价）
        style_d = standardize_style_panel(
            raw_style_panel.filter(
                (pl.col("trade_date") <= d) & pl.col("ts_code").is_in(uni_d)
            )
        )
        # 行业切片：仅 ≤d；行业名 = 切片内出现过的列（PIT，不含未来行业）
        if industry_panel_full.height:
            ind_d = industry_panel_full.filter(
                (pl.col("trade_date") <= d) & pl.col("ts_code").is_in(uni_d)
            )
            ind_names_d = [
                c for c in industry_names_full
                if c in ind_d.columns and ind_d[c].sum() > 0
            ]
        else:
            ind_d = industry_panel_full
            ind_names_d = industry_names_full
        risk_result = RiskModel().build(
            daily_d,
            db_d,
            stocks_d,
            start,
            d_str,
            style_panel=style_d,
            industry_panel=ind_d,
            industry_names=ind_names_d or None,
        )
        codes = risk_result.factor_exposures.codes
        alpha_file = _alpha_file_for_date(panel, d, alpha_tmp / f"{d_str}.parquet")
        adf = pl.read_parquet(alpha_file)
        amap = dict(zip(adf["ts_code"].to_list(), adf["alpha"].to_list(), strict=False))
        alpha = np.array([float(amap.get(c, 0.0)) for c in codes])
        neutral = ([n for n in risk_result.factor_names if n.startswith("ind_")]
                   if industry_neutral else None)
        bench_weights = np.full(len(codes), 1.0 / len(codes)) if industry_neutral else None
        _ind = dict(zip(stocks_d["ts_code"].to_list(), stocks_d["industry"].to_list(), strict=False))
        sectors = [(_ind.get(c) or "") for c in codes]
        # 换手约束需要上期权重（按当日 codes 对齐）；否则 turnover_budget 被静默丢弃
        prev_weights = (
            np.array([float(prev_w_map.get(c, 0.0)) for c in codes]) if prev_w_map else None
        )
        res = run_portfolio(
            alpha, risk_result, codes=codes, stock_returns=np.zeros(len(codes)),
            sectors=sectors, factor_returns_latest={}, risk_aversion=risk_aversion,
            w_max=w_max, neutral_factors=neutral, turnover_budget=turnover,
            prev_weights=prev_weights, bench_weights=bench_weights, signal_date=iso,
            out_dir=str(portfolios_root), run_id=d_str,
            command=(command or ["research", "run"]),
        )
        build_dirs.append(res["run_dir"])
        # 记录本期权重供下期换手约束
        _wdf = pl.read_parquet(Path(res["run_dir"]) / "weights.parquet")
        prev_w_map = dict(zip(_wdf["ts_code"].to_list(), _wdf["target_weight"].to_list(),
                              strict=False))

    # ── 5) sim（一次扫 portfolios_root 下所有调仓 run_dir，拼净值）──
    sim_res = run_portfolio_simulation(
        build_dirs, daily_full, out_dir=str(Path(out_root) / "sim"), run_id=rid,
    )

    # ── 6) report（dashboard HTML）──
    html_path = _write_report(sim_res, build_dirs, out_root, rid)

    # ── 顶层可复现 manifest ──
    research_dir = Path(out_root) / "research" / rid
    research_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest_base(
        command=command or ["research", "run"],
        config={"start": start, "end": end, "universe": universe, "n_trials": n_trials,
                "method": method, "seed": seed, "top_k": top_k, "rebalance_days": rebalance_days,
                "warmup": warmup, "risk_aversion": risk_aversion, "w_max": w_max,
                "turnover": turnover, "industry_neutral": industry_neutral, "lookback": lookback,
                "intraday": intraday, "intraday_freq": intraday_freq,
                "expression": expr, "n_rebalances": len(build_dirs)},
    )
    (research_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"run_id": rid, "expression": expr, "n_rebalances": len(build_dirs),
            "mining_session_dir": mine_res["session_dir"], "portfolios_root": str(portfolios_root),
            "sim_dir": sim_res["run_dir"], "report_html": str(html_path),
            "sharpe": sim_res.get("sharpe"), "ann_ret": sim_res.get("ann_ret")}
