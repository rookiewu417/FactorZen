"""Argparse tree assembly for the FactorZen CLI.

Callbacks are resolved from the supplied command module when the parser is built, so
tests and embedding callers can still replace command functions before dispatch.
"""

from __future__ import annotations

import argparse
from typing import Any, Protocol

from factorzen.config.settings import (
    COMBINATIONS_DIR,
    COMBINE_BACKTESTS_DIR,
    CRYPTO_LAKE,
    FACTOR_LIBRARY_DIR,
    MINE_TEAM_DIR,
    PORTFOLIOS_DIR,
    REPORTS_DIR,
)


class _ArgAdder(Protocol):
    """``ArgumentParser`` 与 ``_ArgumentGroup`` 的最小共用面（仅 ``add_argument``）。"""

    def add_argument(self, *args: Any, **kwargs: Any) -> Any: ...


def _add_factor_common_arguments(parser: argparse.ArgumentParser) -> None:
    """``fz factor eval`` / ``fz factor backtest`` 共用参数面。"""
    parser.add_argument("name", nargs="?", help="Factor name")
    parser.add_argument("--start", default=None, help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD")
    parser.add_argument("--universe", default=None, help="Universe name")
    parser.add_argument(
        "--frequency",
        "--freq",
        dest="frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Factor frequency",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark index code（仅 backtest 轨生效；eval 轨保留参数但忽略）",
    )
    parser.add_argument("--config", default=None, help="YAML run config path")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default 42)")
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Override any config field, repeatable: --set backtest.top_n=30",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without running")
    _add_exec_convention_args(parser)


def _add_report_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", help="Factor name")
    parser.add_argument("--factor", default=None, help="Factor name")
    parser.add_argument("--start", default=None, help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD")
    parser.add_argument("--universe", default=None, help="Universe name")
    parser.add_argument(
        "--frequency",
        "--freq",
        dest="frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Factor frequency",
    )
    parser.add_argument("--reuse", action="store_true", help="Reuse existing artifacts")
    parser.add_argument("--benchmark", default=None, help="Benchmark index code")
    parser.add_argument("--config", default=None, help="YAML run config path")


def _add_freq_arg(p: _ArgAdder) -> None:
    p.add_argument("--freq", choices=["1m", "5m", "15m", "1h", "daily"], default="daily",
                   help="bar 粒度(仅 crypto;ashare 只支持 daily)")


def _add_exec_convention_args(p: _ArgAdder) -> None:
    """成交口径旗标：把「信号什么时候能真的成交」变成显式选择。

    **默认 = 可实现口径** ``--exec-lag 1 --exec-price-col open_adj``
    （open[t+2]/open[t+1]，即 t 日信号 → t+1 开盘进、t+2 开盘出）。
    项目铁律是「t 日算 → t+1 执行」(CLAUDE.md PIT 自查第 8 条)。

    旧口径 ``close[t+1]/close[t]`` 隐含「t 日收盘成交」，而算信号需要 t 日
    收盘价，**实际不可实现**，且系统性高估可实现收益（csi500 实测隔夜段占
    top 桶超额的 100%）。仅对照用时显式 ``--exec-lag 0``（并视需要覆盖
    ``--exec-price-col`` 回 close 列）。

    注意：因子库裁决链（rebuild / lift-test / forward-track）**不**挂本组旗标，
    保持历史库口径一致性。
    """
    p.add_argument(
        "--exec-lag", dest="exec_lag", type=int, default=1,
        help="成交滞后(交易日)。默认 1=可实现口径 open_adj t+1→t+2；"
             "0=旧 close→close（不可实现，仅对照用）",
    )
    p.add_argument(
        "--exec-price-col", dest="exec_price_col", default="open_adj",
        help="成交价格列。默认 open_adj（可实现口径）；"
             "与 --exec-lag 1 合用 = open[t+2]/open[t+1]；"
             "旧口径对照可显式传 close / close_adj",
    )


def _add_set_arg(p: _ArgAdder, *, help_extra: str = "") -> None:
    """万能 ``--set KEY=VALUE``（action=append）：覆盖已砍掉的高级调参。

    解析阶段只收集 raw 字符串；合法键校验 + 类型注入在 handler 入口
    ``_apply_set_overrides``（及测试直接调时）完成。
    """
    help_txt = "高级覆盖 KEY=VALUE（可重复；未知 KEY 失败并列出合法键）"
    if help_extra:
        help_txt = f"{help_txt}；{help_extra}"
    p.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help=help_txt,
    )


def build_parser(commands: Any) -> argparse.ArgumentParser:
    from factorzen.discovery.guardrails import DEFAULT_DSR_ALPHA  # 护栏阈值单一真源，防漂移

    parser = argparse.ArgumentParser(prog="fz", description="FactorZen research CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    factor = sub.add_parser("factor", help="Factor workflows")
    factor_sub = factor.add_subparsers(dest="factor_command", required=True)

    new = factor_sub.add_parser("new", help="Create a user factor template")
    new.add_argument("name")
    new.add_argument(
        "--frequency",
        "--freq",
        dest="freq",
        choices=["daily", "weekly", "monthly", "intraday"],
        default="daily",
    )
    new.add_argument("--force", action="store_true")
    new.set_defaults(func=commands._cmd_factor_new)

    list_cmd = factor_sub.add_parser("list", help="List registered factors")
    list_cmd.add_argument(
        "--frequency",
        "--freq",
        dest="freq",
        choices=["daily", "weekly", "monthly", "intraday"],
        default="daily",
    )
    list_cmd.set_defaults(func=commands._cmd_factor_list)

    eval_cmd = factor_sub.add_parser(
        "eval",
        help="因子研究评估（信号层，毛口径：IC/分层/多空/单调性/换手，不跑日环）",
    )
    _add_factor_common_arguments(eval_cmd)
    # 信号轨自己的旋钮:原先硬编码 5 / 0.0,没有出口
    eval_cmd.add_argument(
        "--n-groups",
        type=int,
        default=5,
        dest="n_groups",
        help="截面分位组数（默认 5）；多空取最高组减最低组",
    )
    eval_cmd.set_defaults(func=commands._cmd_factor_eval)

    backtest_cmd = factor_sub.add_parser(
        "backtest",
        help="模拟交易回测（日环撮合+约束+成本，净口径；含 walk-forward/benchmark）",
    )
    _add_factor_common_arguments(backtest_cmd)
    backtest_cmd.set_defaults(func=commands._cmd_factor_backtest)

    sweep = factor_sub.add_parser("sweep", help="Parameter grid sweep over --set overrides")
    sweep.add_argument("name", nargs="?", help="Factor name (or supply via --config)")
    sweep.add_argument("--config", default=None, help="Base YAML run config path")
    sweep.add_argument(
        "--grid",
        action="append",
        default=None,
        metavar="KEY=V1,V2,...",
        help="Grid dimension, repeatable: --grid backtest.top_n=30,50,100",
    )
    sweep.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Fixed override applied to every combo",
    )
    sweep.add_argument("--start", default=None, help="Start date YYYYMMDD")
    sweep.add_argument("--end", default=None, help="End date YYYYMMDD")
    sweep.add_argument("--universe", default=None, help="Universe name")
    sweep.add_argument(
        "--sort-by",
        default="ir",
        dest="sort_by",
        help="Metric to rank rows by (ir/ic_mean/ic_pos/t)",
    )
    sweep.set_defaults(func=commands._cmd_factor_sweep)

    report = sub.add_parser("report", help="Report workflows")
    report_sub = report.add_subparsers(dest="report_command", required=True)

    build_cmd = report_sub.add_parser("build", help="Build a factor report")
    _add_report_build_arguments(build_cmd)
    build_cmd.set_defaults(func=commands._cmd_report_build)

    path_cmd = report_sub.add_parser("path", help="Print report path for a run")
    path_cmd.add_argument("run_id")
    path_cmd.set_defaults(func=commands._cmd_report_path)

    pf_report = report_sub.add_parser("portfolio", help="Generate portfolio dashboard HTML report")
    pf_report.add_argument(
        "--sim-dir",
        default=None,
        dest="sim_dir",
        help="模拟产物目录（含 metrics.json）",
    )
    pf_report.add_argument(
        "--portfolio-dir",
        default=None,
        dest="portfolio_dir",
        help="组合构建产物目录（含 attribution.csv / risk_summary.csv / manifest.json）",
    )
    pf_report.add_argument(
        "--out",
        default=None,
        dest="out",
        help=f"HTML 输出路径；默认 {REPORTS_DIR}/portfolio_<run_id>.html",
    )
    pf_report.add_argument(
        "--market",
        choices=["ashare", "crypto"],
        default=None,
        help="市场语境(默认从 sim manifest 自动识别；crypto=USDT/365/资金费/sector)",
    )
    pf_report.set_defaults(func=commands._cmd_report_portfolio)

    data = sub.add_parser("data", help="Data workflows")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch", help="Fetch raw data into cache")
    fetch.add_argument(
        "data_type",
        choices=["daily", "daily-basic", "fundamentals", "flows", "margin_detail",
                 "stk_holdernumber", "top_list"],
    )
    fetch.add_argument("--start", required=True, help="Start date YYYYMMDD")
    fetch.add_argument("--end", required=True, help="End date YYYYMMDD")
    fetch.set_defaults(func=commands._cmd_data_fetch)

    crypto_p = data_sub.add_parser("crypto", help="Crypto data lake workflows")
    crypto_sub = crypto_p.add_subparsers(dest="crypto_command", required=True)
    bf = crypto_sub.add_parser("backfill", help="Backfill 1m klines/funding/OI from Binance Vision")
    bf.add_argument("--start", required=True)
    bf.add_argument("--end", required=True)
    bf.add_argument("--symbols", default=None, help="逗号分隔;缺省=按上月成交额 Top-N 自动选池")
    bf.add_argument("--top-n", dest="top_n", type=int, default=50)
    bf.add_argument("--lake-root", dest="lake_root", default=str(CRYPTO_LAKE))
    bf.set_defaults(func=commands._cmd_data_crypto_backfill)

    ifeat = data_sub.add_parser("intraday-features", help="Intraday feature panel workflows")
    ifeat_sub = ifeat.add_subparsers(dest="intraday_features_command", required=True)
    if_build = ifeat_sub.add_parser("build", help="Build daily intraday feature panel from 1min lake")
    if_build.add_argument("--start", required=True, help="Start date YYYYMMDD")
    if_build.add_argument("--end", required=True, help="End date YYYYMMDD")
    if_build.add_argument("--freq", default="5min", help="Bar frequency (default 5min)")
    if_build.add_argument("--version", default="v1", help="Battery version (default v1)")
    if_build.add_argument(
        "--codes",
        default=None,
        help="Comma-separated ts_code filter (optional)",
    )
    if_build.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite when battery_hash mismatches existing manifest",
    )
    if_build.add_argument(
        "--force",
        action="store_true",
        help="Force recompute all months (ignore incremental skip of covered months)",
    )
    if_build.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Month-level process parallelism (default 1). "
            "Peak ~7.6GiB per month; on 24GiB RAM prefer 2; >2 warns"
        ),
    )
    if_build.set_defaults(func=commands._cmd_data_intraday_features_build)

    if_status = ifeat_sub.add_parser("status", help="Show intraday feature manifest and partitions")
    if_status.add_argument("--freq", default="5min", help="Bar frequency (default 5min)")
    if_status.add_argument("--version", default="v1", help="Battery version (default v1)")
    if_status.set_defaults(func=commands._cmd_data_intraday_features_status)

    # config 顶层组已删：validate → fz ops validate-config

    runs = sub.add_parser("runs", help="Run history workflows")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    list_cmd = runs_sub.add_parser("list", help="List recorded runs")
    list_cmd.add_argument("--limit", type=int, default=20, help="Maximum rows to print")
    list_cmd.set_defaults(func=commands._cmd_runs_list)
    # runs show 已删（低频查看；manifest 直接 cat）

    # ── fz mine ──（与 fz factor 并列的顶层命令组）
    mine = sub.add_parser("mine", help="Factor mining workflows")
    mine_sub = mine.add_subparsers(dest="mine_command", required=True)

    m_search = mine_sub.add_parser("search", help="Search candidate factor expressions")
    m_search.add_argument("--start", required=True, help="Start date YYYYMMDD")
    m_search.add_argument("--end", required=True, help="End date YYYYMMDD")
    m_search.add_argument("--universe", default=None, help="Universe name (e.g. csi500)")
    m_search.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                          help="Market profile (default ashare)")
    m_search.add_argument("--top-n", dest="top_n", type=int, default=50,
                          help="crypto/futures/us universe size (default 50)")
    m_search.add_argument("--method", choices=["random", "genetic"], default="random")
    m_search.add_argument("--trials", type=int, default=200)
    m_search.add_argument("--top-k", dest="top_k", type=int, default=10)
    m_search.add_argument("--seed", type=int, default=42)
    _add_freq_arg(m_search)
    _add_exec_convention_args(m_search)
    _add_set_arg(
        m_search,
        help_extra="如 workers/holdout_ratio/objective/no_library；见 docs/reference/cli.md 高级覆盖",
    )
    # 被砍参数硬编码默认（经 --set 可覆盖；handler 启动时 apply）
    m_search.set_defaults(
        func=commands._cmd_mine_search,
        workers=1,
        holdout_ratio=0.2,
        train_ratio=0.7,
        decorr_threshold=0.7,
        min_n_train=5,
        dsr_alpha=DEFAULT_DSR_ALPHA,
        no_library=False,
        no_library_orthogonal=False,
        objective="residual",
        intraday_leaves=False,
        intraday_freq="5min",
    )

    m_lb = mine_sub.add_parser("leaderboard", help="Print a mining session leaderboard")
    m_lb.add_argument("session_dir", help="Path to a mining session directory")
    m_lb.add_argument("--all", action="store_true",
                      help="Show all candidates, including those failing the overfitting guardrails")
    m_lb.set_defaults(func=commands._cmd_mine_leaderboard)

    m_exp = mine_sub.add_parser(
        "export-alpha",
        help="Compute one candidate's cross-sectional alpha → (ts_code,alpha) parquet",
    )
    m_exp.add_argument("--session", required=True,
                       help="Mining session dir (contains candidates.csv)")
    m_exp.add_argument("--rank", type=int, default=1,
                       help="Candidate rank in candidates.csv (1-based, default 1)")
    m_exp.add_argument("--date", required=True, help="Cross-section date YYYYMMDD")
    m_exp.add_argument("--universe", default="all_a", help="Universe name (default all_a)")
    m_exp.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                       help="Market profile (default ashare; crypto=USDT-M perps; futures=商品期货; us=S&P500)")
    m_exp.add_argument("--top-n", dest="top_n", type=int, default=50,
                       help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    m_exp.add_argument("--lookback", type=int, default=60,
                       help="Trade-day lookback for time-series operators (default 60)")
    m_exp.add_argument("--out", required=True,
                       help="Output parquet path (columns: ts_code, alpha)")
    m_exp.add_argument("--all", action="store_true",
                       help="Allow exporting a candidate that failed the overfitting guardrails "
                            "(default: only passed candidates)")
    _add_freq_arg(m_exp)
    m_exp.set_defaults(func=commands._cmd_mine_export_alpha)

    m_agent = mine_sub.add_parser("agent", help="LLM-guided agent factor mining")
    m_agent.add_argument("--start", required=True)
    m_agent.add_argument("--end", required=True)
    m_agent.add_argument("--universe", default=None)
    m_agent.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                         help="Market profile (default ashare)")
    m_agent.add_argument("--symbols", default=None,
                         help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照")
    m_agent.add_argument("--top-n", dest="top_n", type=int, default=50,
                         help="crypto/futures/us universe size (default 50)")
    m_agent.add_argument("--iterations", type=int, default=5)
    m_agent.add_argument("--top-k", dest="top_k", type=int, default=5)
    m_agent.add_argument("--seed", type=int, default=42)
    m_agent.add_argument("--human-review", action="store_true", dest="human_review")
    _add_freq_arg(m_agent)
    _add_exec_convention_args(m_agent)
    _add_set_arg(
        m_agent,
        help_extra="如 heal_rounds/patience/objective/intraday_scout；见 docs/reference/cli.md",
    )
    m_agent.set_defaults(
        func=commands._cmd_mine_agent,
        patience=None,
        heal_rounds=2,
        no_library_orthogonal=False,
        objective="residual",
        intraday_leaves=False,
        intraday_freq="5min",
        intraday_scout=False,
        scout_k=4,
        scout_max_leaves=12,
    )

    m_team = mine_sub.add_parser("team", help="Multi-agent team factor mining")
    m_team.add_argument("--start", required=True)
    m_team.add_argument("--end", required=True)
    m_team.add_argument("--universe", default=None)
    m_team.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                        help="Market profile (default ashare)")
    m_team.add_argument("--symbols", default=None,
                        help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照")
    m_team.add_argument("--top-n", dest="top_n", type=int, default=50,
                        help="crypto/futures/us universe size (default 50)")
    m_team.add_argument("--iterations", type=int, default=5)
    m_team.add_argument("--top-k", dest="top_k", type=int, default=5)
    m_team.add_argument("--seed", type=int, default=42)
    m_team.add_argument("--structured", action="store_true",
                        help="结构化假设(机制/预期符号/证伪判据) + 任务分解后逐任务翻译")
    m_team.add_argument(
        "--pool-subproc", dest="pool_subproc", action="store_true",
        help="池构建放子进程，退出全额归还内存；等效 env FACTORZEN_POOL_SUBPROC=1",
    )
    _add_exec_convention_args(m_team)
    _add_freq_arg(m_team)
    _add_set_arg(
        m_team,
        help_extra="如 llm_workers/heal_rounds/objective/hypotheses_per_round；见 docs/reference/cli.md",
    )
    m_team.set_defaults(
        func=commands._cmd_mine_team,
        index_path=str(MINE_TEAM_DIR / "experiment_index.jsonl"),
        patience=None,
        heal_rounds=2,
        hypotheses_per_round=1,
        no_library=False,
        no_library_orthogonal=False,
        objective="residual",
        no_campaign_prior=False,
        llm_workers=4,
        no_auto_lift=False,
        no_sleeve_gate=False,
        lift_se_mult=1.0,
        lift_workers=None,
        intraday_leaves=False,
        intraday_freq="5min",
        intraday_scout=False,
        scout_k=4,
        scout_max_leaves=12,
    )

    # ── fz mine pool-prebuild ──（原顶层 pool-prebuild；子进程内存隔离）
    pool_pre = mine_sub.add_parser(
        "pool-prebuild",
        help="mine team 库池预构建(子进程内存隔离;产物 parquet 供 --pool-subproc 装载)",
    )
    pool_pre.add_argument("--start", required=True)
    pool_pre.add_argument("--end", required=True)
    pool_pre.add_argument("--universe", default=None)
    pool_pre.add_argument(
        "--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
        help="Market profile (default ashare; 与 m_team 同源)",
    )
    pool_pre.add_argument(
        "--symbols", default=None,
        help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照",
    )
    pool_pre.add_argument(
        "--top-n", dest="top_n", type=int, default=50,
        help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)",
    )
    pool_pre.add_argument(
        "--index-path", dest="index_path",
        default=str(MINE_TEAM_DIR / "experiment_index.jsonl"),
    )
    pool_pre.add_argument(
        "--library-root", dest="library_root", default=None,
        help="因子库根目录（默认=index_path 同级 factor_library）",
    )
    pool_pre.add_argument(
        "--holdout-ratio", dest="holdout_ratio", type=float, default=0.2,
        help="holdout 比例（与 run_team_agent 默认同源）",
    )
    pool_pre.add_argument(
        "--intraday-leaves", dest="intraday_leaves", action="store_true",
        help="启用日内特征叶子 i_* 接入（仅 ashare；默认关）",
    )
    pool_pre.add_argument(
        "--intraday-freq", dest="intraday_freq", default="5min",
        help="日内特征面板频率（默认 5min；仅 ashare + --intraday-leaves）",
    )
    pool_pre.add_argument(
        "--out", required=True,
        help="池缓存输出目录（写 pool_wide.parquet + pool_meta.json）",
    )
    pool_pre.set_defaults(func=commands._cmd_pool_prebuild)

    # ── fz factor-library ──（分市场因子登记簿；render/tag-legacy 已删）
    fl = sub.add_parser(
        "factor-library",
        help="因子库登记簿（分市场·全信息·自动维护）："
             "rebuild/list/show/lift-test/"
             "forward-track/forward-review/store",
    )
    fl_sub = fl.add_subparsers(dest="factor_library_command", required=True)

    fl_rb = fl_sub.add_parser("rebuild",
                              help="从历史产物在统一默认窗口重算并重建某市场的因子库")
    fl_rb.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_rb.add_argument("--start", default=None, help="覆盖默认窗口起点 YYYYMMDD（缺省=最近6年滚动）")
    fl_rb.add_argument("--end", default=None, help="覆盖默认窗口终点 YYYYMMDD（缺省=数据最新端）")
    fl_rb.add_argument("--universe", default=None, help="A股 universe 名（如 csi300）")
    fl_rb.add_argument("--horizon", type=int, default=1, help="前向收益持有期（默认1）")
    fl_rb.add_argument("--top-n", dest="top_n", type=int, default=50,
                       help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    fl_rb.add_argument("--symbols", default=None,
                       help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照")
    fl_rb.add_argument("--decorr-threshold", dest="decorr_threshold", type=float, default=0.7,
                       help="去相关 |corr| 门槛，超此仍收录但标 correlated（默认0.7）")
    fl_rb.add_argument("--holdout-ratio", dest="holdout_ratio", type=float, default=0.2)
    fl_rb.add_argument(
        "--only", nargs="+", default=None,
        help="定向重估：只重估这些表达式（自动规范化，须已在库）。不清库、不重估其余记录、"
             "lift 复审也只覆盖子集；去相关**只降不升**（可下调 correlated，绝不上调 "
             "active——上调要跑全量 rebuild）。与 --only-file 可同时给（取并集）",
    )
    fl_rb.add_argument(
        "--only-file", dest="only_file", default=None,
        help="定向重估：从文件读表达式（一行一条，'#' 开头与空行跳过）；语义同 --only，"
             "供上百条批量补账（如补算存量 lift_metric；admission_ic 仅 lift 轨可补——"
             "single 轨的裸 IC 就是 ic_train）",
    )
    # 分钟派生叶子：库里已有带 i_* 叶子的 lift 记录，复审必须能物化它们，
    # 否则物化失败被当成「无增量」降级（已实际发生，见 rebuild 的求值失败守卫）。
    fl_rb.add_argument(
        "--intraday-leaves", dest="intraday_leaves", action="store_true",
        help="启用日内特征叶子 i_* 接入（仅 ashare；默认关）。库内含 i_* 叶子的 lift "
             "记录复审时必须开，否则物化失败",
    )
    fl_rb.add_argument(
        "--intraday-freq", dest="intraday_freq", default="5min",
        help="日内特征面板频率（默认 5min；仅 ashare + --intraday-leaves）",
    )
    _add_freq_arg(fl_rb)
    fl_rb.set_defaults(func=commands._cmd_factor_library_rebuild)

    fl_ls = fl_sub.add_parser("list", help="列出库内因子（rank/expression/holdout_ic/status）")
    fl_ls.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_ls.set_defaults(func=commands._cmd_factor_library_list)

    fl_sh = fl_sub.add_parser("show", help="单因子全字段")
    fl_sh.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_sh.add_argument("--expression", default=None, help="按表达式（规范形）查")
    fl_sh.add_argument("--rank", type=int, default=None, help="按库内排名查（1-based，holdout_ic 降序）")
    fl_sh.set_defaults(func=commands._cmd_factor_library_show)

    fl_lt = fl_sub.add_parser(
        "lift-test",
        help="灰区候选 / registry python 因子组合增量 lift 实验 → 通过者以 status=probation 入库（第二通道）",
    )
    fl_lt.add_argument(
        "--session", nargs="+", required=False, default=None,
        help="mine_team / mine-agent / mining_session 的 run 目录（含 manifest.json）；"
             "与 --factor 至少一个",
    )
    fl_lt.add_argument(
        "--factor", nargs="+", default=None,
        help="registry 因子名（python 型）；与 --session 至少一个；"
             "要求 market=ashare 且 --universe 必填",
    )
    fl_lt.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_lt.add_argument("--start", required=True, help="评估窗口起点 YYYYMMDD")
    fl_lt.add_argument("--end", required=True, help="评估窗口终点 YYYYMMDD")
    fl_lt.add_argument("--universe", default=None, help="A股 universe 名（如 csi300）")
    fl_lt_write = fl_lt.add_mutually_exclusive_group()
    fl_lt_write.add_argument(
        "--apply", dest="apply", action="store_true",
        help="将通过的候选写入因子库，并将 lift 拒绝写回 experiment_index"
             "（默认 dry-run 只打印、不写库也不写 index）",
    )
    fl_lt_write.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="只打印不写库（当前已是默认行为，保留为兼容旗标）",
    )
    fl_lt.add_argument("--seed", type=int, default=0)
    fl_lt.add_argument("--top-n", dest="top_n", type=int, default=50,
                       help="crypto/futures/us universe size")
    fl_lt.add_argument("--symbols", default=None)
    fl_lt.add_argument(
        "--admission-start", dest="admission_start", default=None,
        help="lift 评分窗起点 YYYYMMDD（覆盖 session manifest holdout 推导）",
    )
    fl_lt.add_argument(
        "--admission-end", dest="admission_end", default=None,
        help="lift 评分窗终点 YYYYMMDD（覆盖 session manifest holdout 推导）",
    )
    _add_freq_arg(fl_lt)
    _add_set_arg(
        fl_lt,
        help_extra="如 top_m/threshold/se_mult/library_root/lift_workers；见 docs/reference/cli.md",
    )
    fl_lt.set_defaults(
        func=commands._cmd_factor_library_lift_test,
        top_m=20,
        queue_ic_floor=None,
        include_sub_floor=False,
        threshold=None,
        library_root=None,
        se_mult=1.0,
        allow_active=False,
        horizon=None,
        lift_workers=None,
        intraday_leaves=False,
        intraday_freq="5min",
    )

    fl_ln = fl_sub.add_parser(
        "lift-null",
        help="lift 统计层 null 校准：H0=无真实 lift 下扫 se_mult×min_blocks 的误准入率",
    )
    fl_ln.add_argument("--n-days", dest="n_days", type=int, default=290,
                       help="配对评分日数（默认 290≈holdout 量级）")
    fl_ln.add_argument("--daily-sigma", dest="daily_sigma", type=float, default=0.01,
                       help="日差分(cand_ic−base_ic)标准差量级")
    fl_ln.add_argument("--ar1", type=float, default=0.3,
                       help="日差分 AR(1) 自相关（重叠前向收益导致，默认 0.3）")
    fl_ln.add_argument("--se-mults", dest="se_mults", default="1.0,1.645,2.0",
                       help="逗号分隔的 SE 乘数网格")
    fl_ln.add_argument("--min-blocks", dest="min_blocks", default="0,6,10",
                       help="逗号分隔的最低块数网格（0=不设，现状）")
    fl_ln.add_argument("--n-sims", dest="n_sims", type=int, default=5000)
    fl_ln.add_argument("--seed", type=int, default=0)
    fl_ln.set_defaults(func=commands._cmd_factor_library_lift_null)

    fl_ft = fl_sub.add_parser(
        "forward-track",
        help="记录 as_of 日库内因子 paper forward RankIC"
             "（确认窗口随真实时间累积；ops 每日链路接线为后续工作）",
    )
    fl_ft.add_argument(
        "--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
    )
    fl_ft.add_argument(
        "--date", default=None,
        help="确认日 YYYYMMDD（缺省=数据最新交易日 latest_data_date）",
    )
    fl_ft.add_argument(
        "--root", default=None,
        help=f"因子库根目录（默认 {FACTOR_LIBRARY_DIR}）",
    )
    fl_ft.add_argument(
        "--universe", default=None,
        help="forward 截面 universe（缺省=库记录准入口径众数；必须与准入一致）",
    )
    fl_ft.add_argument(
        "--allow-backfill", dest="allow_backfill", action="store_true",
        help="允许 as_of 距今超过 max-backfill-days 的补录/初始播种"
             "（仍写真实 recorded_at 供审计；默认拒绝历史回灌）",
    )
    fl_ft.add_argument(
        "--max-backfill-days", dest="max_backfill_days", type=int, default=10,
        help="as_of 相对 wall-clock 允许的最大日历滞后天数（默认 10）",
    )
    fl_ft.set_defaults(func=commands._cmd_factor_library_forward_track)

    fl_fr = fl_sub.add_parser(
        "forward-review",
        help="裁决 probation 因子 paper forward 证据"
             "（确认窗口随真实时间累积；ops 每日链路接线为后续工作）",
    )
    fl_fr.add_argument(
        "--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
    )
    fl_fr.add_argument(
        "--min-days", dest="min_days", type=int, default=60,
        help="最低有效 forward 天数（默认 60）",
    )
    fl_fr.add_argument(
        "--se-mult", dest="se_mult", type=float, default=1.645,
        help="块 SE 乘数（默认 1.645≈单侧 95%%）",
    )
    fl_fr.add_argument(
        "--block-days", dest="block_days", type=int, default=20,
        help="块 SE 块长（交易日，默认 20）",
    )
    fl_fr.add_argument(
        "--apply", dest="apply", action="store_true",
        help="写库：promote→active / demote→no_lift（默认 dry-run 只打印）",
    )
    fl_fr.add_argument(
        "--root", default=None,
        help=f"因子库根目录（默认 {FACTOR_LIBRARY_DIR}）",
    )
    fl_fr.set_defaults(func=commands._cmd_factor_library_forward_review)

    # ── fz factor-library store ──（资产库三件套 meta/py/parquet）
    fl_st = fl_sub.add_parser(
        "store",
        help="因子资产库三件套（meta.json + factor.py + factor.parquet）",
    )
    fl_st_sub = fl_st.add_subparsers(dest="factor_library_store_command", required=True)

    fl_st_sync = fl_st_sub.add_parser(
        "sync",
        help=(
            "从 jsonl 同步资产库：写 meta+py；默认物化 active/probation 的 parquet "
            "（固定 all_a × 2016-01-01~最新已完结交易日，与 jsonl 评估窗分离）"
        ),
    )
    fl_st_sync.add_argument(
        "--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
    )
    fl_st_sync.add_argument(
        "--only",
        default=None,
        help="只同步这些 name（逗号分隔）；缺省=全库",
    )
    fl_st_sync.add_argument(
        "--no-materialize",
        dest="no_materialize",
        action="store_true",
        help="只写 meta+py，不物化 parquet",
    )
    fl_st_sync.add_argument(
        "--root",
        default=None,
        help="资产库根目录（默认 workspace/factor_store）",
    )
    fl_st_sync.add_argument(
        "--lib-root",
        dest="lib_root",
        default=None,
        help=f"因子库 jsonl 根目录（默认 {FACTOR_LIBRARY_DIR}）",
    )
    fl_st_sync.set_defaults(func=commands._cmd_factor_library_store_sync)

    fl_st_ver = fl_st_sub.add_parser(
        "verify",
        help=(
            "校验 meta.expression 与 jsonl 一致，并检查 materialization "
            "是否仍为 store 口径（all_a / 2016-01-01~最新）"
        ),
    )
    fl_st_ver.add_argument(
        "--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
    )
    fl_st_ver.add_argument(
        "--root",
        default=None,
        help="资产库根目录（默认 workspace/factor_store）",
    )
    fl_st_ver.add_argument(
        "--lib-root",
        dest="lib_root",
        default=None,
        help=f"因子库 jsonl 根目录（默认 {FACTOR_LIBRARY_DIR}）",
    )
    fl_st_ver.set_defaults(func=commands._cmd_factor_library_store_verify)

    # ── fz validate ──（与 fz mine 并列的顶层命令组）
    # ── fz research ──（端到端编排：mine → 头部 passed 因子 → 循环 build → sim → report）
    research = sub.add_parser("research", help="End-to-end research orchestration")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    r_run = research_sub.add_parser(
        "run", help="mine → 头部 passed 因子 → 按调仓日循环 build → sim → report（同一 run_id）")
    r_run.add_argument("--start", required=True, help="Start date YYYYMMDD")
    r_run.add_argument("--end", required=True, help="End date YYYYMMDD")
    r_run.add_argument(
        "--universe", default=None,
        help="Universe name（default None → 运行时解析为 all_a）",
    )
    r_run.add_argument("--method", choices=["random", "genetic"], default="random")
    r_run.add_argument("--trials", type=int, default=200)
    r_run.add_argument("--top-k", dest="top_k", type=int, default=10)
    r_run.add_argument("--seed", type=int, default=42)
    r_run.add_argument("--rebalance-days", dest="rebalance_days", type=int, default=20,
                       help="调仓间隔（交易日数，默认 20≈月频）")
    r_run.add_argument("--warmup", type=int, default=60,
                       help="起始跳过的交易日数，留给时序算子 lookback（默认 60）")
    r_run.add_argument("--lookback", type=int, default=60,
                       help="因子计算 lookback 交易日数（默认 60）")
    r_run.add_argument("--lam", type=float, default=1.0, help="风险厌恶系数（默认 1.0）")
    r_run.add_argument("--w-max", dest="w_max", type=float, default=0.05,
                       help="单票权重上限（默认 0.05）")
    r_run.add_argument("--turnover", type=float, default=None, help="换手预算（默认无约束）")
    r_run.add_argument("--industry-neutral", dest="industry_neutral", action="store_true",
                       help="行业中性到 universe 等权基准")
    r_run.add_argument("--run-id", dest="run_id", default=None,
                       help="贯穿全链路的 run_id（默认 research_<seed>_<method>）")
    r_run.add_argument(
        "--intraday-leaves", dest="intraday_leaves", action="store_true",
        help="启用日内特征叶子 i_*（需先 fz data intraday-features build）；仅 ashare",
    )
    r_run.add_argument(
        "--intraday-freq", dest="intraday_freq", default="5min",
        help="日内特征频率",
    )
    _add_exec_convention_args(r_run)
    r_run.set_defaults(func=commands._cmd_research_run)

    validate = sub.add_parser("validate", help="Overfitting / robustness checks")
    validate_sub = validate.add_subparsers(dest="validate_command", required=True)
    vo = validate_sub.add_parser("overfit", help="Deflated Sharpe + bootstrap CI for one factor")
    vo.add_argument("factor", nargs="?", help="Registered factor name (ashare)")
    vo.add_argument("--start", required=True)
    vo.add_argument("--end", required=True)
    vo.add_argument("--universe", default=None)
    vo.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                    help="Market profile (default ashare; crypto/futures/us 需 --expression)")
    vo.add_argument("--expression", default=None,
                    help="Factor expression to validate (required for --market crypto/futures/us)")
    vo.add_argument("--top-n", dest="top_n", type=int, default=50,
                    help="crypto/futures/us universe size (default 50)")
    _add_freq_arg(vo)
    vo.set_defaults(func=commands._cmd_validate_overfit)

    # ── fz risk ──（顶层命令组）
    risk = sub.add_parser("risk", help="Risk model workflows")
    risk_sub = risk.add_subparsers(dest="risk_command", required=True)
    r_build = risk_sub.add_parser("build", help="Build Barra risk model")
    r_build.add_argument("--start", required=True, help="Start date YYYYMMDD")
    r_build.add_argument("--end", required=True, help="End date YYYYMMDD")
    r_build.add_argument("--universe", default="all_a", help="Universe name")
    r_build.add_argument("--cov-half-life", type=int, default=90, dest="cov_half_life")
    r_build.add_argument("--nw-lags", type=int, default=2, dest="nw_lags")
    r_build.add_argument("--spec-half-life", type=int, default=90, dest="spec_half_life")
    r_build.add_argument("--spec-shrinkage", type=float, default=0.3, dest="spec_shrinkage")
    r_build.set_defaults(func=commands._cmd_risk_build)

    # ── fz portfolio ──（顶层命令组）
    portfolio = sub.add_parser("portfolio", help="Portfolio construction & attribution")
    pf_sub = portfolio.add_subparsers(dest="portfolio_command", required=True)
    p_build = pf_sub.add_parser("build", help="Build optimized portfolio + attribution")
    p_build.add_argument("--start", required=True)
    p_build.add_argument("--end", required=True)
    p_build.add_argument("--universe", default="all_a")
    p_build.add_argument(
        "--alpha-file",
        required=True,
        dest="alpha_file",
        help="α 信号文件(parquet/csv: 列 ts_code + alpha)",
    )
    p_build.add_argument("--lam", type=float, default=1.0, dest="lam", help="风险厌恶系数")
    p_build.add_argument("--w-max", type=float, default=0.05, dest="w_max")
    p_build.add_argument("--turnover", type=float, default=None)
    p_build.add_argument("--industry-neutral", action="store_true", dest="industry_neutral")
    p_build.add_argument("--market", choices=["ashare", "crypto"], default="ashare",
                         help="Market profile (default ashare; crypto=市场中性做空)")
    p_build.add_argument("--top-n", dest="top_n", type=int, default=50,
                         help="crypto universe size (default 50)")
    p_build.add_argument("--gross-limit", dest="gross_limit", type=float, default=1.0,
                         help="crypto 毛敞口上限 Σ|w| (default 1.0)")
    p_build.add_argument("--run-id", dest="run_id", default=None,
                         help="产物子目录名(默认=end 日期串)；多期构建须用不同 run_id 避免覆盖")
    p_build.add_argument("--out-dir", dest="out_dir", default=str(PORTFOLIOS_DIR),
                         help=f"组合产物根目录(默认 {PORTFOLIOS_DIR})")
    _add_freq_arg(p_build)
    p_build.set_defaults(func=commands._cmd_portfolio_build)

    # ── fz sim ──（顶层命令组）
    sim = sub.add_parser("sim", help="Portfolio simulation workflows")
    sim_sub = sim.add_subparsers(dest="sim_command", required=True)

    s_run = sim_sub.add_parser("run", help="Run portfolio simulation")
    s_run.add_argument(
        "--portfolio-dir",
        required=True,
        dest="portfolio_dir",
        help="组合产物根目录，其下各 {run_id}/ 含 weights.parquet + manifest.json",
    )
    s_run.add_argument("--start", required=True, help="Start date YYYYMMDD")
    s_run.add_argument("--end", required=True, help="End date YYYYMMDD")
    s_run.add_argument("--run-id", default=None, dest="run_id", help="可选输出 run_id")
    s_run.add_argument("--market", choices=["ashare", "crypto"], default="ashare",
                       help="Market profile (default ashare; crypto=funding+做空 NAV 回测)")
    s_run.add_argument("--top-n", dest="top_n", type=int, default=50,
                       help="crypto universe size (default 50)")
    _add_freq_arg(s_run)
    s_run.set_defaults(func=commands._cmd_sim_run)

    s_show = sim_sub.add_parser("show", help="Show simulation metrics")
    s_show.add_argument(
        "--sim-dir",
        required=True,
        dest="sim_dir",
        help="模拟输出目录（含 metrics.json）",
    )
    s_show.set_defaults(func=commands._cmd_sim_show)

    # ── fz live ──（顶层命令组）
    live = sub.add_parser("live", help="向前执行(纸面/实盘)工作流")
    live_sub = live.add_subparsers(dest="live_command", required=True)
    lp = live_sub.add_parser("replay", help="历史窗口 replay 出向前 NAV(A类)")
    lp.add_argument("--session-dir", required=True, dest="session_dir")
    lp.add_argument("--portfolio-run-dir", action="append", required=True, dest="portfolio_run_dirs")
    lp.add_argument("--start", required=True)   # 行情窗口起(YYYYMMDD)
    lp.add_argument("--end", required=True)      # 行情窗口止
    lp.add_argument("--universe", default=None)
    lp.add_argument("--initial-cash", type=float, default=1_000_000.0, dest="initial_cash")
    lp.add_argument("--broker", choices=["paper"], default="paper")
    lp.add_argument("--from-date", default=None, dest="from_date")  # 可选:窗口内进一步裁剪(YYYY-MM-DD)
    lp.add_argument("--to-date", default=None, dest="to_date")
    lp.add_argument("--seed", type=int, default=0)
    lp.set_defaults(func=commands._cmd_live_replay)

    li = live_sub.add_parser("init", help="初始化向前会话")
    li.add_argument("--session-dir", required=True, dest="session_dir")
    li.add_argument("--initial-cash", type=float, default=1_000_000.0, dest="initial_cash")
    li.add_argument("--slippage-bps", type=float, default=0.0, dest="slippage_bps")
    li.add_argument("--broker", choices=["paper"], default="paper")
    li.set_defaults(func=commands._cmd_live_init)

    ls = live_sub.add_parser("step", help="推进一个交易日(可续跑)")
    ls.add_argument("--session-dir", required=True, dest="session_dir")
    ls.add_argument("--date", required=True)  # YYYYMMDD
    ls.add_argument(
        "--portfolio-run-dir", action="append", required=True, dest="portfolio_run_dirs"
    )
    ls.add_argument("--start", required=True)  # 行情窗口(含ADV回看)
    ls.add_argument("--end", required=True)
    ls.add_argument("--universe", default=None)
    ls.set_defaults(func=commands._cmd_live_step)

    lst = live_sub.add_parser("status", help="打印会话当前状态")
    lst.add_argument("--session-dir", required=True, dest="session_dir")
    lst.set_defaults(func=commands._cmd_live_status)

    lr = live_sub.add_parser("report", help="生成A类分歧归因报告")
    lr.add_argument("--session-dir", required=True, dest="session_dir")
    lr.add_argument(
        "--portfolio-run-dir", action="append", required=True, dest="portfolio_run_dirs"
    )
    lr.add_argument("--start", required=True)
    lr.add_argument("--end", required=True)
    lr.add_argument("--universe", default=None)
    lr.set_defaults(func=commands._cmd_live_report)

    # ── combine:多因子组合 OOS 对比 ──
    combine = sub.add_parser("combine", help="多因子组合 OOS 对比实验")
    combine_sub = combine.add_subparsers(dest="combine_command", required=True)
    cr = combine_sub.add_parser("run", help="四方法(等权/IC加权/max_ir/lgbm)OOS 对比")
    cr.add_argument(
        "--factor", action="append", required=True, dest="factors",
        help="因子 parquet[trade_date,ts_code,factor_value](可多次)",
    )
    cr.add_argument("--ret", required=True, help="前向收益 parquet[trade_date,ts_code,ret]")
    cr.add_argument("--train-days", type=int, default=120, dest="train_days")
    cr.add_argument("--test-days", type=int, default=20, dest="test_days")
    cr.add_argument("--purge-days", type=int, default=5, dest="purge_days")
    cr.add_argument("--embargo-days", type=int, default=0, dest="embargo_days")
    cr.add_argument("--methods", default="all", help="逗号分隔(equal_weight,ic_weighted,max_ir,lgbm)或 all")
    cr.add_argument("--seed", type=int, default=0)
    cr.add_argument("--run-id", default=None, dest="run_id")
    cr.add_argument("--out-dir", default=str(COMBINATIONS_DIR), dest="out_dir")
    cr.set_defaults(func=commands._cmd_combine_run)

    # combine from-session:挖掘因子库 → 物化 → 组合 OOS(端到端接线)
    cfs = combine_sub.add_parser("from-session",
                                 help="从挖掘 session 的因子库直接跑组合 OOS(物化+收益面板自动生成)")
    cfs.add_argument("--session", required=True, nargs="+",
                     help="挖掘 session 目录(含 candidates.csv)，可传多个跨 run 合并去重")
    cfs.add_argument("--start", required=True, help="物化窗口起 YYYYMMDD")
    cfs.add_argument("--end", required=True, help="物化窗口止 YYYYMMDD")
    cfs.add_argument("--universe", default=None, help="票池(默认全A)")
    cfs.add_argument("--horizon", type=int, default=5, help="前向收益持有期(交易日,默认5)")
    cfs.add_argument("--top-n", dest="top_n", type=int, default=None, help="只取库前 N 个因子")
    cfs.add_argument("--decorr-threshold", dest="decorr_threshold", type=float, default=0.7,
                     help="贪心去相关阈值(|corr|>阈值剔除近亲；1.0 关闭，默认0.7)")
    cfs.add_argument("--all", action="store_true", help="含未过护栏的因子(默认只用 passed 库因子)")
    cfs.add_argument("--train-days", type=int, default=120, dest="train_days")
    cfs.add_argument("--test-days", type=int, default=20, dest="test_days")
    cfs.add_argument("--purge-days", type=int, default=5, dest="purge_days")
    cfs.add_argument("--embargo-days", type=int, default=0, dest="embargo_days")
    cfs.add_argument("--methods", default="all", help="逗号分隔或 all")
    cfs.add_argument("--seed", type=int, default=0)
    cfs.add_argument("--run-id", default=None, dest="run_id")
    cfs.add_argument("--out-dir", default=str(COMBINATIONS_DIR), dest="out_dir")
    cfs.set_defaults(func=commands._cmd_combine_from_session)

    # combine from-library: 因子库登记簿选品 → 四方法组合 OOS
    def _parse_combine_statuses(s: str) -> tuple[str, ...]:
        allowed = {"active", "probation", "correlated", "no_lift"}
        parts = tuple(p.strip() for p in str(s).split(",") if p.strip())
        if not parts:
            raise argparse.ArgumentTypeError("--statuses 不能为空")
        bad = [p for p in parts if p not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(
                f"--statuses 非法值 {bad}；允许 {sorted(allowed)}"
            )
        return parts

    cfl = combine_sub.add_parser(
        "from-library",
        help="因子库选品 → 四方法组合 OOS（库登记簿的消费入口）",
    )
    cfl.add_argument(
        "--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
    )
    cfl.add_argument(
        "--statuses", type=_parse_combine_statuses, default=("active",),
        help="逗号分隔 status 过滤（默认 active；可选 active/probation/correlated/no_lift）",
    )
    cfl.add_argument("--library-root", dest="library_root", default=None,
                     help="因子库根目录（默认 workspace/factor_library）")
    cfl.add_argument("--start", required=True, help="物化窗口起 YYYYMMDD")
    cfl.add_argument("--end", required=True, help="物化窗口止 YYYYMMDD")
    cfl.add_argument("--universe", default=None, help="票池(默认全A；含 python 因子时必填)")
    cfl.add_argument("--horizon", type=int, default=5, help="前向收益持有期(交易日,默认5)")
    cfl.add_argument("--top-n", dest="top_n", type=int, default=None, help="只取库前 N 个因子")
    cfl.add_argument("--decorr-threshold", dest="decorr_threshold", type=float, default=0.7,
                     help="贪心去相关阈值(|corr|>阈值剔除近亲；1.0 关闭，默认0.7)")
    cfl.add_argument("--train-days", type=int, default=120, dest="train_days")
    cfl.add_argument("--test-days", type=int, default=20, dest="test_days")
    cfl.add_argument("--purge-days", type=int, default=5, dest="purge_days")
    cfl.add_argument("--embargo-days", type=int, default=0, dest="embargo_days")
    cfl.add_argument("--methods", default="all", help="逗号分隔或 all")
    cfl.add_argument("--seed", type=int, default=0)
    cfl.add_argument("--run-id", default=None, dest="run_id")
    cfl.add_argument("--out-dir", default=str(COMBINATIONS_DIR), dest="out_dir")
    cfl.set_defaults(func=commands._cmd_combine_from_library)

    # combine backtest: OOS 组合分数 → 日环策略回测（桥命令）
    cbt = combine_sub.add_parser(
        "backtest",
        help="组合 OOS 分数面板 → 统一日环策略回测（净值/换手/成本后指标）",
    )
    cbt_src = cbt.add_mutually_exclusive_group(required=True)
    cbt_src.add_argument(
        "--scores",
        default=None,
        help="分数 parquet[trade_date,ts_code,<分数列>]；与 --run-dir 二选一",
    )
    cbt_src.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help="combine 产物目录（读 oos_scores/<method>.parquet）；与 --scores 二选一",
    )
    cbt.add_argument(
        "--method",
        default="equal_weight",
        help="配合 --run-dir：读 oos_scores/<method>.parquet（默认 equal_weight）",
    )
    cbt.add_argument(
        "--score-col",
        dest="score_col",
        default=None,
        help="分数列名；缺省取除 trade_date/ts_code 外唯一数值列，多列则必填",
    )
    cbt.add_argument(
        "--strategy",
        default="quantile_ls_5",
        help="策略名（默认 quantile_ls_5，与 fz factor backtest/daily_single 无 YAML 默认一致）；"
        "支持 quantile_ls_5 / topn_long_only / factor_weighted 等既有 registry 类",
    )
    cbt.add_argument("--start", required=True, help="回测起 YYYYMMDD")
    cbt.add_argument("--end", required=True, help="回测止 YYYYMMDD")
    cbt.add_argument(
        "--universe", default="all_a", help="票池（默认 all_a；PIT membership 过滤）",
    )
    cbt.add_argument(
        "--market", choices=["ashare"], default="ashare",
        help="市场（当前仅 ashare）",
    )
    cbt.add_argument(
        "--cost-bps",
        dest="cost_bps",
        type=float,
        default=None,
        help="单边成本(bps)。缺省=daily_single LinearCostModel 默认费率；"
        "0=零成本；显式数值用 commission=bps/1e4（印花税/滑点/融券置 0）",
    )
    cbt.add_argument(
        "--rebalance-days",
        dest="rebalance_days",
        type=int,
        default=None,
        help="调仓间隔（交易日）。缺省或 1=逐日；k>1 时桥层把分数降采样到每 k 日并"
        "按股票前向填充（非调仓日目标权重不变、换手≈0），引擎仍日环、净值逐日更新",
    )
    cbt.add_argument("--run-id", default=None, dest="run_id")
    cbt.add_argument(
        "--out-dir",
        default=str(COMBINE_BACKTESTS_DIR),
        dest="out_dir",
        help=f"产物根目录（默认 {COMBINE_BACKTESTS_DIR}）",
    )
    cbt.set_defaults(func=commands._cmd_combine_backtest)

    # ── ops:无人值守运营 ──
    ops = sub.add_parser("ops", help="无人值守运营(每日链路)")
    ops_sub = ops.add_subparsers(dest="ops_command", required=True)

    od = ops_sub.add_parser("daily", help="执行一个交易日的无人值守链路")
    od.add_argument("--config", required=True, help="ops.yaml 配置路径")
    od.add_argument("--date", default=None, help="YYYYMMDD,缺省今天")
    od.set_defaults(func=commands._cmd_ops_daily)

    ost = ops_sub.add_parser("status", help="打印某日各阶段状态")
    ost.add_argument("--config", required=True, help="ops.yaml 配置路径")
    ost.add_argument("--date", default=None, help="YYYYMMDD,缺省今天")
    ost.set_defaults(func=commands._cmd_ops_status)

    # 原 fz config validate → 迁入 ops（handler 复用）
    ovc = ops_sub.add_parser("validate-config", help="Validate a YAML run config")
    ovc.add_argument("path", help="YAML run config path")
    ovc.set_defaults(func=commands._cmd_config_validate)

    return parser
