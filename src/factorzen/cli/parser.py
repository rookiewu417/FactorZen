"""Argparse tree assembly for the FactorZen CLI.

Callbacks are resolved from the supplied command module when the parser is built, so
tests and embedding callers can still replace command functions before dispatch.
"""

from __future__ import annotations

import argparse
from typing import Any

from factorzen.config.settings import (
    COMBINATIONS_DIR,
    CRYPTO_LAKE,
    FACTOR_LIBRARY_DIR,
    MINE_TEAM_DIR,
    PORTFOLIOS_DIR,
    REPORTS_DIR,
)


def _add_factor_run_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--benchmark", default=None, help="Benchmark index code")
    parser.add_argument("--config", default=None, help="YAML run config path")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Override any config field, repeatable: --set backtest.top_n=30",
    )
    parser.add_argument("--all", action="store_true", help="Enable deep evaluation preset")
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without running")
    parser.add_argument(
        "--ic-method",
        default=None,
        choices=["rank", "pearson", "both"],
        dest="ic_method",
        help="IC method",
    )
    parser.add_argument("--neutralized-ic", action="store_true", dest="neutralized_ic")
    parser.add_argument("--event-study", action="store_true", dest="event_study")
    parser.add_argument(
        "--llm-explain",
        action="store_true",
        help="Enable LLM explanation; no-config daily runs enable this by default",
    )
    parser.add_argument("--llm-refresh", action="store_true")


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
    parser.add_argument("--all", action="store_true", help="Enable deep report preset")
    parser.add_argument("--benchmark", default=None, help="Benchmark index code")
    parser.add_argument("--config", default=None, help="YAML run config path")
    parser.add_argument(
        "--ic-method",
        default=None,
        choices=["rank", "pearson", "both"],
        dest="ic_method",
        help="IC method",
    )
    parser.add_argument("--neutralized-ic", action="store_true", dest="neutralized_ic")
    parser.add_argument("--event-study", action="store_true", dest="event_study")
    parser.add_argument("--llm-explain", action="store_true")
    parser.add_argument("--llm-refresh", action="store_true")


def _add_freq_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--freq", choices=["1m", "5m", "15m", "1h", "daily"], default="daily",
                   help="bar 粒度(仅 crypto;ashare 只支持 daily)")


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

    run = factor_sub.add_parser("run", help="Run a single factor evaluation")
    _add_factor_run_arguments(run)
    run.set_defaults(func=commands._cmd_factor_test)

    test = factor_sub.add_parser("test", help="Deprecated alias for 'factor run'")
    _add_factor_run_arguments(test)
    test.set_defaults(func=commands._cmd_factor_test)

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
    path_cmd.set_defaults(func=commands._cmd_report_open)

    open_cmd = report_sub.add_parser("open", help="Deprecated alias for 'report path'")
    open_cmd.add_argument("run_id")
    open_cmd.set_defaults(func=commands._cmd_report_open)

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
    if_build.set_defaults(func=commands._cmd_data_intraday_features_build)

    if_status = ifeat_sub.add_parser("status", help="Show intraday feature manifest and partitions")
    if_status.add_argument("--freq", default="5min", help="Bar frequency (default 5min)")
    if_status.add_argument("--version", default="v1", help="Battery version (default v1)")
    if_status.set_defaults(func=commands._cmd_data_intraday_features_status)

    config = sub.add_parser("config", help="Config workflows")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate = config_sub.add_parser("validate", help="Validate a YAML run config")
    validate.add_argument("path", help="YAML run config path")
    validate.set_defaults(func=commands._cmd_config_validate)

    runs = sub.add_parser("runs", help="Run history workflows")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    list_cmd = runs_sub.add_parser("list", help="List recorded runs")
    list_cmd.add_argument("--limit", type=int, default=20, help="Maximum rows to print")
    list_cmd.set_defaults(func=commands._cmd_runs_list)
    show_cmd = runs_sub.add_parser("show", help="Show one run manifest")
    show_cmd.add_argument("run_id")
    show_cmd.set_defaults(func=commands._cmd_runs_show)

    # ── fz mine ──（与 fz factor 并列的顶层命令组）
    mine = sub.add_parser("mine", help="Factor mining workflows")
    mine_sub = mine.add_subparsers(dest="mine_command", required=True)

    m_search = mine_sub.add_parser("search", help="Search candidate factor expressions")
    m_search.add_argument("--start", required=True, help="Start date YYYYMMDD")
    m_search.add_argument("--end", required=True, help="End date YYYYMMDD")
    m_search.add_argument("--universe", default=None, help="Universe name (e.g. csi500)")
    m_search.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                          help="Market profile (default ashare; crypto=USDT-M perps; "
                               "futures=国内商品期货主力连续; us=S&P500 Yahoo 后复权)")
    m_search.add_argument("--top-n", dest="top_n", type=int, default=50,
                          help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    m_search.add_argument("--method", choices=["random", "genetic"], default="random")
    m_search.add_argument("--trials", type=int, default=200)
    m_search.add_argument("--top-k", dest="top_k", type=int, default=10)
    m_search.add_argument("--seed", type=int, default=42)
    m_search.add_argument("--workers", type=int, default=1,
                          help="遗传搜索并行评分线程数(默认 1;同 seed 结果与串行等价)")
    m_search.add_argument("--holdout-ratio", dest="holdout_ratio", type=float, default=0.2,
                          help="永久隔离的 OOS holdout 占比（默认 0.2）")
    m_search.add_argument("--train-ratio", dest="train_ratio", type=float, default=0.7,
                          help="mining 段内 train/valid 切分比例（默认 0.7）")
    m_search.add_argument("--decorr-threshold", dest="decorr_threshold", type=float, default=0.7,
                          help="top-K 贪心去相关的 |corr| 门槛，≥该值视为近重复剔除（默认 0.7）")
    m_search.add_argument("--min-n-train", dest="min_n_train", type=int, default=5,
                          help="候选 train 段最少有效 IC 天数，不足则丢弃（默认 5）")
    m_search.add_argument("--dsr-alpha", dest="dsr_alpha", type=float, default=DEFAULT_DSR_ALPHA,
                          help="护栏 passed 标记的 DSR 显著性阈值（默认 0.10，2026-07 松一档）")
    m_search.add_argument("--no-library", dest="no_library", action="store_true",
                          help=f"关闭收尾自动 upsert 因子库（默认开，passed 候选进 {FACTOR_LIBRARY_DIR}）")
    m_search.add_argument("--no-library-orthogonal", dest="no_library_orthogonal",
                          action="store_true",
                          help="关闭搜索期库级正交过滤（默认开：top-K 贪心去相关时避开库内 active 方向；"
                               "与 --no-library 无关，后者只关收尾 upsert）")
    m_search.add_argument("--objective", choices=["raw", "residual"], default="residual",
                          help="挖掘评估目标：residual=对库内 active 因子截面正交后的残差 IC "
                               "（默认；库空自动退化为 raw）；raw=裸 Rank IC（旧口径）")
    _add_freq_arg(m_search)
    m_search.set_defaults(func=commands._cmd_mine_search)

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
                         help="Market profile (default ashare; crypto=USDT-M perps via Vision lake; "
                              "futures=国内商品期货主力连续 via Tushare; us=S&P500 via Yahoo chart)")
    m_agent.add_argument("--symbols", default=None,
                         help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照")
    m_agent.add_argument("--top-n", dest="top_n", type=int, default=50,
                         help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    m_agent.add_argument("--iterations", type=int, default=5)
    m_agent.add_argument("--top-k", dest="top_k", type=int, default=5)
    m_agent.add_argument("--seed", type=int, default=42)
    m_agent.add_argument("--human-review", action="store_true", dest="human_review")
    m_agent.add_argument("--patience", type=commands._positive_patience, default=None,
                         help="连续 N 轮无新候选则早停（N>=1；默认不早停，跑满 --iterations）")
    m_agent.add_argument("--heal-rounds", dest="heal_rounds", type=int, default=2,
                         help="表达式解析失败时回灌 LLM 修正的最大轮数（0=关闭）")
    m_agent.add_argument("--no-library-orthogonal", dest="no_library_orthogonal",
                         action="store_true",
                         help="关闭搜索期库级正交过滤（默认开：护栏阶段避开库内 active 方向）")
    m_agent.add_argument("--objective", choices=["raw", "residual"], default="residual",
                        help="挖掘评估目标：residual=对库残差 IC（默认；库空→raw）；raw=裸 IC")
    _add_freq_arg(m_agent)
    m_agent.set_defaults(func=commands._cmd_mine_agent)

    m_team = mine_sub.add_parser("team", help="Multi-agent team factor mining")
    m_team.add_argument("--start", required=True)
    m_team.add_argument("--end", required=True)
    m_team.add_argument("--universe", default=None)
    m_team.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
                        help="Market profile (default ashare; crypto=USDT-M perps via Vision lake; "
                             "futures=国内商品期货主力连续 via Tushare; us=S&P500 via Yahoo chart)")
    m_team.add_argument("--symbols", default=None,
                        help="crypto/futures/us only: 逗号分隔 symbols；缺省=universe Top-N 快照")
    m_team.add_argument("--top-n", dest="top_n", type=int, default=50,
                        help="crypto/futures universe size (Top-N by turnover); us=S&P500 静态池截断 (default 50)")
    m_team.add_argument("--iterations", type=int, default=5)
    m_team.add_argument("--top-k", dest="top_k", type=int, default=5)
    m_team.add_argument("--seed", type=int, default=42)
    m_team.add_argument("--index-path", dest="index_path",
                        default=str(MINE_TEAM_DIR / "experiment_index.jsonl"))
    m_team.add_argument("--structured", action="store_true",
                        help="结构化假设(机制/预期符号/证伪判据) + 任务分解后逐任务翻译")
    m_team.add_argument("--patience", type=commands._positive_patience, default=None,
                        help="连续 N 轮无新候选则早停（N>=1；默认不早停，跑满 --iterations）")
    m_team.add_argument("--heal-rounds", dest="heal_rounds", type=int, default=2,
                        help="表达式解析失败时回灌 LLM 修正的最大轮数（0=关闭）")
    m_team.add_argument("--hypotheses-per-round", dest="hypotheses_per_round",
                        type=int, default=1,
                        help="每轮提多少个假设（默认1；>1 提升单轮产能，护栏/Critic 仍每轮一次）")
    m_team.add_argument("--no-library", dest="no_library", action="store_true",
                        help=f"关闭收尾自动 upsert 因子库（默认开，最终候选进 {FACTOR_LIBRARY_DIR}）")
    m_team.add_argument("--no-library-orthogonal", dest="no_library_orthogonal",
                        action="store_true",
                        help="关闭搜索期库级正交过滤（默认开：护栏阶段避开库内 active 方向；"
                             "与 --no-library 无关，后者只关收尾 upsert）")
    m_team.add_argument("--objective", choices=["raw", "residual"], default="residual",
                       help="挖掘评估目标：residual=对库残差 IC（默认；库空→raw）；raw=裸 IC")
    m_team.add_argument("--no-campaign-prior", dest="no_campaign_prior",
                        action="store_true",
                        help="关闭跨 session trial family 记账（默认开：finalize 的 DSR 用"
                             "同评价配置历史唯一表达式∪本 session 的 N，防多重检验清零漏记）")
    m_team.add_argument("--llm-workers", dest="llm_workers", type=int, default=4,
                        help="轮内独立 LLM 调用的并发度（默认 4 提速；1=串行零回归；"
                             "API/pipeline 缺省仍为 1）")
    m_team.add_argument(
        "--no-auto-lift", dest="no_auto_lift", action="store_true",
        help="关闭 session 末自动组 lift 裁决（默认开：lift_queue 候选组测+入库）",
    )
    m_team.add_argument(
        "--lift-se-mult", dest="lift_se_mult", type=float, default=1.0,
        help="lift 准入 SE 乘数（默认 1.0：lift ≥ max(threshold, se_mult×SE)）",
    )
    m_team.add_argument(
        "--lift-workers", dest="lift_workers", type=int, default=None,
        help="session 末 lift 逐候选线程并发（默认自适应可用内存，上限 4；1=串行）",
    )
    _add_freq_arg(m_team)
    m_team.set_defaults(func=commands._cmd_mine_team)

    # ── fz factor-library ──（分市场因子登记簿：rebuild / list / show / render）
    fl = sub.add_parser(
        "factor-library",
        help="因子库登记簿（分市场·全信息·自动维护）："
             "rebuild/list/show/render/lift-test/tag-legacy/"
             "forward-track/forward-review",
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

    fl_rd = fl_sub.add_parser("render", help="重生 {market}.md（不重算）")
    fl_rd.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_rd.set_defaults(func=commands._cmd_factor_library_render)

    fl_lt = fl_sub.add_parser(
        "lift-test",
        help="灰区候选组合增量 lift 实验 → 通过者以 status=probation 入库（第二通道）",
    )
    fl_lt.add_argument(
        "--session", nargs="+", required=True,
        help="mine_team / mine-agent / mining_session 的 run 目录（含 manifest.json）",
    )
    fl_lt.add_argument("--market", choices=["ashare", "crypto", "futures", "us"], default="ashare")
    fl_lt.add_argument("--start", required=True, help="评估窗口起点 YYYYMMDD")
    fl_lt.add_argument("--end", required=True, help="评估窗口终点 YYYYMMDD")
    fl_lt.add_argument("--universe", default=None, help="A股 universe 名（如 csi300）")
    fl_lt.add_argument(
        "--top-m", dest="top_m", type=int, default=None,
        help="按 |residual_ic_train| 取 top-M 控成本（默认全测；显式截断会打印警告）",
    )
    fl_lt.add_argument("--threshold", type=float, default=None,
                       help="RankIC lift 阈值（默认 DEFAULT_LIFT_THRESHOLD=0.001）")
    fl_lt.add_argument("--seed", type=int, default=0)
    fl_lt.add_argument("--library-root", dest="library_root", default=None,
                       help=f"因子库根目录（默认 {FACTOR_LIBRARY_DIR}）")
    fl_lt_write = fl_lt.add_mutually_exclusive_group()
    fl_lt_write.add_argument(
        "--apply", dest="apply", action="store_true",
        help="将通过的候选写入因子库（默认 dry-run 只打印）",
    )
    fl_lt_write.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="只打印不写库（当前已是默认行为，保留为兼容旗标）",
    )
    fl_lt.add_argument(
        "--se-mult", dest="se_mult", type=float, default=1.0,
        help="lift 准入 SE 乘数（默认 1.0：lift ≥ max(threshold, se_mult×SE)）",
    )
    fl_lt.add_argument(
        "--allow-active", dest="allow_active", action="store_true",
        help="允许 lift 裁决直接写 active（默认封顶 probation，待校准）",
    )
    fl_lt.add_argument(
        "--admission-start", dest="admission_start", default=None,
        help="lift 评分窗起点 YYYYMMDD（覆盖 session manifest holdout 推导）",
    )
    fl_lt.add_argument(
        "--admission-end", dest="admission_end", default=None,
        help="lift 评分窗终点 YYYYMMDD（覆盖 session manifest holdout 推导）",
    )
    fl_lt.add_argument(
        "--horizon", type=int, default=None,
        help="lift 前向持有期；默认跟随 session manifest 的 mining horizon，兜底 DEFAULT_HORIZON",
    )
    fl_lt.add_argument(
        "--lift-workers", dest="lift_workers", type=int, default=None,
        help="候选级 lift 线程并发（默认按可用内存自适应，上限 4；1=串行）",
    )
    fl_lt.add_argument("--top-n", dest="top_n", type=int, default=50,
                       help="crypto/futures/us universe size")
    fl_lt.add_argument("--symbols", default=None)
    _add_freq_arg(fl_lt)
    fl_lt.set_defaults(func=commands._cmd_factor_library_lift_test)

    fl_tl = fl_sub.add_parser(
        "tag-legacy",
        help="把 evidence_tier 为 None 的记录标为 legacy（幂等，不改 status）",
    )
    fl_tl.add_argument(
        "--market", choices=["ashare", "crypto", "futures", "us"], default="ashare",
    )
    fl_tl.add_argument(
        "--root", default=None,
        help=f"因子库根目录（默认 {FACTOR_LIBRARY_DIR}）",
    )
    fl_tl.set_defaults(func=commands._cmd_factor_library_tag_legacy)

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
        help="块 SE 乘数（默认 1.645≈单侧 95%）",
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

    # ── fz validate ──（与 fz mine 并列的顶层命令组）
    # ── fz research ──（端到端编排：mine → 头部 passed 因子 → 循环 build → sim → report）
    research = sub.add_parser("research", help="End-to-end research orchestration")
    research_sub = research.add_subparsers(dest="research_command", required=True)
    r_run = research_sub.add_parser(
        "run", help="mine → 头部 passed 因子 → 按调仓日循环 build → sim → report（同一 run_id）")
    r_run.add_argument("--start", required=True, help="Start date YYYYMMDD")
    r_run.add_argument("--end", required=True, help="End date YYYYMMDD")
    r_run.add_argument("--universe", default=None, help="Universe name (default all_a)")
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

    return parser
