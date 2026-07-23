"""全局路径配置。所有模块通过 ``from factorzen.config.settings import ROOT`` 引用路径。"""

from pathlib import Path

# 项目根目录：src/factorzen/config/settings.py 向上三级
ROOT = Path(__file__).resolve().parents[3]

WORKSPACE_DIR = ROOT / "workspace"
WORKSPACE_RUNS = WORKSPACE_DIR / "runs"
WORKSPACE_OPS_DIR = WORKSPACE_DIR / "_ops"  # 运维杂项统一屋（下划线前缀：非产品 stage）
FACTOR_LIBRARY_DIR = WORKSPACE_DIR / "factor_library"
# 因子资产三件套（meta/py/parquet）+ 评估 run / reports 收口根
FACTOR_STORE_DIR = WORKSPACE_DIR / "factors"
MINING_SESSIONS_DIR = WORKSPACE_DIR / "mining_sessions"
MINE_AGENT_DIR = WORKSPACE_DIR / "mine_agent"
MINE_TEAM_DIR = WORKSPACE_DIR / "mine_team"
COMBINATIONS_DIR = WORKSPACE_DIR / "combinations"
COMBINE_BACKTESTS_DIR = WORKSPACE_DIR / "combine_backtests"
RISK_MODELS_DIR = WORKSPACE_DIR / "risk_models"
PORTFOLIOS_DIR = WORKSPACE_DIR / "portfolios"
SIM_DIR = WORKSPACE_DIR / "sim"
STRATEGIES_DIR = WORKSPACE_DIR / "strategies"
# HTML 报告集中收口：workspace/factors/reports/
REPORTS_DIR = FACTOR_STORE_DIR / "reports"
EXECUTION_DIR = WORKSPACE_DIR / "execution"
OPS_DIR = WORKSPACE_DIR / "ops"
OPS_SITE_DIR = OPS_DIR / "site"
OPS_STATE_DIR = OPS_DIR / "state"


# config
CONFIG_DIR = WORKSPACE_DIR / "configs"

# data
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_RAW_DAILY = DATA_RAW / "daily"
DATA_RAW_FINANCE = DATA_RAW / "finance"
DATA_RAW_MINUTE = DATA_RAW / "minute_1min"
DATA_RAW_FUTURES = DATA_RAW / "fut_daily"
DATA_RAW_US = DATA_RAW / "us_daily"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_CACHE = DATA_DIR / "cache"
DATA_DERIVED = DATA_DIR / "derived"
INTRADAY_FEATURES_DIR = DATA_DERIVED / "intraday_features"
CRYPTO_LAKE = DATA_DIR / "crypto_lake"  # crypto 永续数据湖（Binance Vision）

# 评估产物落点：
#   有因子名 → workspace/factors/<market>/<name>/evaluations/{run_id}/
#   无因子名 → workspace/factors/_runs/{run_id}/
# 全局索引：workspace/factors/experiment_index.jsonl
# 旧 runs/artifacts/daily/{factors,results} 中间层已废除。
# HTML 报告收口到 workspace/factors/reports/（前端「报告」栏单点可见）。
OUTPUT_DIR = WORKSPACE_RUNS / "artifacts"  # 仅残留 intraday 等历史约定；日频评估不再写入
OUTPUT_DAILY_REPORTS = REPORTS_DIR / "daily"

OUTPUT_INTRADAY = OUTPUT_DIR / "intraday"
OUTPUT_INTRADAY_FACTORS = OUTPUT_INTRADAY / "factors"
OUTPUT_INTRADAY_RESULTS = OUTPUT_INTRADAY / "results"
OUTPUT_INTRADAY_REPORTS = REPORTS_DIR / "intraday"

# source
COMMON_DIR = ROOT / "src" / "factorzen" / "core"
REPORTING_DIR = ROOT / "src" / "factorzen" / "reports"
NOTEBOOKS_DIR = WORKSPACE_DIR / "notebooks"
TESTS_DIR = ROOT / "tests"


def daily_output_bucket(factor_name: str) -> str | None:
    """Return the output subdirectory for grouped daily report HTML."""
    if factor_name.startswith("qlib_alpha158_"):
        return "qlib158"
    if factor_name.startswith("qlib_alpha360_"):
        return "qlib360"
    return None


def daily_report_output_dir(factor_name: str) -> Path:
    """HTML 报告集中目录（非评估 run 的大文件落点）。"""
    bucket = daily_output_bucket(factor_name)
    return OUTPUT_DAILY_REPORTS / bucket if bucket else OUTPUT_DAILY_REPORTS
