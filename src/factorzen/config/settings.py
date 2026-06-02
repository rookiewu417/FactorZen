"""全局路径配置。所有模块通过 ``from factorzen.config.settings import ROOT`` 引用路径。"""

from pathlib import Path

# 项目根目录：src/factorzen/config/settings.py 向上三级
ROOT = Path(__file__).resolve().parents[3]

WORKSPACE_DIR = ROOT / "workspace"
WORKSPACE_RUNS = WORKSPACE_DIR / "runs"
FACTOR_EVALUATIONS_DIR = WORKSPACE_DIR / "factor_evaluations"

# ── 自动化调度配置 ───────────────────────────────────────────────────────────────
AUTOMATION_OUTPUT = WORKSPACE_RUNS / "automation"
SCHEDULER_TIMEZONE: str = "Asia/Shanghai"
SCHEDULER_CRON_HOUR: int = 16
SCHEDULER_CRON_MINUTE: int = 30
SCHEDULER_MAX_RETRIES: int = 3
SCHEDULER_RETRY_BASE_SECONDS: int = 60

# config
CONFIG_DIR = WORKSPACE_DIR / "configs"

# data
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_RAW_DAILY = DATA_RAW / "daily"
DATA_RAW_FINANCE = DATA_RAW / "finance"
DATA_RAW_MINUTE = DATA_RAW / "minute"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_CACHE = DATA_DIR / "cache"

OUTPUT_DIR = WORKSPACE_RUNS / "artifacts"
OUTPUT_DAILY = OUTPUT_DIR / "daily"
OUTPUT_DAILY_FACTORS = OUTPUT_DAILY / "factors"
OUTPUT_DAILY_RESULTS = OUTPUT_DAILY / "results"
OUTPUT_DAILY_CHARTS = OUTPUT_DAILY / "charts"
OUTPUT_DAILY_REPORTS = OUTPUT_DAILY / "reports"

OUTPUT_INTRADAY = OUTPUT_DIR / "intraday"
OUTPUT_INTRADAY_FACTORS = OUTPUT_INTRADAY / "factors"
OUTPUT_INTRADAY_RESULTS = OUTPUT_INTRADAY / "results"
OUTPUT_INTRADAY_REPORTS = OUTPUT_INTRADAY / "reports"

# source
COMMON_DIR = ROOT / "src" / "factorzen" / "core"
REPORTING_DIR = ROOT / "src" / "factorzen" / "reports"
NOTEBOOKS_DIR = WORKSPACE_DIR / "notebooks"
TESTS_DIR = ROOT / "tests"


def daily_output_bucket(factor_name: str) -> str | None:
    """Return the output subdirectory for grouped daily factor artifacts."""
    if factor_name.startswith("qlib_alpha158_"):
        return "qlib158"
    if factor_name.startswith("qlib_alpha360_"):
        return "qlib360"
    return None


def daily_factor_output_dir(factor_name: str) -> Path:
    bucket = daily_output_bucket(factor_name)
    return OUTPUT_DAILY_FACTORS / bucket if bucket else OUTPUT_DAILY_FACTORS


def daily_result_output_dir(factor_name: str) -> Path:
    bucket = daily_output_bucket(factor_name)
    return OUTPUT_DAILY_RESULTS / bucket if bucket else OUTPUT_DAILY_RESULTS


def daily_report_output_dir(factor_name: str) -> Path:
    bucket = daily_output_bucket(factor_name)
    return OUTPUT_DAILY_REPORTS / bucket if bucket else OUTPUT_DAILY_REPORTS
