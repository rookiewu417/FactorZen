"""全局路径配置。所有模块通过 ``from config.settings import ROOT`` 引用路径。"""

from pathlib import Path

# 项目根目录：本文件向上两级
ROOT = Path(__file__).resolve().parent.parent

# config
CONFIG_DIR = ROOT / "config"

# data
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_RAW_DAILY = DATA_RAW / "daily"
DATA_RAW_FINANCE = DATA_RAW / "finance"
DATA_RAW_MINUTE = DATA_RAW / "minute"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_CACHE = DATA_DIR / "cache"

# output
OUTPUT_DIR = ROOT / "output"
OUTPUT_LFT = OUTPUT_DIR / "lft"
OUTPUT_MFT = OUTPUT_DIR / "mft"
OUTPUT_LFT_FACTORS = OUTPUT_LFT / "factors"
OUTPUT_LFT_RESULTS = OUTPUT_LFT / "results"
OUTPUT_LFT_CHARTS = OUTPUT_LFT / "charts"
OUTPUT_LFT_REPORTS = OUTPUT_LFT / "reports"
OUTPUT_MFT_FACTORS = OUTPUT_MFT / "factors"
OUTPUT_MFT_RESULTS = OUTPUT_MFT / "results"
OUTPUT_MFT_CHARTS = OUTPUT_MFT / "charts"
OUTPUT_MFT_REPORTS = OUTPUT_MFT / "reports"

OUTPUT_INTRADAY = OUTPUT_DIR / "intraday"
OUTPUT_INTRADAY_FACTORS = OUTPUT_INTRADAY / "factors"
OUTPUT_INTRADAY_RESULTS = OUTPUT_INTRADAY / "results"
OUTPUT_INTRADAY_REPORTS = OUTPUT_INTRADAY / "reports"

# source
COMMON_DIR = ROOT / "common"
LFT_DIR = ROOT / "lft"
MFT_DIR = ROOT / "mft"
HFT_DIR = ROOT / "hft"
REPORTING_DIR = ROOT / "reporting"
SCRIPTS_DIR = ROOT / "scripts"
NOTEBOOKS_DIR = ROOT / "notebooks"
TESTS_DIR = ROOT / "tests"
