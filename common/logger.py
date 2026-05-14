"""日志系统。控制台 + 文件双输出，按天轮转。"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from config.settings import ROOT

_initialized = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: Path | None = None) -> None:
    """初始化日志系统。幂等，重复调用不重复添加 handler。

    Args:
        log_dir: 日志输出目录。默认由 settings.ROOT / "output" / "logs" 确定。
    """
    global _initialized
    if _initialized:
        return

    if log_dir is None:
        log_dir = ROOT / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 控制台 handler：INFO 级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件 handler：DEBUG 级别，按天轮转，保留 30 天
    log_file = log_dir / "factor_research.log"
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y%m%d"
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger。自动继承根 logger 的 handler 配置。"""
    return logging.getLogger(name)
