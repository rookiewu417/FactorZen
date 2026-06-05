"""Tushare 连接配置。Token 从环境变量读取，积分/限流参数可配置。"""

import os
from collections.abc import MutableMapping
from pathlib import Path

from factorzen.config.settings import ROOT


def _load_dotenv(path: Path, env: MutableMapping[str, str] | None = None) -> None:
    """读取 .env 填充环境变量（不覆盖已存在的键）。

    容忍 BOM（``utf-8-sig``）、CRLF、键值首尾空白与成对引号——此前用 ``utf-8``
    打开会把 BOM 读进首行，使 ``TUSHARE_TOKEN`` 被存成 ``\\ufeffTUSHARE_TOKEN``
    而静默失效。
    """
    target = env if env is not None else os.environ
    if not path.exists():
        return
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in target:
                target[key] = value


# ── 加载 .env 文件（如果存在）────────────────────────────
_env_file = ROOT / ".env"
_load_dotenv(_env_file)

# ── Token（延迟校验，import 时不崩溃）──────────────────────
_token: str | None = os.environ.get("TUSHARE_TOKEN")
TUSHARE_TOKEN: str = _token or ""


def ensure_token() -> str:
    """返回 token；首次真正需要时调用，避免离线测试场景 import 即崩。"""
    tok = os.environ.get("TUSHARE_TOKEN") or _token
    if not tok:
        raise RuntimeError(
            "请设置 TUSHARE_TOKEN 环境变量\n"
            "  Windows: set TUSHARE_TOKEN=your_token\n"
            "  或在 .env 文件中写入 TUSHARE_TOKEN=your_token"
        )
    return tok


# ── 积分与限流 ─────────────────────────────────────────
TUSHARE_POINTS: int = int(os.environ.get("TUSHARE_POINTS", "2000"))
MAX_RPS: int = int(os.environ.get("TUSHARE_MAX_RPS", "5"))
MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.0
BATCH_SIZE: int = 5000
CACHE_EXPIRE_DAYS: int = 7
