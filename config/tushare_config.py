"""Tushare 连接配置。Token 从环境变量读取，积分/限流参数可配置。"""

import os
from pathlib import Path

# ── 加载 .env 文件（如果存在）────────────────────────────
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if key and key not in os.environ:
                    os.environ[key] = value

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
