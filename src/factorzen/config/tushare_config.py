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
            value = value.strip()
            # 剥行内注释（` #` 空格+井号，dotenv 常见写法 `KEY=val # 说明`）；
            # 无空格的 `#`（如 URL fragment）不当注释。在去引号前处理。
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            value = value.strip('"').strip("'")
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
def _int_env(key: str, default: str) -> int:
    """import 期安全解析 int：剥行内注释、非数字回退默认，避免 .env 写法失误令整个
    CLI 在 import 阶段崩溃（连与 Tushare 无关的离线命令都用不了）。"""
    raw = os.environ.get(key, default)
    try:
        return int(str(raw).split("#", 1)[0].strip())
    except (ValueError, TypeError):
        return int(default)


TUSHARE_POINTS: int = _int_env("TUSHARE_POINTS", "2000")
MAX_RPS: int = _int_env("TUSHARE_MAX_RPS", "5")
MAX_RETRIES: int = 3
RETRY_DELAY: float = 1.0
BATCH_SIZE: int = 5000
CACHE_EXPIRE_DAYS: int = 7
