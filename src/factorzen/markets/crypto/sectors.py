"""crypto sector 分类（对标 A 股行业）。

crypto 无官方行业分类源，用一份策展的静态映射（L1/L2/DeFi/meme/oracle/…）。
未收录标的归 "other"。可通过 sector_map 覆盖（如接入第三方分类）。
"""
from __future__ import annotations

import polars as pl

# 常见 USDT-M 永续 → sector（策展，MVP）
CRYPTO_SECTORS: dict[str, str] = {
    "BTCUSDT": "L1", "ETHUSDT": "L1", "SOLUSDT": "L1", "BNBUSDT": "L1",
    "ADAUSDT": "L1", "AVAXUSDT": "L1", "XRPUSDT": "L1", "TRXUSDT": "L1",
    "DOTUSDT": "L1", "NEARUSDT": "L1", "APTUSDT": "L1", "SUIUSDT": "L1",
    "ARBUSDT": "L2", "OPUSDT": "L2", "MATICUSDT": "L2",
    "UNIUSDT": "DeFi", "AAVEUSDT": "DeFi", "MKRUSDT": "DeFi", "CRVUSDT": "DeFi",
    "LDOUSDT": "DeFi", "COMPUSDT": "DeFi",
    "LINKUSDT": "oracle",
    "DOGEUSDT": "meme", "SHIBUSDT": "meme", "PEPEUSDT": "meme", "WIFUSDT": "meme",
    "FILUSDT": "storage", "RNDRUSDT": "AI", "FETUSDT": "AI",
}

_DEFAULT_SECTOR = "other"


def sector_of(ts_code: str, sector_map: dict[str, str] | None = None) -> str:
    m = CRYPTO_SECTORS if sector_map is None else sector_map
    return m.get(ts_code, _DEFAULT_SECTOR)


def build_sector_frame(
    symbols: list[str], sector_map: dict[str, str] | None = None
) -> pl.DataFrame:
    """标的 → ``[ts_code, industry]``（industry=sector），供 get_industry_dummies 生成 one-hot。"""
    return pl.DataFrame(
        {"ts_code": symbols, "industry": [sector_of(s, sector_map) for s in symbols]}
    )
