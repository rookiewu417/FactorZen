"""crypto 频率表:bar 频率 → ccxt timeframe / polars every / 年化周期数,单一事实源。

provider(timeframe)、resample(every)、calendar(年化)共用此表;未知频率一律
raise,不做静默兜底(替代旧 ``_TIMEFRAME_MAP.get(freq, "1d")``)。``hourly`` 为
``1h`` 别名;weekly/monthly 仅保留年化(calendar 兼容),不是合法 bar 频率。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BarFreq:
    timeframe: str  # ccxt timeframe
    every: str  # polars group_by_dynamic / dt.truncate 的 every
    periods_per_year: float


BAR_FREQS: dict[str, BarFreq] = {
    "1m": BarFreq("1m", "1m", 365.0 * 24 * 60),
    "5m": BarFreq("5m", "5m", 365.0 * 24 * 12),
    "15m": BarFreq("15m", "15m", 365.0 * 24 * 4),
    "1h": BarFreq("1h", "1h", 365.0 * 24),
    "daily": BarFreq("1d", "1d", 365.0),
}
_ALIASES: dict[str, str] = {"hourly": "1h"}
_EXTRA_PERIODS: dict[str, float] = {"weekly": 52.0, "monthly": 12.0}


def normalize_freq(freq: str) -> str:
    f = _ALIASES.get(freq, freq)
    if f not in BAR_FREQS:
        raise ValueError(
            f"未知频率: {freq!r}，支持 {sorted(BAR_FREQS)} + 别名 {sorted(_ALIASES)}"
        )
    return f


def periods_per_year(freq: str) -> float:
    if freq in _EXTRA_PERIODS:
        return _EXTRA_PERIODS[freq]
    return BAR_FREQS[normalize_freq(freq)].periods_per_year
