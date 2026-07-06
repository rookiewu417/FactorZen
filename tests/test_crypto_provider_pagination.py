"""crypto provider funding 分页(M1) + OI timeframe/日聚合(M2)。

M1：fetch_funding 单次 limit=1000(~333天)长区间静默截断；须分页拉全。
M2：fetch_open_interest 不传 timeframe(真实 ccxt 默认 '1h')且不按日聚合→24行/日；
   须请求 '1d' 并按日去重聚合。
"""
from __future__ import annotations

import polars as pl

from factorzen.markets.crypto.provider import CryptoDataProvider

_DAY_MS = 86_400_000
_H8 = 8 * 3600_000
_BASE = 1_704_067_200_000  # 2024-01-01 00:00 UTC


class _PagedCCXT:
    def __init__(self, funding, oi):
        self._f = funding
        self._oi = oi
        self.oi_timeframes: list[str] = []

    def _to_unified(self, s):  # 不用；provider 自己映射
        return s

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        data = [r for r in self._f if since is None or r["timestamp"] >= since]
        return data[:limit]

    def fetch_open_interest_history(self, symbol, timeframe="1h", since=None, limit=1000):
        # 默认 '1h' 模拟真实 ccxt；provider 须显式传 '1d'
        self.oi_timeframes.append(timeframe)
        data = [r for r in self._oi if since is None or r["timestamp"] >= since]
        return data[:limit]

    def load_markets(self):
        return {}


def test_fetch_funding_paginates_beyond_1000():
    # 1200 档 funding（8h 一档）= 400 天，超过单页 1000 档(~333 天)
    funding = [{"timestamp": _BASE + i * _H8, "fundingRate": 0.0001} for i in range(1200)]
    client = _PagedCCXT(funding, [])
    p = CryptoDataProvider(client=client)
    end = "20250204"  # 2024-01-01 + 400 天 ≈ 2025-02-04
    fd = p.fetch_funding(["BTCUSDT"], "20240101", end)
    # 400 天全部拉到（每天 3 档聚合成 1 行）；修复前只 ~333 天
    assert fd.height >= 399, f"长区间 funding 应分页拉全，实得 {fd.height} 天（修复前截断到 ~333）"


def test_fetch_open_interest_daily_timeframe_and_aggregation():
    # 每天 24 个小时级 OI 点，共 2 天 = 48 条
    oi = [{"timestamp": _BASE + d * _DAY_MS + h * 3600_000, "openInterestAmount": 1000.0 + h}
          for d in range(2) for h in range(24)]
    client = _PagedCCXT([], oi)
    p = CryptoDataProvider(client=client)
    result = p.fetch_open_interest(["BTCUSDT"], "20240101", "20240102")
    # 按日聚合 → 每天 1 行（修复前 24 行/日 → 48 行，join 后日频帧爆炸 24 倍）
    assert result.height == 2, f"OI 应按日聚合成 2 行，实得 {result.height}"
    # 每个 (ts_code, trade_date) 唯一
    assert result.select(["ts_code", "trade_date"]).n_unique() == result.height
    # provider 须显式请求 daily timeframe，而非用 ccxt 默认 '1h'
    assert "1d" in client.oi_timeframes, f"应请求 timeframe='1d'，实得 {client.oi_timeframes}"
