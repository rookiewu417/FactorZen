"""vision backfill 当月增量补拉(M3) + 写 meta 打通 universe(M4)。

M3：当月(无月包)首次用日包写入部分月份后，后续扩日期的回填被整月跳过、新增天永久缺失。
M4：backfill 从不写 meta.parquet → lake universe.snapshot 恒空 → backfill→mine 官方链路断。
"""
from __future__ import annotations

import io
import zipfile

import polars as pl

from factorzen.markets.crypto.lake import CryptoLake
from factorzen.markets.crypto.vision import backfill

_HEADER = (b"open_time,open,high,low,close,volume,close_time,quote_volume,"
           b"count,taker_buy_volume,taker_buy_quote_volume,ignore\n")
_D28 = 1782604800000  # 2026-06-28 00:00 UTC
_DAY = 86_400_000


def _zip(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("d.csv", payload)
    return buf.getvalue()


def _day_zip(open_ms: int) -> bytes:
    row = f"{open_ms},1,1,1,1,1,{open_ms + 59999},100,1,1,1,0\n".encode()
    return _zip(_HEADER + row)


def _fetch_daypacks_only(url: str) -> bytes:
    # 当月无月包(/monthly/ 全 404)；日包按 URL 里的日期返回对应时间戳
    if "/daily/klines/" in url and "2026-06-28" in url:
        return _day_zip(_D28)
    if "/daily/klines/" in url and "2026-06-29" in url:
        return _day_zip(_D28 + _DAY)
    raise OSError("404")


def test_backfill_current_month_increment_tops_up(tmp_path):
    lake = CryptoLake(tmp_path)
    # 第一次：只回填 6-28（日包，无月包）
    backfill(lake, ["BTCUSDT"], "20260628", "20260628",
             fetch=_fetch_daypacks_only, log=lambda *a: None)
    assert lake.read_klines(["BTCUSDT"], "20260628", "20260628").height == 1

    # 第二次：扩到 6-29 —— 修复前当月分区已存在被整月跳过，6-29 永久缺失
    backfill(lake, ["BTCUSDT"], "20260628", "20260629",
             fetch=_fetch_daypacks_only, log=lambda *a: None)
    got = lake.read_klines(["BTCUSDT"], "20260628", "20260629")
    dates = set(got.select(pl.col("trade_date").dt.strftime("%Y%m%d")).to_series().to_list())
    assert dates == {"20260628", "20260629"}, f"当月增量应补拉 6-29，实得 {dates}"


def test_backfill_writes_meta_for_universe(tmp_path):
    lake = CryptoLake(tmp_path)
    backfill(lake, ["BTCUSDT"], "20260628", "20260628",
             fetch=_fetch_daypacks_only, log=lambda *a: None)
    meta = lake.read_meta()
    assert meta.height == 1
    assert meta["ts_code"][0] == "BTCUSDT"
    # universe.snapshot 要求 list_date 非空，否则过滤后为空
    assert meta["list_date"][0] is not None
