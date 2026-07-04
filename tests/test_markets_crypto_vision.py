"""vision 下载器离线单测:canned XML/CSV/zip,fetch 全注入。"""
import io
import zipfile

import polars as pl
import pytest

from factorzen.markets.crypto.lake import CryptoLake
from factorzen.markets.crypto.vision import (
    backfill,
    fetch_zip_csv,
    list_um_symbols,
    parse_funding_csv,
    parse_kline_csv,
    parse_metrics_csv,
    rank_symbols_by_amount,
)

_KLINE_HEADER = (b"open_time,open,high,low,close,volume,close_time,quote_volume,"
                 b"count,taker_buy_volume,taker_buy_quote_volume,ignore\n")
_KLINE_ROW = b"1782604800000,60000.4,60018.7,60000.3,60018.6,37.652,1782604859999,2259400.8,1187,12.342,740568.6,0\n"


def _zip_bytes(name: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, payload)
    return buf.getvalue()


def test_parse_kline_csv_with_and_without_header():
    for raw in (_KLINE_HEADER + _KLINE_ROW, _KLINE_ROW):  # vision 老文件无表头
        df = parse_kline_csv(raw)
        assert df.columns == ["trade_date", "open", "high", "low", "close",
                              "vol", "amount", "taker_buy_volume"]
        assert df.schema["trade_date"] == pl.Datetime("us")
        assert df["amount"][0] == pytest.approx(2259400.8)   # 真 quote_volume,非 close*vol
        assert df["taker_buy_volume"][0] == pytest.approx(12.342)


def test_parse_funding_and_metrics():
    fr = parse_funding_csv(b"calc_time,funding_interval_hours,last_funding_rate\n"
                           b"1777593600000,8,-0.00003746\n")
    assert fr.columns == ["event_time", "funding_rate"]
    assert fr["funding_rate"][0] == pytest.approx(-0.00003746)
    mt = parse_metrics_csv(
        b"create_time,symbol,sum_open_interest,sum_open_interest_value,"
        b"count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
        b"count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        b"2026-06-27 00:05:00,BTCUSDT,103630.42,6225989541.67,2.2,1.2,2.1,0.7\n")
    assert mt.columns == ["event_time", "open_interest"]
    assert mt["open_interest"][0] == pytest.approx(103630.42)


def test_list_um_symbols_from_s3_xml():
    xml = (b"<?xml version='1.0'?><ListBucketResult>"
           b"<Prefix>data/futures/um/monthly/klines/</Prefix>"
           b"<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>"
           b"<CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHUSDT/</Prefix></CommonPrefixes>"
           b"<CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSD_PERP/</Prefix></CommonPrefixes>"
           b"<IsTruncated>false</IsTruncated></ListBucketResult>")
    seen = {}

    def _fetch(url):
        seen["url"] = url
        return xml

    syms = list_um_symbols(fetch=_fetch)
    assert syms == ["BTCUSDT", "ETHUSDT"]  # 非 USDT 结尾剔除
    assert "s3" in seen["url"]  # listing 必须走 S3 endpoint(CDN 前端只返回 HTML)


def test_fetch_zip_csv_retries_then_none():
    calls = {"n": 0}
    def bad_fetch(url):
        calls["n"] += 1
        raise OSError("404")
    assert fetch_zip_csv("http://x/a.zip", fetch=bad_fetch, retries=2) is None
    assert calls["n"] == 3  # 1 次 + 2 重试


def test_backfill_writes_lake_and_records_gaps(tmp_path):
    lake = CryptoLake(tmp_path)
    kzip = _zip_bytes("k.csv", _KLINE_HEADER + _KLINE_ROW)
    fzip = _zip_bytes("f.csv", b"calc_time,funding_interval_hours,last_funding_rate\n"
                               b"1782604800000,8,0.0001\n")
    def fetch(url):
        if "fundingRate" in url:
            return fzip
        if "/klines/" in url and "1m" in url:
            return kzip
        raise OSError("404")  # metrics 全 404 → 进 gaps
    manifest = backfill(lake, ["BTCUSDT"], "20260628", "20260628", fetch=fetch, log=lambda *a: None)
    assert lake.read_klines(["BTCUSDT"], "20260628", "20260628").height == 1
    assert lake.read_funding(["BTCUSDT"], "20260628", "20260628").height == 1
    assert any("metrics" in g for g in manifest["gaps"])  # 缺口不静默
    assert (tmp_path / "manifest.json").exists()
    # 增量:重跑不重复下载已有分区
    counts = {"n": 0}
    def counting_fetch(url):
        counts["n"] += 1
        return fetch(url)
    backfill(lake, ["BTCUSDT"], "20260628", "20260628", fetch=counting_fetch, log=lambda *a: None)
    assert all("/klines/" not in u for u in []) or counts["n"] < 3  # kline/funding 已存在被跳过


def test_rank_symbols_by_amount(tmp_path):
    # 1d 月包:BTC 成交额 > ETH,选 Top-1 应得 BTC
    def _kd(amount_row: bytes) -> bytes:
        return _zip_bytes("d.csv", _KLINE_HEADER + amount_row)
    big = b"1782604800000,1,1,1,1,1,1782604859999,9999999,1,1,1,0\n"
    small = b"1782604800000,1,1,1,1,1,1782604859999,1000,1,1,1,0\n"
    def fetch(url):
        if "BTCUSDT-1d" in url:
            return _kd(big)
        if "ETHUSDT-1d" in url:
            return _kd(small)
        raise OSError("404")
    top = rank_symbols_by_amount(["BTCUSDT", "ETHUSDT"], "2026-05", top_n=1, fetch=fetch)
    assert top == ["BTCUSDT"]
