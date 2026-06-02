"""板块涨跌停阈值及 filter_limit 按板块细化测试。"""

import polars as pl

from factorzen.core.universe import _get_board_limit, filter_limit

# ──────────────────────────────────────────────────────────
# _get_board_limit 单元测试
# ──────────────────────────────────────────────────────────


def test_chuang_ye_ban_limit_300():
    """创业板 300xxx → 19.8%。"""
    assert abs(_get_board_limit("300001.SZ") - 0.198) < 1e-6


def test_chuang_ye_ban_limit_301():
    """创业板 301xxx → 19.8%。"""
    assert abs(_get_board_limit("301001.SZ") - 0.198) < 1e-6


def test_ke_chuang_ban_limit_688():
    """科创板 688xxx → 19.8%。"""
    assert abs(_get_board_limit("688001.SH") - 0.198) < 1e-6


def test_ke_chuang_ban_limit_689():
    """科创板 689xxx → 19.8%。"""
    assert abs(_get_board_limit("689001.SH") - 0.198) < 1e-6


def test_bei_jiao_suo_limit():
    """北交所 .BJ 后缀 → 29.8%。"""
    assert abs(_get_board_limit("830001.BJ") - 0.298) < 1e-6


def test_main_board_limit_600():
    """主板 600xxx → 9.8%。"""
    assert abs(_get_board_limit("600001.SH") - 0.098) < 1e-6


def test_main_board_limit_000():
    """主板 000xxx → 9.8%。"""
    assert abs(_get_board_limit("000001.SZ") - 0.098) < 1e-6


def test_main_board_limit_case_insensitive():
    """大小写不敏感。"""
    assert abs(_get_board_limit("600001.sh") - 0.098) < 1e-6


# ──────────────────────────────────────────────────────────
# filter_limit 纯 DataFrame 路径（不依赖日线存储）
#
# filter_limit 正常路径需要 load_parquet；
# 此处通过 monkeypatch 绕过，直接测试过滤逻辑。
# ──────────────────────────────────────────────────────────


def _make_daily(ts_code: str, pct_chg: float) -> pl.DataFrame:
    """构造仅含 ts_code 和 pct_chg 的最小日线 DataFrame。"""
    return pl.DataFrame({
        "ts_code": [ts_code],
        "pct_chg": [pct_chg],
        "vol": [1000.0],
        "amount": [1_000_000.0],
        "open": [10.0],
        "close": [10.0],
    })


def _make_stocks(ts_code: str) -> pl.DataFrame:
    return pl.DataFrame({
        "ts_code": [ts_code],
        "name": ["Test Stock"],
        "list_date": [None],
        "delist_date": [None],
    })


def test_filter_limit_allows_chuang_ye_195pct(monkeypatch):
    """创业板 19.5% 涨幅 < 19.8% 阈值，不应被过滤。"""

    ts_code = "300001.SZ"
    pct_chg = 19.5

    def fake_load(category, start=None, end=None):

        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 1, f"创业板 19.5% 不应被过滤，但 result={result}"


def test_filter_limit_blocks_chuang_ye_198pct(monkeypatch):
    """创业板 19.8% 正好达到阈值，应被过滤（>= 而非 >）。"""
    ts_code = "300001.SZ"
    pct_chg = 19.8

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 0, f"创业板 19.8% 应被过滤，但 result={result}"


def test_filter_limit_blocks_main_board_10pct(monkeypatch):
    """主板 10% > 9.8% 阈值，应被过滤。"""
    ts_code = "600001.SH"
    pct_chg = 10.0

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 0, f"主板 10% 应被过滤，但 result={result}"


def test_filter_limit_allows_main_board_9pct(monkeypatch):
    """主板 9% < 9.8% 阈值，不应被过滤。"""
    ts_code = "600001.SH"
    pct_chg = 9.0

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 1, f"主板 9% 不应被过滤，但 result={result}"


def test_filter_limit_mixed_boards(monkeypatch):
    """主板 10% 被过滤，创业板 19.5% 保留，测试混合场景。"""
    daily_data = pl.DataFrame({
        "ts_code": ["600001.SH", "300001.SZ"],
        "pct_chg": [10.0, 19.5],
        "vol": [1000.0, 1000.0],
        "amount": [1_000_000.0, 1_000_000.0],
        "open": [10.0, 10.0],
        "close": [10.0, 10.0],
    })

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return daily_data

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)

    stocks = pl.DataFrame({
        "ts_code": ["600001.SH", "300001.SZ"],
        "name": ["Main", "ChiNext"],
        "list_date": [None, None],
        "delist_date": [None, None],
    })
    result = filter_limit(stocks, "20240101")
    assert len(result) == 1
    assert result["ts_code"][0] == "300001.SZ"
