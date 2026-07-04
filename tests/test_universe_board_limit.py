"""板块涨跌停阈值及 filter_limit 按板块细化测试。"""

from datetime import date

import polars as pl
import pytest

from factorzen.core.universe import _get_board_limit, filter_limit


@pytest.fixture(autouse=True)
def _no_namechange_by_default(monkeypatch):
    """默认 namechange 不可用，filter_limit 统一走降级（按 name 字符串匹配）路径。

    universe.py 用 ``from factorzen.core.loader import fetch_namechange`` 在
    模块级绑定，须 patch ``factorzen.core.universe.fetch_namechange`` 才能
    生效（patch ``factorzen.core.loader.fetch_namechange`` 对已绑定的引用
    无效）。避免本机 .env 配了真实 token 时意外触发真实网络请求。
    """

    def _boom() -> pl.DataFrame:
        raise RuntimeError("namechange unavailable in offline tests")

    monkeypatch.setattr("factorzen.core.universe.fetch_namechange", _boom)


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
# _get_board_limit(is_st=True) — ST 主板收窄阈值
# ──────────────────────────────────────────────────────────


def test_main_board_st_limit_is_4_8pct():
    """主板 ST/*ST 股票 is_st=True → 4.8%（5% 真实限额 - 0.2pp 容差）。"""
    assert abs(_get_board_limit("600001.SH", is_st=True) - 0.048) < 1e-6


def test_main_board_st_default_is_st_false_unchanged():
    """is_st 默认 False，行为与未引入该参数前完全一致（9.8%）。"""
    assert abs(_get_board_limit("600001.SH") - 0.098) < 1e-6
    assert abs(_get_board_limit("600001.SH", is_st=False) - 0.098) < 1e-6


def test_chuang_ye_ban_is_st_does_not_affect_limit():
    """创业板不受 is_st 影响（2020 年注册制改革后 ST 与非 ST 涨跌幅规则相同）。"""
    assert abs(_get_board_limit("300001.SZ", is_st=True) - 0.198) < 1e-6


def test_ke_chuang_ban_is_st_does_not_affect_limit():
    """科创板不受 is_st 影响（同上）。"""
    assert abs(_get_board_limit("688001.SH", is_st=True) - 0.198) < 1e-6


def test_bei_jiao_suo_is_st_does_not_affect_limit():
    """北交所不受 is_st 影响。"""
    assert abs(_get_board_limit("830001.BJ", is_st=True) - 0.298) < 1e-6


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


# ──────────────────────────────────────────────────────────
# filter_limit — ST 主板收窄阈值（4.8%），经 namechange PIT 判断
# ──────────────────────────────────────────────────────────


def _namechange_st_df(ts_code: str, start_date: date = date(2024, 1, 1)) -> pl.DataFrame:
    """构造单只股票当前处于 ST 状态的 namechange 记录。"""
    return pl.DataFrame(
        {
            "ts_code": [ts_code],
            "name": ["ST测试股"],
            "start_date": [start_date],
            "end_date": [None],
            "ann_date": [start_date],
            "change_reason": ["ST"],
        }
    )


def test_filter_limit_st_main_board_5pct_blocked(monkeypatch):
    """主板 ST 股票涨幅约 +5.0%（除法构造而非字面量），namechange 标记 ST 后
    应被 filter_limit 判定涨停过滤（阈值 4.8%）。
    """
    ts_code = "600001.SH"
    pct_chg = (10.5 / 10.0 - 1.0) * 100  # ≈5.0，由除法构造

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)
    monkeypatch.setattr(
        "factorzen.core.universe.fetch_namechange",
        lambda: _namechange_st_df(ts_code),
    )

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 0, f"ST 主板 5% 涨幅应被判定涨停过滤，实际 result={result}"


def test_filter_limit_non_st_5pct_not_blocked(monkeypatch):
    """同样约 +5.0% 涨幅，非 ST 主板不应被过滤（主板非 ST 阈值 9.8%）。"""
    ts_code = "600001.SH"
    pct_chg = (10.5 / 10.0 - 1.0) * 100  # ≈5.0，由除法构造

    def fake_load(category, start=None, end=None):
        class LazyWrapper:
            def collect(self):
                return _make_daily(ts_code, pct_chg)

        return LazyWrapper()

    monkeypatch.setattr("factorzen.core.storage.load_parquet", fake_load)
    # namechange 可用但无该代码的 ST 记录 → 判定为非 ST
    monkeypatch.setattr(
        "factorzen.core.universe.fetch_namechange",
        lambda: _namechange_st_df("000999.SZ"),
    )

    stocks = _make_stocks(ts_code)
    result = filter_limit(stocks, "20240101")
    assert len(result) == 1, f"非 ST 5% 涨幅不应被过滤，实际 result={result}"
