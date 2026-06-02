"""S1-B 防回归：验证 get_universe("all_a") 实现 PIT 过滤（幸存者偏差消除）。

策略：用 monkeypatch 替换 fetch_stock_basic，注入含退市股的合成数据，
验证 get_universe 在指定日期只返回彼时在市的股票，不包含未来退市股和未来上市股。
"""

from datetime import date

import polars as pl
import pytest

from factorzen.core.universe import get_universe


@pytest.fixture
def synthetic_stock_basic(monkeypatch):
    """注入含退市股的合成股票基本信息。"""
    df = pl.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "symbol": ["000001", "000002", "000003", "000004"],
            "name": ["股票A", "股票B（已退市）", "股票C（未来上市）", "股票D（无退市日）"],
            "area": ["深圳"] * 4,
            "industry": ["银行"] * 4,
            "market": ["主板"] * 4,
            "list_date": [
                date(2005, 1, 1),  # 2005 上市，至今在市
                date(2010, 1, 1),  # 2010 上市，2023-12-31 退市
                date(2025, 1, 1),  # 2025 上市（基准日之后），不应出现
                date(2008, 1, 1),  # 2008 上市，无退市日（仍在市）
            ],
            "delist_date": [
                None,  # A：无退市日，仍在市
                date(2023, 12, 31),  # B：2023-12-31 退市
                None,  # C：未来上市
                None,  # D：无退市日，仍在市
            ],
        }
    )
    monkeypatch.setattr("factorzen.core.universe.get_stock_basic", lambda: df)
    return df


class TestUniversePIT:
    def test_all_a_excludes_delisted(self, synthetic_stock_basic):
        """基准日 2024-01-15：已于 2023-12-31 退市的 000002.SZ 不应出现。"""
        result = get_universe("20240115", "all_a")
        codes = result["ts_code"].to_list()
        assert "000002.SZ" not in codes, "退市股 000002.SZ 不应出现在 2024-01-15 的股票池"

    def test_all_a_excludes_future_listed(self, synthetic_stock_basic):
        """基准日 2024-01-15：2025 年上市的 000003.SZ 不应出现。"""
        result = get_universe("20240115", "all_a")
        codes = result["ts_code"].to_list()
        assert "000003.SZ" not in codes, "未上市股 000003.SZ 不应出现在 2024-01-15 的股票池"

    def test_all_a_includes_active_stocks(self, synthetic_stock_basic):
        """基准日 2024-01-15：2005 上市、仍在市的 000001.SZ 应出现。"""
        result = get_universe("20240115", "all_a")
        codes = result["ts_code"].to_list()
        assert "000001.SZ" in codes, "在市股 000001.SZ 应出现在 2024-01-15 的股票池"
        assert "000004.SZ" in codes, "在市股 000004.SZ 应出现在 2024-01-15 的股票池"

    def test_all_a_includes_stock_before_delist(self, synthetic_stock_basic):
        """基准日 2023-06-01：000002.SZ 尚未退市（2023-12-31 才退），应出现。"""
        result = get_universe("20230601", "all_a")
        codes = result["ts_code"].to_list()
        assert "000002.SZ" in codes, "尚未退市的 000002.SZ 应出现在 2023-06-01 的股票池"

    def test_all_a_excludes_stock_on_delist_date(self, synthetic_stock_basic):
        """基准日 2023-12-31（退市日当天）：000002.SZ 应已被排除（delist_date > date 严格大于）。"""
        result = get_universe("20231231", "all_a")
        codes = result["ts_code"].to_list()
        assert "000002.SZ" not in codes, (
            "退市当日 000002.SZ 不应出现在股票池（delist_date 严格大于）"
        )

    def test_pit_count_varies_by_date(self, synthetic_stock_basic):
        """不同日期的股票池大小应不同（PIT 过滤生效）。"""
        pre_delist = get_universe("20230601", "all_a")  # B 尚在市 → 3 只
        post_delist = get_universe("20240115", "all_a")  # B 已退市 → 2 只
        assert len(pre_delist) > len(post_delist), (
            f"2023-06-01 ({len(pre_delist)} 只) 应多于 2024-01-15 ({len(post_delist)} 只)"
        )
