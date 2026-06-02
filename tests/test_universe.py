"""测试指数成分股加载。"""

import os

import pytest

from factorzen.core.universe import get_universe

# ── helpers ────────────────────────────────────────────────────────────────

needs_tushare = pytest.mark.skipif(
    not os.environ.get("TUSHARE_TOKEN"),
    reason="TUSHARE_TOKEN 未设置，跳过 Tushare 集成测试",
)

# 使用近期交易日，确保 Tushare 有数据
FIXTURE_DATE = "20260512"
FIXTURE_INDEX_CSI300 = "000300.SH"
FIXTURE_INDEX_CSI500 = "000905.SH"


# ── index members ──────────────────────────────────────────────────────────


@needs_tushare
def test_get_index_members_csi300():
    """CSI300 成分股应返回 200-350 只股票（而非全 A 股 ~5500 只）。"""
    result = get_universe(FIXTURE_DATE, "csi300")

    assert not result.is_empty(), "CSI300 不应为空"
    assert "ts_code" in result.columns
    assert "name" in result.columns

    count = result.height
    assert 200 <= count <= 350, f"CSI300 预期 200-350 只，实际 {count} 只"


@needs_tushare
def test_csi800_is_union():
    """CSI800 = CSI300 ∪ CSI500，去重后数量应 ≈ CSI300 + CSI500。"""
    csi300_codes = set(get_universe(FIXTURE_DATE, "csi300")["ts_code"].to_list())
    csi500_codes = set(get_universe(FIXTURE_DATE, "csi500")["ts_code"].to_list())
    csi800_codes = set(get_universe(FIXTURE_DATE, "csi800")["ts_code"].to_list())

    n800 = len(csi800_codes)

    # CSI800 应为 union 去重
    expected_union = csi300_codes | csi500_codes
    assert expected_union == csi800_codes, "CSI800 应为 CSI300 ∪ CSI500"

    assert n800 == len(expected_union), (
        f"CSI800({n800}) 应等于 union 去重结果({len(expected_union)})"
    )
