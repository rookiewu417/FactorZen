from __future__ import annotations


def test_daily_basic_cols_include_new_fields():
    from factorzen.core.loader import DAILY_BASIC_COLS
    for f in ["turnover_rate", "turnover_rate_f", "volume_ratio", "float_share"]:
        assert f in DAILY_BASIC_COLS, f"DAILY_BASIC_COLS missing {f}"
    # 原有字段仍在
    for f in ["trade_date", "ts_code", "pe_ttm", "pb", "total_mv", "circ_mv"]:
        assert f in DAILY_BASIC_COLS
