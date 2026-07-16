"""reporting/ — 因子报告生成。

提供单因子 Tear Sheet（极简单页）与组合 Dashboard 报告。

Usage:
    >>> from factorzen.reports.tear_sheet import generate_tear_sheet
    >>> html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
"""

from factorzen.reports.tear_sheet import generate_tear_sheet

__all__ = ["generate_tear_sheet"]
