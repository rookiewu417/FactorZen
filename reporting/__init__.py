"""reporting/ — Factor Tear Sheet 报告生成引擎。

提供因子研究完整报告的 HTML 生成功能，包含 6 个面板：
  1. Overview — 关键指标卡片
  2. Returns Analysis — 分组收益 + 多空 NAV
  3. IC Analysis — Rank IC 时间序列
  4. Turnover Analysis — 换手率 + 迁移矩阵
  5. Risk Attribution — 行业/市值/市场状态分层
  6. Summary — 综合评估表

Usage:
    >>> from reporting.tear_sheet import generate_tear_sheet
    >>> html = generate_tear_sheet("momentum_20d", ic_result, bt_result, to_result)
"""
