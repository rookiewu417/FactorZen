"""财报类月频因子（asset_growth/roe_ttm）的 required_data 须如实声明依赖。

历史缺陷：二者读 finance parquet、pipeline 还需 daily 算前向收益，却误声明
required_data=["daily_basic"]（从不读）——导致 ensure 拉错数据，且缺 "daily"
声明使 daily_single 调 ctx.daily 时 raise "daily data not declared"，用空返回
掩盖 finance 缺失的真实根因。
"""
from __future__ import annotations


def test_finance_monthly_factors_declare_finance_and_daily():
    from factorzen.builtin_factors.monthly.asset_growth import AssetGrowthMonthly
    from factorzen.builtin_factors.monthly.profitability import RoeTtmMonthly

    for cls in (AssetGrowthMonthly, RoeTtmMonthly):
        rd = cls.required_data
        assert "finance" in rd, f"{cls.name} compute 读 finance parquet，应声明 finance"
        assert "daily" in rd, f"{cls.name} pipeline 需 ctx.daily 算前向收益，应声明 daily"
        assert "daily_basic" not in rd, f"{cls.name} 从不读 daily_basic，不应声明"
