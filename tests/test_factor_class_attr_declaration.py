"""日频因子的类属性声明必须在实例上生效。

历史缺陷：`DailyFactor` 曾是 `@dataclass`，而其子类（含全部内置日频因子与
`workspace/factors/` 模板教的写法）都用**无注解的普通类属性**声明
`lookback_days` / `frequency`。无注解 → 不是 dataclass 字段 → 继承来的
`__init__` 在实例化时用基类默认值把它们覆盖掉。

后果：所有 python 日频因子的预热窗口恒为 20 个交易日，声明更大窗口的因子
在请求区间起点处拿不到足够历史；`frequency` 同样被打回 "daily"。
消费方读的都是实例属性（daily_single.py / python_factor.py），因此静默生效。
"""

from __future__ import annotations

import polars as pl

from factorzen.daily.factors.base import DailyFactor


class _PlainDeclared(DailyFactor):
    """按 workspace/factors/*/TEMPLATE.md 教的写法声明——无注解的类属性。"""

    name = "plain_declared_probe"
    category = "weekly"
    frequency = "weekly"
    lookback_days = 30
    description = "探针因子"

    def compute(self, ctx: object) -> pl.DataFrame:  # pragma: no cover - 不求值
        return pl.DataFrame()


def test_plain_class_attrs_survive_instantiation():
    probe = _PlainDeclared()
    assert probe.lookback_days == 30, "子类声明的 lookback_days 被基类默认值覆盖"
    assert probe.frequency == "weekly", "子类声明的 frequency 被基类默认值覆盖"
    assert probe.category == "weekly"
    assert probe.name == "plain_declared_probe"


def test_builtin_weekly_factor_keeps_declared_window():
    """真实内置因子的回归：momentum_weekly 声明 30 日窗 + weekly 频率。"""
    from factorzen.builtin_factors.weekly.momentum import MomentumWeekly

    factor = MomentumWeekly()
    assert factor.lookback_days == 30
    assert factor.frequency == "weekly"


def test_no_daily_factor_loses_its_declaration():
    """全量守卫：任何内置日频因子的类声明都不得在实例化时丢失。"""
    from factorzen.daily.factors.registry import get_factor, list_factors

    drifted: list[str] = []
    for name in list_factors():
        cls = get_factor(name)
        if not (isinstance(cls, type) and issubclass(cls, DailyFactor)):
            continue
        try:
            inst = cls()
        except TypeError:  # 需要构造参数的因子不在本守卫范围
            continue
        for attr in ("lookback_days", "frequency", "category"):
            declared = getattr(cls, attr, None)
            actual = getattr(inst, attr, None)
            if declared != actual:
                drifted.append(f"{cls.__name__}.{attr}: 声明 {declared!r} → 实例 {actual!r}")

    assert not drifted, "以下因子的类属性声明在实例化时丢失:\n" + "\n".join(drifted)
