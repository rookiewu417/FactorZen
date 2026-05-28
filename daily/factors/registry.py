"""Daily 因子注册中心（代理到 common.registry.FactorRegistry）。"""

from common.registry import FactorRegistry
from daily.factors.base import DailyFactor

_registry = FactorRegistry(
    base_cls=DailyFactor,
    scan_packages=[
        "daily.factors.daily",
        "daily.factors.weekly",
        "daily.factors.monthly",
        "daily.factors.qlib",
    ],
)
# 模块加载时自动扫描（与之前行为保持一致）
_registry.discover()


def get_factor(name: str):
    return _registry.get(name)


def list_factors(category: str | None = None) -> list[str]:
    return _registry.list(category)
