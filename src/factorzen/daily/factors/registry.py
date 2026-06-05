"""Daily 因子注册中心（代理到 common.registry.FactorRegistry）。"""

from factorzen.core.registry import FactorRegistry
from factorzen.daily.factors.base import DailyFactor

_registry = FactorRegistry(
    base_cls=DailyFactor,
    scan_packages=[
        # 框架自带因子（随包分发）
        "factorzen.builtin_factors.daily",
        "factorzen.builtin_factors.weekly",
        "factorzen.builtin_factors.monthly",
        "factorzen.builtin_factors.qlib",
        # 用户自定义因子（workspace 在后，同名时覆盖内置）
        "workspace.factors.daily",
        "workspace.factors.weekly",
        "workspace.factors.monthly",
        "workspace.factors.qlib",
    ],
)
# 模块加载时自动扫描（与之前行为保持一致）
_registry.discover()


def get_factor(name: str):
    return _registry.get(name)


def list_factors(category: str | None = None) -> list[str]:
    return _registry.list(category)
