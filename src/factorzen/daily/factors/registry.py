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
        # 用户 python 因子由 library_provider 从 factor_store 动态注入，
        # 不再扫描 workspace.factors.*（该树已退役）
    ],
)
# 模块加载时自动扫描（与之前行为保持一致）
_registry.discover()


def get_factor(name: str):
    return _registry.get(name)


def list_factors(category: str | None = None) -> list[str]:
    return _registry.list(category)
