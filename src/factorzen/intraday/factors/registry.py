"""Intraday 因子注册中心（代理到 common.registry.FactorRegistry）。"""

from factorzen.core.registry import FactorRegistry
from factorzen.intraday.factors.base import IntradayFactor

_registry = FactorRegistry(
    base_cls=IntradayFactor,
    scan_packages=[
        "factorzen.builtin_factors.intraday",
        "workspace.factors.intraday",
    ],
)
_registry.discover()


def get_factor(name: str):
    return _registry.get(name)


def list_factors(category: str | None = None) -> list[str]:
    return _registry.list(category)
