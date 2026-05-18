"""Intraday 因子注册中心（代理到 common.registry.FactorRegistry）。"""

from common.registry import FactorRegistry
from intraday.factors.base import IntradayFactor

_registry = FactorRegistry(
    base_cls=IntradayFactor,
    scan_packages=["intraday.factors.demo", "intraday.factors.technical"],
)
_registry.discover()


def get_factor(name: str):
    return _registry.get(name)


def list_factors(category: str | None = None) -> list[str]:
    return _registry.list(category)
