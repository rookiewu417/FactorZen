"""Intraday 因子注册中心（代理到 common.registry.FactorRegistry）。"""

from common.registry import FactorRegistry
from intraday.factors.base import MFTFactor

_registry = FactorRegistry(
    base_cls=MFTFactor,
    scan_packages=["intraday.factors.demo"],
)
_registry.discover()


def get_factor(name: str):
    return _registry.get(name)


def list_factors(category: str | None = None) -> list[str]:
    return _registry.list(category)
