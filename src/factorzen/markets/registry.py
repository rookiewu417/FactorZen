"""市场注册表：register / get / list。

用 builder（工厂函数）注册而非直接注册 profile，使 profile 惰性构造
（避免 import 时就创建 ccxt client 等重资源）。首次 ``get`` 构造并缓存。
"""
from __future__ import annotations

from collections.abc import Callable

from factorzen.markets.base import MarketProfile

_BUILDERS: dict[str, Callable[[], MarketProfile]] = {}
_CACHE: dict[str, MarketProfile] = {}


def register(name: str, builder: Callable[[], MarketProfile]) -> None:
    """注册一个市场的 profile builder（覆盖同名）。"""
    _BUILDERS[name] = builder
    _CACHE.pop(name, None)


def get(name: str) -> MarketProfile:
    """取市场 profile（惰性构造 + 缓存）。未注册抛 KeyError。"""
    if name not in _BUILDERS:
        raise KeyError(f"未注册的市场: {name!r}。已注册: {sorted(_BUILDERS)}")
    if name not in _CACHE:
        _CACHE[name] = _BUILDERS[name]()
    return _CACHE[name]


def list_markets() -> list[str]:
    """已注册市场名列表（排序）。"""
    return sorted(_BUILDERS)
