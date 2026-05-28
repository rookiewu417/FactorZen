"""通用因子注册中心。支持 daily/intraday 因子的自动发现与注册。

用法:
    # daily 频率
    from common.registry import FactorRegistry
    daily_registry = FactorRegistry(
        base_cls=DailyFactor,
        scan_packages=[
            "daily.factors.personal.daily",
            "daily.factors.personal.weekly",
            "daily.factors.personal.monthly",
            "daily.factors.qlib",
        ],
    )
    factor_cls = daily_registry.get("momentum_20d")

    # intraday 频率
    intraday_registry = FactorRegistry(
        base_cls=IntradayFactor,
        scan_packages=["intraday.factors.demo"],
    )
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, cast

from common.factor import BaseFactor
from common.logger import get_logger

logger = get_logger(__name__)


class FactorRegistry:
    """线程不安全（单进程使用），同一进程中可多次实例化不同频率的注册表。"""

    def __init__(self, base_cls: type[Any], scan_packages: list[str]) -> None:
        self._base_cls = base_cls
        self._scan_packages = scan_packages
        self._registry: dict[str, type[BaseFactor]] = {}
        self._discovered = False

    def discover(self) -> dict[str, type[BaseFactor]]:
        """扫描所有配置包，注册 base_cls 的子类。结果缓存，只扫描一次。"""
        if self._discovered:
            return self._registry

        for package in self._scan_packages:
            try:
                pkg_mod = importlib.import_module(package)
                pkg_path = pkg_mod.__path__
            except ModuleNotFoundError:
                logger.debug(f"包 {package} 不存在，跳过")
                continue

            for _, mod_name, _ in pkgutil.iter_modules(pkg_path, prefix=package + "."):
                try:
                    mod = importlib.import_module(mod_name)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, self._base_cls)
                            and attr is not self._base_cls
                        ):
                            factor_cls = cast(type[BaseFactor], attr)
                            instance = factor_cls()
                            if instance.name:
                                self._registry[instance.name] = factor_cls
                except Exception as e:
                    # 记录 import 失败，不静默吞噬（方便调试缺失依赖）
                    logger.warning(f"导入 {mod_name} 时失败: {e}", exc_info=True)

        self._discovered = True
        return self._registry

    def get(self, name: str) -> type[BaseFactor]:
        """按名称获取因子类。若未找到抛出 KeyError。"""
        self.discover()
        if name not in self._registry:
            raise KeyError(f"因子 '{name}' 未注册。可用: {sorted(self._registry)}")
        return self._registry[name]

    def list(self, category: str | None = None) -> list[str]:
        """列出所有已注册因子名称，可按 category 过滤。"""
        self.discover()
        names = [
            n
            for n, cls in self._registry.items()
            if category is None or getattr(cls, "category", None) == category
        ]
        return sorted(names)

    def reset(self) -> None:
        """清空注册表（用于测试时重置状态）。"""
        self._registry.clear()
        self._discovered = False
