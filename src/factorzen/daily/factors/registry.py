"""Daily 因子注册中心（代理到 common.registry.FactorRegistry）。"""

from __future__ import annotations

from factorzen.core.logger import get_logger
from factorzen.core.registry import FactorRegistry
from factorzen.daily.factors.base import DailyFactor

logger = get_logger(__name__)

_registry = FactorRegistry(
    base_cls=DailyFactor,
    scan_packages=[
        # 框架自带因子（随包分发）
        "factorzen.builtin_factors.daily",
        "factorzen.builtin_factors.weekly",
        "factorzen.builtin_factors.monthly",
        "factorzen.builtin_factors.qlib",
        # 用户自定义因子（workspace 在后，同名时覆盖内置）
        # qlib 因子由框架经 builtin_factors.qlib 生成，用户不在 workspace 手写
        "workspace.factors.daily",
        "workspace.factors.weekly",
        "workspace.factors.monthly",
    ],
)
# 模块加载时自动扫描（与之前行为保持一致）
_registry.discover()


def get_factor(name: str):
    return _registry.get(name)


def list_factors(category: str | None = None) -> list[str]:
    return _registry.list(category)


def load_library_factors(market: str = "ashare", root: str | None = None) -> int:
    """从 factor_library 注入 expression 型记录到 daily registry（显式调用，无 import 副作用）。

    - 只注册 kind=expression（python 型本就在 registry，不重复）
    - status 不过滤（correlated/probation 也可 run，复现用途）
    - builtin/workspace 同名优先：register(override=False) 让位并 warning
    - 返回成功注册数

    约束：不在模块 import 时自动执行，避免测试污染。
    """
    from factorzen.discovery.factor import ExpressionFactor, lookback_for_expression
    from factorzen.discovery.factor_library import (
        DEFAULT_ROOT,
        _is_python_record,
        _normalize,
        default_name_for_expression,
        load_library,
    )

    lib_root = root if root is not None else DEFAULT_ROOT
    records = load_library(market, root=lib_root)
    n_ok = 0
    for r in records:
        # python 型已由包扫描进 registry；expression 哨兵旧行也跳过
        if _is_python_record(r) or r.kind != "expression":
            continue
        expr = (r.expression or "").strip()
        if not expr:
            continue
        name = (r.name or "").strip() or default_name_for_expression(_normalize(expr))
        if not name:
            continue
        # 幂等：本 provider 已注入过的 LibFactor_* 静默跳过，避免二次 load 刷 warning
        try:
            existing = _registry.get(name)
        except KeyError:
            existing = None
        if existing is not None and getattr(existing, "__name__", "").startswith("LibFactor_"):
            continue
        lookback = lookback_for_expression(expr)
        status = r.status or "active"
        # 类名仅作调试标识；注册键用 instance.name
        cls_name = f"LibFactor_{name}"
        cls = type(
            cls_name,
            (ExpressionFactor,),
            {
                "name": name,
                "expression": expr,
                "mined_name": name,
                "lookback_days": lookback,
                "frequency": "daily",
                "description": f"[{status}] mined: {expr}",
            },
        )
        if _registry.register(cls, override=False):
            n_ok += 1
    return n_ok
