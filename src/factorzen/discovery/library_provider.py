"""library provider：factor_library expression 型记录 → daily registry 动态注入。

放在 discovery 侧的原因（架构约束）：daily 是底层能力层，不许反向依赖 discovery
（daily→discovery→daily 成环）；本模块自身依赖 discovery.factor / factor_library，
而 discovery→daily 边已存在（ExpressionFactor 继承 DailyFactor）。
"""
from __future__ import annotations

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


def load_library_factors(market: str = "ashare", root: str | None = None) -> int:
    """从 factor_library 注入 expression 型记录到 daily registry（显式调用，无 import 副作用）。

    - 只注册 kind=expression（python 型本就在 registry，不重复）
    - status 不过滤（correlated/probation 也可 run，复现用途）
    - builtin/workspace 同名优先：register(override=False) 让位并 warning
    - 返回成功注册数

    约束：不在模块 import 时自动执行，避免测试污染。
    """
    from factorzen.daily.factors.registry import _registry
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
