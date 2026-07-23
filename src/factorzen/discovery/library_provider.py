"""library provider：factor_library + factor_store → daily registry 动态注入。

放在 discovery 侧的原因（架构约束）：daily 是底层能力层，不许反向依赖 discovery
（daily→discovery→daily 成环）；本模块自身依赖 discovery.factor / factor_library，
而 discovery→daily 边已存在（ExpressionFactor 继承 DailyFactor）。

- expression 型：从 jsonl 动态生成 ExpressionFactor 子类
- python 型：从 ``workspace/factors/<market>/<name>/factor.py`` 加载（唯一用户路径）
"""

from __future__ import annotations

from factorzen.core.logger import get_logger

logger = get_logger(__name__)


def load_library_factors(
    market: str = "ashare",
    root: str | None = None,
    *,
    store_root: str | None = None,
) -> int:
    """从 factor_library 注入 expression 型 + 从 factor_store 注入 python 型。

    - expression：kind=expression，status 不过滤（correlated/probation 也可 run）
    - python：扫描 factor_store 下 kind=python 的 factor.py
    - builtin 同名优先：register(override=False) 让位并 warning
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
    from factorzen.discovery.factor_store import (
        register_python_factors_from_store,
        store_root_for_library,
    )

    lib_root = root if root is not None else DEFAULT_ROOT
    s_root = store_root if store_root is not None else store_root_for_library(lib_root)

    records = load_library(market, root=lib_root)
    n_ok = 0
    for r in records:
        # python 型走 factor_store 扫描；expression 哨兵旧行也跳过
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

    # python 型：factor_store 扫描（唯一用户 python 因子路径）
    try:
        n_ok += register_python_factors_from_store(s_root, market=market, override=False)
    except Exception as e:
        logger.warning(f"register_python_factors_from_store 跳过: {e}")

    return n_ok
