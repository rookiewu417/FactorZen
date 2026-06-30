"""把挖出的表达式渲染成独立 .py，落入 workspace/factors/daily/ 供 registry 发现。"""
from __future__ import annotations

from pathlib import Path


def _class_name(name: str) -> str:
    return "".join(p.capitalize() for p in name.replace("-", "_").split("_"))


def render_factor_file(expression: str, name: str) -> str:
    cls = _class_name(name)
    return f'''"""Mined factor: {name}. 由 fz mine 自动生成。表达式: {expression}"""

from factorzen.discovery.factor import ExpressionFactor


class {cls}(ExpressionFactor):
    name = "{name}"
    frequency = "daily"
    expression = "{expression}"
    mined_name = "{name}"
    lookback_days = 60


{cls}()  # 模块级实例化供 registry 自动发现
'''


def export_candidate(expression: str, name: str, dest_dir: str) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.py"
    path.write_text(render_factor_file(expression, name), encoding="utf-8")
    return path
