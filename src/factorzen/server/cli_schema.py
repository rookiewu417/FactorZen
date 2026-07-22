"""从真实 argparse 树导出 CLI schema（供 Web 命令启动器自动生成表单）。

用 ``factorzen.cli.main.build_parser()`` 构建完整 parser（main 模块自身作为
commands，set_defaults 可取到真实 handler 属性）；导出时**跳过** ``func`` 与
``-h``，default 中不可 JSON 序列化对象转 str。结果模块级 lru_cache。
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Any


def _jsonable_default(value: Any) -> Any:
    """将 argparse default 转为 JSON 可序列化值。"""
    if value is None or value is argparse.SUPPRESS:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_default(v) for v in value]
    # Path、type 对象等
    return str(value)


def _export_parser(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    help_text: str | None,
) -> dict[str, Any]:
    """递归导出单个 parser 节点。"""
    options: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []

    for action in parser._actions:
        # 子命令树
        if isinstance(action, argparse._SubParsersAction):
            for choice_name, sub in action.choices.items():
                sub_help = None
                if isinstance(sub, argparse.ArgumentParser):
                    sub_help = sub.description or getattr(sub, "help", None)
                # ArgumentParser 无 help 属性时从 _name_parser_map 旁路拿不到，用 choice 即可
                children.append(
                    _export_parser(
                        sub,
                        name=choice_name,
                        help_text=sub_help or "",
                    )
                )
            continue

        # 跳过 help / func（set_defaults 的 func 不在 _actions，但 dest=help 在）
        if action.dest in ("help", "func"):
            continue
        if action.option_strings and set(action.option_strings) <= {"-h", "--help"}:
            continue

        is_flag = isinstance(
            action,
            (argparse._StoreTrueAction, argparse._StoreFalseAction),
        )
        is_positional = not bool(action.option_strings)

        nargs = action.nargs
        if nargs is not None and not isinstance(nargs, str):
            # int nargs → 字符串化，保持 JSON 友好
            nargs_out: str | int | None = nargs
        else:
            nargs_out = nargs

        choices = None
        if action.choices is not None:
            try:
                choices = [str(c) for c in action.choices]
            except TypeError:
                choices = [str(action.choices)]

        options.append(
            {
                "flags": list(action.option_strings),
                "dest": action.dest,
                "help": action.help,
                "required": bool(getattr(action, "required", False)),
                "default": _jsonable_default(action.default),
                "choices": choices,
                "nargs": nargs_out if isinstance(nargs_out, (str, int)) or nargs_out is None else str(nargs_out),
                "is_flag": is_flag,
                "is_positional": is_positional,
            }
        )

    return {
        "name": name,
        "help": help_text or "",
        "children": children,
        "options": options,
    }


@lru_cache(maxsize=1)
def get_cli_schema() -> dict[str, Any]:
    """导出整棵 fz CLI schema（缓存）。"""
    from factorzen.cli.main import build_parser

    parser = build_parser()
    return _export_parser(
        parser,
        name="fz",
        help_text=parser.description or "FactorZen research CLI",
    )
