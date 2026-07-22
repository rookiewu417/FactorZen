"""CLI schema 导出测试。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from factorzen.server.api import create_app
from factorzen.server.cli_schema import get_cli_schema


def _has_child(node: dict, name: str) -> bool:
    return any(c.get("name") == name for c in node.get("children") or [])


def _walk(node: dict):
    yield node
    for c in node.get("children") or []:
        yield from _walk(c)


def test_schema_nonempty_has_factor_strategies():
    # 清缓存，确保新鲜
    get_cli_schema.cache_clear()
    schema = get_cli_schema()
    assert schema["name"] == "fz"
    assert schema.get("children"), "顶层子命令不应为空"
    assert _has_child(schema, "factor")
    assert _has_child(schema, "strategies")


def test_schema_json_serializable_no_func_leak():
    get_cli_schema.cache_clear()
    schema = get_cli_schema()
    # 全树可 dumps
    raw = json.dumps(schema)
    assert isinstance(raw, str)
    assert len(raw) > 100

    # 无 func 泄漏
    assert '"func"' not in raw or not any(
        opt.get("dest") == "func"
        for node in _walk(schema)
        for opt in node.get("options") or []
    )
    for node in _walk(schema):
        for opt in node.get("options") or []:
            assert opt.get("dest") != "func"
            # default 必须 JSON 友好（已 dumps 过，这里再断言无 callable 字面量）
            assert not callable(opt.get("default"))


def test_api_cli_schema_200(tmp_path):
    get_cli_schema.cache_clear()
    client = TestClient(create_app(tmp_path))
    r = client.get("/api/cli/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "fz"
    names = [c["name"] for c in body["children"]]
    assert "factor" in names
    assert "strategies" in names
