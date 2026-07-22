"""文件管理器 FileManager 与 /api/files* 端点测试。"""

from __future__ import annotations

import json
import math
from datetime import datetime

import polars as pl
import pytest
from fastapi.testclient import TestClient

from factorzen.server import files as files_mod
from factorzen.server.api import create_app
from factorzen.server.files import FILE_MAX_BYTES, FileManager


def _client(tmp_path):
    return TestClient(create_app(tmp_path))


def _tree(tmp_path):
    """造标准文件树。"""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "note.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / "a" / "cfg.json").write_text(
        json.dumps({"x": 1}), encoding="utf-8"
    )
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02\xff")
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    nested = tmp_path / "nested" / "deep"
    nested.mkdir(parents=True)
    (nested / "leaf.md").write_text("# leaf\n", encoding="utf-8")
    return tmp_path


# ---- 列目录 ----


def test_list_root(tmp_path):
    _tree(tmp_path)
    r = _client(tmp_path).get("/api/files", params={"path": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == ""
    dir_names = [d["name"] for d in body["dirs"]]
    file_names = [f["name"] for f in body["files"]]
    assert "a" in dir_names
    assert "empty_dir" in dir_names
    assert "nested" in dir_names
    assert "bin.dat" in file_names
    # 按 name 排序
    assert dir_names == sorted(dir_names)
    assert file_names == sorted(file_names)
    # mtime 是 iso 字符串
    assert body["dirs"][0]["mtime"]
    assert "T" in body["dirs"][0]["mtime"]


def test_list_subdir(tmp_path):
    _tree(tmp_path)
    r = _client(tmp_path).get("/api/files", params={"path": "a"})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "a"
    names = {f["name"] for f in body["files"]}
    assert names == {"note.txt", "cfg.json"}
    assert body["dirs"] == []


def test_list_missing_404(tmp_path):
    r = _client(tmp_path).get("/api/files", params={"path": "nope"})
    assert r.status_code == 404


# ---- 文本读写 ----


def test_text_read_write_roundtrip(tmp_path):
    _tree(tmp_path)
    client = _client(tmp_path)

    r = client.get("/api/files/content", params={"path": "a/note.txt"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "text"
    assert body["content"] == "hello\n"
    assert body["size"] == len(b"hello\n")

    r2 = client.put(
        "/api/files/content",
        json={"path": "a/note.txt", "content": "world\n"},
    )
    assert r2.status_code == 200
    assert r2.json()["path"] == "a/note.txt"
    assert r2.json()["size"] == len(b"world\n")

    r3 = client.get("/api/files/content", params={"path": "a/note.txt"})
    assert r3.json()["content"] == "world\n"


def test_write_no_ext_as_text(tmp_path):
    (tmp_path / "LICENSE").write_text("MIT\n", encoding="utf-8")
    r = _client(tmp_path).get("/api/files/content", params={"path": "LICENSE"})
    assert r.status_code == 200
    assert r.json()["kind"] == "text"
    assert r.json()["content"] == "MIT\n"


def test_write_illegal_ext_400(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"abc")
    r = _client(tmp_path).put(
        "/api/files/content",
        json={"path": "x.bin", "content": "nope"},
    )
    assert r.status_code == 400


def test_write_missing_parent_404(tmp_path):
    r = _client(tmp_path).put(
        "/api/files/content",
        json={"path": "ghost/dir/a.txt", "content": "x"},
    )
    assert r.status_code == 404


# ---- parquet 预览 ----


def test_parquet_preview(tmp_path):
    df = pl.DataFrame(
        {
            "ts": [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 2, 12, 0, 0),
            ],
            "val": [1.5, float("nan")],
            "flag": [True, False],
            "n": [1, None],
        }
    )
    pq = tmp_path / "sample.parquet"
    df.write_parquet(pq)

    r = _client(tmp_path).get(
        "/api/files/content", params={"path": "sample.parquet"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "parquet"
    assert body["n_rows"] == 2
    schema_names = [c["name"] for c in body["schema"]]
    assert schema_names == ["ts", "val", "flag", "n"]
    assert len(body["head"]) == 2

    # 响应必须可 JSON 序列化（无 NaN 字面量）
    raw = json.dumps(body)
    assert "NaN" not in raw
    assert "Infinity" not in raw

    # head 行内 NaN → null
    row1 = body["head"][1]
    assert row1["val"] is None
    assert row1["n"] is None
    # datetime → iso 字符串
    assert isinstance(body["head"][0]["ts"], str)
    assert "2024" in body["head"][0]["ts"]
    # bool / int 保留
    assert body["head"][0]["flag"] is True
    assert body["head"][0]["n"] == 1


def test_parquet_read_fail_422(tmp_path):
    (tmp_path / "bad.parquet").write_bytes(b"not a parquet")
    r = _client(tmp_path).get(
        "/api/files/content", params={"path": "bad.parquet"}
    )
    assert r.status_code == 422


# ---- 二进制 ----


def test_binary_kind(tmp_path):
    _tree(tmp_path)
    r = _client(tmp_path).get("/api/files/content", params={"path": "bin.dat"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "binary"
    assert body["size"] == 4
    assert "content" not in body


# ---- 路径遍历 ----


@pytest.mark.parametrize(
    "evil",
    ["../x", "/etc/passwd", "a\\b", "a/../../etc/passwd", ".."],
)
def test_path_traversal_404(tmp_path, evil):
    _tree(tmp_path)
    client = _client(tmp_path)
    assert client.get("/api/files", params={"path": evil}).status_code == 404
    assert (
        client.get("/api/files/content", params={"path": evil}).status_code
        == 404
    )
    assert (
        client.put(
            "/api/files/content", json={"path": evil, "content": "x"}
        ).status_code
        == 404
    )
    assert (
        client.delete("/api/files", params={"path": evil}).status_code == 404
    )


# ---- 删除 ----


def test_delete_file(tmp_path):
    _tree(tmp_path)
    client = _client(tmp_path)
    r = client.delete("/api/files", params={"path": "a/note.txt"})
    assert r.status_code == 200
    assert r.json()["deleted"] == "a/note.txt"
    assert not (tmp_path / "a" / "note.txt").exists()


def test_delete_empty_dir(tmp_path):
    _tree(tmp_path)
    r = _client(tmp_path).delete(
        "/api/files", params={"path": "empty_dir", "recursive": False}
    )
    assert r.status_code == 200
    assert not (tmp_path / "empty_dir").exists()


def test_delete_nonempty_without_recursive_409(tmp_path):
    _tree(tmp_path)
    r = _client(tmp_path).delete(
        "/api/files", params={"path": "a", "recursive": False}
    )
    assert r.status_code == 409
    assert (tmp_path / "a").is_dir()


def test_delete_nonempty_with_recursive(tmp_path):
    _tree(tmp_path)
    r = _client(tmp_path).delete(
        "/api/files", params={"path": "a", "recursive": True}
    )
    assert r.status_code == 200
    assert r.json()["deleted"] == "a"
    assert not (tmp_path / "a").exists()


def test_delete_root_400(tmp_path):
    client = _client(tmp_path)
    # 空 path
    r = client.delete("/api/files", params={"path": ""})
    # FastAPI 可能因 Query(...) 必填而 422；我们实现上 path="" 应 400
    # 若 Query 允许空串，则进 FileManager → 400
    assert r.status_code in (400, 422)

    r2 = client.delete("/api/files", params={"path": "."})
    assert r2.status_code == 400


# ---- 大文件 413 ----


def test_text_too_large_413(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "FILE_MAX_BYTES", 16)
    (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")
    r = _client(tmp_path).get("/api/files/content", params={"path": "big.txt"})
    assert r.status_code == 413


def test_file_max_bytes_is_module_constant():
    assert FILE_MAX_BYTES == 1_000_000


# ---- FileManager 单元：jsonable 边界 ----


def test_jsonable_nan_inf():
    from factorzen.server.files import _jsonable_cell

    assert _jsonable_cell(float("nan")) is None
    assert _jsonable_cell(float("inf")) is None
    assert _jsonable_cell(float("-inf")) is None
    assert _jsonable_cell(1.5) == 1.5
    assert _jsonable_cell(None) is None
    assert not math.isnan(_jsonable_cell(0.0))  # type: ignore[arg-type]


def test_manager_list_and_delete_direct(tmp_path):
    _tree(tmp_path)
    fm = FileManager(tmp_path)
    listing = fm.list_dir("")
    assert any(d["name"] == "a" for d in listing["dirs"])
    fm.delete("bin.dat")
    assert not (tmp_path / "bin.dat").exists()
