"""因子库 / 因子资产 API 与 FactorLibraryIndex 测试。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from factorzen.server.api import create_app
from factorzen.server.library import FactorLibraryIndex


def _client(tmp_path):
    return TestClient(create_app(tmp_path))


def _write_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_library_list_count_by_status_skips_bad_lines(tmp_path):
    """jsonl 含坏行时跳过；count/by_status 正确。"""
    lib = tmp_path / "factor_library"
    _write_jsonl(
        lib / "ashare.jsonl",
        [
            json.dumps(
                {
                    "expression": "ts_mean(close, 5)",
                    "status": "active",
                    "ic_train": 0.02,
                }
            ),
            "{ not valid json",
            json.dumps(
                {
                    "expression": "ts_std(volume, 10)",
                    "status": "correlated",
                    "ic_train": -0.01,
                }
            ),
            json.dumps(
                {
                    "expression": "div(high, low)",
                    "status": "active",
                    "ic_train": 0.03,
                }
            ),
        ],
    )

    r = _client(tmp_path).get("/api/library/ashare")
    assert r.status_code == 200
    body = r.json()
    assert body["market"] == "ashare"
    assert body["count"] == 3
    assert body["by_status"] == {"active": 2, "correlated": 1}
    assert len(body["factors"]) == 3
    assert body["factors"][0]["expression"] == "ts_mean(close, 5)"


def test_library_missing_file_returns_empty(tmp_path):
    r = _client(tmp_path).get("/api/library/crypto")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["factors"] == []
    assert body["by_status"] == {}


def test_library_invalid_market_404(tmp_path):
    assert _client(tmp_path).get("/api/library/bitcoin").status_code == 404


def test_library_track_filter_and_sort(tmp_path):
    """track 按 expression 过滤，date 升序。"""
    track = tmp_path / "factor_library" / "forward_track"
    expr = "div(high, low)"
    other = "ts_mean(close, 5)"
    _write_jsonl(
        track / "ashare.jsonl",
        [
            json.dumps(
                {
                    "date": "20260610",
                    "expression": expr,
                    "ic": 0.02,
                    "n_stocks": 100,
                }
            ),
            json.dumps(
                {
                    "date": "20260605",
                    "expression": expr,
                    "ic": -0.01,
                    "n_stocks": 100,
                }
            ),
            json.dumps(
                {
                    "date": "20260605",
                    "expression": other,
                    "ic": 0.5,
                    "n_stocks": 50,
                }
            ),
            "bad line",
        ],
    )

    r = _client(tmp_path).get(
        "/api/library/ashare/track",
        params={"expression": expr},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["expression"] == expr
    assert len(body["points"]) == 2
    assert body["points"][0]["date"] == "20260605"
    assert body["points"][0]["ic"] == -0.01
    assert body["points"][1]["date"] == "20260610"
    assert body["points"][1]["n_stocks"] == 100


def test_library_track_empty(tmp_path):
    r = _client(tmp_path).get(
        "/api/library/us/track",
        params={"expression": "x"},
    )
    assert r.status_code == 200
    assert r.json()["points"] == []


def test_store_list_and_detail(tmp_path):
    """meta.json + factor.py 读取。"""
    d = tmp_path / "factor_store" / "ashare" / "alpha_1"
    d.mkdir(parents=True)
    meta = {
        "name": "alpha_1",
        "kind": "expression",
        "expression": "ts_mean(close, 20)",
        "created_at": "2026-07-19",
        "ledger_snapshot": {"status": "active", "ic_train": 0.01},
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "factor.py").write_text("# source\nprint(1)\n", encoding="utf-8")

    # 无 meta 的目录应跳过
    (tmp_path / "factor_store" / "ashare" / "orphan").mkdir(parents=True)

    r = _client(tmp_path).get("/api/store/ashare")
    assert r.status_code == 200
    body = r.json()
    assert body["market"] == "ashare"
    assert len(body["entries"]) == 1
    assert body["entries"][0]["name"] == "alpha_1"
    assert body["entries"][0]["expression"] == "ts_mean(close, 20)"

    r2 = _client(tmp_path).get("/api/store/ashare/alpha_1")
    assert r2.status_code == 200
    detail = r2.json()
    assert detail["name"] == "alpha_1"
    assert detail["meta"]["kind"] == "expression"
    assert detail["source"] == "# source\nprint(1)\n"


def test_store_detail_missing_source_null(tmp_path):
    d = tmp_path / "factor_store" / "crypto" / "f1"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps({"name": "f1", "expression": "x"}), encoding="utf-8"
    )
    r = _client(tmp_path).get("/api/store/crypto/f1")
    assert r.status_code == 200
    assert r.json()["source"] is None


def test_store_path_traversal_404(tmp_path):
    """name 含 ../ 路径遍历期望 404。"""
    # 在 store 外放一个 secret
    evil = tmp_path / "secret"
    evil.mkdir()
    (evil / "meta.json").write_text('{"secret": 1}', encoding="utf-8")
    (tmp_path / "factor_store" / "ashare").mkdir(parents=True)

    client = _client(tmp_path)
    for name in ("../secret", "../../secret", "..%2Fsecret"):
        # FastAPI 会解码 path param；直接传 ../secret
        r = client.get(f"/api/store/ashare/{name}")
        # 路径遍历或找不到都应 404
        assert r.status_code == 404, f"name={name!r} got {r.status_code}"

    # 直接测 Index
    import pytest

    with pytest.raises(FileNotFoundError):
        FactorLibraryIndex(tmp_path).store_detail("ashare", "../secret")


def test_store_invalid_market_404(tmp_path):
    assert _client(tmp_path).get("/api/store/xyz").status_code == 404


def test_store_missing_404(tmp_path):
    (tmp_path / "factor_store" / "ashare").mkdir(parents=True)
    assert _client(tmp_path).get("/api/store/ashare/nope").status_code == 404


# ---- 手写因子合并 + 改状态 ----


def _write_store_python(tmp_path, market, name, *, status=None, extra_snap=None):
    """写入 kind=python 的 factor_store 资产。"""
    d = tmp_path / "factor_store" / market / name
    d.mkdir(parents=True)
    snap = {"status": status, "ic_train": 0.01, "holdout_ic": None}
    if extra_snap:
        snap.update(extra_snap)
    meta = {
        "name": name,
        "kind": "python",
        "expression": f"py::{name}",
        "created_at": "2026-07-20",
        "ledger_snapshot": snap,
    }
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def test_list_factors_merges_handwritten_store(tmp_path):
    """library 2 条 + store 两个 python：一个已在 lib 不重复、一个以 manual+store 出现。"""
    lib = tmp_path / "factor_library"
    _write_jsonl(
        lib / "ashare.jsonl",
        [
            json.dumps(
                {
                    "expression": "ts_mean(close, 5)",
                    "status": "active",
                    "ic_train": 0.02,
                    "extra_unknown": "keep-me",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "expression": "py::already_in_lib",
                    "status": "correlated",
                    "kind": "python",
                    "name": "already_in_lib",
                },
                ensure_ascii=False,
            ),
        ],
    )
    # 已在 library：不应重复出现
    _write_store_python(tmp_path, "ashare", "already_in_lib", status="active")
    # 不在 library：应以 manual + source=store 出现
    _write_store_python(tmp_path, "ashare", "hand_alpha", status=None)
    # 非 python 资产不并入
    expr_dir = tmp_path / "factor_store" / "ashare" / "expr_only"
    expr_dir.mkdir(parents=True)
    (expr_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "expr_only",
                "kind": "expression",
                "expression": "ts_std(volume, 10)",
            }
        ),
        encoding="utf-8",
    )

    r = _client(tmp_path).get("/api/library/ashare")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert body["by_status"]["active"] == 1
    assert body["by_status"]["correlated"] == 1
    assert body["by_status"]["manual"] == 1

    by_expr = {f["expression"]: f for f in body["factors"]}
    assert set(by_expr) == {
        "ts_mean(close, 5)",
        "py::already_in_lib",
        "py::hand_alpha",
    }
    assert by_expr["ts_mean(close, 5)"]["source"] == "library"
    assert by_expr["py::already_in_lib"]["source"] == "library"
    assert by_expr["py::already_in_lib"]["status"] == "correlated"  # 用 library 那条
    hand = by_expr["py::hand_alpha"]
    assert hand["source"] == "store"
    assert hand["status"] == "manual"
    assert hand["admission_track"] == "manual"
    assert hand["kind"] == "python"
    assert hand["name"] == "hand_alpha"


def test_update_status_library_preserves_unknown_fields(tmp_path):
    """按行改写：目标行 status 变；其它行未知字段原样；行数不变。"""
    path = tmp_path / "factor_library" / "ashare.jsonl"
    line_a = json.dumps(
        {
            "expression": "factor_a",
            "status": "active",
            "weird_field": {"nested": 1},
            "keep_me": "yes",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    line_b = json.dumps(
        {
            "expression": "factor_b",
            "status": "correlated",
            "another_unknown": [1, 2, 3],
            "z_extra": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    bad = "{ not json"
    _write_jsonl(path, [line_a, bad, line_b])

    idx = FactorLibraryIndex(tmp_path)
    out = idx.update_status("ashare", "factor_a", "probation", "library")
    assert out["status"] == "probation"
    assert out["source"] == "library"

    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 3
    assert lines[1] == bad  # 坏行原样

    parsed = []
    for ln in lines:
        try:
            parsed.append(json.loads(ln))
        except json.JSONDecodeError:
            parsed.append(None)

    assert parsed[0]["status"] == "probation"
    assert parsed[0]["weird_field"] == {"nested": 1}
    assert parsed[0]["keep_me"] == "yes"
    assert parsed[0]["expression"] == "factor_a"

    # 未改行：原文完全一致（未知字段 + 格式）
    assert lines[2] == line_b
    assert parsed[2]["another_unknown"] == [1, 2, 3]
    assert parsed[2]["z_extra"] is True
    assert parsed[2]["status"] == "correlated"

    # 非法 status
    import pytest

    with pytest.raises(ValueError, match="非法 status"):
        idx.update_status("ashare", "factor_a", "rejected", "library")


def test_update_status_store_and_path_traversal(tmp_path):
    """store：改 ledger_snapshot.status；路径遍历 → FileNotFoundError。"""
    _write_store_python(tmp_path, "ashare", "hand_beta", status=None)
    idx = FactorLibraryIndex(tmp_path)

    out = idx.update_status("ashare", "py::hand_beta", "active", "store")
    assert out == {
        "market": "ashare",
        "expression": "py::hand_beta",
        "status": "active",
        "source": "store",
    }
    meta = json.loads(
        (tmp_path / "factor_store" / "ashare" / "hand_beta" / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["ledger_snapshot"]["status"] == "active"
    # 其它字段保留
    assert meta["name"] == "hand_beta"
    assert meta["kind"] == "python"

    import pytest

    with pytest.raises(FileNotFoundError):
        idx.update_status("ashare", "py::../x", "manual", "store")


def test_api_update_status_endpoints(tmp_path):
    """POST /status：200 / 非法 market 404 / 非法 status 400 / store 路径遍历 404。"""
    lib = tmp_path / "factor_library"
    _write_jsonl(
        lib / "ashare.jsonl",
        [
            json.dumps(
                {"expression": "ts_mean(close, 5)", "status": "active", "x": 1}
            ),
        ],
    )
    _write_store_python(tmp_path, "ashare", "hand_g", status="manual")
    client = _client(tmp_path)

    # 正常 library
    r = client.post(
        "/api/library/ashare/status",
        json={
            "expression": "ts_mean(close, 5)",
            "status": "no_lift",
            "source": "library",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {
        "market": "ashare",
        "expression": "ts_mean(close, 5)",
        "status": "no_lift",
        "source": "library",
    }

    # 正常 store
    r2 = client.post(
        "/api/library/ashare/status",
        json={
            "expression": "py::hand_g",
            "status": "probation",
            "source": "store",
        },
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["status"] == "probation"

    # 非法 market
    r3 = client.post(
        "/api/library/bitcoin/status",
        json={"expression": "x", "status": "manual", "source": "library"},
    )
    assert r3.status_code == 404

    # 非法 status
    r4 = client.post(
        "/api/library/ashare/status",
        json={
            "expression": "ts_mean(close, 5)",
            "status": "rejected",
            "source": "library",
        },
    )
    assert r4.status_code == 400

    # store 路径遍历
    r5 = client.post(
        "/api/library/ashare/status",
        json={
            "expression": "py::../secret",
            "status": "manual",
            "source": "store",
        },
    )
    assert r5.status_code == 404
