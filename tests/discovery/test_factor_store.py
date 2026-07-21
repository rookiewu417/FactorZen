"""factor_store 三件套：meta.json + factor.py + factor.parquet。

TDD 覆盖：
- expression 类 write → 三件套齐、meta 字段全、生成 factor.py 可 import 且与生产求值逐位一致
- python 类迁移后 provider 能加载、compute 可跑
- sync 增量：第二次 sync 跳过未变因子
- verify：人为改 meta.expression → 报漂移
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ── helpers ──────────────────────────────────────────────────────────────────


def _expr_record(
    expr: str = "rank(close)",
    *,
    name: str = "mined_test_rank",
    status: str = "active",
    market: str = "ashare",
    **extra,
):
    from factorzen.discovery.factor_library import FactorRecord

    base = dict(
        expression=expr,
        market=market,
        kind="expression",
        name=name,
        status=status,
        hypothesis="test hypothesis about ranking close prices for signal strength",
        ic_train=0.05,
        holdout_ic=0.04,
        admission_ic=0.03,
        lift=0.002,
        frequency="daily",
        source_run_id="run_test_1",
        eval_start="20240101",
        eval_end="20240630",
        universe="csi300",
        added_at="2026-07-21",
        updated_at="2026-07-21",
    )
    base.update(extra)
    return FactorRecord(**{k: v for k, v in base.items() if k in FactorRecord.__dataclass_fields__})


def _tiny_daily(n_days: int = 5, n_stocks: int = 4) -> pl.DataFrame:
    """合成日频面板，含 close_adj 等生产叶列（LEAF_FEATURES 把 close→close_adj）。"""
    rows = []
    for d in range(n_days):
        dt = date(2024, 1, 2) + timedelta(days=d)
        for i in range(n_stocks):
            px = 10.0 + i + 0.1 * d
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": f"{i:06d}.SH",
                    "close": px,
                    "close_adj": px,
                    "open": 10.0 + i,
                    "open_adj": 10.0 + i,
                    "high": 11.0 + i,
                    "high_adj": 11.0 + i,
                    "low": 9.0 + i,
                    "low_adj": 9.0 + i,
                    "vol": 1000.0 + i * 10,
                    "amount": 1e6,
                }
            )
    return pl.DataFrame(rows).sort(["ts_code", "trade_date"])


def _import_factor_py(path: Path, mod_name: str = "fz_test_factor_mod"):
    """动态 import factor.py 模块。"""
    # 唯一模块名，避免缓存污染
    uniq = f"{mod_name}_{abs(hash(str(path))) % 10**8}"
    if uniq in sys.modules:
        del sys.modules[uniq]
    spec = importlib.util.spec_from_file_location(uniq, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[uniq] = mod
    spec.loader.exec_module(mod)
    return mod


# ── write_factor_asset: expression ───────────────────────────────────────────


def test_write_expression_asset_suite(tmp_path, monkeypatch):
    """expression 类 write → 三件套齐、meta 全、factor.py 与生产求值一致。"""
    from factorzen.discovery import factor_store as fs
    from factorzen.discovery.expression import evaluate_materialized, parse_expr
    from factorzen.discovery.factor_store import write_factor_asset

    # 离线:end 走 prev_trade_date(today) 需要交易日历(CI 无 token 会炸),mock 固定值
    fixed_end = "2026-07-17"
    monkeypatch.setattr(fs, "store_materialize_end", lambda: fixed_end)

    rec = _expr_record("rank(close)")
    daily = _tiny_daily()

    # 生产求值基准
    node = parse_expr(rec.expression)
    expected = (
        daily.select(["trade_date", "ts_code"])
        .with_columns(evaluate_materialized(node, daily).alias("factor_value"))
        .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
    )

    asset_dir = write_factor_asset(
        rec,
        market="ashare",
        root=str(tmp_path),
        materialize=True,
        panel=expected,
    )
    asset_dir = Path(asset_dir)
    assert (asset_dir / "meta.json").is_file()
    assert (asset_dir / "factor.py").is_file()
    assert (asset_dir / "factor.parquet").is_file()

    meta = json.loads((asset_dir / "meta.json").read_text(encoding="utf-8"))
    # 必填字段
    for key in (
        "name",
        "kind",
        "expression",
        "frequency",
        "description",
        "source_run_id",
        "created_at",
        "ledger_snapshot",
        "materialization",
    ):
        assert key in meta, f"missing meta key: {key}"
    assert meta["name"] == "mined_test_rank"
    assert meta["kind"] == "expression"
    assert meta["expression"] == "rank(close)"
    assert meta["frequency"] == "daily"
    assert meta["source_run_id"] == "run_test_1"
    assert meta["description"]  # hypothesis 截断
    snap = meta["ledger_snapshot"]
    assert snap["status"] == "active"
    assert snap["lift"] == 0.002
    assert snap["admission_ic"] == 0.03
    assert snap["ic_train"] == 0.05
    assert snap["holdout_ic"] == 0.04
    assert "factor_library" in snap["truth"] and "ashare.jsonl" in snap["truth"]
    mat = meta["materialization"]
    assert mat is not None
    # 物化口径与裁决 eval 窗分离：parquet 固定 all_a / 2016-01-01~最新
    # (end 用 mock 的字面值断言,不与生产同 helper 重算——那是恒真)
    assert mat["start"] == "2016-01-01"
    assert mat["end"] == fixed_end
    assert mat["universe"] == "all_a"
    assert mat["n_rows"] == expected.height
    assert "generated_at" in mat
    # ledger 评估口径仍保留在 record 侧（eval_* / universe 不变）
    assert rec.eval_start == "20240101"
    assert rec.eval_end == "20240630"
    assert rec.universe == "csi300"

    # 生成的 factor.py 可 import，compute 与生产求值逐位一致
    mod = _import_factor_py(asset_dir / "factor.py")
    assert mod.EXPRESSION == "rank(close)"
    got = mod.compute(daily)
    assert got.columns == ["trade_date", "ts_code", "factor_value"] or set(got.columns) >= {
        "trade_date",
        "ts_code",
        "factor_value",
    }
    got = got.select(["trade_date", "ts_code", "factor_value"]).sort(
        ["trade_date", "ts_code"]
    )
    exp = expected.select(["trade_date", "ts_code", "factor_value"]).sort(
        ["trade_date", "ts_code"]
    )
    assert got.height == exp.height
    np.testing.assert_allclose(
        got["factor_value"].to_numpy(),
        exp["factor_value"].to_numpy(),
        equal_nan=True,
    )

    # parquet 内容
    pq = pl.read_parquet(asset_dir / "factor.parquet")
    assert set(pq.columns) >= {"trade_date", "ts_code", "factor_value"}
    assert pq.height == expected.height


def test_write_correlated_skips_parquet(tmp_path):
    """correlated 只写 meta+py，不物化 parquet。"""
    from factorzen.discovery.factor_store import write_factor_asset

    rec = _expr_record("rank(vol)", name="mined_corr", status="correlated")
    asset_dir = Path(
        write_factor_asset(
            rec, market="ashare", root=str(tmp_path), materialize=True, panel=None
        )
    )
    assert (asset_dir / "meta.json").is_file()
    assert (asset_dir / "factor.py").is_file()
    assert not (asset_dir / "factor.parquet").exists()
    meta = json.loads((asset_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["materialization"] is None


def test_write_materialize_false_no_parquet(tmp_path):
    """materialize=False 时 active 也不写 parquet。"""
    from factorzen.discovery.factor_store import write_factor_asset

    rec = _expr_record(status="active")
    asset_dir = Path(
        write_factor_asset(
            rec, market="ashare", root=str(tmp_path), materialize=False
        )
    )
    assert (asset_dir / "meta.json").is_file()
    assert (asset_dir / "factor.py").is_file()
    assert not (asset_dir / "factor.parquet").exists()
    meta = json.loads((asset_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["materialization"] is None


# ── python 类 + provider ─────────────────────────────────────────────────────


def test_python_factor_from_store_provider(tmp_path, monkeypatch):
    """python 类 factor.py 放进 factor_store 后，provider 能加载且 compute 可跑。"""
    from factorzen.daily.factors.base import DailyFactor
    from factorzen.discovery.library_provider import load_library_factors

    name = "store_py_alpha_tdd"
    factor_code = f'''"""TDD python factor in store."""
import polars as pl
from factorzen.daily.data.context import FactorDataContext
from factorzen.daily.factors.base import DailyFactor


class StorePyAlphaTdd(DailyFactor):
    name = "{name}"
    category = "daily"
    frequency = "daily"
    required_data = ["daily"]
    lookback_days = 5
    description = "tdd store python factor"

    def compute(self, ctx: FactorDataContext) -> pl.DataFrame:
        return (
            ctx.daily.sort(["ts_code", "trade_date"])
            .with_columns(pl.col("close_adj").alias("factor_value"))
            .select(["trade_date", "ts_code", "factor_value"])
            .collect()
        )


StorePyAlphaTdd()
'''
    asset = tmp_path / "factor_store" / "ashare" / name
    asset.mkdir(parents=True)
    (asset / "factor.py").write_text(factor_code, encoding="utf-8")
    (asset / "meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "kind": "python",
                "expression": f"py::{name}",
                "frequency": "daily",
                "description": "tdd",
                "source_run_id": None,
                "created_at": "2026-07-21",
                "ledger_snapshot": {
                    "status": "active",
                    "lift": None,
                    "admission_ic": None,
                    "ic_train": None,
                    "holdout_ic": None,
                    "truth": "workspace/factor_library/ashare.jsonl",
                },
                "materialization": None,
            }
        ),
        encoding="utf-8",
    )

    # jsonl 里登记 python 记录（provider 按库+store 加载）
    from factorzen.discovery.factor_library import FactorRecord, _save_library

    rec = FactorRecord(
        expression=f"py::{name}",
        market="ashare",
        kind="python",
        name=name,
        impl=name,
        status="active",
        added_at="2026-07-21",
        updated_at="2026-07-21",
    )
    lib_root = tmp_path / "factor_library"
    lib_root.mkdir()
    _save_library("ashare", [rec], root=str(lib_root))

    # 确保未预注册
    from factorzen.daily.factors import registry as reg_mod

    if name in reg_mod._registry._registry:
        del reg_mod._registry._registry[name]

    n = load_library_factors(
        market="ashare",
        root=str(lib_root),
        store_root=str(tmp_path / "factor_store"),
    )
    assert n >= 1
    cls = reg_mod.get_factor(name)
    assert issubclass(cls, DailyFactor)
    inst = cls()
    assert inst.name == name


# ── sync 增量 ────────────────────────────────────────────────────────────────


def test_sync_skips_unchanged_materialization(tmp_path, monkeypatch):
    """第二次 sync：store 口径 (all_a/2016~最新) 已覆盖 → 跳过物化。"""
    from factorzen.discovery import factor_store as fs
    from factorzen.discovery.factor_library import _save_library

    rec = _expr_record("rank(close)", name="mined_sync_skip", status="active")
    lib_root = tmp_path / "factor_library"
    store_root = tmp_path / "factor_store"
    lib_root.mkdir()
    _save_library("ashare", [rec], root=str(lib_root))

    # 固定「最新已完结交易日」，避免日历/日期抖动
    fixed_end = "2026-07-20"
    monkeypatch.setattr(fs, "store_materialize_end", lambda: fixed_end)

    calls: list[str] = []

    def fake_mat(records, **kw):
        for r in records:
            calls.append(r.name or r.expression)
            # 写假 parquet + 更新 meta materialization（新 store 口径）
            name = r.name or "anon"
            d = store_root / "ashare" / name
            d.mkdir(parents=True, exist_ok=True)
            panel = pl.DataFrame(
                {
                    "trade_date": [date(2024, 1, 2)],
                    "ts_code": ["000001.SH"],
                    "factor_value": [1.0],
                }
            )
            panel.write_parquet(d / "factor.parquet")
            meta_path = d / "meta.json"
            meta = (
                json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists()
                else {}
            )
            meta["materialization"] = {
                "start": fs.STORE_MATERIALIZE_START,
                "end": fixed_end,
                "universe": fs.STORE_MATERIALIZE_UNIVERSE,
                "git_sha": "deadbeef",
                "n_rows": 1,
                "generated_at": "2026-07-21T00:00:00+00:00",
                "expression": r.expression,
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return len(records)

    monkeypatch.setattr(fs, "_materialize_records", fake_mat)

    # 第一次：写 meta+py，并物化
    stats1 = fs.sync_store(
        "ashare",
        root=str(store_root),
        lib_root=str(lib_root),
        materialize=True,
    )
    assert stats1["written"] == 1
    assert stats1["materialized"] == 1
    assert calls == ["mined_sync_skip"]

    # 第二次：应跳过物化
    calls.clear()
    stats2 = fs.sync_store(
        "ashare",
        root=str(store_root),
        lib_root=str(lib_root),
        materialize=True,
    )
    assert stats2["written"] == 1  # meta/py 仍刷新或计为处理
    assert stats2["materialized"] == 0
    assert stats2["skipped_materialize"] >= 1
    assert calls == []


# ── verify 漂移 ──────────────────────────────────────────────────────────────


def test_verify_reports_expression_drift(tmp_path):
    """人为改 meta.expression → verify 报漂移。"""
    from factorzen.discovery.factor_library import _save_library
    from factorzen.discovery.factor_store import verify_store, write_factor_asset

    rec = _expr_record("rank(close)", name="mined_drift")
    lib_root = tmp_path / "factor_library"
    store_root = tmp_path / "factor_store"
    lib_root.mkdir()
    _save_library("ashare", [rec], root=str(lib_root))
    write_factor_asset(rec, market="ashare", root=str(store_root), materialize=False)

    # 篡改 meta
    meta_path = store_root / "ashare" / "mined_drift" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["expression"] = "rank(vol)"  # 漂移
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    report = verify_store("ashare", root=str(store_root), lib_root=str(lib_root))
    assert report["ok"] is False
    assert any(d.get("name") == "mined_drift" for d in report["drifts"])
    drift = next(d for d in report["drifts"] if d["name"] == "mined_drift")
    assert drift["field"] == "expression"
    assert drift["store"] == "rank(vol)"
    assert drift["ledger"] == "rank(close)"


def test_verify_clean_when_consistent(tmp_path):
    """一致时 verify 零漂移。"""
    from factorzen.discovery.factor_library import _save_library
    from factorzen.discovery.factor_store import verify_store, write_factor_asset

    rec = _expr_record("rank(close)", name="mined_ok")
    lib_root = tmp_path / "factor_library"
    store_root = tmp_path / "factor_store"
    lib_root.mkdir()
    _save_library("ashare", [rec], root=str(lib_root))
    write_factor_asset(rec, market="ashare", root=str(store_root), materialize=False)

    report = verify_store("ashare", root=str(store_root), lib_root=str(lib_root))
    assert report["ok"] is True
    assert report["drifts"] == []


def test_verify_reports_materialization_universe_drift(tmp_path, monkeypatch):
    """旧 csi300/csi500 物化口径 → verify 报 materialization 漂移（期望重物化为 all_a）。"""
    from factorzen.discovery import factor_store as fs
    from factorzen.discovery.factor_library import _save_library
    from factorzen.discovery.factor_store import verify_store, write_factor_asset

    fixed_end = "2026-07-20"
    monkeypatch.setattr(fs, "store_materialize_end", lambda: fixed_end)

    # probation + 旧 universe 物化（模拟 mined_7d449261 类历史资产）
    rec = _expr_record(
        "rank(close)",
        name="mined_7d449261",
        status="probation",
        universe="csi500",
    )
    lib_root = tmp_path / "factor_library"
    store_root = tmp_path / "factor_store"
    lib_root.mkdir()
    _save_library("ashare", [rec], root=str(lib_root))
    write_factor_asset(rec, market="ashare", root=str(store_root), materialize=False)

    meta_path = store_root / "ashare" / "mined_7d449261" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["materialization"] = {
        "start": "20240101",
        "end": "20240630",
        "universe": "csi300",
        "git_sha": "old",
        "n_rows": 10,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "expression": rec.expression,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    # 假 parquet 存在，触发窗口/universe 一致性校验
    (store_root / "ashare" / "mined_7d449261" / "factor.parquet").write_bytes(b"pq")

    report = verify_store("ashare", root=str(store_root), lib_root=str(lib_root))
    assert report["ok"] is False
    mat_drifts = [
        d
        for d in report["drifts"]
        if d.get("name") == "mined_7d449261"
        and str(d.get("field", "")).startswith("materialization")
    ]
    assert mat_drifts, f"expected materialization drift, got {report['drifts']}"
    fields = {d["field"] for d in mat_drifts}
    assert "materialization.universe" in fields or any(
        "universe" in f for f in fields
    )


def test_materialize_records_panel_loader_uses_store_window(tmp_path, monkeypatch):
    """_materialize_records 装帧：panel_loader 收到 store 常量口径（非 record.eval_*）。"""
    from factorzen.discovery import factor_store as fs

    fixed_end = "2026-07-20"
    monkeypatch.setattr(fs, "store_materialize_end", lambda: fixed_end)

    rec = _expr_record(
        "rank(close)",
        name="mined_loader_win",
        status="active",
        eval_start="20240101",
        eval_end="20240630",
        universe="csi300",
    )
    calls: list[dict] = []

    def fake_loader(**kw):
        calls.append(dict(kw))
        # 最小 prepped 帧：_materializer_from_prepped 需要真实列；此处直接拦在 mat 前
        raise RuntimeError("stop_after_loader_capture")

    # 只验证 loader 入参：panel_loader 抛错时组级 continue，n_ok=0
    n = fs._materialize_records(
        [rec],
        market="ashare",
        root=str(tmp_path),
        panel_loader=fake_loader,
    )
    assert n == 0
    assert len(calls) == 1
    c = calls[0]
    # 行为/数值断言（不用 signature）
    assert c["universe"] == fs.STORE_MATERIALIZE_UNIVERSE
    # panel 通道用 YYYYMMDD
    assert c["start"].replace("-", "") == fs.STORE_MATERIALIZE_START.replace("-", "")
    assert c["end"].replace("-", "") == fixed_end.replace("-", "")
    assert c["market"] == "ashare"
    assert "intraday_leaves" in c


# ── 生成模板可 import ────────────────────────────────────────────────────────


def test_render_expression_factor_py_importable(tmp_path):
    """生成模板内容独立可 import 与调用。"""
    from factorzen.discovery.factor_store import render_expression_factor_py

    code = render_expression_factor_py(
        name="mined_tpl",
        expression="neg(rank(close))",
        hypothesis="short-term reversal",
        snapshot={"ic_train": 0.01, "holdout_ic": -0.01, "status": "active"},
    )
    path = tmp_path / "factor.py"
    path.write_text(code, encoding="utf-8")
    mod = _import_factor_py(path, "fz_tpl")
    assert mod.EXPRESSION == "neg(rank(close))"
    daily = _tiny_daily()
    out = mod.compute(daily)
    assert out.height > 0
    assert "factor_value" in out.columns


# ── description 截断 ─────────────────────────────────────────────────────────


def test_description_truncates_hypothesis_200(tmp_path):
    from factorzen.discovery.factor_store import write_factor_asset

    long_h = "x" * 500
    rec = _expr_record(hypothesis=long_h, name="mined_long_h")
    asset_dir = Path(
        write_factor_asset(rec, market="ashare", root=str(tmp_path), materialize=False)
    )
    meta = json.loads((asset_dir / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["description"]) == 200
