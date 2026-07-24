"""factor_store 三件套：meta.json + factor.py + factor.parquet。

TDD 覆盖：
- expression 类 write → 三件套齐、meta 字段全、生成 factor.py 可 import 且与生产求值逐位一致
- python 类迁移后 provider 能加载、compute 可跑
- sync 增量：第二次 sync 跳过未变因子
- verify：人为改 meta.expression → 报漂移

生产默认 ``STORE_FACTOR_PARQUET_ENABLED=False``；本文件测试物化路径时 monkeypatch 开回。
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
    asset = tmp_path / "factors" / "ashare" / name
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
        store_root=str(tmp_path / "factors"),
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
    store_root = tmp_path / "factors"
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
    store_root = tmp_path / "factors"
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
    store_root = tmp_path / "factors"
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
    store_root = tmp_path / "factors"
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


# ── load_materialized_factor 门控 ─────────────────────────────────────────────


def _write_mini_asset(
    root: Path,
    *,
    name: str = "f_rank_close",
    expression: str = "rank(close)",
    universe: str = "all_a",
    mat_start: str = "2016-01-01",
    mat_end: str = "2024-12-31",
    dates: list[date] | None = None,
    values: list[tuple[date, str, float]] | None = None,
) -> Path:
    """手写微型 store 资产（meta + parquet），离线字面量。"""
    d = root / "ashare" / name
    d.mkdir(parents=True, exist_ok=True)
    if values is None:
        if dates is None:
            dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        values = [(dt, "000001.SZ", 1.0 + i * 0.1) for i, dt in enumerate(dates)]
    pq = pl.DataFrame(
        {
            "trade_date": [v[0] for v in values],
            "ts_code": [v[1] for v in values],
            "factor_value": [v[2] for v in values],
        }
    ).with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("ts_code").cast(pl.Utf8),
        pl.col("factor_value").cast(pl.Float64),
    )
    pq.write_parquet(d / "factor.parquet")
    meta = {
        "name": name,
        "kind": "expression",
        "expression": expression,
        "frequency": "daily",
        "description": "",
        "materialization": {
            "start": mat_start,
            "end": mat_end,
            "universe": universe,
            "git_sha": "deadbeef",
            "n_rows": pq.height,
            "generated_at": "2026-07-01T00:00:00+00:00",
            "expression": expression,
        },
    }
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (d / "factor.py").write_text("# stub\n", encoding="utf-8")
    return d


def test_load_materialized_factor_gates(tmp_path):
    """loader 门逐个：universe / 窗下界 / 窗上界 / expression / 全对齐切片。"""
    from factorzen.discovery.factor_store import load_materialized_factor

    store = tmp_path / "factors"
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    values = [
        (date(2024, 1, 2), "000001.SZ", 1.0),
        (date(2024, 1, 3), "000001.SZ", 2.0),
        (date(2024, 1, 4), "000001.SZ", 3.0),
    ]
    _write_mini_asset(store, values=values, dates=dates)
    rec = _expr_record("rank(close)", name="f_rank_close")

    # universe 不匹配
    df, reason, meta = load_materialized_factor(
        rec, market="ashare", root=str(store),
        start="20240102", end="20240104", universe="csi300",
    )
    assert df is None and reason == "universe_mismatch" and meta is None

    # 请求 start 早于 parquet 最小日
    df, reason, meta = load_materialized_factor(
        rec, market="ashare", root=str(store),
        start="20240101", end="20240104", universe="all_a",
    )
    assert df is None and reason == "window_start_uncovered"

    # materialization.end 早于请求 end
    store2 = tmp_path / "store_short_end"
    _write_mini_asset(store2, mat_end="2024-01-03", values=values)
    df, reason, meta = load_materialized_factor(
        rec, market="ashare", root=str(store2),
        start="20240102", end="20240104", universe="all_a",
    )
    assert df is None and reason == "window_end_uncovered"

    # expression 不一致
    rec_bad = _expr_record("rank(vol)", name="f_rank_close")
    df, reason, meta = load_materialized_factor(
        rec_bad, market="ashare", root=str(store),
        start="20240102", end="20240104", universe="all_a",
    )
    assert df is None and reason == "expression_mismatch"

    # 全对齐：返回帧逐值等于手写期望切片 [01-03, 01-04]
    df, reason, meta = load_materialized_factor(
        rec, market="ashare", root=str(store),
        start="20240103", end="20240104", universe="all_a",
    )
    assert reason is None and df is not None
    assert meta is not None and meta.get("git_sha") == "deadbeef"
    expect = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 3), date(2024, 1, 4)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "factor_value": [2.0, 3.0],
        }
    ).with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("ts_code").cast(pl.Utf8),
        pl.col("factor_value").cast(pl.Float64),
    )
    assert df.sort(["trade_date", "ts_code"]).equals(
        expect.sort(["trade_date", "ts_code"])
    )


def test_default_root_equals_factor_store_dir():
    """DEFAULT_ROOT 与 settings.FACTOR_STORE_DIR 一字不差。"""
    from factorzen.config.settings import FACTOR_STORE_DIR
    from factorzen.discovery.factor_store import DEFAULT_ROOT

    assert str(FACTOR_STORE_DIR) == DEFAULT_ROOT


# ── materialize_assets / allow_warmup_head ───────────────────────────────────


def test_materialize_assets_expr_python_skip_and_errors(tmp_path, monkeypatch):
    """tmp store 造 2 个资产(表达式+假 python)：物化后 parquet/meta 齐、二次 skip、errors 不炸。"""
    from factorzen.discovery import factor_store as fs
    from factorzen.discovery.factor_store import materialize_assets

    fixed_end = "2026-07-20"
    monkeypatch.setattr(fs, "store_materialize_end", lambda: fixed_end)

    store = tmp_path / "factors"
    # 1) expression 资产
    expr_name = "mat_assets_rank"
    expr_dir = store / "ashare" / expr_name
    expr_dir.mkdir(parents=True)
    (expr_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": expr_name,
                "kind": "expression",
                "expression": "rank(close)",
                "frequency": "daily",
                "description": "",
                "source_run_id": None,
                "created_at": "2026-07-21",
                "ledger_snapshot": {
                    "status": "correlated",
                    "lift": None,
                    "admission_ic": None,
                    "ic_train": 0.01,
                    "holdout_ic": 0.01,
                    "truth": "workspace/factor_library/ashare.jsonl",
                },
                "materialization": None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (expr_dir / "factor.py").write_text("# stub\n", encoding="utf-8")

    # 2) python 资产（假 py::）
    py_name = "mat_assets_py_fake"
    py_dir = store / "ashare" / py_name
    py_dir.mkdir(parents=True)
    (py_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": py_name,
                "kind": "python",
                "expression": f"py::{py_name}",
                "frequency": "daily",
                "description": "fake",
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
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (py_dir / "factor.py").write_text("# stub python\n", encoding="utf-8")

    # 3) 坏资产：无 expression → errors
    bad_name = "mat_assets_bad"
    bad_dir = store / "ashare" / bad_name
    bad_dir.mkdir(parents=True)
    (bad_dir / "meta.json").write_text(
        json.dumps({"name": bad_name, "kind": "expression", "expression": ""}) + "\n",
        encoding="utf-8",
    )

    daily = _tiny_daily(n_days=5, n_stocks=3)

    def fake_loader(**kw):
        return daily

    # 拦截 materializer：expression 返回真实 rank 面板；python 返回假面板
    from factorzen.discovery.expression import evaluate_materialized, parse_expr

    def fake_mat_factory(prepped, leaf_map, **kw):
        def _mat(expr: str):
            if expr.startswith("py::"):
                return pl.DataFrame(
                    {
                        "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
                        "ts_code": ["000001.SH", "000001.SH"],
                        "factor_value": [0.1, 0.2],
                    }
                )
            node = parse_expr(expr)
            return (
                prepped.select(["trade_date", "ts_code"])
                .with_columns(
                    evaluate_materialized(node, prepped).alias("factor_value")
                )
                .filter(
                    pl.col("factor_value").is_not_null()
                    & pl.col("factor_value").is_finite()
                )
            )

        return _mat

    monkeypatch.setattr(
        "factorzen.discovery.lift_test._materializer_from_prepped", fake_mat_factory
    )

    stats1 = materialize_assets(
        "ashare",
        names=[expr_name, py_name, bad_name],
        root=str(store),
        panel_loader=fake_loader,
    )
    assert stats1["total"] == 3
    assert stats1["materialized"] == 2
    assert stats1["skipped"] == 0
    assert bad_name in stats1["errors"]
    assert (expr_dir / "factor.parquet").is_file()
    assert (py_dir / "factor.parquet").is_file()

    for d, expected_expr in (
        (expr_dir, "rank(close)"),
        (py_dir, f"py::{py_name}"),
    ):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        mat = meta["materialization"]
        assert mat is not None
        for key in ("start", "end", "universe", "generated_at", "git_sha", "n_rows"):
            assert key in mat, f"missing mat key {key}"
        assert mat["start"] == fs.STORE_MATERIALIZE_START
        assert mat["end"] == fixed_end
        assert mat["universe"] == fs.STORE_MATERIALIZE_UNIVERSE
        assert mat["expression"] == expected_expr
        # ledger 状态保留（correlated 不被改写）
        if d == expr_dir:
            assert meta["ledger_snapshot"]["status"] == "correlated"

    # 二次调用 → skip
    stats2 = materialize_assets(
        "ashare",
        names=[expr_name, py_name],
        root=str(store),
        panel_loader=fake_loader,
    )
    assert stats2["materialized"] == 0
    assert stats2["skipped"] == 2
    assert stats2["errors"] == []


def test_load_materialized_factor_allow_warmup_head(tmp_path):
    """预热头部：parquet min > start 时 allow_warmup_head True 命中、False miss。"""
    from factorzen.discovery.factor_store import load_materialized_factor

    store = tmp_path / "factors"
    # parquet 从 2024-01-10 起（模拟预热吃掉 01-01~01-09）
    values = [
        (date(2024, 1, 10), "000001.SZ", 1.0),
        (date(2024, 1, 11), "000001.SZ", 2.0),
        (date(2024, 1, 12), "000001.SZ", 3.0),
    ]
    _write_mini_asset(
        store,
        name="warmup_f",
        expression="rank(close)",
        mat_start="2016-01-01",
        mat_end="2024-12-31",
        values=values,
    )
    rec = _expr_record("rank(close)", name="warmup_f")

    # 默认：parquet min > start → miss
    df, reason, meta = load_materialized_factor(
        rec,
        market="ashare",
        root=str(store),
        start="20240101",
        end="20240112",
        universe="all_a",
        allow_warmup_head=False,
    )
    assert df is None and reason == "window_start_uncovered"

    # 放宽：meta.start(2016) <= 请求 start → 命中，滤到 [start,end] 后非空
    df, reason, meta = load_materialized_factor(
        rec,
        market="ashare",
        root=str(store),
        start="20240101",
        end="20240112",
        universe="all_a",
        allow_warmup_head=True,
    )
    assert reason is None and df is not None
    assert df.height == 3
    assert meta is not None


def test_materialization_window_fresh_accepts_superset_start(monkeypatch):
    """回归：eval 补头把 mat.start 写早于 2016 后，sync 必须判新鲜（严格相等会乒乓重物化丢头部）。"""
    from factorzen.discovery import factor_store as fs

    monkeypatch.setattr(fs, "store_materialize_end", lambda: "2026-07-22")
    base = {"universe": "all_a", "end": "2026-07-22"}
    assert fs._materialization_window_fresh({**base, "start": "2016-01-01"})
    assert fs._materialization_window_fresh({**base, "start": "2015-06-01"})
    assert not fs._materialization_window_fresh({**base, "start": "2017-01-01"})
    assert not fs._materialization_window_fresh({**base, "start": None})


def test_finalize_factor_panel_normalizes_ts_code_dtype():
    """dtype 单点收口：Categorical ts_code（mining prepped 帧）落盘前必须转 Utf8。"""
    import polars as pl

    from factorzen.discovery.factor_store import finalize_factor_panel

    panel = pl.DataFrame(
        {
            "trade_date": [__import__("datetime").date(2024, 1, 2)],
            "ts_code": ["000001.SZ"],
            "factor_value": [1.0],
            "factor_clean": [0.5],
        }
    ).with_columns(pl.col("ts_code").cast(pl.Categorical))
    out = finalize_factor_panel(panel)
    assert out["ts_code"].dtype == pl.Utf8
    assert out["trade_date"].dtype == pl.Date
    assert out["factor_value"].dtype == pl.Float64


def test_load_python_factor_module_reuses_until_source_changes(tmp_path):
    """回归：批量重扫时源文件未变必须复用同一模块对象（del+重载会破坏
    multiprocessing 因子的 pickle 身份 → not the same object）；改文件后须重载。"""
    import os
    import time

    from factorzen.discovery.factor_store import load_python_factor_module

    d = tmp_path / "myfac"
    d.mkdir()
    py = d / "factor.py"
    py.write_text("MARK = 1\n", encoding="utf-8")

    m1 = load_python_factor_module(py)
    m2 = load_python_factor_module(py)
    assert m1 is m2, "源文件未变必须复用同一模块实例"

    py.write_text("MARK = 2\n", encoding="utf-8")
    # 文件系统 mtime 分辨率可能到秒级：显式推 1s，别赌粒度
    t = time.time_ns() + 10**9
    os.utime(py, ns=(t, t))
    m3 = load_python_factor_module(py)
    assert m3 is not m1 and m3.MARK == 2, "源文件变更必须重载"
