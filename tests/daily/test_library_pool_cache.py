"""
test_library_pool_compact.py：库池 compact（单骨架宽面板）内存路径：与 legacy 数值 parity + 自动开关。
test_python_panel_cache.py：python 因子面板磁盘缓存单测。
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


# ==== 来自 test_library_pool_compact.py ====
def _mk_daily(n_days: int = 100, n_stocks: int = 20, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days: list[dt.date] = []
    d = dt.date(2021, 1, 4)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    for c in [f"{600000 + i:06d}.SH" for i in range(n_stocks)]:
        base = rng.uniform(8, 15)
        for i, dd in enumerate(days):
            px = base * (1 + 0.001 * i) + rng.normal(0, 0.1)
            rows.append({
                "trade_date": dd, "ts_code": c,
                "close": px, "open": px, "high": px * 1.01, "low": px * 0.99,
                "close_adj": px, "open_adj": px, "high_adj": px * 1.01, "low_adj": px * 0.99,
                "pre_close": px / (1 + 0.001 * max(i, 1)),
                "vol": 1e6 + rng.normal(0, 1e4), "amount": 1e7 + rng.normal(0, 1e5),
            })
    return pl.DataFrame(rows)


def _write_lib(root: Path, market: str, records: list[dict]) -> None:
    path = root / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _seed_lib(tmp_path: Path) -> Path:
    _write_lib(tmp_path, "ashare", [
        {"expression": "rank(close)", "market": "ashare", "status": "active",
         "ic_train": 0.05},
        {"expression": "rank(vol)", "market": "ashare", "status": "active",
         "ic_train": 0.04},
        {"expression": "rank(amount)", "market": "ashare", "status": "active",
         "ic_train": 0.03},
    ])
    return tmp_path


# ── parity: compact vs legacy ────────────────────────────────────────────────


def test_compact_legacy_getitem_values_equal(tmp_path):
    """同一表达式长表 filter 后 factor_value 与键 f64 全等。"""
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
    )

    daily = _mk_daily()
    root = str(_seed_lib(tmp_path))
    legacy = build_library_pool("ashare", daily, root=root, compact=False)
    compact = build_library_pool("ashare", daily, root=root, compact=True)
    assert isinstance(compact, CompactLibraryPool)
    assert set(legacy.keys()) == set(compact.keys())
    for expr in legacy:
        a = legacy[expr].sort(["trade_date", "ts_code"])
        b = compact[expr].sort(["trade_date", "ts_code"])
        assert a.height == b.height
        assert a["trade_date"].to_list() == b["trade_date"].to_list()
        assert a["ts_code"].to_list() == b["ts_code"].to_list()
        va = a["factor_value"].to_numpy()
        vb = b["factor_value"].to_numpy()
        np.testing.assert_array_equal(va, vb)


def test_compact_legacy_corr_panel_and_max_corr_equal(tmp_path):
    """build_library_corr_panel + max_correlation 两模式 f64 全等。"""
    from factorzen.discovery.factor_library import build_library_pool
    from factorzen.discovery.scoring import (
        build_library_corr_panel,
        max_correlation,
        max_correlation_detail,
    )

    daily = _mk_daily()
    root = str(_seed_lib(tmp_path))
    legacy = build_library_pool("ashare", daily, root=root, compact=False)
    compact = build_library_pool("ashare", daily, root=root, compact=True)

    p_leg = build_library_corr_panel(legacy)
    p_cmp = build_library_corr_panel(compact)
    assert p_leg is not None and p_cmp is not None
    assert p_leg.names == p_cmp.names
    assert p_leg.dates == p_cmp.dates
    assert p_leg.stocks == p_cmp.stocks
    # present=None 新契约:掩码经 present_block 推导(直接 np.where(None,...) 会把
    # None 当 False 标量退化成恒真比较——陷阱#1)
    pres_leg = p_leg.present_block(0, len(p_leg.dates))
    pres_cmp = p_cmp.present_block(0, len(p_cmp.dates))
    np.testing.assert_array_equal(pres_leg, pres_cmp)
    assert pres_leg.any()  # 掩码非空,比较有判别力
    # 值：null 位已由 present 标；有限位须 bit-identical
    np.testing.assert_array_equal(
        np.where(pres_leg, p_leg.values, 0.0),
        np.where(pres_cmp, p_cmp.values, 0.0),
    )

    # 候选 = 库内第一因子
    cand = legacy[next(iter(legacy))]
    mc_l, n_l = max_correlation_detail(cand, legacy, panel=p_leg)
    mc_c, n_c = max_correlation_detail(cand, compact, panel=p_cmp)
    assert mc_l == mc_c
    assert n_l == n_c
    assert max_correlation(cand, legacy, panel=p_leg) == max_correlation(
        cand, compact, panel=p_cmp,
    )


def test_compact_legacy_residual_ic_equal(tmp_path):
    """residual LibraryPanel + compute_residual_ic 两模式一致。"""
    from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns
    from factorzen.discovery.factor_library import build_library_pool
    from factorzen.discovery.residual import (
        ResidualProjector,
        build_library_panel,
        compute_residual_ic,
    )

    daily = _mk_daily()
    root = str(_seed_lib(tmp_path))
    legacy = build_library_pool("ashare", daily, root=root, compact=False)
    compact = build_library_pool("ashare", daily, root=root, compact=True)

    panel_l = build_library_panel(legacy)
    panel_c = build_library_panel(compact)
    assert panel_l is not None and panel_c is not None
    assert panel_l.factor_names == panel_c.factor_names
    assert panel_l.dates == panel_c.dates
    assert panel_l.stocks == panel_c.stocks
    np.testing.assert_allclose(panel_l.X, panel_c.X, rtol=0, atol=0)

    cand = legacy[next(iter(legacy))]
    # 用略扰动的候选避免与库列完全共线导致数值病态差异放大
    cand2 = cand.with_columns(
        (pl.col("factor_value") + 0.01 * pl.col("factor_value").rank().over("trade_date")
         / pl.col("factor_value").count().over("trade_date")).alias("factor_value")
    )
    sorted_daily = daily.sort(["ts_code", "trade_date"])
    fwd = compute_fwd_returns(sorted_daily, price_col="close_adj")
    proj_l = ResidualProjector.from_panel(panel_l)
    proj_c = ResidualProjector.from_panel(panel_c)
    r_l = compute_residual_ic(cand2, panel_l, fwd, projector=proj_l)
    r_c = compute_residual_ic(cand2, panel_c, fwd, projector=proj_c)
    assert r_l.n_days == r_c.n_days
    if r_l.n_days > 0:
        assert r_l.ic_mean == r_c.ic_mean


# ── 自动开关 ────────────────────────────────────────────────────────────────


def test_auto_compact_when_over_threshold(tmp_path, capsys):
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
    )

    daily = _mk_daily(n_days=30, n_stocks=10)
    root = str(_seed_lib(tmp_path))
    # 阈值调到极小 → 必走 compact
    pool = build_library_pool(
        "ashare", daily, root=root, compact=None, compact_threshold=1,
    )
    assert isinstance(pool, CompactLibraryPool)
    out = capsys.readouterr().out
    assert "库池 compact 模式" in out


def test_auto_legacy_on_small_frame(tmp_path):
    from factorzen.discovery.factor_library import (
        CompactLibraryPool,
        build_library_pool,
    )

    daily = _mk_daily(n_days=30, n_stocks=10)
    root = str(_seed_lib(tmp_path))
    pool = build_library_pool("ashare", daily, root=root, compact=None)
    assert isinstance(pool, dict)
    assert not isinstance(pool, CompactLibraryPool)


def test_should_use_compact_pool_math():
    from factorzen.discovery.factor_library import (
        POOL_KEY_BYTES_PER_ROW,
        estimate_library_pool_key_bytes,
        should_use_compact_pool,
    )

    n_f, n_r = 84, 10_925_813
    est = estimate_library_pool_key_bytes(n_f, n_r)
    assert est == n_f * n_r * POOL_KEY_BYTES_PER_ROW
    assert should_use_compact_pool(n_f, n_r, threshold=8 * 1024**3)
    assert not should_use_compact_pool(3, 2000, threshold=8 * 1024**3)


def test_compact_filter_dates(tmp_path):
    from factorzen.discovery.factor_library import CompactLibraryPool, build_library_pool

    daily = _mk_daily(n_days=40, n_stocks=8)
    root = str(_seed_lib(tmp_path))
    pool = build_library_pool("ashare", daily, root=root, compact=True)
    assert isinstance(pool, CompactLibraryPool)
    dates = sorted(daily["trade_date"].unique().to_list())
    half = dates[: len(dates) // 2]
    sliced = pool.filter_dates(half)
    assert isinstance(sliced, CompactLibraryPool)
    assert sliced.wide["trade_date"].max() <= max(half)
    assert len(sliced) > 0


def test_compact_panel_row_set_matches_legacy_with_warmup_nulls(tmp_path):
    """滚动窗因子的预热期全 null 行:legacy 行集=「至少一因子有限」;compact 必须同行集。

    否则全缺行(带 ret)混进 LGBM 训练面板与 fold 日期轴——同数据 compact/legacy
    静默数值漂移(预热期真实场景必现,满覆盖 mock 测不到)。
    """
    from factorzen.discovery.factor_library import build_library_pool
    from factorzen.research.combination.models import build_panel

    _write_lib(tmp_path, "ashare", [
        {"expression": "ts_mean(close, 10)", "market": "ashare",
         "status": "active", "ic_train": 0.05},
    ])
    daily = _mk_daily(40, 6)
    legacy = build_library_pool("ashare", daily, None, root=str(tmp_path), compact=False)
    comp = build_library_pool("ashare", daily, None, root=str(tmp_path), compact=True)
    ret = daily.select(
        [pl.col("trade_date").cast(pl.Utf8), "ts_code"]
    ).with_columns(pl.lit(0.01).alias("ret"))

    p_l = build_panel(legacy, ret)
    p_c = build_panel(comp, ret)
    assert p_c.height == p_l.height, \
        f"compact 面板混入全 null 预热行: compact={p_c.height} legacy={p_l.height}"
    key = ["trade_date", "ts_code"]
    assert p_c.sort(key).select(p_l.columns).equals(p_l.sort(key)), "行集/值不一致"

# ==== 来自 test_python_panel_cache.py ====
def _install_factor_module(tmp_path: Path, name: str, body: str) -> type:
    """写真实 .py 再 import，保证 inspect.getsourcefile 可用。"""
    mod_path = tmp_path / f"{name}.py"
    mod_path.write_text(textwrap.dedent(body), encoding="utf-8")
    mod_name = f"_cache_test_{name}_{mod_path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.CachedFactor  # type: ignore[attr-defined]


_FACTOR_BODY = '''
from datetime import datetime, timedelta
import polars as pl
from factorzen.daily.factors.base import DailyFactor

_COMPUTE_COUNT = 0

class CachedFactor(DailyFactor):
    name = "cached_factor"
    lookback_days = 2
    required_data = ["daily"]
    description = "panel cache test"

    def compute(self, ctx):
        global _COMPUTE_COUNT
        _COMPUTE_COUNT += 1
        start_d = datetime.strptime(ctx.start, "%Y%m%d").date()
        end_d = datetime.strptime(ctx.end, "%Y%m%d").date()
        exp_d = datetime.strptime(ctx.expanded_start, "%Y%m%d").date()
        rows = []
        d = exp_d
        while d <= end_d:
            if d.weekday() < 5:
                for i in range(2):
                    rows.append({
                        "trade_date": d,
                        "ts_code": f"{i:06d}.SH",
                        "factor_value": 1.0 + i + (0.0 if d >= start_d else -99.0),
                    })
            d += timedelta(days=1)
        return pl.DataFrame(rows)
'''

def _patch_materialize_offline(monkeypatch, factor_cls, tmp_path: Path):
    """registry / universe / calendar / DATA_CACHE 全部离线。"""
    import factorzen.config.settings as settings
    import factorzen.daily.data.context as ctx_mod
    import factorzen.daily.factors.registry as reg_mod
    from factorzen.discovery import python_factor as pyf

    monkeypatch.setattr(settings, "DATA_CACHE", tmp_path / "cache")
    monkeypatch.setattr(reg_mod, "get_factor", lambda name: factor_cls)
    monkeypatch.setattr(
        pyf, "_load_universe_codes",
        lambda start, end, universe: ["000000.SH", "000001.SH"],
    )

    def _fake_expanded(self):
        d = datetime.strptime(self.start, "%Y%m%d").date() - timedelta(
            days=self.lookback_days + 2
        )
        return d.strftime("%Y%m%d")

    monkeypatch.setattr(
        ctx_mod.FactorDataContext, "expanded_start", property(_fake_expanded),
    )
    return pyf


def test_panel_cache_hit_skips_recompute(tmp_path, monkeypatch):
    """首调 compute 1 次；二调 0 次且 frame 相等。"""
    factor_cls = _install_factor_module(tmp_path, "hit_factor", _FACTOR_BODY)
    pyf = _patch_materialize_offline(monkeypatch, factor_cls, tmp_path)

    # 通过模块全局计数
    mod = sys.modules[factor_cls.__module__]
    assert mod._COMPUTE_COUNT == 0

    start, end = "20240110", "20240115"
    out1 = pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare", use_cache=True,
    )
    assert mod._COMPUTE_COUNT == 1
    assert out1.height > 0

    out2 = pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare", use_cache=True,
    )
    assert mod._COMPUTE_COUNT == 1  # 命中，不再 compute
    assert out1.equals(out2)

    # 缓存文件落在 DATA_CACHE/python_factor_panels/...
    cache_root = tmp_path / "cache" / "python_factor_panels"
    assert any(cache_root.rglob("*.parquet"))


def test_panel_cache_source_change_busts_key(tmp_path, monkeypatch):
    """改写 .py 源码 → impl_sha 变 → 重算（不命中旧缓存）。"""
    from factorzen.discovery.python_factor import _impl_source_sha, _panel_cache_key

    mod_path = tmp_path / "bust_factor.py"
    mod_path.write_text(textwrap.dedent(_FACTOR_BODY), encoding="utf-8")

    def _load():
        # 每次新模块名，避免 sys.modules 缓存旧代码
        mod_name = f"_bust_{mod_path.stat().st_mtime_ns}_{len(sys.modules)}"
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    mod1 = _load()
    sha1 = _impl_source_sha(mod1.CachedFactor)
    assert sha1 is not None
    pyf = _patch_materialize_offline(monkeypatch, mod1.CachedFactor, tmp_path)

    start, end = "20240110", "20240115"
    out1 = pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare",
    )
    assert mod1._COMPUTE_COUNT == 1
    key1 = _panel_cache_key(
        "ashare", "cached_factor", start, end, "csi300", sha1, lookback_days=2,
    )

    # 改源码：追加注释即可变 impl_sha（不依赖值断言）
    mod_path.write_text(
        textwrap.dedent(_FACTOR_BODY) + "\n# source-bust marker v2\n",
        encoding="utf-8",
    )
    mod2 = _load()
    sha2 = _impl_source_sha(mod2.CachedFactor)
    assert sha2 is not None and sha2 != sha1
    key2 = _panel_cache_key(
        "ashare", "cached_factor", start, end, "csi300", sha2, lookback_days=2,
    )
    assert key2 != key1

    import factorzen.daily.factors.registry as reg_mod

    monkeypatch.setattr(reg_mod, "get_factor", lambda name: mod2.CachedFactor)

    out2 = pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare",
    )
    assert mod2._COMPUTE_COUNT == 1  # 新键未命中 → 重算
    # 结果仍合法三列面板
    assert set(out2.columns) == {"trade_date", "ts_code", "factor_value"}
    assert out2.height == out1.height


def test_panel_cache_corrupt_recomputes(tmp_path, monkeypatch):
    """损坏 parquet → 重算不崩、坏文件被清。"""
    factor_cls = _install_factor_module(tmp_path, "corrupt_factor", _FACTOR_BODY)
    pyf = _patch_materialize_offline(monkeypatch, factor_cls, tmp_path)
    mod = sys.modules[factor_cls.__module__]

    start, end = "20240110", "20240115"
    pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare",
    )
    assert mod._COMPUTE_COUNT == 1

    # 把缓存写成垃圾
    cache_files = list((tmp_path / "cache" / "python_factor_panels").rglob("*.parquet"))
    assert cache_files
    bad = cache_files[0]
    bad.write_bytes(b"not a parquet file!!!")

    out = pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare",
    )
    assert mod._COMPUTE_COUNT == 2  # 重算
    assert out.height > 0
    # 坏文件已被替换为合法 parquet（或至少可读）
    assert bad.exists()
    reloaded = pl.read_parquet(bad)
    assert {"trade_date", "ts_code", "factor_value"}.issubset(set(reloaded.columns))


def test_panel_cache_use_cache_false(tmp_path, monkeypatch):
    """use_cache=False 全程不读不写。"""
    factor_cls = _install_factor_module(tmp_path, "nocache_factor", _FACTOR_BODY)
    pyf = _patch_materialize_offline(monkeypatch, factor_cls, tmp_path)
    mod = sys.modules[factor_cls.__module__]

    start, end = "20240110", "20240115"
    pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare", use_cache=False,
    )
    pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare", use_cache=False,
    )
    assert mod._COMPUTE_COUNT == 2
    cache_root = tmp_path / "cache" / "python_factor_panels"
    assert not cache_root.exists() or not any(cache_root.rglob("*.parquet"))


def test_panel_cache_dynamic_class_no_cache(tmp_path, monkeypatch):
    """type() 动态类无源文件 → 不缓存不崩。"""
    from factorzen.daily.factors.base import DailyFactor

    count = {"n": 0}

    def compute(self, ctx):
        count["n"] += 1
        start_d = datetime.strptime(ctx.start, "%Y%m%d").date()
        end_d = datetime.strptime(ctx.end, "%Y%m%d").date()
        rows = []
        d = start_d
        while d <= end_d:
            if d.weekday() < 5:
                rows.append({
                    "trade_date": d,
                    "ts_code": "000000.SH",
                    "factor_value": 1.0,
                })
            d += timedelta(days=1)
        return pl.DataFrame(rows)

    Dyn = type(
        "DynFactor",
        (DailyFactor,),
        {
            "name": "dyn_factor",
            "lookback_days": 1,
            "required_data": ["daily"],
            "description": "dynamic",
            "compute": compute,
        },
    )
    # type() 类通常 getsourcefile → None
    assert py_impl_sha_is_none(Dyn)

    pyf = _patch_materialize_offline(monkeypatch, Dyn, tmp_path)
    start, end = "20240110", "20240115"
    out1 = pyf.materialize_python_panel(
        "dyn_factor", start, end, "csi300", market="ashare", use_cache=True,
    )
    out2 = pyf.materialize_python_panel(
        "dyn_factor", start, end, "csi300", market="ashare", use_cache=True,
    )
    assert count["n"] == 2  # 无缓存，每次都算
    assert out1.height == out2.height
    cache_root = tmp_path / "cache" / "python_factor_panels"
    assert not cache_root.exists() or not any(cache_root.rglob("*.parquet"))


def py_impl_sha_is_none(cls) -> bool:
    from factorzen.discovery.python_factor import _impl_source_sha

    return _impl_source_sha(cls) is None


_EMPTY_FACTOR_BODY = '''
import polars as pl
from factorzen.daily.factors.base import DailyFactor

_COMPUTE_COUNT = 0

class CachedFactor(DailyFactor):
    name = "empty_factor"
    lookback_days = 2
    required_data = ["daily"]
    description = "empty panel cache test"

    def compute(self, ctx):
        global _COMPUTE_COUNT
        _COMPUTE_COUNT += 1
        return pl.DataFrame(schema={"trade_date": pl.Date, "ts_code": pl.Utf8,
                                    "factor_value": pl.Float64})
'''


def test_panel_cache_skips_empty_panel(tmp_path, monkeypatch):
    """空面板不写缓存：数据未回补的空结果落盘会在回补后持续命中（文件存在≠数据完整）。"""
    import sys as _sys

    factor_cls = _install_factor_module(tmp_path, "empty_factor", _EMPTY_FACTOR_BODY)
    pyf = _patch_materialize_offline(monkeypatch, factor_cls, tmp_path)
    mod = _sys.modules[factor_cls.__module__]

    out1 = pyf.materialize_python_panel(
        "empty_factor", "20240110", "20240115", "csi300",
        market="ashare", use_cache=True,
    )
    assert out1.is_empty()
    assert mod._COMPUTE_COUNT == 1
    cache_root = tmp_path / "cache" / "python_factor_panels"
    assert not any(cache_root.rglob("*.parquet"))  # 空面板未落盘

    out2 = pyf.materialize_python_panel(
        "empty_factor", "20240110", "20240115", "csi300",
        market="ashare", use_cache=True,
    )
    assert out2.is_empty()
    assert mod._COMPUTE_COUNT == 2  # 无缓存可命中 → 重算


def test_panel_cache_key_includes_lookback_days():
    """其余参数相同、lookback 不同 → 缓存键必须不同。"""
    from factorzen.discovery.python_factor import _panel_cache_key

    common = ("ashare", "cached_factor", "20240110", "20240115", "csi300", "deadbeef")
    key20 = _panel_cache_key(*common, lookback_days=20)
    key40 = _panel_cache_key(*common, lookback_days=40)
    assert key20 != key40


def test_panel_cache_lookback_change_busts_key(tmp_path, monkeypatch):
    """monkeypatch 类 lookback_days 后二次物化不得命中第一次写的缓存路径。"""
    factor_cls = _install_factor_module(tmp_path, "lb_factor", _FACTOR_BODY)
    # 源文件 lookback=2；先以 5 物化，再改成 40
    factor_cls.lookback_days = 5
    pyf = _patch_materialize_offline(monkeypatch, factor_cls, tmp_path)
    mod = sys.modules[factor_cls.__module__]

    start, end = "20240110", "20240115"
    pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare", use_cache=True,
    )
    assert mod._COMPUTE_COUNT == 1
    cache_root = tmp_path / "cache" / "python_factor_panels"
    files_after_first = {p.resolve() for p in cache_root.rglob("*.parquet")}
    assert files_after_first

    factor_cls.lookback_days = 40
    pyf.materialize_python_panel(
        "cached_factor", start, end, "csi300", market="ashare", use_cache=True,
    )
    # lookback 入键 → 不命中旧文件，必须重算并写出新路径
    assert mod._COMPUTE_COUNT == 2
    files_after_second = {p.resolve() for p in cache_root.rglob("*.parquet")}
    assert files_after_second - files_after_first

