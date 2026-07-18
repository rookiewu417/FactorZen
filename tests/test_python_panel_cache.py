"""python 因子面板磁盘缓存单测。"""
from __future__ import annotations

import importlib.util
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


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
    key1 = _panel_cache_key("ashare", "cached_factor", start, end, "csi300", sha1)

    # 改源码：追加注释即可变 impl_sha（不依赖值断言）
    mod_path.write_text(
        textwrap.dedent(_FACTOR_BODY) + "\n# source-bust marker v2\n",
        encoding="utf-8",
    )
    mod2 = _load()
    sha2 = _impl_source_sha(mod2.CachedFactor)
    assert sha2 is not None and sha2 != sha1
    key2 = _panel_cache_key("ashare", "cached_factor", start, end, "csi300", sha2)
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
