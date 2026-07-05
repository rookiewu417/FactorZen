"""因子重要性 explain 的测试:gain / shap(可选) / 缺 shap 回退。"""
from __future__ import annotations

import builtins
import importlib.util

import numpy as np
import polars as pl
import pytest

from factorzen.research.combination.importance import explain
from factorzen.research.combination.models import LGBMCombiner, build_panel

_HAS_SHAP = importlib.util.find_spec("shap") is not None


def _fitted():
    """在 ret=0.8*fa-0.4*fb 合成数据上 fit,fa 贡献强于 fb。"""
    rng = np.random.default_rng(0)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(150)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(40)
        fb = rng.standard_normal(40)
        ret = 0.8 * fa - 0.4 * fb + rng.standard_normal(40) * 0.3
        for s in range(40):
            c = f"{s:04d}.SZ"
            ra.append({"trade_date": d, "ts_code": c, "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": c, "factor_value": float(fb[s])})
            rr.append({"trade_date": d, "ts_code": c, "ret": float(ret[s])})
    factor_dfs = {"fa": pl.DataFrame(ra), "fb": pl.DataFrame(rb)}
    panel = build_panel(factor_dfs, pl.DataFrame(rr))
    combiner = LGBMCombiner(min_child_samples=20, n_estimators=60, seed=0)
    combiner.fit(panel.select(["fa", "fb"]), panel["ret"])
    return combiner, panel.select(["fa", "fb"])


def test_explain_gain():
    c, x = _fitted()
    out = explain(c, x, method="gain")
    assert set(out.columns) == {"factor", "importance", "method"}
    assert (out["method"] == "gain").all()
    d = dict(zip(out["factor"].to_list(), out["importance"].to_list(), strict=True))
    assert d["fa"] > d["fb"]


@pytest.mark.skipif(not _HAS_SHAP, reason="shap 未安装")
def test_explain_auto_uses_shap_when_available():
    c, x = _fitted()
    out = explain(c, x, method="auto")
    assert (out["method"] == "shap").all()
    d = dict(zip(out["factor"].to_list(), out["importance"].to_list(), strict=True))
    assert d["fa"] > d["fb"]


def test_explain_auto_falls_back_to_gain_without_shap(monkeypatch):
    c, x = _fitted()
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "shap":
            raise ImportError("模拟 shap 未安装")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = explain(c, x, method="auto")
    assert (out["method"] == "gain").all()


def test_explain_unknown_method_raises():
    c, x = _fitted()
    with pytest.raises(ValueError):
        explain(c, x, method="nonsense")
