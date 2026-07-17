"""W0-fix-1：build_panel 覆盖警告改口径——逐列非空率，非行齐全率。"""
from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from factorzen.research.combination.models import (
    LOW_FEATURE_COVERAGE_WARN,
    _warn_incomplete,
    build_panel,
)


def _feat(name: str, n: int, coverage: float, *, seed: int = 0) -> pl.DataFrame:
    """coverage = 非空比例；其余为 null。"""
    rng = np.random.default_rng(seed)
    n_ok = max(0, round(n * coverage))
    vals = [float(x) for x in rng.normal(size=n_ok)] + [None] * (n - n_ok)
    rng.shuffle(vals)
    return pl.DataFrame({
        "trade_date": [f"202001{i+1:02d}" for i in range(n)],
        "ts_code": ["000001.SZ"] * n,
        "factor_value": vals,
    })


def _ret(n: int) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [f"202001{i+1:02d}" for i in range(n)],
        "ts_code": ["000001.SZ"] * n,
        "ret": [0.01] * n,
    })


def test_low_feature_coverage_constant():
    assert LOW_FEATURE_COVERAGE_WARN == 0.30


def test_warn_when_one_column_below_30pct():
    """一列 20% 覆盖 → warn，文案含该列名。"""
    n = 100
    # 两列健康、一列 20%
    dfs = {
        "ok_a": _feat("ok_a", n, 0.9, seed=1),
        "ok_b": _feat("ok_b", n, 0.8, seed=2),
        "sparse_x": _feat("sparse_x", n, 0.20, seed=3),
    }
    # full join 后行齐全率几乎必 <70%（互补稀疏）——旧口径恒 warn
    panel = build_panel(dfs, _ret(n))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_incomplete(panel)
        msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert msgs, "应触发逐列覆盖警告"
    joined = " ".join(msgs)
    assert "sparse_x" in joined
    assert "20%" in joined or "0.2" in joined or "20" in joined


def test_no_warn_when_all_cols_above_30pct_even_if_row_complete_low():
    """全列 ≥30% 非空 → 不 warn，即便行齐全率 <70%（回归旧恒真行为）。"""
    n = 100
    # 三列各 50% 但互补缺失 → 行齐全率很低，但 min 列覆盖 =50% ≥30%
    rng = np.random.default_rng(42)
    dates = [f"202001{i+1:02d}" for i in range(n)]
    codes = ["000001.SZ"] * n

    def col_half(seed: int) -> list:
        mask = rng.random(n) < 0.5
        return [float(rng.normal()) if m else None for m in mask]

    # 手工拼宽表走 _warn_incomplete（与 build_panel 同路径）
    from factorzen.research.combination.models import _factor_panel, _join_ret

    feat = {
        "f1": pl.DataFrame({
            "trade_date": dates, "ts_code": codes, "factor_value": col_half(1),
        }),
        "f2": pl.DataFrame({
            "trade_date": dates, "ts_code": codes, "factor_value": col_half(2),
        }),
        "f3": pl.DataFrame({
            "trade_date": dates, "ts_code": codes, "factor_value": col_half(3),
        }),
    }
    wide = _join_ret(_factor_panel(feat), _ret(n))
    # 确认行齐全率 <70%（旧口径会 warn）
    names = [c for c in wide.columns if c not in ("trade_date", "ts_code", "ret")]
    complete_pct = wide.drop_nulls(subset=names).height / wide.height
    assert complete_pct < 0.7, f"本测依赖行齐全率低，得到 {complete_pct}"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_incomplete(wide)
        cov_warns = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "build_panel" in str(x.message)
        ]
    assert cov_warns == [], f"全列≥30% 不应 warn: {[str(x.message) for x in cov_warns]}"
