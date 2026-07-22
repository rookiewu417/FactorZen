"""S2：sleeve / quantile_group 权重语义手算断言（离线字面量面板）。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from factorzen.strategies.quantile_group import (
    build_group_weights,
    generate_quantile_group_products,
)
from factorzen.strategies.sleeve import build_sleeve_weights, generate_sleeve_products


def test_sleeve_weights_hand_computed(tmp_path):
    """手写微型因子面板 → 选中票、权重、Σw 与草稿语义手算一致。

    场景（top_n=2, holding_days=2, direction=top）::

        d0: A=3, B=2, C=1 → top2 = A,B；层权 1/(2*2)=0.25
        d1: A=1, B=3, C=2 → top2 = B,C；层权 0.25
        d2: A=2, B=1, C=3 → top2 = C,A；层权 0.25

    期望::

        d0: A=0.25, B=0.25                     Σ=0.50（预热 1 层）
        d1: A=0.25, B=0.25+0.25=0.50, C=0.25  Σ=1.00
        d2: B=0.25, C=0.25+0.25=0.50, A=0.25  Σ=1.00
    """
    d0, d1, d2 = date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)
    scores = pl.DataFrame(
        {
            "trade_date": [d0, d0, d0, d1, d1, d1, d2, d2, d2],
            "ts_code": ["A.SZ", "B.SZ", "C.SZ"] * 3,
            "score": [3.0, 2.0, 1.0, 1.0, 3.0, 2.0, 2.0, 1.0, 3.0],
        }
    )
    trade_dates = [d0, d1, d2]
    weights = build_sleeve_weights(
        scores,
        score_col="score",
        top_n=2,
        holding_days=2,
        trade_dates=trade_dates,
        direction="top",
    )

    def _wmap(d: date) -> dict[str, float]:
        df = weights[d]
        return dict(zip(df["ts_code"].to_list(), df["target_weight"].to_list(), strict=True))

    m0 = _wmap(d0)
    assert set(m0) == {"A.SZ", "B.SZ"}
    assert abs(m0["A.SZ"] - 0.25) < 1e-12
    assert abs(m0["B.SZ"] - 0.25) < 1e-12
    assert abs(sum(m0.values()) - 0.50) < 1e-12

    m1 = _wmap(d1)
    assert abs(m1["A.SZ"] - 0.25) < 1e-12
    assert abs(m1["B.SZ"] - 0.50) < 1e-12
    assert abs(m1["C.SZ"] - 0.25) < 1e-12
    assert abs(sum(m1.values()) - 1.0) < 1e-12

    m2 = _wmap(d2)
    assert abs(m2["A.SZ"] - 0.25) < 1e-12
    assert abs(m2["B.SZ"] - 0.25) < 1e-12
    assert abs(m2["C.SZ"] - 0.50) < 1e-12
    assert abs(sum(m2.values()) - 1.0) < 1e-12

    # 产物形态
    run_dirs = generate_sleeve_products(
        str(tmp_path / "sleeve"),
        scores,
        score_col="score",
        top_n=2,
        holding_days=2,
        trade_dates=trade_dates,
    )
    assert len(run_dirs) == 3
    w = pl.read_parquet(Path(run_dirs[1]) / "weights.parquet")
    assert abs(float(w["target_weight"].sum()) - 1.0) < 1e-12
    mf = json.loads((Path(run_dirs[1]) / "manifest.json").read_text())
    assert mf["signal_date"] == d1.isoformat()
    assert mf["status"] == "optimal"


def test_quantile_group_weights_hand_computed_and_degenerate(tmp_path):
    """分位组：组内等权 + 截面不足 n_groups 退化不建仓。

    n_groups=3, group=1（最高组）::

        d0 五票分数 1,2,3,4,5 → rank 升序 1..5
            g = (rk-1)*3//5 → 0,0,1,1,2 → 最高组 g=2 → E(5)
            组内等权 w=1.0
        d1 仅 2 票（<3）→ 不建仓
    """
    d0, d1 = date(2023, 1, 3), date(2023, 1, 4)
    scores = pl.DataFrame(
        {
            "trade_date": [d0] * 5 + [d1, d1],
            "ts_code": [
                "A.SZ",
                "B.SZ",
                "C.SZ",
                "D.SZ",
                "E.SZ",
                "A.SZ",
                "B.SZ",
            ],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 8.0],
        }
    )
    weights = build_group_weights(
        scores, score_col="score", n_groups=3, group=1
    )
    assert d0 in weights
    m0 = dict(
        zip(
            weights[d0]["ts_code"].to_list(),
            weights[d0]["target_weight"].to_list(),
            strict=True,
        )
    )
    assert set(m0) == {"E.SZ"}
    assert abs(m0["E.SZ"] - 1.0) < 1e-12
    assert abs(sum(m0.values()) - 1.0) < 1e-12

    # 组内不足 n_groups：d1 不应出现
    assert d1 not in weights

    # 中间组 group=2 → code_group=1 → C,D 等权 0.5
    mid = build_group_weights(scores, score_col="score", n_groups=3, group=2)
    m_mid = dict(
        zip(
            mid[d0]["ts_code"].to_list(),
            mid[d0]["target_weight"].to_list(),
            strict=True,
        )
    )
    assert set(m_mid) == {"C.SZ", "D.SZ"}
    assert abs(m_mid["C.SZ"] - 0.5) < 1e-12
    assert abs(m_mid["D.SZ"] - 0.5) < 1e-12

    # 产物 + 显式 trade_dates 覆盖空仓日
    run_dirs = generate_quantile_group_products(
        str(tmp_path / "qg"),
        scores,
        score_col="score",
        n_groups=3,
        group=1,
        trade_dates=[d0, d1],
    )
    assert len(run_dirs) == 2
    w0 = pl.read_parquet(Path(run_dirs[0]) / "weights.parquet")
    assert w0.height == 1 and abs(float(w0["target_weight"][0]) - 1.0) < 1e-12
    w1 = pl.read_parquet(Path(run_dirs[1]) / "weights.parquet")
    assert w1.height == 0, "截面不足 n_groups 应空仓"
    mf = json.loads((Path(run_dirs[0]) / "manifest.json").read_text())
    assert mf["signal_date"] == d0.isoformat() and mf["status"] == "optimal"
