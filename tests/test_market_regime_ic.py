"""测试分市场状态 IC：按市场上涨/下跌/高波等状态分组计算 IC。"""

import polars as pl

from daily.evaluation.advanced import MarketRegimeICResult, compute_market_regime_ic


def _make_regime_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """构造因子收益数据和市场状态数据。"""
    stocks = [f"s{i}" for i in range(30)]
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    factor_rows = []
    for d in dates:
        for s in stocks:
            factor_rows.append({"trade_date": d, "ts_code": s})
    factor = pl.DataFrame(factor_rows).with_columns([
        pl.Series("factor_value", [i / 30 for i in range(30)] * 3),
        pl.Series("fwd_ret", [i / 30 * 0.02 for i in range(30)] * 3),
    ])
    # 市场状态：上涨日、下跌日、震荡日
    market = pl.DataFrame({
        "trade_date": dates,
        "market_return": [0.02, -0.02, 0.001],
        "market_volatility": [0.15, 0.25, 0.08],
    })
    return factor, market


def test_market_regime_ic_returns_dataframe():
    """compute_market_regime_ic 返回包含各状态 IC 的 DataFrame。"""
    factor_df, market_df = _make_regime_data()
    result = compute_market_regime_ic(
        factor_df=factor_df,
        market_df=market_df,
        factor_col="factor_value",
        ret_col="fwd_ret",
        regime_type="direction",  # up/down
    )
    assert isinstance(result, pl.DataFrame)
    assert "regime" in result.columns
    assert "ic" in result.columns


def test_market_regime_ic_two_directions():
    """direction 模式应包含 up 和 down 两个状态。"""
    factor_df, market_df = _make_regime_data()
    result = compute_market_regime_ic(
        factor_df=factor_df, market_df=market_df,
        factor_col="factor_value", ret_col="fwd_ret",
        regime_type="direction",
    )
    regimes = set(result["regime"].to_list())
    assert "up" in regimes or "bull" in regimes, "上涨状态应存在"
    assert "down" in regimes or "bear" in regimes, "下跌状态应存在"


def test_market_regime_ic_returns_result_object():
    """compute_market_regime_ic 可返回 MarketRegimeICResult 对象。"""
    factor_df, market_df = _make_regime_data()
    result = compute_market_regime_ic(
        factor_df=factor_df, market_df=market_df,
        factor_col="factor_value", ret_col="fwd_ret",
        regime_type="direction", return_object=True,
    )
    assert isinstance(result, MarketRegimeICResult)
    assert hasattr(result, "regime_ic")
    assert hasattr(result, "regime_type")


def test_market_regime_ic_by_volatility():
    """volatility 模式按波动率状态分组。"""
    factor_df, market_df = _make_regime_data()
    result = compute_market_regime_ic(
        factor_df=factor_df, market_df=market_df,
        factor_col="factor_value", ret_col="fwd_ret",
        regime_type="volatility", n_regimes=3,
    )
    assert isinstance(result, pl.DataFrame)
    assert result.height <= 3