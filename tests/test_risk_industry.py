import polars as pl


def make_stocks(n_stocks=8):
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    industries = ["银行", "医药", "电子", "食品饮料"]
    return pl.DataFrame({
        "ts_code": codes,
        "industry": [industries[i % len(industries)] for i in range(n_stocks)],
    })


def test_industry_dummies_one_hot_per_stock():
    from factorzen.risk.industry_factors import get_industry_dummies
    dummies = get_industry_dummies(make_stocks())
    ind_cols = [c for c in dummies.columns if c.startswith("ind_")]
    assert len(ind_cols) == 4  # 4 个唯一行业
    # 每只股票恰属一个行业：ind_* 列之和 == 1
    row_sums = dummies.select(ind_cols).sum_horizontal()
    assert row_sums.to_list() == [1.0] * dummies.height


def test_industry_names_sorted_bare():
    from factorzen.risk.industry_factors import get_industry_names
    names = get_industry_names(make_stocks())
    assert names == sorted(names)
    assert set(names) == {"银行", "医药", "电子", "食品饮料"}
    assert all(not n.startswith("ind_") for n in names)  # 裸名，无前缀


def test_industry_dummies_missing_col_raises():
    import pytest

    from factorzen.risk.industry_factors import get_industry_dummies
    with pytest.raises(ValueError):
        get_industry_dummies(pl.DataFrame({"ts_code": ["000001.SZ"]}))
