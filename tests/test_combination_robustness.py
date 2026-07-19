"""组合层健壮性:因子库常含空因子/覆盖异质,组合器须丢弃退化因子 + 外连接容缺,
不能因个别因子拖垮整个 OOS run(实测 csi300 池化库触发过两类崩溃)。"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from factorzen.research.combination.cv import PurgedWalkForwardCV
from factorzen.research.combination.models import combine_lgbm
from factorzen.research.combination.oos import combine_oos


def _panel(n_days=120, n_stocks=40, seed=0, stocks=None):
    """ret = 0.7*fa - 0.3*fb + 噪声。stocks 指定则只在这些股票上出因子(制造覆盖异质)。"""
    rng = np.random.default_rng(seed)
    dates = [f"2025{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n_days)]
    codes = [f"{s:04d}.SZ" for s in range(n_stocks)]
    ra, rb, rr = [], [], []
    for d in dates:
        fa = rng.standard_normal(n_stocks)
        fb = rng.standard_normal(n_stocks)
        ret = 0.7 * fa - 0.3 * fb + rng.standard_normal(n_stocks) * 0.3
        for s in range(n_stocks):
            rr.append({"trade_date": d, "ts_code": codes[s], "ret": float(ret[s])})
            ra.append({"trade_date": d, "ts_code": codes[s], "factor_value": float(fa[s])})
            rb.append({"trade_date": d, "ts_code": codes[s], "factor_value": float(fb[s])})
    return {"fa": pl.DataFrame(ra), "fb": pl.DataFrame(rb)}, pl.DataFrame(rr), codes


def test_lgbm_drops_empty_factor_instead_of_crashing():
    """库里混入一个物化为空(0 行)的因子(陈旧基本面 ts_skew 退化)→ 丢弃它、用其余因子照常组合。

    复现:实测 csi300 池化库里 `ts_skew(netprofit_margin,60)` 物化 0 行 → inner join 塌空
    → 「训练面板为空」崩。修后应丢弃空因子、正常产出。
    """
    factor_dfs, ret_df, _ = _panel()
    factor_dfs["empty"] = pl.DataFrame(
        schema={"trade_date": pl.Utf8, "ts_code": pl.Utf8, "factor_value": pl.Float64}
    )
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    out = combine_lgbm(factor_dfs, ret_df, cv, n_estimators=30, min_child_samples=20)
    assert out.height > 0


def test_lgbm_survives_heterogeneous_coverage():
    """两因子覆盖的股票**不相交** → inner join 全空(train 与 test 都空)→ 原实现崩。

    复现:实测覆盖率异质(部分因子 ~71k 行 vs 满 144k)→ 某 test 折 inner join 空
    → lightgbm「Input data must be 2 dimensional and non empty」崩。
    修后:外连接取并集、缺失特征交给 LGBM 原生 NaN 处理 → 覆盖并集、不崩。
    """
    fa_dfs, ret_df, codes = _panel(n_stocks=40, seed=1)
    half = len(codes) // 2
    left, right = set(codes[:half]), set(codes[half:])
    fa = fa_dfs["fa"].filter(pl.col("ts_code").is_in(left))
    fb = fa_dfs["fb"].filter(pl.col("ts_code").is_in(right))
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    out = combine_lgbm({"fa": fa, "fb": fb}, ret_df, cv, n_estimators=30, min_child_samples=20)
    assert out.height > 0
    covered = set(out["ts_code"].to_list())
    assert covered & left and covered & right  # 并集覆盖两组,不是只剩交集


def test_lgbm_all_null_factors_raise_clear_error():
    """所有因子都全缺 → 去除退化因子后 0 个有效 → 明确报错(不静默产垃圾)。"""
    factor_dfs, ret_df, _ = _panel(n_days=80)
    for k in list(factor_dfs):
        factor_dfs[k] = factor_dfs[k].with_columns(
            pl.lit(None, dtype=pl.Float64).alias("factor_value")
        )
    cv = PurgedWalkForwardCV(train_days=40, test_days=20, purge_days=5)
    with pytest.raises(ValueError):
        combine_lgbm(factor_dfs, ret_df, cv, n_estimators=20)


def test_combine_oos_survives_heterogeneous_coverage():
    """线性路径同样容缺:覆盖不相交的两因子外连接 + 缺失补 0(中性)→ 覆盖并集不崩。"""
    fa_dfs, ret_df, codes = _panel(n_stocks=40, seed=2)
    half = len(codes) // 2
    fa = fa_dfs["fa"].filter(pl.col("ts_code").is_in(set(codes[:half])))
    fb = fa_dfs["fb"].filter(pl.col("ts_code").is_in(set(codes[half:])))
    cv = PurgedWalkForwardCV(train_days=60, test_days=20, purge_days=5)
    out = combine_oos({"fa": fa, "fb": fb}, ret_df, cv, method="ic_weighted")
    assert out.height > 0
    covered = set(out["ts_code"].to_list())
    assert covered & set(codes[:half]) and covered & set(codes[half:])


def test_duplicate_join_keys_do_not_explode_rows():
    """因子面板含少量重复 (trade_date, ts_code) 时，链式 outer join 不得笛卡尔积爆炸。

    **实测根因（2026-07-19）**：`_zscore_and_merge` 把 k 个因子链式 full join，
    每次遇重复键行数相乘。生产物化产物里几乎每个因子面板都有 3 行重复
    （2026-06-30 的 6 只 603xxx，每键 4 条），重复率仅 0.0006%——
    但 62 个因子链式 join 下按 4^n 放大：逐步打点实测

        join #5: 6786 行 → join #10: 1,051,266 行（5 次 join 涨 155 倍）

    最终 anon-rss 打满 23GB 被 OOM killer 杀。P1-① 阶段 2 的四次尝试全折在这里。

    键唯一性是链式 join 的**前提**，不该假设成立而不校验——上游任何一处重复
    都会被指数放大成 OOM。
    """
    from factorzen.research.combination.methods import _zscore_and_merge

    n_days, n_stocks, k = 6, 20, 8
    dates = [f"2025010{i + 1}" for i in range(n_days)]
    codes = [f"{s:04d}.SZ" for s in range(n_stocks)]
    rng = np.random.default_rng(3)

    factor_dfs = {}
    for j in range(k):
        rows = [
            {"trade_date": d, "ts_code": c, "factor_value": float(rng.standard_normal())}
            for d in dates
            for c in codes
        ]
        # 每个因子在同一个 (日, 股) 上多出 3 条重复（模拟生产的 4 条/键）
        dup_key = (dates[0], codes[0])
        rows += [
            {"trade_date": dup_key[0], "ts_code": dup_key[1],
             "factor_value": float(rng.standard_normal())}
            for _ in range(3)
        ]
        factor_dfs[f"f{j}"] = pl.DataFrame(rows)

    merged, names = _zscore_and_merge(factor_dfs)

    assert len(names) == k
    # 唯一键数就是行数上界；未防御时这里是 4**8 = 65536 量级
    assert merged.height == n_days * n_stocks, (
        f"链式 join 把 {k} 个因子的重复键放大到 {merged.height} 行"
        f"（应为 {n_days * n_stocks}）"
    )
    assert merged.select(["trade_date", "ts_code"]).unique().height == merged.height


def test_duplicate_join_keys_emit_warning():
    """重复键必须告警——静默去重会掩盖上游数据缺陷（本例真实来源仍未查清）。"""
    from factorzen.research.combination.methods import _zscore_and_merge

    base = [
        {"trade_date": "20250101", "ts_code": "0001.SZ", "factor_value": 1.0},
        {"trade_date": "20250101", "ts_code": "0002.SZ", "factor_value": 2.0},
    ]
    dfs = {
        "clean": pl.DataFrame(base),
        "dirty": pl.DataFrame([
            *base,
            {"trade_date": "20250101", "ts_code": "0001.SZ", "factor_value": 9.0},
        ]),
    }
    with pytest.warns(UserWarning, match="重复"):
        merged, _ = _zscore_and_merge(dfs)
    assert merged.height == 2
