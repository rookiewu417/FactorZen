"""多因子合成方法。

估权(estimate_*)与应用(apply_weights)拆开:估权只吃「因子 + 前向收益」产出权重向量,
应用只吃「因子 + 权重」产出合成因子。三种公开方法(equal_weight/ic_weighted/max_ir)
是「估权 + 应用」的薄包装;OOS 协议(oos.combine_oos)则逐折用 train 估权、test 应用。
所有方法在截面 z-score 化后的因子值上操作,输入需含 trade_date, ts_code, factor_value。

性能:IC 序列按日独立(RankIC 截面),可全样本算一次再按 train 日期切片;
截面 z-score 同样按日独立,可全样本标准化一次再切片。OOS 热路径走预计算缓存。
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from factorzen.core.stats import spearman_avg_rank

# IC 缓存: {factor_name: (dates_sorted list[str], ic ndarray)}
IcCache = dict[str, tuple[list[str], np.ndarray]]


def _zscore_factor(df: pl.DataFrame, col: str = "factor_value") -> pl.DataFrame:
    """截面 z-score 标准化。"""
    return (
        df.with_columns(
            [
                pl.col(col).mean().over("trade_date").alias("_mean"),
                pl.col(col).std(ddof=1).over("trade_date").alias("_std"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_std") > 0)
            .then((pl.col(col) - pl.col("_mean")) / pl.col("_std"))
            .otherwise(0.0)
            .alias(col)
        )
        .drop(["_mean", "_std"])
    )


def _rank_ic_numpy(fv: np.ndarray, rv: np.ndarray) -> float | None:
    """单日 RankIC: average-rank Spearman（core.stats）；截面 n<10 跳过。"""
    if fv.size < 10:
        return None
    return spearman_avg_rank(fv, rv)


def _compute_ic_dated(
    factor_df: pl.DataFrame,
    ret_df: pl.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """计算因子 vs 前向收益的截面 IC 序列,返回 (dates_sorted, ics)。

    RankIC 按日独立,全样本一次计算后可按 train 日期切片,数值与逐段重算一致。
    实现:一次 join + 按日排序后纯 numpy 扫组,避免 polars group_by 逐日物化。
    """
    merged = (
        factor_df.select(["trade_date", "ts_code", "factor_value"])
        .rename({"factor_value": "_fv"})
        .with_columns(pl.col("trade_date").cast(pl.Utf8))
        .join(
            ret_df.select(["trade_date", "ts_code", "ret"])
            .rename({"ret": "_ret"})
            .with_columns(pl.col("trade_date").cast(pl.Utf8)),
            on=["trade_date", "ts_code"],
            how="inner",
        )
        .drop_nulls(subset=["_fv", "_ret"])
    )
    if merged.height == 0:
        return [], np.array([], dtype=float)

    # 按日期稳定排序,再扫连续组
    merged = merged.sort("trade_date")
    dates_arr = merged["trade_date"].to_numpy()
    fv_all = merged["_fv"].to_numpy().astype(float, copy=False)
    rv_all = merged["_ret"].to_numpy().astype(float, copy=False)

    ic_dates: list[str] = []
    ic_vals: list[float] = []
    n = len(dates_arr)
    i = 0
    while i < n:
        j = i + 1
        d = dates_arr[i]
        while j < n and dates_arr[j] == d:
            j += 1
        ic = _rank_ic_numpy(fv_all[i:j], rv_all[i:j])
        if ic is not None:
            ic_dates.append(str(d))
            ic_vals.append(ic)
        i = j
    return ic_dates, np.asarray(ic_vals, dtype=float)


def _compute_ic_series(
    factor_df: pl.DataFrame,
    ret_df: pl.DataFrame,
    factor_name: str,
) -> np.ndarray:
    """计算因子 vs 前向收益的截面 IC 序列（Pearson(rank(f), rank(r))）。"""
    _ = factor_name  # 保留签名兼容调用方
    _, ics = _compute_ic_dated(factor_df, ret_df)
    return ics


def build_ic_cache(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
) -> IcCache:
    """全样本一次算各因子 IC 序列(按日),供 OOS 多方法/多折复用。"""
    rdf = ret_df.with_columns(pl.col("trade_date").cast(pl.Utf8))
    return {
        name: _compute_ic_dated(
            df.with_columns(pl.col("trade_date").cast(pl.Utf8)), rdf
        )
        for name, df in factor_dfs.items()
    }


def _slice_ic_to_train(
    dates: list[str], ics: np.ndarray, train_dates: set[str]
) -> np.ndarray:
    """从全样本 IC 中按 train 日期集合切片(保时序)。"""
    if not dates:
        return np.array([], dtype=float)
    return np.asarray(
        [ic for d, ic in zip(dates, ics, strict=True) if d in train_dates],
        dtype=float,
    )


def pre_zscore_factors(
    factor_dfs: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """各因子全样本截面 z-score(按日独立,可再按 fold 切片)。"""
    out: dict[str, pl.DataFrame] = {}
    for name, df in factor_dfs.items():
        z = _zscore_factor(
            df.select(["trade_date", "ts_code", "factor_value"]).with_columns(
                pl.col("trade_date").cast(pl.Utf8)
            )
        )
        out[name] = z
    return out


def _zscore_and_merge(
    factor_dfs: dict[str, pl.DataFrame],
    *,
    already_zscored: bool = False,
) -> tuple[pl.DataFrame, list[str]]:
    """各因子截面 z-score 后 **outer join** 成宽表(列名 `_f_<name>`),缺失补 0。

    因子库覆盖常异质(不同因子覆盖的股票/日期不同)。inner join 会把并集缩到交集、
    甚至塌空;改外连接取并集,某股票缺某因子时该因子补 0(z-score 后 0=截面均值=中性),
    等价于「缺失因子不表态」,不至于整行被丢或组合崩。

    **键唯一性是链式 join 的前提,必须校验**(2026-07-19 实测 OOM 根因):
    每次 full join 遇重复键行数相乘,k 个因子链式下按 重复数^k 指数放大。
    生产物化产物里几乎每个因子面板都有 3 行重复(2026-06-30 的 6 只 603xxx,
    每键 4 条),重复率仅 0.0006%——62 个因子链式 join 逐步打点实测
    join#5 6786 行 → join#10 105 万行,最终 anon-rss 打满 23GB 被 OOM killer 杀。
    故 join 前按 (trade_date, ts_code) 去重(保留首行,确定性),
    并**汇总告警**:静默去重会掩盖上游数据缺陷。
    """
    if not factor_dfs:
        raise ValueError("factor_dfs 不能为空")
    normed = []
    dup_report: list[tuple[str, int]] = []
    for name, df in factor_dfs.items():
        base = df.select(["trade_date", "ts_code", "factor_value"])
        n_raw = base.height
        # maintain_order 保证「保留首行」在多次运行间一致(可复现铁律)
        base = base.unique(subset=["trade_date", "ts_code"], keep="first", maintain_order=True)
        if base.height < n_raw:
            dup_report.append((name, n_raw - base.height))
        z = base if already_zscored else _zscore_factor(base)
        normed.append(z.rename({"factor_value": f"_f_{name}"}))
    if dup_report:
        total = sum(n for _, n in dup_report)
        head = ", ".join(f"{nm}({n})" for nm, n in dup_report[:5])
        more = f" 等 {len(dup_report)} 个因子" if len(dup_report) > 5 else ""
        warnings.warn(
            f"_zscore_and_merge: {len(dup_report)} 个因子面板含重复 "
            f"(trade_date, ts_code)，共 {total} 行，已按首行去重后再 join："
            f"{head}{more}。链式 outer join 会把重复按 重复数^因子数 放大"
            f"（实测 62 因子 × 每键 4 条 → OOM），**上游重复源应单独排查**。",
            UserWarning,
            stacklevel=2,
        )
    merged = normed[0]
    for z in normed[1:]:
        merged = merged.join(z, on=["trade_date", "ts_code"], how="full", coalesce=True)
    fcols = [f"_f_{n}" for n in factor_dfs]
    merged = merged.with_columns([pl.col(c).fill_null(0.0) for c in fcols])
    return merged, list(factor_dfs.keys())


def apply_weights(
    factor_dfs: dict[str, pl.DataFrame],
    weights: dict[str, float],
    *,
    already_zscored: bool = False,
) -> pl.DataFrame:
    """按给定权重加权合成(默认先各因子截面 z-score)。

    Args:
        factor_dfs: {factor_name: DataFrame(trade_date, ts_code, factor_value)}
        weights: {factor_name: weight}
        already_zscored: True 时跳过 z-score(输入已是截面标准化后的值)

    Returns:
        DataFrame(trade_date, ts_code, factor_value) — 加权合成因子
    """
    merged, names = _zscore_and_merge(factor_dfs, already_zscored=already_zscored)
    exprs = [pl.col(f"_f_{n}") * weights[n] for n in names]
    expr: pl.Expr = exprs[0]
    for e in exprs[1:]:
        expr = expr + e
    return merged.with_columns(expr.alias("factor_value")).select(
        ["trade_date", "ts_code", "factor_value"]
    )


def estimate_equal_weights(factor_dfs: dict[str, pl.DataFrame]) -> dict[str, float]:
    """等权:每因子 1/k。"""
    if not factor_dfs:
        raise ValueError("factor_dfs 不能为空")
    k = len(factor_dfs)
    return {n: 1.0 / k for n in factor_dfs}


def estimate_ic_weights(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    ic_window: int = 60,
    *,
    ic_cache: IcCache | None = None,
    train_dates: set[str] | None = None,
    allow_negative: bool = False,
) -> dict[str, float]:
    """IC 加权:权重 = max(0, IC_mean) 归一化;全非正则退化等权。

    Args:
        ic_window: 计算 IC 的最近窗口天数(-1 表示全历史)。
        ic_cache: 可选全样本 IC 缓存;与 train_dates 联用时按 train 切片,避免重算。
        train_dates: 训练集日期集合(与 ic_cache 联用)。
        allow_negative: 允许负权(见下)。默认 False = 现语义,A 股基线零回归。

    ``allow_negative=True``:权重 = IC_mean **不裁剪**,按 **L1**(``Σ|w|=1``)归一化。
    动机(P1-①):准入用残差口径、部署用裸值,已实锤裸 IC 为负的因子够格准入;
    裁到 0 等于把它携带的信息整条丢掉,让它带负权才是"权重自己处理符号"。
    归一化必须换 L1——负权下 ``Σw`` 可能≈0,除之会让权重爆炸甚至整体翻号。

    ⚠️ **该动机已被实测证伪,默认保持 False**:真实库 OOS 上 signed 版反而更差
    (0.0276 vs clipped 0.0374)。裁剪是有效的隐式正则,不是历史包袱。
    详见 ``_solve_max_ir_weights`` 里的完整对照表与解释。
    """
    weights: dict[str, float] = {}
    for name, df in factor_dfs.items():
        if ic_cache is not None and name in ic_cache and train_dates is not None:
            dates, ics = ic_cache[name]
            ic_series = _slice_ic_to_train(dates, ics, train_dates)
        elif ic_cache is not None and name in ic_cache and train_dates is None:
            ic_series = ic_cache[name][1]
        else:
            ic_series = _compute_ic_series(df, ret_df, name)
        if len(ic_series) == 0:
            weights[name] = 0.0
        else:
            tail = ic_series[-ic_window:] if ic_window > 0 else ic_series
            raw = float(np.mean(tail))
            weights[name] = raw if allow_negative else max(0.0, raw)
    # 归一化基数:signed 走 L1(Σ|w|),clipped 沿用 Σw(此时二者等价,逐位不变)
    total_w = sum(abs(w) for w in weights.values()) if allow_negative \
        else sum(weights.values())
    if total_w < 1e-12:
        return {n: 1.0 / len(factor_dfs) for n in factor_dfs}
    return {n: w / total_w for n, w in weights.items()}


def _solve_max_ir_weights(
    mu: np.ndarray, sigma: np.ndarray, *, allow_negative: bool = False,
) -> np.ndarray:
    """max-IR 闭式解 ``w ∝ Σ⁻¹μ`` 的求解 + 归一化(纯数值,便于手算对拍)。

    ``allow_negative=False``(默认):裁到非负再按 ``Σw`` 归一化——现语义。
    ``allow_negative=True``:保留闭式解符号,按 **L1** 归一化。
    退化(和≈0)一律回等权,不除零。

    ⚠️ **实测结论:``allow_negative=True`` 在真实库上更差,故默认保持 False。**
    csi300 / 2020-2026 / 85 因子(其中 21 条 ``ic_train`` 为负)六方法同协议 OOS:

    ====================  =============  =====
    method                rank_ic_mean    ICIR
    ====================  =============  =====
    equal_weight               0.0587     0.241
    max_ir                     0.0536     0.247
    ic_weighted                0.0374     0.146
    ic_weighted_signed         0.0276     0.102
    max_ir_signed              0.0186     0.139
    ====================  =============  =====

    即"裁剪到非负"**不是**未经审视的历史包袱,而是有效的**隐式正则**
    ——与 Jagannathan & Ma (2003)"禁止做空约束等价于协方差收缩"一致:
    ``Σ⁻¹μ`` 的闭式最优性建立在 μ/Σ 估准的前提上,而 85 个因子的 IC 估计噪声很大,
    放开负权只是让噪声被放大。等权跑赢一切也是同一现象(1/N 难以击败)。
    保留本参数是为了让这个结论**可复现、可再检验**,不是推荐使用。
    """
    k = len(mu)
    try:
        sigma_inv = np.linalg.inv(sigma + np.eye(k) * 1e-6)
    except np.linalg.LinAlgError:
        sigma_inv = np.eye(k)
    w_raw = sigma_inv @ mu
    if allow_negative:
        denom = float(np.abs(w_raw).sum())
        if denom < 1e-12:
            return np.ones(k) / k
        return w_raw / denom
    w_pos = np.maximum(w_raw, 0.0)
    if w_pos.sum() < 1e-12:
        w_pos = np.ones(k)
    return w_pos / w_pos.sum()


def estimate_max_ir_weights(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    lookback: int = 120,
    *,
    ic_cache: IcCache | None = None,
    train_dates: set[str] | None = None,
    allow_negative: bool = False,
) -> dict[str, float] | None:
    """最大化 IR 闭式解 w = Σ⁻¹μ(Ledoit-Wolf 收缩)。数据不足返回 None(调用方退化等权)。

    ``allow_negative``:见 ``_solve_max_ir_weights``。默认 False = 现语义(裁到非负)。
    """
    names = list(factor_dfs.keys())
    k = len(names)
    ic_series_map: dict[str, np.ndarray] = {}
    min_len: int | None = None
    for name, df in factor_dfs.items():
        if ic_cache is not None and name in ic_cache and train_dates is not None:
            dates, ics = ic_cache[name]
            ic = _slice_ic_to_train(dates, ics, train_dates)
        elif ic_cache is not None and name in ic_cache and train_dates is None:
            ic = ic_cache[name][1]
        else:
            ic = _compute_ic_series(df, ret_df, name)
        ic_series_map[name] = ic
        if min_len is None or len(ic) < min_len:
            min_len = len(ic)
    if min_len is None or min_len < k + 1:
        return None
    tail_len = min(lookback, min_len)
    ic_mat = np.column_stack([ic_series_map[n][-tail_len:] for n in names])  # (T, K)
    mu = ic_mat.mean(axis=0)
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore[import]

        sigma = LedoitWolf().fit(ic_mat).covariance_
    except ImportError:
        sigma = np.cov(ic_mat, rowvar=False) + np.eye(k) * 1e-6
    w = _solve_max_ir_weights(mu, sigma, allow_negative=allow_negative)
    return dict(zip(names, w.tolist(), strict=True))


def equal_weight(factor_dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """等权合成(薄包装:估等权 + 应用)。"""
    return apply_weights(factor_dfs, estimate_equal_weights(factor_dfs))


def ic_weighted(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    ic_window: int = 60,
) -> pl.DataFrame:
    """IC 加权合成(薄包装;in-sample 研究口径,OOS 请用 oos.combine_oos)。"""
    return apply_weights(factor_dfs, estimate_ic_weights(factor_dfs, ret_df, ic_window))


def max_ir(
    factor_dfs: dict[str, pl.DataFrame],
    ret_df: pl.DataFrame,
    lookback: int = 120,
) -> pl.DataFrame:
    """最大化 IR 合成(薄包装;数据不足退化等权)。"""
    w = estimate_max_ir_weights(factor_dfs, ret_df, lookback)
    if w is None:
        return equal_weight(factor_dfs)
    return apply_weights(factor_dfs, w)
