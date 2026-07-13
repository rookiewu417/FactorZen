# src/factorzen/discovery/residual.py
"""候选因子对库内 active 因子的同日截面残差化 + 残差 Rank IC。

PIT 安全契约
------------
**全部运算在单日截面内完成**，无跨日状态、无跨日拟合。对每个交易日 t 独立：

1. 库因子 X_t 做截面 z-score，null→0（与 combination ``_zscore_and_merge`` 缺失补 0 同口径）
2. 候选 y_t 只取非空有限行
3. β_t = lstsq([1 | X_t[y 非空行]], y_t)（含截距）
4. 残差 r_t = y_t − X̂_t @ β_t
5. Spearman(r_t, fwd_ret_t) 写入 IC 序列

日守卫：候选有效行 n_t < max(_MIN_CROSS_SAMPLES, k+10) → 跳过该日。
全日跳过 → residual IC = NaN、n_days=0（走覆盖门语义）。

为何库 upsert/rebuild **不**用残差口径
------------------------------------
因子库是参照系：对参照系自身做「对库残差化」是循环定义。库准入维持裸 IC + 覆盖门
（见 ``factor_library.upsert`` / ``rebuild``）；残差目标只用于**挖掘评估**，测「对库的真增量」。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from factorzen.daily.evaluation.ic_analysis import _MIN_CROSS_SAMPLES

# 与 ic_analysis 同门槛；日守卫再取 max(本值, k+10)
MIN_CROSS_SAMPLES = _MIN_CROSS_SAMPLES


@dataclass(frozen=True)
class ResidualICResult:
    """残差 Rank IC 点估计 + 有效天数。

    ``n_days=0`` 时 ``ic_mean`` 为 NaN（覆盖门用 n_days，不读 0.0 哨兵）。
    """

    ic_mean: float
    n_days: int


@dataclass(frozen=True)
class LibraryPanel:
    """库因子紧凑面板：逐日截面 z-score + null→0 后的 (date × stock × k) float64。

    构建一次、多候选复用。``factor_names`` 与 ``X[..., j]`` 列序一致。
    """

    dates: tuple  # sorted unique trade_date
    stocks: tuple  # sorted unique ts_code
    date_idx: dict
    stock_idx: dict
    X: np.ndarray  # (n_dates, n_stocks, k) float64
    factor_names: tuple[str, ...]

    @property
    def k(self) -> int:
        return int(self.X.shape[2]) if self.X.ndim == 3 else 0

    @property
    def n_dates(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_stocks(self) -> int:
        return int(self.X.shape[1])


def _panel_to_matrix(
    panel: pl.DataFrame,
    date_idx: dict,
    stock_idx: dict,
    d_n: int,
    s_n: int,
) -> np.ndarray:
    """[trade_date, ts_code, factor_value] → (d_n, s_n) float64，缺测为 NaN。"""
    m = np.full((d_n, s_n), np.nan, dtype=np.float64)
    if panel is None or panel.is_empty():
        return m
    r = np.fromiter(
        (date_idx.get(d, -1) for d in panel["trade_date"].to_list()),
        dtype=np.int64, count=panel.height,
    )
    c = np.fromiter(
        (stock_idx.get(s, -1) for s in panel["ts_code"].to_list()),
        dtype=np.int64, count=panel.height,
    )
    v = panel["factor_value"].to_numpy().astype(np.float64, copy=False)
    keep = (r >= 0) & (c >= 0) & np.isfinite(v)
    m[r[keep], c[keep]] = v[keep]
    return m


def _cs_zscore_null0(row: np.ndarray) -> np.ndarray:
    """单日截面 z-score（ddof=1，与 ``_zscore_factor`` 一致），null/非有限 → 0。"""
    out = np.zeros(row.shape[0], dtype=np.float64)
    mask = np.isfinite(row)
    n = int(mask.sum())
    if n < 2:
        return out
    vals = row[mask]
    mu = float(vals.mean())
    # ddof=1 与 combination._zscore_factor / polars std 默认一致
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    if not np.isfinite(sd) or sd <= 0.0:
        return out
    out[mask] = (vals - mu) / sd
    return out


def build_library_panel(lib_pool: dict[str, pl.DataFrame] | None) -> LibraryPanel | None:
    """把 ``build_library_pool`` 产物转紧凑矩阵并**一次性**做逐日 z-score + null→0。

    空/None → None（调用方据此把 objective 退化为 raw）。
    """
    if not lib_pool:
        return None
    names = tuple(lib_pool.keys())
    dates: set = set()
    stocks: set = set()
    for p in lib_pool.values():
        if p is None or p.is_empty():
            continue
        dates |= set(p["trade_date"].to_list())
        stocks |= set(p["ts_code"].to_list())
    if not dates or not stocks:
        return None
    date_list = tuple(sorted(dates))
    stock_list = tuple(sorted(stocks))
    date_idx = {d: i for i, d in enumerate(date_list)}
    stock_idx = {s: i for i, s in enumerate(stock_list)}
    d_n, s_n, k = len(date_list), len(stock_list), len(names)
    X = np.zeros((d_n, s_n, k), dtype=np.float64)
    for j, name in enumerate(names):
        raw = _panel_to_matrix(lib_pool[name], date_idx, stock_idx, d_n, s_n)
        for di in range(d_n):
            X[di, :, j] = _cs_zscore_null0(raw[di])
    return LibraryPanel(
        dates=date_list, stocks=stock_list,
        date_idx=date_idx, stock_idx=stock_idx,
        X=X, factor_names=names,
    )


def residualize_cross_section(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """**单日**截面残差：r = y − [1|X] @ β，β = lstsq([1|X], y)。

    参数均为**同一交易日**的行对齐数组；禁止跨日拼接后调用（PIT 结构守卫依赖此签名）。
    ``y`` shape (n,)，``X`` shape (n, k)；k=0 时残差 = y（减截距后零均值，但 k=0 不应调用）。
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = y.shape[0]
    if X.shape[0] != n:
        raise ValueError(f"y/X 行数不一致: y={n}, X={X.shape[0]}")
    if n == 0:
        return y.copy()
    A = np.column_stack([np.ones(n, dtype=np.float64), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _day_min_samples(k: int) -> int:
    return max(MIN_CROSS_SAMPLES, int(k) + 10)


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """单日 Spearman = Pearson(rank, rank)；退化截面 → None。"""
    if a.size < 2:
        return None
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    # 平均秩（ties 取组内平均），与 ic_analysis 的 polars rank(method="average") 口径一致。
    def _avg_rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(x.size, dtype=np.float64)
        i = 0
        while i < x.size:
            j = i + 1
            while j < x.size and x[order[j]] == x[order[i]]:
                j += 1
            avg = 0.5 * (i + j - 1) + 1.0  # 1-based average rank
            ranks[order[i:j]] = avg
            i = j
        return ranks

    ra = _avg_rank(a)
    rb = _avg_rank(b)
    c = float(np.corrcoef(ra, rb)[0, 1])
    return c if np.isfinite(c) else None


def compute_residual_ic(
    candidate: pl.DataFrame,
    lib_panel: LibraryPanel,
    fwd_returns: pl.DataFrame,
    *,
    ret_col: str = "fwd_ret_1d",
) -> ResidualICResult:
    """对候选面板逐日残差化后算 Rank IC 均值（与 ``compute_rank_ic`` 同口径：逐日 Spearman 均值）。

    ``candidate``: [trade_date, ts_code, factor_value]
    ``fwd_returns``: 须含 trade_date, ts_code, ``ret_col``（通常由 ``compute_fwd_returns`` 产出）。
    只在单日截面内 lstsq；无跨日状态。
    """
    if lib_panel is None or lib_panel.k == 0:
        return ResidualICResult(float("nan"), 0)
    if candidate is None or candidate.is_empty():
        return ResidualICResult(float("nan"), 0)
    if ret_col not in fwd_returns.columns:
        raise ValueError(f"fwd_returns 缺列 {ret_col!r}")

    cand = candidate.filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )
    if cand.is_empty():
        return ResidualICResult(float("nan"), 0)

    # 只 join 收益：残差在 numpy 侧做，避免把库矩阵拉回 polars
    joined = cand.join(
        fwd_returns.select(["trade_date", "ts_code", ret_col]),
        on=["trade_date", "ts_code"], how="inner",
    ).filter(pl.col(ret_col).is_not_null() & pl.col(ret_col).is_finite())
    if joined.is_empty():
        return ResidualICResult(float("nan"), 0)

    k = lib_panel.k
    min_n = _day_min_samples(k)
    ics: list[float] = []

    # 按日 group：Python 层循环保证「无跨日拟合」的可审计结构
    for date, day_df in joined.group_by("trade_date", maintain_order=True):
        # polars group_by key 可能是 tuple
        d = date[0] if isinstance(date, tuple) else date
        di = lib_panel.date_idx.get(d)
        if di is None:
            continue
        codes = day_df["ts_code"].to_list()
        y = day_df["factor_value"].to_numpy().astype(np.float64, copy=False)
        ret = day_df[ret_col].to_numpy().astype(np.float64, copy=False)
        # 对齐到 panel 股票轴
        si = np.fromiter(
            (lib_panel.stock_idx.get(c, -1) for c in codes),
            dtype=np.int64, count=len(codes),
        )
        valid = si >= 0
        if int(valid.sum()) < min_n:
            continue
        si_v = si[valid]
        y_v = y[valid]
        ret_v = ret[valid]
        X_day = lib_panel.X[di, si_v, :]  # (n, k) 已 z-score + null→0
        n_t = y_v.shape[0]
        if n_t < min_n:
            continue
        resid = residualize_cross_section(y_v, X_day)
        ic = _spearman(resid, ret_v)
        if ic is not None:
            ics.append(ic)

    if not ics:
        return ResidualICResult(float("nan"), 0)
    return ResidualICResult(float(np.mean(ics)), len(ics))


def resolve_objective(objective: str | None, lib_nonempty: bool) -> str:
    """解析有效 objective：``"residual"`` 仅当库非空；否则退化为 ``"raw"``。

    ``objective=None`` → 默认 residual（库空则 raw）。非法值 → ValueError。
    """
    obj = "residual" if objective is None else str(objective).strip().lower()
    if obj not in ("raw", "residual"):
        raise ValueError(f"未知 objective={objective!r}，应为 'raw' 或 'residual'")
    if obj == "residual" and not lib_nonempty:
        return "raw"
    return obj
