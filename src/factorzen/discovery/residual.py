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

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

from factorzen.core.stats import spearman_avg_rank as _spearman
from factorzen.daily.evaluation.ic_analysis import _MIN_CROSS_SAMPLES

# 与 ic_analysis 同门槛；日守卫再取 max(本值, k+10)
MIN_CROSS_SAMPLES = _MIN_CROSS_SAMPLES

# 残差面板 f32 阈值(格数 d×s×k):全 A ~997M 格时 f64 X+逐日 Q 各 ~8G;
# f32 减半,数学处 numpy 自动升精度。csi800 ~163M 格不触发保 f64 零回归。
RESIDUAL_PANEL_F32_CELLS = 500_000_000


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


def build_library_panel(
    lib_pool: Mapping[str, pl.DataFrame] | None,
) -> LibraryPanel | None:
    """把 ``build_library_pool`` 产物转紧凑矩阵并**一次性**做逐日 z-score + null→0。

    空/None → None（调用方据此把 objective 退化为 raw）。
    识别 ``CompactLibraryPool``：从单骨架宽表一次散射，避免 dict-of-frames 键副本。
    """
    if not lib_pool:
        return None
    from factorzen.discovery.factor_library import CompactLibraryPool

    if isinstance(lib_pool, CompactLibraryPool):
        return _build_library_panel_from_wide(lib_pool)

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


def _build_library_panel_from_wide(pool: object) -> LibraryPanel | None:
    """CompactLibraryPool.wide → LibraryPanel（键并集=任因子有限值的行，对齐旧 filter 并集）。"""
    from factorzen.discovery.factor_library import CompactLibraryPool

    assert isinstance(pool, CompactLibraryPool)
    names = pool.factor_names
    if not names:
        return None
    wide = pool.wide
    # 与旧路径「各因子 filter 后键并集」对齐：至少一列有限
    any_finite = pl.any_horizontal(
        [pl.col(n).is_not_null() & pl.col(n).is_finite() for n in names]
    )
    keys = wide.filter(any_finite).select(["trade_date", "ts_code"])
    if keys.is_empty():
        return None
    date_list = tuple(sorted(keys["trade_date"].unique().to_list()))
    stock_list = tuple(sorted(keys["ts_code"].unique().to_list()))
    if not date_list or not stock_list:
        return None
    date_idx = {d: i for i, d in enumerate(date_list)}
    stock_idx = {s: i for i, s in enumerate(stock_list)}
    d_n, s_n, k = len(date_list), len(stock_list), len(names)
    # X f32(仅超阈值大面板):全 A 2055×5776×84 ≈ 997M 格,f64 X + 逐日 Q 缓存
    # 各 ~8G(v19 探针死于池后残差投影)。f32 后各 ~4G;投影/QR 的 numpy 数学
    # 自动升精度,1e-6 级差异远低于裁决阈值。csi800 级 ~163M 格保 f64 零回归。
    _dtype = (
        np.float32 if d_n * s_n * k >= RESIDUAL_PANEL_F32_CELLS else np.float64
    )
    X = np.zeros((d_n, s_n, k), dtype=_dtype)

    # 索引 join 只用键列(84 列宽帧 join 会整帧复制 ~3G);值列逐列从 wide 直取,
    # 行序与 keys join 结果对齐依赖 wide 行序不变——故 join 后按行序回查。
    # (P4c:ts_code 可能 Categorical,小帧 align)
    from factorzen.discovery.scoring import _align_join_key

    date_map = pl.DataFrame({"trade_date": list(date_list), "_di": list(range(d_n))})
    stock_map = pl.DataFrame({"ts_code": list(stock_list), "_si": list(range(s_n))})
    keys_only = wide.select(["trade_date", "ts_code"])
    date_map = _align_join_key(date_map, "trade_date", keys_only)
    stock_map = _align_join_key(stock_map, "ts_code", keys_only)
    # left join 保全行(全 null 行的 date/stock 不在 any_finite 并集轴上 → 哨兵 -1
    # 掩码丢弃,与原 wide_sel inner join 的丢行语义一致);行序 maintain_order 保序,
    # 值列按原始行位从 wide 直取。
    indexed = (
        keys_only
        .join(date_map, on="trade_date", how="left", maintain_order="left")
        .join(stock_map, on="ts_code", how="left", maintain_order="left")
    )
    r = indexed["_di"].fill_null(-1).cast(pl.Int64).to_numpy()
    c = indexed["_si"].fill_null(-1).cast(pl.Int64).to_numpy()
    on_axis = (r >= 0) & (c >= 0)
    for j, name in enumerate(names):
        raw = np.full((d_n, s_n), np.nan, dtype=np.float64)
        v = wide[name].to_numpy()
        # polars null → 可能是 None 在 object；统一 isfinite
        v64 = np.asarray(v, dtype=np.float64)
        keep = np.isfinite(v64) & on_axis
        raw[r[keep], c[keep]] = v64[keep]
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


def _qr_rcond_tol(A: np.ndarray, diag: np.ndarray) -> float:
    """与 ``np.linalg.lstsq(..., rcond=None)`` 同阶的数值秩阈值。"""
    scale = float(diag.max()) if diag.size else 1.0
    return float(np.finfo(np.float64).eps * max(A.shape) * scale)


def _qr_basis(X: np.ndarray) -> np.ndarray | None:
    """对 ``A=[1|X]`` 做 reduced QR，返回满秩列对应的 Q；秩亏 → None。

    秩亏时 Householder QR 无列主元，截断 R 对角不能保证张成完整 col(A)；
    返回 None 让呼叫方走 ``lstsq``（SVD）慢路径，残差与 ``residualize_cross_section`` 一致。
    """
    # 保持输入精度:大面板 X 为 f32 时 Q 缓存同存 f32(逐日 Q 全集 ~8G→4G,全 A 关键);
    # 投影 Q@(Qᵀy) 与 f64 y 相乘时 numpy 自动升精度。小面板 f64 不变。
    _dt = np.float32 if getattr(X, "dtype", None) == np.float32 else np.float64
    X = np.asarray(X, dtype=_dt)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = int(X.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=_dt)
    A = np.column_stack([np.ones(n, dtype=_dt), X])
    Q, R = np.linalg.qr(A, mode="reduced")
    if R.size == 0:
        return Q
    diag = np.abs(np.diag(R))
    tol = _qr_rcond_tol(A, diag)
    if np.any(diag <= tol):
        return None
    return Q


def _project_with_Q_or_lstsq(y: np.ndarray, X: np.ndarray, Q: np.ndarray | None) -> np.ndarray:
    """``r = y - Q @ (Q.T @ y)``；Q 不可用时回退 lstsq。"""
    if Q is not None and Q.shape[0] == y.shape[0]:
        # 满秩 reduced QR：正交投影 residual，与 lstsq 最小二乘残差一致
        return y - Q @ (Q.T @ y)
    return residualize_cross_section(y, X)


class ResidualProjector:
    """库面板 per-date QR 预计算：多候选残差化时复用每日 Q。

    构造时对每个 ``trade_date`` 的设计矩阵 ``[1 | X_lib]``（面板全体 ``ts_code`` 行）
    做 reduced QR，缓存 Q 与行索引（即 ``panel.stock_idx`` / ``panel.stocks``）。

    ``residualize`` / ``project_day`` 语义对齐 ``residualize_cross_section`` +
    ``compute_residual_ic`` 的日守卫与对齐规则（候选∩库、``max(30, k+10)``、NaN/null）。

    快路径：对齐后的股票集覆盖全日面板轴时，直接 ``y - Q@(Q.T@y)``（两次 matvec）。
    子集日：对子矩阵现场 reduced QR（仍避免 SVD lstsq）；子矩阵秩亏则 lstsq 回退。
    """

    def __init__(self, panel: LibraryPanel) -> None:
        if panel is None or panel.k == 0:
            raise ValueError("ResidualProjector 需要 k>0 的 LibraryPanel")
        self.panel = panel
        self._min_n = _day_min_samples(panel.k)
        # di → Q (n_stocks, k+1) 或 None（该日全截面秩亏 → project 走 lstsq）
        self._Q: list[np.ndarray | None] = [
            _qr_basis(panel.X[di]) for di in range(panel.n_dates)
        ]

    @classmethod
    def from_panel(cls, panel: LibraryPanel) -> ResidualProjector:
        return cls(panel)

    def project_day(
        self, di: int, y_v: np.ndarray, si_v: np.ndarray,
    ) -> np.ndarray:
        """单日投影：``y_v`` 与 ``si_v``（panel 股票轴下标）行对齐。"""
        y_v = np.asarray(y_v, dtype=np.float64).reshape(-1)
        si_v = np.asarray(si_v, dtype=np.int64).reshape(-1)
        if y_v.shape[0] != si_v.shape[0]:
            raise ValueError(
                f"y/si 行数不一致: y={y_v.shape[0]}, si={si_v.shape[0]}"
            )
        X_day = self.panel.X[di, si_v, :]
        n_full = self.panel.n_stocks
        Q_full = self._Q[di]
        # 全日覆盖（任意行置换）：置换后的 Q 列仍标准正交，可直接投影
        if (
            Q_full is not None
            and si_v.shape[0] == n_full
            and np.unique(si_v).size == n_full
        ):
            Q = Q_full[si_v]
            return y_v - Q @ (Q.T @ y_v)
        # 子集：对 [1|X_sub] 现场 QR；秩亏 → lstsq
        Q_sub = _qr_basis(X_day)
        return _project_with_Q_or_lstsq(y_v, X_day, Q_sub)

    def residualize(self, factor_df: pl.DataFrame) -> pl.DataFrame:
        """候选面板逐日残差化 → ``[trade_date, ts_code, factor_value]``（值为残差）。

        与 ``compute_residual_ic`` 相同的对齐/守卫（无收益 join）：
        - ``factor_value``：``fill_nan(None)`` 后只留有限非空
        - 库无 ``trade_date`` / 库外 ``ts_code`` → 丢弃
        - 有效行 ``n_t < max(30, k+10)`` → 整日丢弃
        """
        empty = pl.DataFrame(
            schema={
                "trade_date": pl.Date,
                "ts_code": pl.Utf8,
                "factor_value": pl.Float64,
            }
        )
        if factor_df is None or factor_df.is_empty():
            return empty

        cand = factor_df.with_columns(pl.col("factor_value").fill_nan(None)).filter(
            pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
        )
        if cand.is_empty():
            return empty

        panel = self.panel
        min_n = self._min_n
        out_dates: list = []
        out_codes: list = []
        out_vals: list[float] = []

        for date, day_df in cand.group_by("trade_date", maintain_order=True):
            d = date[0] if isinstance(date, tuple) else date
            di = panel.date_idx.get(d)
            if di is None:
                continue
            codes = day_df["ts_code"].to_list()
            y = day_df["factor_value"].to_numpy().astype(np.float64, copy=False)
            si = np.fromiter(
                (panel.stock_idx.get(c, -1) for c in codes),
                dtype=np.int64, count=len(codes),
            )
            valid = si >= 0
            n_valid = int(valid.sum())
            if n_valid < min_n:
                continue
            si_v = si[valid]
            y_v = y[valid]
            if y_v.shape[0] < min_n:
                continue
            resid = self.project_day(di, y_v, si_v)
            codes_v = [c for c, ok in zip(codes, valid, strict=True) if ok]
            out_dates.extend([d] * len(codes_v))
            out_codes.extend(codes_v)
            out_vals.extend(resid.tolist())

        if not out_vals:
            return empty
        return pl.DataFrame({
            "trade_date": out_dates,
            "ts_code": out_codes,
            "factor_value": out_vals,
        })


def compute_residual_ic(
    candidate: pl.DataFrame,
    lib_panel: LibraryPanel,
    fwd_returns: pl.DataFrame,
    *,
    ret_col: str = "fwd_ret_1d",
    projector: ResidualProjector | None = None,
) -> ResidualICResult:
    """对候选面板逐日残差化后算 Rank IC 均值（与 ``compute_rank_ic`` 同口径：逐日 Spearman 均值）。

    ``candidate``: [trade_date, ts_code, factor_value]
    ``fwd_returns``: 须含 trade_date, ts_code, ``ret_col``（通常由 ``compute_fwd_returns`` 产出）。
    只在单日截面内 lstsq / QR 投影；无跨日状态。

    ``projector``: 可选；传入时走 ``ResidualProjector.project_day`` 快路径（语义不变）。
    """
    if lib_panel is None or lib_panel.k == 0:
        return ResidualICResult(float("nan"), 0)
    if candidate is None or candidate.is_empty():
        return ResidualICResult(float("nan"), 0)
    if ret_col not in fwd_returns.columns:
        raise ValueError(f"fwd_returns 缺列 {ret_col!r}")

    # 与 residualize 同口径：NaN ≠ null，先 fill_nan
    cand = candidate.with_columns(pl.col("factor_value").fill_nan(None)).filter(
        pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
    )
    if cand.is_empty():
        return ResidualICResult(float("nan"), 0)

    # 只 join 收益：残差在 numpy 侧做，避免把库矩阵拉回 polars
    # P4c：fwd 与 candidate 的 ts_code 可能一侧 Categorical、一侧 Utf8
    from factorzen.discovery.scoring import _align_join_key

    fwd_sel = fwd_returns.select(["trade_date", "ts_code", ret_col])
    fwd_sel = _align_join_key(fwd_sel, "ts_code", cand)
    joined = cand.join(
        fwd_sel, on=["trade_date", "ts_code"], how="inner",
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
        n_t = y_v.shape[0]
        if n_t < min_n:
            continue
        if projector is not None:
            resid = projector.project_day(di, y_v, si_v)
        else:
            X_day = lib_panel.X[di, si_v, :]  # (n, k) 已 z-score + null→0
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
