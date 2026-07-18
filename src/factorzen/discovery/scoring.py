# src/factorzen/discovery/scoring.py
"""候选因子快速评估：两段式中的「内循环」——只算 Rank IC/IR，不跑回测。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from factorzen.daily.evaluation.correlation import compute_factor_correlation
from factorzen.daily.evaluation.ic_analysis import compute_fwd_returns, compute_rank_ic
from factorzen.daily.preprocessing.normalizer import cross_sectional_zscore
from factorzen.discovery.expression import Node
from factorzen.discovery.expression import complexity as _complexity


def _cut_literal(df: pl.DataFrame, yyyymmdd: str):
    """"YYYYMMDD" → 与 df.trade_date dtype 匹配的比较字面量(Date→date,Datetime→当日零点)。

    日频帧行为与旧 ``.date()`` 完全一致;intraday(Datetime 键)返回 datetime,
    避免 polars Datetime 列与 date 字面量比较的类型错误。
    """
    from datetime import datetime
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    return dt if isinstance(df.schema["trade_date"], pl.Datetime) else dt.date()


@dataclass
class DataBundle:
    daily: pl.DataFrame
    fwd_returns: pl.DataFrame
    train_end: str  # "YYYYMMDD"，train 段含此日及之前

    @classmethod
    def build(cls, daily: pl.DataFrame, train_ratio: float = 0.7) -> DataBundle:
        daily = daily.sort(["ts_code", "trade_date"])
        # compute_fwd_returns 在整表 with_columns 后原样返回（正式 IC decay 仍需全列语义）；
        # 挖掘 bundle 只消费键 + fwd_ret_*（quick_fitness / residual_ic / pool_pbo / ic_overfit），
        # 在此 select 收窄，消灭一份近全宽 mining 副本。不动 compute_fwd_returns 本身。
        fwd = compute_fwd_returns(
            daily, price_col="close_adj" if "close_adj" in daily.columns else "close",
        )
        fwd_cols = [c for c in fwd.columns if c.startswith("fwd_ret_")]
        fwd = fwd.select(["trade_date", "ts_code", *fwd_cols])
        dates = sorted(daily["trade_date"].unique().to_list())
        cut = dates[min(int(len(dates) * train_ratio), len(dates) - 1)]
        train_end = cut.strftime("%Y%m%d") if hasattr(cut, "strftime") else str(cut)
        # P5：全仓无 bundle.daily 读点（只写 train_end/fwd_returns）；长驻只留键列，
        # 消灭 sort(mining) 全宽幽灵副本（全 A ~2.2G）。build 内部短暂全宽仅用于 fwd。
        keys = daily.select(["trade_date", "ts_code"])
        return cls(daily=keys, fwd_returns=fwd, train_end=train_end)

    def _segment_mask(self, df: pl.DataFrame, segment: str) -> pl.DataFrame:
        cut = _cut_literal(df, self.train_end)
        if segment == "train":
            return df.filter(pl.col("trade_date") <= cut)
        return df.filter(pl.col("trade_date") > cut)


def quick_fitness(factor_df: pl.DataFrame, bundle: DataBundle,
                  segment: Literal["train", "valid"] = "train") -> dict:
    """factor_df: [trade_date, ts_code, factor_value] → {ic_mean, ir, tstat, n}。

    ``tstat`` 为 IC 序列的 Newey-West HAC t 统计量（``compute_rank_ic`` 已算），
    仅当有效 IC 天数 >4 且 ic_std>0 时非零，天然惩罚低样本 —— 用作排序键可避免
    小样本 ic_std 虚低把 IR 撑爆的假象（见 score_candidate）。
    """
    seg = bundle._segment_mask(factor_df, segment)
    if seg.is_empty():
        return {"ic_mean": 0.0, "ir": 0.0, "tstat": 0.0, "n": 0}
    # 截面 zscore（cross_sectional_zscore 新增列 factor_value_z）
    clean = cross_sectional_zscore(seg, col="factor_value").rename({"factor_value_z": "factor_clean"})
    ret = bundle._segment_mask(bundle.fwd_returns, segment)
    # 挖掘路径只消费 1d Rank IC/IR/tstat（candidates.csv / score / 护栏均不读
    # ic_decay 5/10/20d）。显式 horizons=[1]，避免默认 [1,5,10,20] 重复截面相关。
    # 正式 factor run / ic_overfit_report 仍走 compute_rank_ic 默认多 horizon。
    res = compute_rank_ic(
        clean.select(["trade_date", "ts_code", "factor_clean"]),
        ret, factor_col="factor_clean", frequency="daily", horizons=[1],
    )
    return {"ic_mean": res.ic_mean, "ir": res.ir, "tstat": res.ic_tstat, "n": res.n_periods}


# 去相关 |corr| 门槛的单一真源——session 池去相关、库级正交、upsert 默认共用。
DEFAULT_DECORR_THRESHOLD = 0.7

# 与 compute_factor_correlation 逐日门槛一致
_MIN_CORR_CROSS = 30

# corr panel values f32 阈值: n_d×n_s×n_f×8B ≥ 此值时 values 存 float32。
# 全 A ~2055×5776×84×8 ≈ 8G 必触发;csi800 级 ~1.1G 不触发,f64 零回归。
# 计算侧 (_max_corr_detail_panel) 仍升 f64,与残差 X/Q 的 f32 存储风格一致。
# 仅 legacy dict 构建路径使用;compact 路径同量级优先 lazy-wide(见下)。
CORR_PANEL_F32_BYTES_THRESHOLD = 2 * 1024**3  # 2 GiB

# compact 超阈值免物化: n_d×n_s×n_f×8B ≥ 此值时返回 LazyWideCorrGrid。
# 与 f32 阈值同值 → 全 A 走 lazy 优先,不再物化 (n_d,n_s,n_f) 网格;
# csi800 级 ~1.1G 不触发,f64 物化零回归。
CORR_PANEL_LAZY_BYTES_THRESHOLD = 2 * 1024**3  # 2 GiB

# 分块归约预算:块内 f64 临时 (c_m/l_m/乘积等 ~5 份) 约此字节上限。
# 按日独立 sum → 分块不改变任一日浮点求和顺序,f64 下与整面板逐位等价。
CORR_PANEL_CHUNK_BYTES = 256 * 1024**2  # 256 MiB


@dataclass(frozen=True)
class LibraryCorrPanel:
    """库池一次对齐的宽面板，供候选 vs 库逐对相关向量化。

    语义与 ``compute_factor_correlation`` 逐对路径一致（见 ``max_correlation_detail``）：
    - ``present``：``None`` 时从 ``~np.isnan(values)`` 推导（**新契约: values 中 NaN ⇔ absent**）。
      两条构建路径的输入都保证 present 位有限——compact 池 wide 列非有限已在构建时转 null,
      legacy 池帧构建时已 filter finite; null → 散射为 NaN(不再 fill 0) → isnan 推导与旧
      bool 掩码逐格等价。显式 bool 数组仍兼容(手工构造/测试)。
    - 缺行 / null → values=NaN → present 推导为 False（该 (date,stock) 不参与该对）
    - 候选侧 NaN 毒化由 ``_scatter_candidate_to_panel`` 的显式 present 保留（与库契约分离）
    - ``names`` 保持 pool 插入序（并列 max|corr| 取后出现者）
    - ``values`` 可为 float64 或 float32（超 ``CORR_PANEL_F32_BYTES_THRESHOLD`` 时 f32 存储）
    """

    names: tuple[str, ...]
    dates: tuple  # sorted unique trade_date
    stocks: tuple  # sorted unique ts_code
    date_idx: dict
    stock_idx: dict
    values: np.ndarray  # (n_dates, n_stocks, n_factors) float64|float32
    present: np.ndarray | None  # (n_dates, n_stocks, n_factors) bool; None → ~isnan(values)

    def present_block(self, d0: int, d1: int) -> np.ndarray:
        """日期切片 [d0:d1] 的 present 掩码；``present is None`` 时由 values 推导。"""
        if self.present is None:
            return ~np.isnan(self.values[d0:d1])
        return self.present[d0:d1]

    def block(self, d0: int, d1: int) -> tuple[np.ndarray, np.ndarray]:
        """日期切片 [d0:d1] → (vals_f64, present_bool)，形状 (d1-d0, n_s, n_f)。"""
        vals = self.values[d0:d1]
        if vals.dtype != np.float64:
            vals = vals.astype(np.float64)
        return vals, self.present_block(d0, d1)


class LazyWideCorrGrid:
    """超阈值 compact 库池：免物化 (n_d,n_s,n_f) 网格，按日块从 wide 散射。

    与 ``LibraryCorrPanel`` duck-type 兼容字段：``names/dates/stocks/date_idx/
    stock_idx``、``present is None``。无 ``values`` 大网格；``block(d0,d1)``
    用 polars gather 按块散射后即弃。散射语义与物化面板逐位等价（同 f32/f64
    源值、计算同升 f64）。
    """

    __slots__ = (
        "_day_starts",
        "_di_by_day",
        "_row_by_day",
        "_si_by_day",
        "_wide",
        "date_idx",
        "dates",
        "names",
        "present",
        "stock_idx",
        "stocks",
    )

    def __init__(
        self,
        *,
        names: tuple[str, ...],
        dates: tuple,
        stocks: tuple,
        date_idx: dict,
        stock_idx: dict,
        wide: pl.DataFrame,
        row_by_day: np.ndarray,
        si_by_day: np.ndarray,
        di_by_day: np.ndarray,
    ) -> None:
        self.names = names
        self.dates = dates
        self.stocks = stocks
        self.date_idx = date_idx
        self.stock_idx = stock_idx
        self.present = None
        self._wide = wide  # 引用 CompactLibraryPool.wide，不复制
        self._row_by_day = row_by_day
        self._si_by_day = si_by_day
        self._di_by_day = di_by_day
        n_d = len(dates)
        self._day_starts = np.searchsorted(di_by_day, np.arange(n_d + 1))

    def block(self, d0: int, d1: int) -> tuple[np.ndarray, np.ndarray]:
        """日期切片 [d0:d1] → (vals_f64, present_bool)，形状 (d1-d0, n_s, n_f)。

        每块**一次**整行 take(84 列小帧)→ to_numpy 单矩阵 → 一次 3D scatter。
        逐因子逐块 gather 曾是 v25 探针死因:每候选 ~13k 次小分配的碎片/滞留
        在全 A 把 WSL VM 顶穿;单次 take 把分配次数降两个量级。
        绝不整列 ``to_numpy()``(含 null 的 f32 列整列转换物化 10.9M NaN 副本)。
        """
        n_s = len(self.stocks)
        n_f = len(self.names)
        a = int(self._day_starts[d0])
        b = int(self._day_starts[d1])
        rows = self._row_by_day[a:b]
        sis = self._si_by_day[a:b]
        dis = self._di_by_day[a:b] - d0
        vals = np.full((d1 - d0, n_s, n_f), np.nan, dtype=np.float64)
        if a < b:
            # 值列全同 dtype(f32 或 f64)→ to_numpy 单矩阵零逐列对象;null→NaN 天然
            sub = self._wide.select(list(self.names))[rows]
            m = sub.to_numpy()
            vals[dis, sis, :] = m  # f32→f64 赋值自动升位
        return vals, ~np.isnan(vals)

    def present_block(self, d0: int, d1: int) -> np.ndarray:
        """日期切片 present；热路径请用 ``block`` 一次取 vals+pres。"""
        return self.block(d0, d1)[1]


def _corr_panel_value_dtype(n_d: int, n_s: int, n_f: int) -> np.dtype:
    """按估算字节数选 values dtype;超阈值时 print 一行(对齐 library-pool f32 提示)。"""
    est = int(n_d) * int(n_s) * int(n_f) * 8
    if est >= CORR_PANEL_F32_BYTES_THRESHOLD:
        print(
            f"[corr-panel] values f32 模式(估算 {est / (1024**3):.1f}G"
            f"≥阈值 {CORR_PANEL_F32_BYTES_THRESHOLD / (1024**3):.0f}G)",
            flush=True,
        )
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _factor_col_name(df: pl.DataFrame) -> str:
    if "factor_value" in df.columns:
        return "factor_value"
    if "factor_clean" in df.columns:
        return "factor_clean"
    raise ValueError(
        f"因子帧须含 factor_value 或 factor_clean，实得列={list(df.columns)}"
    )


def _index_maps_from_keys(dates: tuple, stocks: tuple) -> tuple[dict, dict, pl.DataFrame, pl.DataFrame]:
    date_idx = {d: i for i, d in enumerate(dates)}
    stock_idx = {s: i for i, s in enumerate(stocks)}
    date_map = pl.DataFrame({"trade_date": list(dates), "_di": list(range(len(dates)))})
    stock_map = pl.DataFrame({"ts_code": list(stocks), "_si": list(range(len(stocks)))})
    return date_idx, stock_idx, date_map, stock_map


def _align_join_key(small: pl.DataFrame, col: str, like: pl.DataFrame) -> pl.DataFrame:
    """把 small[col] cast 到 like[col] 的 dtype（Categorical↔Utf8 join 防 SchemaError）。"""
    if col not in small.columns or col not in like.columns:
        return small
    tgt = like.schema[col]
    if small.schema[col] != tgt:
        return small.with_columns(pl.col(col).cast(tgt))
    return small


def _scatter_frame_to_slice(
    sub: pl.DataFrame,
    date_map: pl.DataFrame,
    stock_map: pl.DataFrame,
    n_d: int,
    n_s: int,
) -> tuple[np.ndarray, np.ndarray]:
    """[trade_date, ts_code, _v] → (values, present) 二维切片。

    null → values=NaN（to_numpy 天然;不再 fill 0）。present 仍由 ``~is_null`` 得到——
    候选路径需要区分「null 缺席」与「非 null 的 float NaN 毒化」；库路径只消费 values,
    present 由 ``LibraryCorrPanel.present is None`` 契约从 isnan 推导。
    """
    vals = np.full((n_d, n_s), np.nan, dtype=np.float64)
    pres = np.zeros((n_d, n_s), dtype=bool)
    if sub.is_empty():
        return vals, pres
    date_map = _align_join_key(date_map, "trade_date", sub)
    stock_map = _align_join_key(stock_map, "ts_code", sub)
    joined = (
        sub.join(date_map, on="trade_date", how="inner")
        .join(stock_map, on="ts_code", how="inner")
    )
    if joined.is_empty():
        return vals, pres
    r = joined["_di"].to_numpy().astype(np.int64, copy=False)
    c = joined["_si"].to_numpy().astype(np.int64, copy=False)
    is_null = joined["_v"].is_null().to_numpy()
    # null → NaN；非 null 的 float NaN 原样保留（候选毒化语义）
    arr = joined["_v"].to_numpy()
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64, copy=False)
    vals[r, c] = arr
    pres[r, c] = ~is_null
    return vals, pres


def build_library_corr_panel(
    pool: Mapping[str, pl.DataFrame] | None,
) -> LibraryCorrPanel | LazyWideCorrGrid | None:
    """把库池对齐成 (date × stock × k) 矩阵；空/None → None。

    Session 级构建一次、整 session 复用。不改池因子数值，只做散射对齐。
    识别 ``CompactLibraryPool``：从单骨架宽表散射，避免 dict-of-frames 键副本。

    ``present`` 恒为 ``None``（由 values 的 NaN 推导）；legacy dict 超阈值时
    values 存 float32；compact 超 ``CORR_PANEL_LAZY_BYTES_THRESHOLD`` 时返回
    ``LazyWideCorrGrid``（免物化大网格）。
    """
    if not pool:
        return None
    from factorzen.discovery.factor_library import CompactLibraryPool

    if isinstance(pool, CompactLibraryPool):
        return _build_library_corr_panel_from_wide(pool)

    names = tuple(pool.keys())
    prepared: list[pl.DataFrame] = []
    pieces: list[pl.DataFrame] = []
    for name in names:
        df = pool[name]
        col = _factor_col_name(df)
        sub = df.select(
            ["trade_date", "ts_code", pl.col(col).alias("_v")]
        )
        prepared.append(sub)
        if not sub.is_empty():
            pieces.append(sub.select(["trade_date", "ts_code"]))

    if not pieces:
        dates: tuple = ()
        stocks: tuple = ()
    else:
        keys = pl.concat(pieces).unique()
        dates = tuple(sorted(keys["trade_date"].unique().to_list()))
        stocks = tuple(sorted(keys["ts_code"].unique().to_list()))

    date_idx, stock_idx, date_map, stock_map = _index_maps_from_keys(dates, stocks)
    n_d, n_s, n_f = len(dates), len(stocks), len(names)
    dtype = _corr_panel_value_dtype(n_d, n_s, n_f)
    values = np.full((n_d, n_s, n_f), np.nan, dtype=dtype)

    for fi, sub in enumerate(prepared):
        if sub.is_empty() or n_d == 0:
            continue
        v_sl, p_sl = _scatter_frame_to_slice(sub, date_map, stock_map, n_d, n_s)
        # 非 null 的 float NaN:与 compute_factor_correlation 一致,毒化该日整截面
        # (corrcoef 遇 NaN → 跳过该日)。present=None 契约下 NaN⇔absent,故把毒化日
        # 全日置 NaN → 该日 n=0,与逐对「整日跳过」等价。生产路径(finite filter)不触发。
        poison_days = np.any(p_sl & np.isnan(v_sl), axis=1)
        if np.any(poison_days):
            v_sl = np.array(v_sl, copy=True, dtype=np.float64)
            v_sl[poison_days, :] = np.nan
        values[:, :, fi] = v_sl

    return LibraryCorrPanel(
        names=names,
        dates=dates,
        stocks=stocks,
        date_idx=date_idx,
        stock_idx=stock_idx,
        values=values,
        present=None,
    )


def _wide_panel_key_index(
    wide: pl.DataFrame,
    date_map: pl.DataFrame,
    stock_map: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """键窄 join + ``_ri`` 钉序 → ``(di_all, si_all, ri_all)`` 与 ``wide`` 行对齐。

    物化与 lazy-wide 共用，避免双路径漂移。只 join 键两列 + 行号，
    不触碰值列（避免 84 列 join 复制大帧）。
    """
    wide_keys = wide.select(["trade_date", "ts_code"]).with_row_index("_ri")
    date_map = _align_join_key(date_map, "trade_date", wide_keys)
    stock_map = _align_join_key(stock_map, "ts_code", wide_keys)
    indexed = (
        wide_keys
        .join(date_map, on="trade_date", how="inner")
        .join(stock_map, on="ts_code", how="inner")
        .sort("_ri")  # 钉回原 wide 行序，使 ri 与 wide[name] 对齐
    )
    di_all = indexed["_di"].to_numpy().astype(np.int32, copy=False)
    si_all = indexed["_si"].to_numpy().astype(np.int32, copy=False)
    ri_all = indexed["_ri"].to_numpy().astype(np.int64, copy=False)
    return di_all, si_all, ri_all


def _build_library_corr_panel_from_wide(
    pool: object,
) -> LibraryCorrPanel | LazyWideCorrGrid | None:
    """CompactLibraryPool.wide → 物化 ``LibraryCorrPanel`` 或 ``LazyWideCorrGrid``。

    超 ``CORR_PANEL_LAZY_BYTES_THRESHOLD`` 时免物化大网格；否则逐因子散射
    （无 84 列 join 大帧）。键窄 join / ``_ri`` 钉序经 ``_wide_panel_key_index`` 共享。
    """
    from factorzen.discovery.factor_library import CompactLibraryPool

    assert isinstance(pool, CompactLibraryPool)
    names = pool.factor_names
    if not names:
        return None
    wide = pool.wide
    any_present = pl.any_horizontal(
        [pl.col(n).is_not_null() for n in names]
    )
    keys = wide.filter(any_present).select(["trade_date", "ts_code"])
    if keys.is_empty():
        dates: tuple = ()
        stocks: tuple = ()
    else:
        dates = tuple(sorted(keys["trade_date"].unique().to_list()))
        stocks = tuple(sorted(keys["ts_code"].unique().to_list()))

    date_idx, stock_idx, date_map, stock_map = _index_maps_from_keys(dates, stocks)
    n_d, n_s, n_f = len(dates), len(stocks), len(names)
    est = int(n_d) * int(n_s) * int(n_f) * 8

    if n_d == 0:
        return LibraryCorrPanel(
            names=names, dates=dates, stocks=stocks,
            date_idx=date_idx, stock_idx=stock_idx,
            values=np.full((0, n_s, n_f), np.nan, dtype=np.float64),
            present=None,
        )

    di_all, si_all, ri_all = _wide_panel_key_index(wide, date_map, stock_map)

    if est >= CORR_PANEL_LAZY_BYTES_THRESHOLD:
        print(
            f"[corr-panel] lazy-wide 模式(估算 {est / (1024**3):.1f}G"
            f"≥阈值 {CORR_PANEL_LAZY_BYTES_THRESHOLD / (1024**3):.0f}G,免物化)",
            flush=True,
        )
        order = np.argsort(di_all, kind="stable")
        return LazyWideCorrGrid(
            names=names,
            dates=dates,
            stocks=stocks,
            date_idx=date_idx,
            stock_idx=stock_idx,
            wide=wide,
            row_by_day=ri_all[order],
            si_by_day=si_all[order],
            di_by_day=di_all[order],
        )

    # 低于 lazy 阈值：物化；dtype 仍走 f32 阈值（测试/边界；生产同量级已 lazy）
    dtype = _corr_panel_value_dtype(n_d, n_s, n_f)
    values = np.full((n_d, n_s, n_f), np.nan, dtype=dtype)
    for fi, name in enumerate(names):
        # polars null→NaN 天然；f32 列保持 f32。逐因子散射后即弃，无宽值帧。
        arr = wide[name].to_numpy()
        values[di_all, si_all, fi] = arr[ri_all]

    return LibraryCorrPanel(
        names=names,
        dates=dates,
        stocks=stocks,
        date_idx=date_idx,
        stock_idx=stock_idx,
        values=values,
        present=None,
    )


def _scatter_candidate_to_panel(
    factor_df: pl.DataFrame, panel: LibraryCorrPanel | LazyWideCorrGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """候选散射到 panel 网格 → (values, present)，形状 (n_dates, n_stocks)。

    候选帧可能含真实 NaN 值（非 null）：present 内保留 NaN 以毒化该日——不能用 isnan 推导。
    只消费 ``dates/stocks`` 键序，``LibraryCorrPanel`` / ``LazyWideCorrGrid`` 均兼容。
    """
    n_d, n_s = len(panel.dates), len(panel.stocks)
    if factor_df.is_empty() or n_d == 0:
        return (
            np.full((n_d, n_s), np.nan, dtype=np.float64),
            np.zeros((n_d, n_s), dtype=bool),
        )
    col = _factor_col_name(factor_df)
    sub = factor_df.select(["trade_date", "ts_code", pl.col(col).alias("_v")])
    # 复用 panel 键序建临时 map（小表，join 比 Python dict fromiter 快）
    # ts_code 可能是 Categorical（P4c）；map 由 str list 建 → 在 scatter 内 align
    date_map = pl.DataFrame({"trade_date": list(panel.dates), "_di": list(range(n_d))})
    stock_map = pl.DataFrame({"ts_code": list(panel.stocks), "_si": list(range(n_s))})
    return _scatter_frame_to_slice(sub, date_map, stock_map, n_d, n_s)


def _max_corr_detail_panel(
    factor_df: pl.DataFrame, panel: LibraryCorrPanel | LazyWideCorrGrid,
) -> tuple[float, str | None]:
    """矩阵化逐对相关：按日期块流式归约，语义对齐 compute_factor_correlation。

    只驻留 6 个 (n_d, n_f) 归约量；块内 both/c_m/l_m 用完即弃。
    按日独立 ⇒ f64 下与整面板实现逐位等价；f32 存储时计算仍升 f64。
    候选 NaN（cand_p=True）→ c_m 含 NaN → 该日 sums NaN → corr NaN → ok=False。
    ``panel.block`` 统一物化/lazy 接口。
    """
    if not panel.names:
        return 0.0, None
    cand_v, cand_p = _scatter_candidate_to_panel(factor_df, panel)
    n_d = len(panel.dates)
    n_s = len(panel.stocks)
    n_f = len(panel.names)
    # 块大小:块内 f64 临时(c_m/l_m/乘积 ~5 份)预算 CORR_PANEL_CHUNK_BYTES
    rows_per_day = n_s * n_f
    blk = max(1, int(CORR_PANEL_CHUNK_BYTES / max(1, rows_per_day * 8 * 5)))

    n = np.zeros((n_d, n_f), dtype=np.float64)
    sum_c = np.zeros((n_d, n_f), dtype=np.float64)
    sum_l = np.zeros((n_d, n_f), dtype=np.float64)
    sum_c2 = np.zeros((n_d, n_f), dtype=np.float64)
    sum_l2 = np.zeros((n_d, n_f), dtype=np.float64)
    sum_cl = np.zeros((n_d, n_f), dtype=np.float64)

    for d0 in range(0, n_d, blk):
        d1 = min(d0 + blk, n_d)
        vals, pres = panel.block(d0, d1)
        both = cand_p[d0:d1, :, None] & pres
        c_m = np.where(both, cand_v[d0:d1, :, None], 0.0)
        l_m = np.where(both, vals, 0.0)
        n[d0:d1] = both.sum(axis=1)
        sum_c[d0:d1] = c_m.sum(axis=1)
        sum_l[d0:d1] = l_m.sum(axis=1)
        sum_c2[d0:d1] = (c_m * c_m).sum(axis=1)
        sum_l2[d0:d1] = (l_m * l_m).sum(axis=1)
        sum_cl[d0:d1] = (c_m * l_m).sum(axis=1)
        del c_m, l_m, both, vals, pres

    with np.errstate(invalid="ignore", divide="ignore"):
        # Pearson ≡ np.corrcoef；(std==0 ddof=0) ⇔ n·Σx²−(Σx)² == 0
        den_c = n * sum_c2 - sum_c * sum_c
        den_l = n * sum_l2 - sum_l * sum_l
        num = n * sum_cl - sum_c * sum_l
        corr = num / np.sqrt(den_c * den_l)
    ok = (n >= _MIN_CORR_CROSS) & (den_c > 0) & (den_l > 0) & np.isfinite(corr)
    cnt = ok.sum(axis=0)  # (n_f,)
    cum = np.where(ok, corr, 0.0).sum(axis=0)

    best = 0.0
    nearest: str | None = None
    for fi, name in enumerate(panel.names):
        if int(cnt[fi]) <= 0:
            c = 0.0
        else:
            c = abs(float(cum[fi] / cnt[fi]))
        if c == c and c >= best:
            best, nearest = c, name
    return best, nearest


def max_correlation(
    factor_df: pl.DataFrame,
    pool: Mapping[str, pl.DataFrame],
    panel: LibraryCorrPanel | LazyWideCorrGrid | None = None,
) -> float:
    """factor_df 与 pool 中每个因子的截面相关性绝对值的最大值。pool 为空时返回 0。

    逐对(pairwise)计算：候选与池中**每个**因子单独算相关。这样一个退化的池因子
    (截面 std==0 / 不足 30 只 / NaN) 只会让它自己那一对得 0，不会污染其它对。
    历史 bug：把候选 + 全池一次性 inner-join 交给 compute_factor_correlation，任一
    池因子退化就 continue 丢整条截面 → count=0 → 所有真实高相关一起被抹成 0.0，
    数学等价簇因此逃过 0.7 去重门槛。不动 compute_factor_correlation（daily 报告仍用其语义）。

    ``panel``：可选预构建库面板（``LibraryCorrPanel`` 或 ``LazyWideCorrGrid``）；
    传入时走矩阵化路径（与逐对数值等价）。
    """
    return max_correlation_detail(factor_df, pool, panel=panel)[0]


def max_correlation_detail(
    factor_df: pl.DataFrame,
    pool: Mapping[str, pl.DataFrame],
    panel: LibraryCorrPanel | LazyWideCorrGrid | None = None,
) -> tuple[float, str | None]:
    """同 ``max_correlation``，额外返回最相近的 pool key（表达式）。pool 空 → (0.0, None)。

    ``panel`` 非 None 时走矩阵化路径，须由同一 ``pool`` 经 ``build_library_corr_panel``
    构建（物化 ``LibraryCorrPanel`` 或超阈值 ``LazyWideCorrGrid``）。
    """
    if not pool:
        return 0.0, None
    if panel is not None:
        return _max_corr_detail_panel(factor_df, panel)
    cand = (factor_df.rename({"factor_value": "factor_clean"})
            if "factor_value" in factor_df.columns else factor_df)
    best = 0.0
    nearest: str | None = None
    for name, df in pool.items():
        other = df.rename({"factor_value": "factor_clean"}) if "factor_value" in df.columns else df
        res = compute_factor_correlation({"__fz_cand__": cand, name: other}, factor_col="factor_clean")
        if len(res.factor_names) < 2:
            continue
        c = abs(float(res.corr_matrix[0][1]))  # [cand, other] 按插入序，[0][1]=候选对该因子
        if c == c and c >= best:  # 排除 NaN；并列取后出现者亦可
            best, nearest = c, name
    return best, nearest


def library_orthogonal_check(
    factor_df: pl.DataFrame,
    lib_pool: Mapping[str, pl.DataFrame] | None,
    *,
    threshold: float = DEFAULT_DECORR_THRESHOLD,
    panel: LibraryCorrPanel | LazyWideCorrGrid | None = None,
) -> tuple[bool, float, str | None]:
    """库相关度量：与库池 max|corr| 是否 ``>= threshold``。

    返回 ``(ok, max_corr_library, nearest_expr)``——``ok=True`` 当且仅当 max|corr| < threshold。
    ``lib_pool`` 空/None → 恒通过、corr=0（零回归）。

    **阈值由调用方按政策传入**（本函数只做度量 + 比较，不做硬拒/软信号语义）：
    - 硬拒重复：``threshold=DEFAULT_DUPLICATE_CORR``（0.95）
    - 快速通道/旧默认：``threshold=DEFAULT_DECORR_THRESHOLD``（0.7，向后兼容）
    M1 与 team/agent 双路径必须调本函数，禁止各自内联相关计算（架构守卫锁死）。

    ``panel``：可选 ``LibraryCorrPanel`` / ``LazyWideCorrGrid``（session 级构建一次）；
    不传则逐对原路径。
    """
    if not lib_pool:
        return True, 0.0, None
    mc, nearest = max_correlation_detail(factor_df, lib_pool, panel=panel)
    if mc >= threshold:
        return False, mc, nearest
    return True, mc, nearest


def score_candidate(factor_df: pl.DataFrame, node: Node, bundle: DataBundle,
                    pool: dict[str, pl.DataFrame], lam: float = 0.5,
                    gamma: float = 0.002) -> dict:
    train = quick_fitness(factor_df, bundle, "train")
    mc = max_correlation(factor_df, pool)
    cplx = _complexity(node)
    # 排序键用 t-stat 而非裸 IR：t-stat 自带 n>4 门槛（低样本→0），避免小样本 ic_std
    # 虚低把 IR 撑成假象（历史 rank1: ic≈2.4e-16 却 IR=14.68、n=7 排第一）。
    # ir_train 仍保留在结果里供 DSR / CSV 使用。
    tstat = train["tstat"]
    fitness = tstat - lam * mc - gamma * cplx
    return {"fitness": fitness, "ic_train": train["ic_mean"], "ir_train": train["ir"],
            "tstat_train": tstat, "max_corr": mc, "complexity": cplx, "n_train": train["n"]}


def ic_overfit_report(
    factor_df: pl.DataFrame, daily: pl.DataFrame, train_ratio: float = 1.0
) -> dict:
    """市场无关的单因子防过拟合报告：全样本 IC/IR + bootstrap IC 95%CI + DSR(N=1)。

    ``factor_df``: ``[trade_date, ts_code, factor_value]``；``daily`` 用于算前向收益。
    A 股 ``fz validate overfit`` 与 crypto 单表达式验证共用此路径（避免双实现）。
    """
    from factorzen.discovery.guardrails import DeflationBasis, deflated_pvalue
    from factorzen.validation.bootstrap import block_bootstrap_ic_ci

    bundle = DataBundle.build(daily, train_ratio=train_ratio)
    clean = cross_sectional_zscore(factor_df, col="factor_value").rename(
        {"factor_value_z": "factor_clean"}
    )
    ic_res = compute_rank_ic(
        clean.select(["trade_date", "ts_code", "factor_clean"]),
        bundle.fwd_returns, factor_col="factor_clean", frequency="daily",
    )
    ic_vals = ic_res.ic_series["ic"].drop_nulls().drop_nans().to_numpy()
    lo, hi = block_bootstrap_ic_ci(ic_vals)
    # 单因子验证：语义上不存在 trial 池，N=1 → expected_max_sharpe 返回 0（无 deflation）。
    # 仍走共享入口，使 deflated_sharpe 的导入收口在 guardrails.py 一处（架构守卫测试强制）。
    _dsr, p = deflated_pvalue(ic_res.ir, DeflationBasis(n_trials=1, sharpe_variance=1.0),
                              len(ic_vals))
    return {"ic_mean": float(ic_res.ic_mean), "ir": float(ic_res.ir),
            "dsr_p": float(p), "ci_lo": float(lo), "ci_hi": float(hi), "n": len(ic_vals)}
