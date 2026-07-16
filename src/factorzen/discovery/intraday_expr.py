"""Bar 级逐元素表达式 → 日频 ``ix_*`` 叶子：求值、筛选、注册表与复现缓存。

落在 ``discovery`` 层：可静态依赖 ``intraday.sessions`` 与本包 ``expression`` /
``operators``（DAG：discovery → intraday → daily）。v1 DSL 仅允许逐元素算子
（无 ``ts_*`` / 截面），日聚合后产出日频面板，供挖掘注入与 factor run / lift /
forward 复现共用。
"""

from __future__ import annotations

import gc
import hashlib
import json
import warnings
from calendar import monthrange
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import DATA_RAW, INTRADAY_FEATURES_DIR
from factorzen.core.storage import load_parquet, save_parquet
from factorzen.discovery.expression import (
    OpNode,
    evaluate_materialized,
    parse_expr,
    to_expr_string,
)
from factorzen.discovery.operators import OPERATORS
from factorzen.intraday.sessions import (
    ASHARE_BAR_FREQS,
    canonicalize_minute,
    normalize_freq,
    resample_intraday,
)

# bar 级叶子：恒等映射（列名 = 叶名）
BAR_LEAVES: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "vol": "vol",
    "amount": "amount",
    "bar_ret": "bar_ret",
}


ELEMENTWISE_OPS: frozenset[str] = frozenset(
    name
    for name, spec in OPERATORS.items()
    if spec.category == "arith" and not name.startswith("ts_")
)

AGG_FUNCS: dict[str, Callable[[str], pl.Expr]] = {
    "sum": lambda c: pl.col(c).sum(),
    "mean": lambda c: pl.col(c).mean(),
    "std": lambda c: pl.col(c).std(),
    "skew": lambda c: pl.col(c).skew(),
    "min": lambda c: pl.col(c).min(),
    "max": lambda c: pl.col(c).max(),
    "last": lambda c: pl.col(c).last(),
    "first": lambda c: pl.col(c).first(),
    "median": lambda c: pl.col(c).median(),
}

_KEYS_DAY = ["ts_code", "trade_date"]
_EPS_DEGENERATE = 1e-12


def validate_bar_expr(bar_expr: str) -> Any:
    """解析并校验 bar 级表达式：仅允许 ``ELEMENTWISE_OPS`` 中的算子。

    Returns:
        discovery.expression.Node AST。

    Raises:
        ValueError: 未知叶子/算子，或含 ts_*/截面算子。
    """
    node = parse_expr(bar_expr, BAR_LEAVES)

    def walk(n: Any) -> None:
        if isinstance(n, OpNode):
            if n.op not in ELEMENTWISE_OPS:
                raise ValueError(
                    f"bar 级表达式禁止算子 {n.op!r}（v1 DSL 仅逐元素："
                    f"{sorted(ELEMENTWISE_OPS)}）"
                )
            for c in n.children:
                walk(c)

    walk(node)
    return node


@dataclass(frozen=True)
class IntradayExprSpec:
    """日内 bar 表达式规格：日聚合后的 ``ix_*`` 叶子定义。"""

    name: str
    bar_expr: str
    agg: str
    freq: str
    hypothesis: str = ""


def make_expr_spec(
    bar_expr: str,
    agg: str,
    *,
    freq: str = "5min",
    hypothesis: str = "",
) -> IntradayExprSpec:
    """校验并规范化，生成以哈希为名的 ``IntradayExprSpec``。

    ``name = "ix_" + sha1(f"{agg}|{canonical_expr}|{freq}")[:8]``，
    同 (agg, 规范化表达式, freq) 天然去重。
    """
    if agg not in AGG_FUNCS:
        raise ValueError(f"未知聚合: {agg!r}，支持 {sorted(AGG_FUNCS)}")
    freq_n = normalize_freq(freq)
    node = validate_bar_expr(bar_expr)
    canonical = to_expr_string(node)
    digest = hashlib.sha1(f"{agg}|{canonical}|{freq_n}".encode()).hexdigest()[:8]
    return IntradayExprSpec(
        name=f"ix_{digest}",
        bar_expr=canonical,
        agg=agg,
        freq=freq_n,
        hypothesis=hypothesis,
    )


def _month_windows(start: str, end: str) -> list[tuple[str, str, str]]:
    """``[(YYYY-MM, month_start_YYYYMMDD, month_end_YYYYMMDD), ...]`` 与 [start,end] 求交。"""
    s = datetime.strptime(start, "%Y%m%d").date()
    e = datetime.strptime(end, "%Y%m%d").date()
    if e < s:
        return []
    out: list[tuple[str, str, str]] = []
    y, m = s.year, s.month
    while True:
        first = date(y, m, 1)
        last = date(y, m, monthrange(y, m)[1])
        w0 = max(s, first)
        w1 = min(e, last)
        if w0 <= w1:
            out.append((f"{y:04d}-{m:02d}", w0.strftime("%Y%m%d"), w1.strftime("%Y%m%d")))
        if (y, m) == (e.year, e.month):
            break
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _empty_ix_panel(names: Sequence[str]) -> pl.DataFrame:
    schema: dict[str, Any] = {"trade_date": pl.Date, "ts_code": pl.String}
    for n in names:
        schema[n] = pl.Float64
    return pl.DataFrame(schema=schema)


def _add_bar_ret(bars: pl.DataFrame) -> pl.DataFrame:
    """按 (ts_code, 日) 排序派生 ``bar_ret``：首 bar = close/open−1，其余 close/prev−1。

    时间轴列须已命名为 ``trade_date``（Datetime，operators / crypto 先例）。
    """
    day = pl.col("trade_date").dt.date()
    return (
        bars.sort(["ts_code", "trade_date"])
        .with_columns(day.alias("_day"))
        .with_columns(pl.col("close").shift(1).over(["ts_code", "_day"]).alias("_pc"))
        .with_columns(
            pl.when(pl.col("_pc").is_null())
            .then(pl.col("close") / pl.col("open") - 1.0)
            .otherwise(pl.col("close") / pl.col("_pc") - 1.0)
            .alias("bar_ret")
        )
        .drop(["_day", "_pc"])
    )


def materialize_expr_features(
    specs: Sequence[IntradayExprSpec],
    start: str,
    end: str,
    *,
    freq: str = "5min",
    source_dir: Path | None = None,
    min_bar_coverage: float = 0.8,
) -> pl.DataFrame:
    """逐月物化 bar 表达式特征为日频面板。

    流程：``load_parquet(minute_1min)`` → canonicalize → resample → 时间轴改名
    ``trade_date`` → 派生 ``bar_ret`` → 全 specs 求值 → 按日聚合 → 覆盖守卫 →
    ``fill_nan(None)``。

    Args:
        specs: 表达式规格列表；**全部 freq 必须一致**。
        start / end: ``YYYYMMDD`` 闭区间。
        freq: 默认频率（与 specs 对齐校验）。
        source_dir: 1min 源湖根，默认 ``DATA_RAW``。
        min_bar_coverage: 有效 bar 覆盖率门槛。

    Returns:
        ``[trade_date(Date), ts_code, ix_*...]``，按 (trade_date, ts_code) 排序。

    Raises:
        ValueError: specs 空、混频、或与 ``freq`` 参数不一致。
    """
    if not specs:
        return _empty_ix_panel([])

    freq_n = normalize_freq(freq)
    freqs = {normalize_freq(s.freq) for s in specs}
    if len(freqs) != 1:
        raise ValueError(f"specs 混频: {sorted(freqs)}")
    spec_freq = next(iter(freqs))
    if spec_freq != freq_n:
        raise ValueError(f"specs freq={spec_freq!r} 与参数 freq={freq_n!r} 不一致")

    names = [s.name for s in specs]
    n_bars = ASHARE_BAR_FREQS[freq_n].bars_per_day
    src = DATA_RAW if source_dir is None else Path(source_dir)
    windows = _month_windows(start, end)
    parts: list[pl.DataFrame] = []

    for _label, m_start, m_end in windows:
        try:
            lf = load_parquet(
                "minute_1min",
                start=m_start,
                end=m_end,
                date_col="trade_time",
                base_dir=src,
            )
            minute = lf.collect()
        except Exception:
            continue

        if minute.is_empty():
            del minute
            gc.collect()
            continue

        panel = _materialize_month(minute, specs, freq_n, min_bar_coverage, n_bars, names)
        del minute
        gc.collect()
        if not panel.is_empty():
            parts.append(panel)
        else:
            del panel
            gc.collect()

    if not parts:
        return _empty_ix_panel(names)
    out = pl.concat(parts, how="vertical_relaxed")
    return out.sort(["trade_date", "ts_code"])


def _materialize_month(
    minute: pl.DataFrame,
    specs: Sequence[IntradayExprSpec],
    freq_n: str,
    min_bar_coverage: float,
    n_bars: int,
    names: Sequence[str],
) -> pl.DataFrame:
    """单月 1min 帧 → 日频 ix 面板。"""
    canon = canonicalize_minute(minute.lazy()).collect()
    if canon.is_empty():
        return _empty_ix_panel(names)

    bars = resample_intraday(canon, freq_n)
    if bars.is_empty():
        return _empty_ix_panel(names)

    # 时间轴改名 trade_date（operators 约定；crypto 先例）
    work = bars.rename({"trade_time": "trade_date"})
    work = _add_bar_ret(work)

    # 全部 bar_expr 求值到中间列
    mid_names: list[str] = []
    for spec in specs:
        node = parse_expr(spec.bar_expr, BAR_LEAVES)
        series = evaluate_materialized(node, work, BAR_LEAVES)
        mid = f"__ix_{spec.name}"
        work = work.with_columns(series.alias(mid))
        mid_names.append(mid)

    day_col = pl.col("trade_date").dt.date().alias("trade_date")
    work = work.with_columns(day_col)

    # 有效 bar：close 非空（覆盖守卫）
    work = work.with_columns(pl.col("close").is_not_null().alias("_valid_bar"))

    agg_exprs: list[pl.Expr] = [
        pl.col("_valid_bar").sum().cast(pl.Int32).alias("_n_valid"),
    ]
    for spec, mid in zip(specs, mid_names, strict=True):
        agg_fn = AGG_FUNCS[spec.agg]
        agg_exprs.append(agg_fn(mid).alias(spec.name))

    panel = (
        work.sort(["ts_code", "trade_date"])
        .group_by(["ts_code", "trade_date"], maintain_order=True)
        .agg(agg_exprs)
    )

    threshold = min_bar_coverage * float(n_bars)
    null_feats = [
        pl.when(pl.col("_n_valid").cast(pl.Float64) < threshold)
        .then(None)
        .otherwise(pl.col(name))
        .alias(name)
        for name in names
    ]
    out = (
        panel.select(
            pl.col("trade_date").cast(pl.Date),
            pl.col("ts_code").cast(pl.String),
            *null_feats,
        )
        .with_columns([pl.col(c).fill_nan(None) for c in names])
        .sort(["trade_date", "ts_code"])
    )
    return out


def screen_expr_panel(
    panel: pl.DataFrame,
    reference: pl.DataFrame | None = None,
    *,
    min_coverage: float = 0.6,
    max_abs_corr: float = 0.9,
) -> dict[str, str]:
    """廉价筛选每个 ``ix_*`` 列：keep / low_coverage / degenerate / correlated:<col>。

    - low_coverage：非空比例 < ``min_coverage``
    - degenerate：逐日截面 std 的中位数 ≈ 0
    - correlated:<col>：与 reference 中某列的日频截面 rank 相关中位 |ρ| ≥ ``max_abs_corr``
    """
    ix_cols = [c for c in panel.columns if c.startswith("ix_")]
    verdict: dict[str, str] = {}
    if not ix_cols or panel.is_empty():
        return {c: "low_coverage" for c in ix_cols}

    n = panel.height
    for col in ix_cols:
        nn = int(panel[col].is_not_null().sum())
        if n == 0 or (nn / n) < min_coverage:
            verdict[col] = "low_coverage"
            continue

        # 逐日截面 std → 中位
        day_std = (
            panel.group_by("trade_date")
            .agg(pl.col(col).std().alias("_s"))
            .select(pl.col("_s").median())
        )
        med_std = day_std[0, 0]
        if med_std is None or (
            isinstance(med_std, (int, float)) and abs(float(med_std)) < _EPS_DEGENERATE
        ):
            verdict[col] = "degenerate"
            continue

        if reference is not None and not reference.is_empty():
            ref_cols = [
                c
                for c in reference.columns
                if c not in ("trade_date", "ts_code") and c != col
            ]
            if ref_cols:
                joined = panel.select(["trade_date", "ts_code", col]).join(
                    reference.select(
                        ["trade_date", "ts_code", *[c for c in ref_cols if c in reference.columns]]
                    ),
                    on=["trade_date", "ts_code"],
                    how="inner",
                )
                rejected = False
                for rc in ref_cols:
                    if rc not in joined.columns:
                        continue
                    # 日频截面 rank 相关
                    ranked = joined.select(
                        pl.col("trade_date"),
                        pl.col(col).rank().over("trade_date").alias("_a"),
                        pl.col(rc).rank().over("trade_date").alias("_b"),
                    )
                    daily_rho = (
                        ranked.group_by("trade_date")
                        .agg(pl.corr("_a", "_b").alias("_rho"))
                        .select(pl.col("_rho").abs().median())
                    )
                    med = daily_rho[0, 0]
                    if med is not None and float(med) >= max_abs_corr:
                        verdict[col] = f"correlated:{rc}"
                        rejected = True
                        break
                if rejected:
                    continue

        verdict[col] = "keep"
    return verdict


# ── 注册表 ──────────────────────────────────────────────────────────────


def registry_path(base_dir: Path | None = None) -> Path:
    """``{INTRADAY_FEATURES_DIR}/expr_registry.jsonl``。"""
    base = INTRADAY_FEATURES_DIR if base_dir is None else Path(base_dir)
    return base / "expr_registry.jsonl"


def load_expr_registry(base_dir: Path | None = None) -> dict[str, IntradayExprSpec]:
    """加载表达式注册表；文件不存在 → 空 dict。"""
    path = registry_path(base_dir)
    if not path.exists():
        return {}
    out: dict[str, IntradayExprSpec] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or "name" not in rec:
            continue
        name = str(rec["name"])
        try:
            out[name] = IntradayExprSpec(
                name=name,
                bar_expr=str(rec["bar_expr"]),
                agg=str(rec["agg"]),
                freq=str(rec.get("freq", "5min")),
                hypothesis=str(rec.get("hypothesis", "")),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def register_expr_features(
    specs: Sequence[IntradayExprSpec],
    *,
    session: str,
    base_dir: Path | None = None,
) -> None:
    """将 specs 追加写入注册表；同 name 幂等跳过。"""
    path = registry_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_expr_registry(base_dir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8") as f:
        for spec in specs:
            if spec.name in existing:
                continue
            rec = {
                **asdict(spec),
                "session": session,
                "created_at": now,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing[spec.name] = spec


def _exp_cache_data_type(freq: str, name: str) -> str:
    return f"exp/{normalize_freq(freq)}/{name}"


def _cached_months_covered(
    cache_dir: Path,
) -> set[tuple[int, int]]:
    """扫描 ``year=/month=/`` 分区，返回已有 (year, month) 集合。"""
    covered: set[tuple[int, int]] = set()
    if not cache_dir.exists():
        return covered
    for ydir in cache_dir.glob("year=*"):
        try:
            year = int(ydir.name.split("=", 1)[1])
        except ValueError:
            continue
        for mdir in ydir.glob("month=*"):
            try:
                month = int(mdir.name.split("=", 1)[1])
            except ValueError:
                continue
            if (mdir / "data.parquet").exists():
                covered.add((year, month))
    return covered


def ensure_expr_panel(
    name: str,
    start: str,
    end: str,
    *,
    base_dir: Path | None = None,
    source_dir: Path | None = None,
) -> pl.DataFrame:
    """复现路径：查 registry → 读/补缓存 → 返回 ``[trade_date, ts_code, name]``。

    缓存布局：``{base_dir}/exp/{freq}/{name}/year=/month=/``。
    未覆盖月份现场物化并落盘。

    Raises:
        ValueError: registry 无此 name。
    """
    base = INTRADAY_FEATURES_DIR if base_dir is None else Path(base_dir)
    reg = load_expr_registry(base)
    if name not in reg:
        raise ValueError(
            f"日内表达式叶子 {name!r} 未注册；"
            f"请先 register_expr_features（registry={registry_path(base)}）"
        )
    spec = reg[name]
    freq_n = normalize_freq(spec.freq)
    data_type = _exp_cache_data_type(freq_n, name)
    cache_root = base / data_type

    # 需要的自然月
    needed = _month_windows(start, end)
    covered = _cached_months_covered(cache_root)
    missing_windows: list[tuple[str, str, str]] = []
    for label, m0, m1 in needed:
        y, m = int(label[:4]), int(label[5:7])
        if (y, m) not in covered:
            missing_windows.append((label, m0, m1))

    if missing_windows:
        # 合并缺失窗口的起止，一次物化后按月落缓存
        miss_start = min(w[1] for w in missing_windows)
        miss_end = max(w[2] for w in missing_windows)
        panel = materialize_expr_features(
            [spec],
            miss_start,
            miss_end,
            freq=freq_n,
            source_dir=source_dir,
        )
        if not panel.is_empty():
            # 只保留 name 列写入缓存
            to_save = panel.select(["trade_date", "ts_code", name])
            save_parquet(
                to_save,
                data_type=data_type,
                date_col="trade_date",
                base_dir=base,
                mode="overwrite",
            )
        del panel
        gc.collect()

    # 读缓存
    try:
        lf = load_parquet(
            data_type,
            start=start,
            end=end,
            date_col="trade_date",
            base_dir=base,
        )
        out = lf.collect()
    except Exception:
        out = _empty_ix_panel([name])

    if out.is_empty():
        return _empty_ix_panel([name])
    keep = [c for c in ("trade_date", "ts_code", name) if c in out.columns]
    if name not in out.columns:
        return _empty_ix_panel([name])
    return out.select(keep).sort(["trade_date", "ts_code"])


# ── attach：复现消费方 join ix_* ───────────────────────────────────────


def _frame_date_bounds(df: pl.DataFrame) -> tuple[str | None, str | None]:
    """帧内最早/最晚 trade_date → ``YYYYMMDD``；空帧 → (None, None)。"""
    if df.is_empty() or "trade_date" not in df.columns:
        return None, None
    col = df["trade_date"]
    try:
        mn, mx = col.min(), col.max()
    except Exception:
        return None, None
    return _to_yyyymmdd(mn), _to_yyyymmdd(mx)


def _to_yyyymmdd(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.strftime("%Y%m%d")
    if isinstance(v, date):
        return v.strftime("%Y%m%d")
    s = str(v).strip().replace("-", "")
    if len(s) >= 8 and s[:8].isdigit():
        return s[:8]
    return None


def _align_trade_date(sel: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """把面板 trade_date 类型对齐到 frame（Date / Utf8 双向兼容）。"""
    if "trade_date" not in sel.columns or "trade_date" not in frame.columns:
        return sel
    src_dt = sel["trade_date"].dtype
    tgt_dt = frame["trade_date"].dtype
    if src_dt == tgt_dt:
        return sel
    if tgt_dt == pl.Date and src_dt in (pl.Utf8, pl.String):
        return sel.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        )
    if tgt_dt in (pl.Utf8, pl.String) and src_dt == pl.Date:
        return sel.with_columns(pl.col("trade_date").dt.strftime("%Y%m%d"))
    if tgt_dt == pl.Date:
        return sel.with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y%m%d", strict=False)
        )
    return sel


def _ensure_expr_null_cols(
    frame: pl.DataFrame, names: Sequence[str],
) -> pl.DataFrame:
    missing = [c for c in names if c not in frame.columns]
    if missing:
        frame = frame.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing]
        )
    return frame


def attach_expr_leaves(
    frame: pl.DataFrame,
    names: Sequence[str],
    *,
    require: bool = False,
    base_dir: Path | None = None,
    source_dir: Path | None = None,
) -> pl.DataFrame:
    """对每个 ``ix_*`` 名 ``ensure_expr_panel`` 后 left-join 进日频帧。

    Args:
        frame: 日频帧，须含 ``trade_date`` / ``ts_code``（可与 builtin ``i_*`` 已 join）。
        names: 要 attach 的 ``ix_*`` 叶名。
        require: registry 缺失 / 面板空时 ``True`` → raise；``False`` → null 列 + warn。
        base_dir: registry/缓存根，默认 ``INTRADAY_FEATURES_DIR``。
        source_dir: 1min 源湖，默认 ``DATA_RAW``。

    Returns:
        含请求 ``ix_*`` 列的帧（缺失列为 Float64 null）。
    """
    if not names:
        return frame

    start_s, end_s = _frame_date_bounds(frame)
    if start_s is None or end_s is None:
        return _ensure_expr_null_cols(frame, names)

    out = frame
    for name in names:
        if not name or not str(name).startswith("ix_"):
            continue
        name_s = str(name)
        try:
            ep = ensure_expr_panel(
                name_s,
                start_s,
                end_s,
                base_dir=base_dir,
                source_dir=source_dir,
            )
        except ValueError:
            if require:
                raise
            warnings.warn(
                f"日内表达式叶子 {name_s} 未注册或不可用，补 null；"
                f"leaf_health 将摘除零覆盖叶子。",
                stacklevel=2,
            )
            out = _ensure_expr_null_cols(out, [name_s])
            continue
        except Exception as exc:
            if require:
                raise ValueError(
                    f"日内表达式叶子 {name_s} 物化失败: {type(exc).__name__}: {exc}"
                ) from exc
            warnings.warn(
                f"日内表达式叶子 {name_s} 物化失败（{type(exc).__name__}），补 null。",
                stacklevel=2,
            )
            out = _ensure_expr_null_cols(out, [name_s])
            continue

        if ep is None or ep.is_empty() or name_s not in ep.columns:
            if require:
                raise ValueError(
                    f"日内表达式叶子 {name_s} 面板为空；请检查 registry 与 minute 源湖"
                )
            warnings.warn(
                f"日内表达式叶子 {name_s} 面板为空，补 null。",
                stacklevel=2,
            )
            out = _ensure_expr_null_cols(out, [name_s])
            continue

        sel = ep.select(["trade_date", "ts_code", name_s])
        sel = _align_trade_date(sel, out)
        if name_s in out.columns:
            out = out.drop(name_s)
        out = out.join(sel, on=["trade_date", "ts_code"], how="left")
    return out


__all__ = [
    "AGG_FUNCS",
    "BAR_LEAVES",
    "ELEMENTWISE_OPS",
    "IntradayExprSpec",
    "attach_expr_leaves",
    "ensure_expr_panel",
    "load_expr_registry",
    "make_expr_spec",
    "materialize_expr_features",
    "register_expr_features",
    "registry_path",
    "screen_expr_panel",
    "validate_bar_expr",
]
