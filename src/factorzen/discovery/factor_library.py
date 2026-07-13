# src/factorzen/discovery/factor_library.py
"""因子库登记系统（分市场 · 全信息 · 自动维护）。

用**当前统一标准**（library gate：真+有信号，复用 `guardrails.acceptance_reasons(gate="library")`）
+ **统一默认窗口**（`backtest_window.default_window`）重算，把散落各处的合格因子收敛成一份
可比、带全信息、按市场分文件、能自动增量维护的登记簿。

**故意不用残差 IC 准入**：因子库是参照系；对参照系自身做「对库残差化」是循环定义。
库 upsert/rebuild 维持裸 IC + 覆盖门。残差目标只用于挖掘评估（``discovery.residual`` /
``node_guardrails`` / ``run_session``），测候选对库的真增量——本模块不接入 objective=residual。

- 分市场分文件：``{root}/{market}.jsonl``（机器读写）+ ``{market}.md``（人类汇总）+ ``summary.md``。
- 去相关 = **方案 A**：库内高相关因子仍收录但打 ``status="correlated"`` + ``correlated_with``，
  不替用户丢弃（看全貌，用户自己挑）。逐对相关走**紧凑 float32 矩阵**（内存有界，修真实 A股
  rebuild OOM），`_avg_cs_corr_matrices` 精确复刻 `compute_factor_correlation` 语义（parity 测试锁死）。
- 门槛复用 `acceptance_reasons(gate="library")`，不另写、不放松。
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import polars as pl

from factorzen.discovery.expression import evaluate_materialized, parse_expr, to_expr_string
from factorzen.discovery.guardrails import acceptance_reasons

_LOG = logging.getLogger(__name__)

DEFAULT_ROOT = "workspace/factor_library"

# 去相关默认参数（内存有界）：每批评估表达式数、去相关紧凑矩阵的日期采样上限。
DEFAULT_EVAL_BATCH = 64
DEFAULT_DECORR_MAX_DATES = 500

# 因子值面板物化器：规范表达式 → [trade_date, ts_code, factor_value]（或 None=无法物化）。
Materializer = Callable[[str], "pl.DataFrame | None"]
# 紧凑矩阵物化器：规范表达式 → (date × stock) float32 矩阵（NaN=缺，或 None=无法物化）。
# 去相关走它而非完整面板，把内存从「面板×因子数」压到「小矩阵×因子数」，随因子数有界。
CompactMaterializer = Callable[[str], "np.ndarray | None"]


def _avg_cs_corr_matrices(a: np.ndarray, b: np.ndarray) -> float:
    """两个 (date×stock) 因子矩阵的**逐日截面相关性均值**（带符号）。

    精确复刻 `daily.evaluation.correlation.compute_factor_correlation` 语义（去相关唯一真源，
    有 parity 测试锁死）：逐日只用两因子**都有值**的股票（inner join），该日有效股 <30 或任一
    方差为 0 则跳过；对每有效日算 Pearson，再跨日平均。全向量化（无 python 逐日循环），
    内存只用几个 date×stock 临时数组，随矩阵尺寸有界。空/全无效 → 0.0。
    """
    both = np.isfinite(a) & np.isfinite(b)
    cnt = both.sum(axis=1)
    ok = cnt >= 30
    if not ok.any():
        return 0.0
    # sum/count 代替 nanmean：全空日 cnt=0 不触发 "Mean of empty slice"，ok 行语义不变
    cnt_col = np.maximum(cnt, 1).astype(a.dtype, copy=False)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(both, a, 0.0).sum(axis=1, keepdims=True) / cnt_col
        mb = np.where(both, b, 0.0).sum(axis=1, keepdims=True) / cnt_col
        da = np.where(both, a - ma, 0.0)
        db = np.where(both, b - mb, 0.0)
        cov = (da * db).sum(axis=1)
        va = (da * da).sum(axis=1)
        vb = (db * db).sum(axis=1)
        denom = np.sqrt(va * vb)
        per_date = np.where(ok & (denom > 0), cov / denom, np.nan)
    vals = per_date[np.isfinite(per_date)]
    return float(vals.mean()) if vals.size else 0.0


def make_compact_materializer(
    prepped: pl.DataFrame, leaf_map: dict[str, str] | None = None, *,
    max_dates: int = DEFAULT_DECORR_MAX_DATES,
) -> CompactMaterializer:
    """在已预处理帧 ``prepped`` 上构造紧凑矩阵物化器，供去相关（内存有界）。

    固定 (date × stock) 网格（日期超 ``max_dates`` 时等距下采样封顶），每个因子求值后**立即**
    散射进 float32 矩阵（丢掉重的 ts_code 字符串列）→ 单因子 ≤ max_dates×n_stock×4 字节。
    ``prepped`` 须已含派生列、按 (ts_code, trade_date) 排序（与求值一致）。
    """
    dates = sorted(prepped["trade_date"].unique().to_list())
    if len(dates) > max_dates:
        stride = math.ceil(len(dates) / max_dates)
        dates = dates[::stride]
    stocks = sorted(prepped["ts_code"].unique().to_list())
    date_idx = {d: i for i, d in enumerate(dates)}
    stock_idx = {s: i for i, s in enumerate(stocks)}
    d_n, s_n = len(dates), len(stocks)
    ridx = np.fromiter((date_idx.get(d, -1) for d in prepped["trade_date"].to_list()),
                       dtype=np.int64, count=prepped.height)
    cidx = np.fromiter((stock_idx.get(s, -1) for s in prepped["ts_code"].to_list()),
                       dtype=np.int64, count=prepped.height)
    row_ok = ridx >= 0  # 未被采样的日期行丢弃

    def _compact(expr: str) -> np.ndarray | None:
        try:
            node = parse_expr(expr, leaf_map)
            v = evaluate_materialized(node, prepped, leaf_map).to_numpy()
        except Exception:
            return None
        m = np.full((d_n, s_n), np.nan, dtype=np.float32)
        fin = row_ok & np.isfinite(v)
        m[ridx[fin], cidx[fin]] = v[fin].astype(np.float32)
        return m

    return _compact


@dataclass
class FactorRecord:
    """一条因子登记记录（全字段）。绝对日期 ``added_at``/``updated_at`` 由调用方从系统时间取一次传入。"""

    expression: str
    market: str
    hypothesis: str | None = None
    ic_train: float | None = None
    ir_train: float | None = None
    holdout_ic: float | None = None
    holdout_ir: float | None = None
    dsr: float | None = None
    dsr_pvalue: float | None = None
    n_train: int | None = None
    ic_ci_low: float | None = None
    ic_ci_high: float | None = None
    pbo: float | None = None
    turnover: float | None = None
    status: str = "active"           # active（独立）/ correlated（与库内已有高相关，仍收录）
    max_corr_in_lib: float | None = None
    correlated_with: str | None = None
    eval_start: str | None = None
    eval_end: str | None = None
    universe: str | None = None
    horizon: int | None = None
    source_run_id: str | None = None
    source_session_dir: str | None = None
    git_sha: str | None = None
    added_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict:
        """JSON 安全序列化：NaN/inf → None（jsonl 须合法 JSON）。"""
        return {k: _safe(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> FactorRecord:
        """容忍未知/缺失字段（向前兼容）。"""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class UpsertResult:
    added: int = 0
    updated: int = 0
    correlated: int = 0        # 本批被标记为 correlated 的数量
    skipped: int = 0           # 未过 library gate 被跳过的数量
    records: list[FactorRecord] = field(default_factory=list)  # 本批新增/更新的记录（含最终 status）


# ── 序列化辅助 ───────────────────────────────────────────────────────────────

def _safe(v):
    """NaN/inf 浮点 → None，其余原样（保证 jsonl 合法、md 不出 'nan'）。"""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _normalize(expr: str, leaf_map: dict[str, str] | None = None) -> str:
    """规范形（去重主键）。parse 失败 → 原串（与 experiment_index._normalize 同容错）。"""
    try:
        return to_expr_string(parse_expr(expr, leaf_map))
    except ValueError:
        return expr


def library_path(market: str, root: str = DEFAULT_ROOT) -> Path:
    return Path(root) / f"{market}.jsonl"


def load_library(market: str, root: str = DEFAULT_ROOT) -> list[FactorRecord]:
    """读 jsonl → 记录列表。文件不存在 → 空列表。损坏行跳过。"""
    path = library_path(market, root)
    if not path.exists():
        return []
    out: list[FactorRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(FactorRecord.from_dict(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return out


def library_covered_expressions(
    market: str, *, k: int = 10, statuses: tuple[str, ...] = ("active",),
    root: str = DEFAULT_ROOT,
) -> list[str]:
    """库内 status∈statuses 因子按 |ic_train| 降序取前 k 的表达式（供 LLM 提示）。

    文件不存在/空 → []。不物化、不求值。
    """
    recs = [r for r in load_library(market, root=root) if r.status in statuses]
    recs.sort(key=lambda r: (-abs(r.ic_train or 0.0), r.expression))
    return [r.expression for r in recs[:k]]


def build_library_pool(
    market: str,
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None = None,
    *,
    statuses: tuple[str, ...] = ("active",),
    root: str = DEFAULT_ROOT,
    eval_start=None,
) -> dict[str, pl.DataFrame]:
    """把库内因子物化为 mining/评估帧上的因子值面板，供搜索期库级正交去相关。

    - 取 status∈statuses 记录，按 |ic_train| 降序。
    - 每条 expression 用 ``evaluate_materialized`` 在 ``daily`` 上算
      ``[trade_date, ts_code, factor_value]``（与挖掘物化路径一致）。
    - 非法/求值失败/全 null 的表达式跳过并计数——一条坏记录不许崩整个 pool。
    - 库文件不存在/空 → {}。
    - ``eval_start``：可选，求值后裁到该日起（team holdout 口径扩窗预热时传入 holdout 起点）。

    调用方负责 ``daily`` 已与挖掘同 prep（派生列/停牌掩码等）；本函数不再二次预处理。
    """
    recs = [r for r in load_library(market, root=root) if r.status in statuses]
    if not recs:
        return {}
    recs.sort(key=lambda r: (-abs(r.ic_train or 0.0), r.expression))

    df = daily.sort(["ts_code", "trade_date"])
    pool: dict[str, pl.DataFrame] = {}
    n_skip = 0
    for r in recs:
        try:
            node = parse_expr(r.expression, leaf_map)
            series = evaluate_materialized(node, df, leaf_map)
            panel = (
                df.select(["trade_date", "ts_code"])
                .with_columns(series.alias("factor_value"))
                .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
            )
            if eval_start is not None:
                panel = panel.filter(pl.col("trade_date") >= eval_start)
            if panel.is_empty():
                n_skip += 1
                continue
            pool[r.expression] = panel
        except Exception as exc:
            n_skip += 1
            _LOG.debug("build_library_pool skip %r: %s: %s",
                       r.expression, type(exc).__name__, exc)
            continue
    if n_skip:
        _LOG.info("build_library_pool(%s): skipped %d / kept %d",
                  market, n_skip, len(pool))
    return pool


def _save_library(market: str, records: list[FactorRecord], root: str = DEFAULT_ROOT) -> None:
    path = library_path(market, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records)
    path.write_text(payload, encoding="utf-8")


# ── upsert ───────────────────────────────────────────────────────────────────

def _record_from_candidate(
    cand: dict, norm_expr: str, market: str, eval_window: tuple[str, str],
    universe: str | None, horizon: int | None, run_id: str | None,
    session_dir: str | None, git_sha: str | None, now: str,
    prev: FactorRecord | None,
) -> FactorRecord:
    """候选 dict + 上下文 → FactorRecord。缺字段以 None 兜底（mining/agents 候选字段口径不一）。"""
    def g(*keys):
        for k in keys:
            if k in cand and cand[k] is not None:
                return cand[k]
        return None

    eval_start, eval_end = eval_window
    return FactorRecord(
        expression=norm_expr,
        market=market,
        hypothesis=g("hypothesis"),
        ic_train=g("ic_train"),
        ir_train=g("ir_train"),
        holdout_ic=g("holdout_ic"),
        holdout_ir=g("holdout_ir"),
        dsr=g("dsr"),
        dsr_pvalue=g("dsr_pvalue"),
        n_train=g("n_train"),
        ic_ci_low=g("ic_ci_low"),
        ic_ci_high=g("ic_ci_high"),
        pbo=g("pbo"),
        turnover=g("turnover"),
        # status/max_corr/correlated_with 由去相关阶段填
        eval_start=eval_start,
        eval_end=eval_end,
        universe=universe,
        horizon=horizon,
        source_run_id=run_id,
        source_session_dir=session_dir,
        git_sha=git_sha,
        added_at=prev.added_at if prev is not None else now,   # 保留原入库日
        updated_at=now,
    )


def _panel_to_compact(panel: pl.DataFrame, date_idx: dict, stock_idx: dict,
                      d_n: int, s_n: int, *, dtype=np.float32) -> np.ndarray:
    """把 [trade_date, ts_code, factor_value] 面板散射成固定网格的紧凑矩阵。

    默认 ``float32``（库 rebuild 内存有界）；组合层贪心去相关传 ``float64`` 以锁
    与 ``max_correlation`` 的 corr 数值 parity（≤1e-9）。
    """
    m = np.full((d_n, s_n), np.nan, dtype=dtype)
    r = np.fromiter((date_idx.get(d, -1) for d in panel["trade_date"].to_list()),
                    dtype=np.int64, count=panel.height)
    c = np.fromiter((stock_idx.get(s, -1) for s in panel["ts_code"].to_list()),
                    dtype=np.int64, count=panel.height)
    v = panel["factor_value"].to_numpy().astype(dtype, copy=False)
    keep = (r >= 0) & (c >= 0) & np.isfinite(v)
    m[r[keep], c[keep]] = v[keep]
    return m


def _compact_of_from_panels(exprs: list[str], materialize: Materializer) -> CompactMaterializer:
    """由面板物化器构造紧凑矩阵物化器（供只有 panel 接口的调用方/测试）。

    先一趟扫所有面板收集 (date, stock) 并集建**共享网格**（供跨因子对齐），再逐因子转紧凑矩阵。
    仅用于小规模（测试/单次挖掘收尾）；大规模真实 rebuild 走 `make_compact_materializer`
    （网格来自 prepped，无需预扫、无二次物化）。
    """
    dates: set = set()
    stocks: set = set()
    panels: dict[str, pl.DataFrame | None] = {}
    for e in exprs:
        try:
            p = materialize(e)
        except Exception:
            p = None
        panels[e] = p
        if p is not None and p.height:
            dates |= set(p["trade_date"].to_list())
            stocks |= set(p["ts_code"].to_list())
    date_idx = {d: i for i, d in enumerate(sorted(dates))}
    stock_idx = {s: i for i, s in enumerate(sorted(stocks))}
    d_n, s_n = len(date_idx), len(stock_idx)

    def _compact(expr: str) -> np.ndarray | None:
        p = panels.get(expr)
        if p is None or not p.height:
            return None
        return _panel_to_compact(p, date_idx, stock_idx, d_n, s_n)

    return _compact


def _decorrelate(
    affected: list[FactorRecord], unchanged: list[FactorRecord],
    compact_of: CompactMaterializer | None, decorr_threshold: float,
) -> int:
    """方案 A 去相关（内存有界）：对 affected 逐个与「其它 active 因子」算逐对相关。

    贪心：按 |ir_train| 降序处理（强者先占 active 位），弱者若与已 active 者超阈 → 标 correlated
    （仍收录）+ 记 ``max_corr_in_lib``/``correlated_with``（最相关者）；否则 active 并入池。
    ``unchanged``：本批未触及、状态稳定的库内 active 记录，始终在池中。
    ``compact_of``：expr → 紧凑 (date×stock) 矩阵；缓存的是**小矩阵**（非完整面板），内存随
    因子数有界。``None`` → 不去相关，全部 active。返回被标记 correlated 的数量。
    """
    if compact_of is None:
        for r in affected:
            r.status = "active"
        return 0

    mat_cache: dict[str, np.ndarray | None] = {}

    def mat(expr: str) -> np.ndarray | None:
        if expr not in mat_cache:
            try:
                mat_cache[expr] = compact_of(expr)
            except Exception:
                mat_cache[expr] = None
        return mat_cache[expr]

    active_pool: list[str] = [r.expression for r in unchanged if r.status == "active"]
    ordered = sorted(affected, key=lambda r: (-abs(r.ir_train or 0.0), r.expression))
    correlated = 0
    for rec in ordered:
        cand = mat(rec.expression)
        if cand is None:
            rec.status = "active"      # 无法物化 → 无从去相关，保守留 active
            active_pool.append(rec.expression)
            continue
        best_val, best_expr = 0.0, None
        for other in active_pool:
            if other == rec.expression:
                continue
            om = mat(other)
            if om is None:
                continue
            c = abs(_avg_cs_corr_matrices(cand, om))
            if c > best_val:
                best_val, best_expr = c, other
        rec.max_corr_in_lib = round(float(best_val), 4)
        if best_expr is not None and best_val > decorr_threshold:
            rec.status = "correlated"
            rec.correlated_with = best_expr
            correlated += 1
        else:
            rec.status = "active"
            active_pool.append(rec.expression)
    return correlated


def upsert(
    market: str, candidates: list[dict], *,
    eval_window: tuple[str, str], universe: str | None, horizon: int | None,
    run_id: str | None, session_dir: str | None, git_sha: str | None, now: str,
    decorr_threshold: float = 0.7, materialize: Materializer | None = None,
    compact_materialize: CompactMaterializer | None = None,
    leaf_map: dict[str, str] | None = None, root: str = DEFAULT_ROOT,
) -> UpsertResult:
    """把候选 upsert 进 ``{market}.jsonl`` + 重生 md。

    1. 每个候选先过 `acceptance_reasons(gate="library")`——不过则跳过（库只收合格的）。
    2. 规范形去重：已在库→更新指标 + ``updated_at=now``（保留原 ``added_at``）；新→新增。
    3. 去相关（方案 A，**内存有界**）：与库内其它 active 因子算逐对相关（紧凑矩阵），超阈标
       correlated 仍收录。去相关物化优先用 ``compact_materialize``（真实大规模调用方传，网格来自
       prepped、单因子只驻小矩阵）；否则退回 ``materialize``（面板，仅小规模/测试，内部转紧凑）。
    4. 写回 jsonl + 重生 ``{market}.md`` + 刷新 ``summary.md``。
    """
    existing = load_library(market, root=root)
    by_expr: dict[str, FactorRecord] = {r.expression: r for r in existing}

    res = UpsertResult()
    affected: list[FactorRecord] = []
    affected_exprs: set[str] = set()
    for cand in candidates:
        raw = cand.get("expression")
        if not raw:
            continue
        # holdout_n_days：覆盖门（P1）。upsert 是 M1/M5-M6 之外的第三条 gate 路径
        # （rebuild 走它）——漏传会让稀薄 holdout（如北向季末残留）靠运气混进库。
        if acceptance_reasons(gate="library", ic_train=cand.get("ic_train"),
                              holdout_ic=cand.get("holdout_ic"),
                              dsr_pvalue=cand.get("dsr_pvalue"),
                              holdout_n_days=cand.get("n_holdout_days")):
            res.skipped += 1
            continue
        norm = _normalize(raw, leaf_map)
        prev = by_expr.get(norm)
        rec = _record_from_candidate(cand, norm, market, eval_window, universe, horizon,
                                     run_id, session_dir, git_sha, now, prev)
        if prev is None:
            res.added += 1
        else:
            res.updated += 1
        by_expr[norm] = rec
        if norm not in affected_exprs:
            affected.append(rec)
            affected_exprs.add(norm)

    # 去相关：affected 对照「未触及的库内 active 记录」（都要紧凑矩阵）。
    unchanged = [r for e, r in by_expr.items() if e not in affected_exprs]
    compact_of = compact_materialize
    if compact_of is None and materialize is not None:
        needed = [r.expression for r in affected] + \
                 [r.expression for r in unchanged if r.status == "active"]
        compact_of = _compact_of_from_panels(needed, materialize)
    res.correlated = _decorrelate(affected, unchanged, compact_of, decorr_threshold)
    res.records = affected

    _save_library(market, list(by_expr.values()), root=root)
    render_markdown(market, root=root)
    return res


# ── markdown 渲染 ────────────────────────────────────────────────────────────

def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


def _sort_key(r: FactorRecord):
    """按 holdout_ic 降序（OOS 信号），次级 ir_train。None/NaN 视作 -inf 沉底。"""
    def val(x):
        return x if (isinstance(x, (int, float)) and math.isfinite(x)) else float("-inf")
    return (-val(r.holdout_ic), -val(r.ir_train))


def render_markdown(market: str, root: str = DEFAULT_ROOT) -> str:
    """生成并写 ``{market}.md``（统计行 + 降序表）+ 刷新 ``summary.md``。空库不崩。"""
    records = load_library(market, root=root)
    n_active = sum(1 for r in records if r.status == "active")
    n_corr = sum(1 for r in records if r.status == "correlated")
    windows = sorted({(r.eval_start, r.eval_end) for r in records if r.eval_start})
    win_str = ", ".join(f"{s}–{e}" for s, e in windows) if windows else "-"
    updated = max((r.updated_at for r in records if r.updated_at), default="-")

    lines = [
        f"# 因子库 · {market}",
        "",
        f"- active: **{n_active}** · correlated: **{n_corr}** · 合计 {len(records)}",
        f"- 评估窗口: {win_str}",
        f"- 更新时间: {updated}",
        "",
    ]
    if records:
        lines += [
            "| # | expression | market | ic_train | holdout_ic | dsr_pvalue | n_train | status | eval 窗口 | added_at |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(sorted(records, key=_sort_key), 1):
            win = f"{r.eval_start or '-'}–{r.eval_end or '-'}"
            corr = f" (~{r.correlated_with})" if r.status == "correlated" and r.correlated_with else ""
            lines.append(
                f"| {i} | `{r.expression}` | {r.market} | {_fmt(r.ic_train)} | "
                f"{_fmt(r.holdout_ic)} | {_fmt(r.dsr_pvalue)} | {_fmt(r.n_train)} | "
                f"{r.status}{corr} | {win} | {r.added_at or '-'} |"
            )
    else:
        lines.append("_（空库——该市场当前统一标准+默认窗口下无合格因子）_")
    md = "\n".join(lines) + "\n"

    out = Path(root) / f"{market}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    render_summary(root=root)
    return md


def render_summary(root: str = DEFAULT_ROOT) -> str:
    """跨市场总览 ``summary.md``：各市场 active/correlated 数 + 最强因子（按 holdout_ic）。"""
    lines = ["# 因子库总览", "", "| market | active | correlated | 最强因子 (holdout_ic) |",
             "|---|---|---|---|"]
    for market in ("ashare", "crypto", "futures", "us"):
        records = load_library(market, root=root)
        n_active = sum(1 for r in records if r.status == "active")
        n_corr = sum(1 for r in records if r.status == "correlated")
        best = min(records, key=_sort_key, default=None)  # _sort_key 用负号，min=最强
        best_str = (f"`{best.expression}` ({_fmt(best.holdout_ic)})"
                    if best is not None else "-")
        lines.append(f"| {market} | {n_active} | {n_corr} | {best_str} |")
    md = "\n".join(lines) + "\n"
    out = Path(root) / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return md


# ── rebuild：候选源收集 + 统一窗口评估器 ─────────────────────────────────────

def collect_source_expressions(
    market: str, *,
    mine_team_root: str = "workspace/mine_team",
    mining_sessions_root: str = "workspace/mining_sessions",
    experiment_index_paths: list[str] | None = None,
) -> list[str]:
    """扫历史产物收集该市场的候选表达式（去重、保序）。缺失/损坏文件跳过。

    源：``experiment_index[_{market}].jsonl`` 的表达式（按 window market 过滤）
    + ``mine_team/*/manifest.json`` 的候选（按 params.market 过滤）
    + ``mining_sessions/*/candidates.csv`` 的候选（无市场归属 → 全收，跨市场表达式在
      重算时 parse 失败自然被剔）。
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(expr):
        if expr and expr not in seen:
            seen.add(expr)
            out.append(expr)

    mt = Path(mine_team_root)
    # experiment_index：默认扫无后缀 + 市场后缀两个文件
    idx_paths = experiment_index_paths or [
        str(mt / "experiment_index.jsonl"), str(mt / f"experiment_index_{market}.jsonl")]
    for ip in idx_paths:
        p = Path(ip)
        if not p.is_file():
            continue
        suffixed = p.name == f"experiment_index_{market}.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            dw = rec.get("data_window") or {}
            # 无后缀文件按记录 window 的 market 过滤；后缀文件全收
            if suffixed or dw.get("market") == market or (
                    market == "ashare" and not dw.get("market")):
                _add(rec.get("expression"))

    # mine_team 每 run 的 manifest.json（按 params.market 过滤）
    if mt.is_dir():
        for man in sorted(mt.glob("*/manifest.json")):
            try:
                m = json.loads(man.read_text(encoding="utf-8"))
            except Exception:
                continue
            mk = (m.get("params") or {}).get("market", "ashare")
            if mk != market:
                continue
            for c in m.get("candidates") or []:
                _add(c.get("expression") if isinstance(c, dict) else None)

    # mining_sessions 每 run 的 candidates.csv（无市场归属，全收）
    ms = Path(mining_sessions_root)
    if ms.is_dir():
        for csv in sorted(ms.glob("*/candidates.csv")):
            try:
                df = pl.read_csv(csv)
            except Exception:
                continue
            if "expression" in df.columns:
                for e in df["expression"].to_list():
                    _add(e)
    return out


def build_library_evaluator(
    daily: pl.DataFrame, *, holdout_ratio: float = 0.2, eval_start: str | None = None,
    leaf_map: dict[str, str] | None = None, profile=None,
    batch_size: int = DEFAULT_EVAL_BATCH, decorr_max_dates: int = DEFAULT_DECORR_MAX_DATES,
) -> tuple[Callable[[list[str]], list[dict]], CompactMaterializer]:
    """在给定 ``daily``（含预热前缀）上构造 ``(evaluate, compact_materialize)`` 供 `rebuild`。

    **复用挖掘评估，别重写**：train 段指标走 agent `evaluate_expressions`（含预热门/预处理/
    leaf_map/profile），holdout 段用 `evaluate_materialized` 在完整帧上求值再裁末段（扩窗预热，
    同挖掘路径），DSR 用池 IR 经验方差 deflation（与 M1/M5 同配方）。

    **内存有界**（修真实 A股 rebuild OOM）：``evaluate`` 把候选切 ``batch_size`` 一批逐批评估
    （批间释放，不一次性把数百表达式全塞 `evaluate_expressions`）；holdout 全帧面板逐表达式瞬态
    （算完即弃，不缓存）；去相关走 ``compact_materialize``（紧凑 float32 矩阵，单因子只驻小矩阵，
    见 `make_compact_materializer`），内存随因子数有界而非随因子数爆。
    """
    from datetime import datetime as _dt

    from factorzen.agents.evaluation import _preprocess_daily, evaluate_expressions
    from factorzen.discovery.guardrails import DeflationBasis, deflated_pvalue
    from factorzen.discovery.scoring import DataBundle
    from factorzen.validation.holdout import holdout_ic_result, split_holdout

    prepped = _preprocess_daily(daily, profile).sort(["ts_code", "trade_date"])
    es_date = _dt.strptime(eval_start, "%Y%m%d").date() if eval_start else None
    sample = prepped if es_date is None else prepped.filter(pl.col("trade_date") >= es_date)
    mining_df, holdout_df, holdout_start = split_holdout(sample, holdout_ratio=holdout_ratio)
    bundle = DataBundle.build(mining_df)
    train_end = _dt.strptime(bundle.train_end, "%Y%m%d").date()

    def _holdout_ic_of(expr: str):
        """单表达式的 holdout IC/CI/有效天数：全帧求值→裁末段→算 IC。**瞬态面板，算完即弃**。

        n_days 是覆盖门燃料（guardrails DEFAULT_HOLDOUT_MIN_DAYS）：求值失败/稀薄 → 0，
        由 upsert 的 acceptance_reasons 拒绝，不再让稀薄 holdout 靠点估计运气入库。
        """
        try:
            node = parse_expr(expr, leaf_map)
            s = evaluate_materialized(node, prepped, leaf_map)
            panel = (prepped.select(["trade_date", "ts_code"])
                     .with_columns(s.alias("factor_value"))
                     .filter(pl.col("factor_value").is_not_null()
                             & pl.col("factor_value").is_finite())
                     .filter(pl.col("trade_date") >= holdout_start))
        except Exception:
            return float("nan"), float("nan"), float("nan"), 0
        if panel.height < 20:
            return float("nan"), float("nan"), float("nan"), 0
        hres = holdout_ic_result(panel, holdout_df)
        return hres.ic_mean, hres.ci[0], hres.ci[1], hres.n_days

    def evaluate(exprs: list[str]) -> list[dict]:
        exprs = list(exprs)
        rows: list[dict] = []
        # 分批评估：批间 evaluate_expressions 的 prepped/中间帧被释放，内存不随候选总数增长。
        for i in range(0, len(exprs), max(1, batch_size)):
            batch = exprs[i:i + max(1, batch_size)]
            results = evaluate_expressions(batch, daily, bundle, eval_start=es_date,
                                           eval_end=train_end, profile=profile)
            for r in results:
                if r["ic_train"] is None:  # 编译失败/预热不足/死表达式 → 不入候选
                    continue
                h_ic, ci_lo, ci_hi, n_hold = _holdout_ic_of(r["expression"])
                rows.append({"expression": r["expression"], "ic_train": r["ic_train"],
                             "ir_train": r["ir_train"], "holdout_ic": h_ic,
                             "n_train": r["n_train"], "turnover": r.get("turnover"),
                             "ic_ci_low": ci_lo, "ic_ci_high": ci_hi,
                             "n_holdout_days": n_hold})
        # DSR（池 IR 经验方差 deflation，N=有效评估唯一表达式数），与挖掘同配方
        basis = DeflationBasis.from_ir_pool([x["ir_train"] for x in rows])
        for x in rows:
            _dsr, p = deflated_pvalue(x["ir_train"], basis, x["n_train"] or 1)
            x["dsr"] = round(float(_dsr), 4) if _dsr == _dsr else None
            x["dsr_pvalue"] = round(float(p), 4) if p == p else None
        return rows

    compact_materialize = make_compact_materializer(prepped, leaf_map, max_dates=decorr_max_dates)
    return evaluate, compact_materialize


def rebuild(
    market: str, *, sources: list[str], eval_window: tuple[str, str],
    universe: str | None, horizon: int | None,
    evaluate: Callable[[list[str]], list[dict]], git_sha: str | None, now: str,
    materialize: Materializer | None = None,
    compact_materialize: CompactMaterializer | None = None, decorr_threshold: float = 0.7,
    leaf_map: dict[str, str] | None = None, root: str = DEFAULT_ROOT,
    manifest_extra: dict | None = None, fresh: bool = True,
) -> UpsertResult:
    """在统一窗口重算历史因子并**从头重建**该市场库（"可比"且"权威"的关键）。

    ``sources``：从历史产物收集的候选表达式（`collect_source_expressions`）。
    ``evaluate``：``list[expr] -> list[候选 dict]``——复用挖掘的数据装配+评估（CLI 注入闭包，
    测试注入 mock）。别在此重写评估。窗口=``eval_window``，holdout 由 evaluate 内部切末段。
    ``compact_materialize``：去相关的紧凑矩阵物化器（`build_library_evaluator` 返回；内存有界）。
    ``fresh``（默认 True）：rebuild 语义 = **从零重算全部历史源**，先清空该市场旧库再写入——
    否则旧库里已失效的记录（如 P0 修复前误收的前视因子、或已从算子库移除的表达式）会残留，
    rebuild 就不再"权威"。清空只针对本 market 文件，不动别的市场。
    落 ``rebuild_{market}_manifest.json``（窗口/源/git_sha/时间，可复现）。
    """
    if fresh:
        library_path(market, root).unlink(missing_ok=True)  # 从零重建，清旧库（仅本 market）

    uniq: list[str] = []
    seen: set[str] = set()
    for e in sources:
        n = _normalize(e, leaf_map)
        if n not in seen:
            seen.add(n)
            uniq.append(e)          # 原串喂 evaluate（其内部自行规范化）

    cand_dicts = evaluate(uniq) if uniq else []
    res = upsert(
        market, cand_dicts, eval_window=eval_window, universe=universe, horizon=horizon,
        run_id=f"rebuild_{now}", session_dir=None, git_sha=git_sha, now=now,
        decorr_threshold=decorr_threshold, materialize=materialize,
        compact_materialize=compact_materialize, leaf_map=leaf_map, root=root,
    )

    manifest = {
        "market": market, "eval_start": eval_window[0], "eval_end": eval_window[1],
        "universe": universe, "horizon": horizon, "git_sha": git_sha, "rebuilt_at": now,
        "n_sources": len(sources), "n_unique": len(uniq), "n_evaluated": len(cand_dicts),
        "added": res.added, "updated": res.updated, "correlated": res.correlated,
        "skipped": res.skipped,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / f"rebuild_{market}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return res
