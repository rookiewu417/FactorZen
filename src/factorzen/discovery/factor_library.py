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

import contextlib
import hashlib
import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factorzen.config.settings import (
    FACTOR_LIBRARY_DIR,
    MINE_TEAM_DIR,
    MINING_SESSIONS_DIR,
)
from factorzen.discovery.expression import evaluate_materialized, parse_expr, to_expr_string
from factorzen.discovery.guardrails import DEFAULT_LIFT_THRESHOLD, acceptance_reasons

# 容器类落 research/combination/pool(被依赖侧,避免 discovery↔research 环);
# 此处 re-export 保旧调用方 `from factorzen.discovery.factor_library import ...` 不变。
from factorzen.research.combination.pool import (  # noqa: F401
    POOL_COMPACT_BYTES_THRESHOLD,
    POOL_KEY_BYTES_PER_ROW,
    POOL_VALUE_F32_BYTES_THRESHOLD,
    CompactLibraryPool,
    HybridLibraryPool,
    estimate_library_pool_key_bytes,
    is_compact_pool,
    should_use_compact_pool,
)

_LOG = logging.getLogger(__name__)

DEFAULT_ROOT = str(FACTOR_LIBRARY_DIR)

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
    status: str = "active"           # active / correlated / probation / no_lift
    max_corr_in_lib: float | None = None
    correlated_with: str | None = None
    # 准入轨道：single=单因子裸口径；lift=组合 OOS 增量通道
    admission_track: str = "single"  # "single" | "lift"
    # 组合增量 lift 通道字段（lift 轨；single 轨通常为 None）
    lift: float | None = None
    lift_baseline: float | None = None
    lift_se: float | None = None
    lift_first_half: float | None = None
    lift_second_half: float | None = None
    # 单因子 admission 窗 RankIC（方向权威；forward_review 调向优先用此字段）
    # 非组合 candidate_rank_ic（组合后≈恒正无判别力）
    admission_ic: float | None = None
    # 审计：holdout 有效覆盖天数（single 轨 upsert 若调用方传入则落盘）
    holdout_n_days: int | None = None
    # paper forward 确认（forward_review --apply 写入）：不进 schema 会被
    # from_dict 丢弃 → 下一次 load→save 循环静默洗掉，故必须正式建字段
    forward_confirmed_at: str | None = None
    forward_n_days: int | None = None
    # 统计裁决原文（cap 前）："active"/"probation"；status 可能被运营护栏压到 probation
    admission_decision: str | None = None
    # 证据层级：legacy=历史入库；v2=新口径写入；None=未标注（语义≈legacy 但未落盘）
    evidence_tier: str | None = None  # "legacy" | "v2"
    eval_start: str | None = None
    eval_end: str | None = None
    universe: str | None = None
    horizon: int | None = None
    # lift 准入 provenance（可重放：窗/CV/block/baseline/profile；旧行缺失→None）
    admission_start: str | None = None
    admission_end: str | None = None
    scored_start: str | None = None
    scored_end: str | None = None
    block_days: int | None = None
    cv_train_days: int | None = None
    cv_test_days: int | None = None
    lift_threshold: float | None = None
    lift_se_mult: float | None = None
    baseline_hash: str | None = None
    profile_name: str | None = None
    frequency: str | None = None
    # 注：target_price / execution_lag 属执行配置 provenance，lift 流程当前不持有，
    # 不加空壳字段；后续从市场执行配置线程接入。
    source_run_id: str | None = None
    source_session_dir: str | None = None
    git_sha: str | None = None
    added_at: str | None = None
    updated_at: str | None = None
    # 因子形态：expression=DSL；python=手写 DailyFactor（expression 存 py::{name} 哨兵）
    # 旧行 from_dict 缺字段 → 默认 expression，零迁移
    kind: str = "expression"  # "expression" | "python"
    name: str | None = None  # 业务名；python 型必填=registry 名；expression 型可空后回填
    impl: str | None = None  # python 型实现引用（一期=registry 名；预留 import path）
    # 日内特征叶子溯源（旧 jsonl 无此字段 → from_dict 向前兼容读入不崩）
    intraday_leaves: list[str] | None = None
    intraday_panel: str | None = None

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
    # rebuild 专用：lift 轨复审失败原因（None=未失败/无 lift 记录）。调用方（CLI）
    # 必须检查——复审失败时旧记录已恢复，静默返回会造成「表面成功、实际跳过」。
    lift_review_error: str | None = None


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


# python 型身份哨兵：expression 存 "py::{name}"，by_expr / 池键 / 台账键零改动；
# _normalize 对 parse 失败原样返回，天然兼容（禁止改 _normalize 容错语义）。
PY_IDENTITY_PREFIX = "py::"


def python_identity(name: str) -> str:
    """python 型因子的 expression 哨兵串。"""
    return f"{PY_IDENTITY_PREFIX}{name}"


def is_python_identity(expr: str | None) -> bool:
    """是否为 python 型身份哨兵（``py::{name}``）。"""
    return isinstance(expr, str) and expr.startswith(PY_IDENTITY_PREFIX) and len(expr) > len(
        PY_IDENTITY_PREFIX
    )


def default_name_for_expression(norm_expr: str) -> str:
    """expression 型缺省业务名：确定性 mined_{sha1[:8]}（回填幂等）。"""
    digest = hashlib.sha1(norm_expr.encode()).hexdigest()[:8]
    return f"mined_{digest}"


def _python_name_from_expression(expr: str) -> str | None:
    """从 ``py::{name}`` 剥出 name；非哨兵 → None。"""
    if not is_python_identity(expr):
        return None
    return expr[len(PY_IDENTITY_PREFIX) :]


def _is_python_record(r: FactorRecord) -> bool:
    """kind 显式 python，或 expression 为哨兵（旧行可能仅有哨兵）。"""
    return r.kind == "python" or is_python_identity(r.expression)


def library_path(market: str, root: str = DEFAULT_ROOT) -> Path:
    return Path(root) / f"{market}.jsonl"


def library_file_hash(market: str, root: str = DEFAULT_ROOT) -> str | None:
    """库 jsonl 内容 sha256 hexdigest 前 16 位;文件不存在 → None。"""
    path = library_path(market, root)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_pool_cache(pool, cache_dir, *, meta: dict) -> None:
    """把库池写到 cache_dir(parquet + meta.json)。

    - ``pool``:``CompactLibraryPool`` 或 ``{}``(空库)。
    - 非空先写 ``pool_wide.parquet``,**最后**写 ``pool_meta.json``
      (meta 存在 = 完整性 sentinel,写序不许颠倒)。
    - 函数内补全 ``factor_names``/``n_factors``/``value_dtype``/``n_rows``;
      调用方负责 ``market/statuses/eval_start/library_hash/prepped_*`` 等指纹字段。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_meta = dict(meta)

    if not pool:
        out_meta["factor_names"] = []
        out_meta["n_factors"] = 0
        out_meta["value_dtype"] = None
        out_meta["n_rows"] = 0
    else:
        # 非空 CompactLibraryPool
        pool.write_parquet(cache_dir / "pool_wide.parquet")
        names = list(pool.factor_names)
        out_meta["factor_names"] = names
        out_meta["n_factors"] = len(names)
        out_meta["n_rows"] = int(pool.wide.height)
        if names:
            dt = pool.wide.schema[names[0]]
            if dt == pl.Float32:
                out_meta["value_dtype"] = "f32"
            elif dt == pl.Float64:
                out_meta["value_dtype"] = "f64"
            else:
                out_meta["value_dtype"] = str(dt)
        else:
            out_meta["value_dtype"] = None

    # meta.json 最后写:存在即视为完整缓存
    (cache_dir / "pool_meta.json").write_text(
        json.dumps(out_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_pool_cache(
    cache_dir,
    *,
    market: str,
    root: str,
    statuses,
    eval_start,
    expect_height: int,
    expect_date_min,
    expect_date_max,
) -> CompactLibraryPool | dict | None:
    """装载并校验池缓存;失效返回 None(调用方回落进程内重建)。

    校验项:market/statuses/library_hash/eval_start/prepped_height/date_min/max。
    ``n_factors==0`` → 返回 ``{}``;否则 from_parquet(factor_names)。
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "pool_meta.json"
    if not meta_path.exists():
        return None

    def _invalidate(reason: str) -> None:
        print(
            f"[library-pool] 池缓存失效({reason})→ 进程内重建",
            flush=True,
        )

    def _as_str(x):
        return str(x) if x is not None else None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _invalidate(f"meta 不可读:{exc}")
        return None

    if meta.get("market") != market:
        _invalidate(f"market {meta.get('market')!r}≠{market!r}")
        return None
    if list(meta.get("statuses", [])) != list(statuses):
        _invalidate(f"statuses {meta.get('statuses')!r}≠{list(statuses)!r}")
        return None

    current_hash = library_file_hash(market, root)
    if meta.get("library_hash") != current_hash:
        _invalidate(
            f"library_hash {meta.get('library_hash')!r}≠{current_hash!r}",
        )
        return None

    if _as_str(meta.get("eval_start")) != _as_str(eval_start):
        _invalidate(
            f"eval_start {meta.get('eval_start')!r}≠{eval_start!r}",
        )
        return None
    if meta.get("prepped_height") != expect_height:
        _invalidate(
            f"prepped_height {meta.get('prepped_height')!r}≠{expect_height!r}",
        )
        return None
    if _as_str(meta.get("prepped_date_min")) != _as_str(expect_date_min):
        _invalidate(
            f"prepped_date_min {meta.get('prepped_date_min')!r}"
            f"≠{expect_date_min!r}",
        )
        return None
    if _as_str(meta.get("prepped_date_max")) != _as_str(expect_date_max):
        _invalidate(
            f"prepped_date_max {meta.get('prepped_date_max')!r}"
            f"≠{expect_date_max!r}",
        )
        return None

    n_factors = int(meta.get("n_factors") or 0)
    if n_factors == 0:
        print(
            f"[library-pool] 池缓存命中 {cache_dir}(n_factors=0)",
            flush=True,
        )
        return {}

    try:
        pool = CompactLibraryPool.from_parquet(
            cache_dir / "pool_wide.parquet",
            meta.get("factor_names"),
        )
    except Exception as exc:
        _invalidate(f"parquet 缺失/读失败:{type(exc).__name__}: {exc}")
        return None

    print(
        f"[library-pool] 池缓存命中 {cache_dir}(n_factors={n_factors})",
        flush=True,
    )
    return pool


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

    默认只取 ``active``——**probation 不进挖掘正交参照系**（试用因子不该挡新候选）。
    文件不存在/空 → []。不物化、不求值。
    """
    recs = [r for r in load_library(market, root=root) if r.status in statuses]
    recs.sort(key=lambda r: (-abs(r.ic_train or 0.0), r.expression))
    return [r.expression for r in recs[:k]]


def library_covered_by_family(
    market: str,
    *,
    per_family: int = 2,
    max_total: int = 12,
    statuses: tuple[str, ...] = ("active",),
    crowded_min: int = 3,
    root: str = DEFAULT_ROOT,
) -> tuple[list[str], list[tuple[str, int]]]:
    """库内因子按叶子族聚类后取代表作 + 拥挤叶统计。

    - 族键 = ``frozenset(feature_names(parse_expr(expression)))``
      （parse 失败的记录单独成族，键用原串）；
    - 族内按 |ic_train| 降序取 ``per_family`` 条；
    - 族间按族内最佳 |ic_train| 排序，总数截 ``max_total``；
    - 第二返回值 = 拥挤叶列表：叶 → 含它的 active 数，仅保留 ≥ ``crowded_min``，
      按数量降序。

    旧 ``library_covered_expressions`` 原样保留。
    """
    from factorzen.discovery.expression import feature_names, parse_expr

    recs = [r for r in load_library(market, root=root) if r.status in statuses]
    # leaf → count（全体 active，不按族截断）
    leaf_counts: dict[str, int] = {}
    families: dict[object, list] = {}
    for r in recs:
        try:
            feats = frozenset(feature_names(parse_expr(r.expression)))
            fam_key: object = feats
        except (ValueError, TypeError, IndexError):
            fam_key = r.expression  # parse 失败：单独成族
            feats = frozenset()
        for leaf in feats:
            leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
        families.setdefault(fam_key, []).append(r)

    # 族内排序 + 截断；族间按最佳 |ic|
    family_blocks: list[tuple[float, list[str]]] = []
    for _key, members in families.items():
        members.sort(key=lambda x: (-abs(x.ic_train or 0.0), x.expression))
        best = abs(members[0].ic_train or 0.0) if members else 0.0
        picked = [m.expression for m in members[:per_family]]
        family_blocks.append((best, picked))
    family_blocks.sort(key=lambda t: -t[0])

    covered: list[str] = []
    for _best, exprs in family_blocks:
        for e in exprs:
            if len(covered) >= max_total:
                break
            covered.append(e)
        if len(covered) >= max_total:
            break

    crowded = [
        (name, n) for name, n in leaf_counts.items() if n >= crowded_min
    ]
    crowded.sort(key=lambda t: (-t[1], t[0]))
    return covered, crowded


def build_library_pool(
    market: str,
    daily: pl.DataFrame,
    leaf_map: dict[str, str] | None = None,
    *,
    statuses: tuple[str, ...] = ("active",),
    root: str = DEFAULT_ROOT,
    eval_start=None,
    compact: bool | None = None,
    compact_threshold: int = POOL_COMPACT_BYTES_THRESHOLD,
    cache_dir: str | Path | None = None,
    universe: str | None = None,
    python_materializer: Callable[[str], pl.DataFrame | None] | None = None,
) -> dict[str, pl.DataFrame] | CompactLibraryPool:
    """把库内因子物化为 mining/评估帧上的因子值面板，供搜索期库级正交去相关。

    - 取 status∈statuses 记录，按 |ic_train| 降序。
    - 默认只取 ``active``——**probation 不进挖掘正交参照系**（试用因子不该挡新候选；
      组合 staging 是否纳入由调用方在 staging csv 控制，本函数不替组合做决策）。
    - expression 型：``evaluate_materialized`` 在 ``daily`` 上算
      ``[trade_date, ts_code, factor_value]``（与挖掘物化路径一致）。
    - python 型：``python_materializer(name)``（测试注入）或
      ``materialize_python_panel(name, start, end, universe)``，再 inner-join 到
      ``daily`` 的 (trade_date, ts_code) 网格；池键 = ``r.expression``（``py::`` 哨兵）。
    - 非法/求值失败/全 null 的表达式跳过并计数——一条坏记录不许崩整个 pool。
    - 库文件不存在/空 → {}。
    - ``eval_start``：可选，求值后裁到该日起（team holdout 口径扩窗预热时传入 holdout 起点）。
    - ``compact``：``None``（默认）按 ``n_recs × n_rows × POOL_KEY_BYTES_PER_ROW`` 是否
      ≥ ``compact_threshold`` 自动选择；``True`` 强制单骨架宽面板；``False`` 强制旧
      dict-of-frames（**零回归默认路径**，小帧/测试保持逐字节行为）。
    - ``cache_dir``：可选池缓存目录；非 None 时先试 ``load_pool_cache``（指纹含库 hash /
      statuses / eval_start / prepped 窗），命中则跳过求值直接返回；默认 None 零回归。
    - ``universe`` / ``python_materializer`` 皆空时 python 记录全部跳过（expression 不受影响）。

    调用方负责 ``daily`` 已与挖掘同 prep（派生列/停牌掩码等）；本函数不再二次预处理。
    """
    if cache_dir is not None:
        cached = load_pool_cache(
            cache_dir,
            market=market,
            root=root,
            statuses=statuses,
            eval_start=eval_start,
            expect_height=daily.height,
            expect_date_min=daily["trade_date"].min(),
            expect_date_max=daily["trade_date"].max(),
        )
        if cached is not None:
            return cached

    recs = [r for r in load_library(market, root=root) if r.status in statuses]
    if not recs:
        return {}
    recs.sort(key=lambda r: (-abs(r.ic_train or 0.0), r.expression))

    df = daily.sort(["ts_code", "trade_date"])
    auto = compact is None
    use_compact = (
        should_use_compact_pool(
            len(recs), df.height, threshold=compact_threshold,
        )
        if auto
        else bool(compact)
    )
    if use_compact:
        est = estimate_library_pool_key_bytes(len(recs), df.height)
        thr = compact_threshold
        reason = (
            f"估算 {est / (1024**3):.2f}G>阈值 {thr / (1024**3):.2f}G"
            if auto
            else "compact=True"
        )
        _LOG.info(
            "库池 compact 模式(%s, n_rec=%d, n_rows=%d)",
            reason, len(recs), df.height,
        )
        print(f"库池 compact 模式({reason})", flush=True)
        return _build_library_pool_compact(
            recs, df, leaf_map, eval_start=eval_start, market=market,
            universe=universe, python_materializer=python_materializer,
        )
    return _build_library_pool_legacy(
        recs, df, leaf_map, eval_start=eval_start, market=market,
        universe=universe, python_materializer=python_materializer,
    )


def _pool_date_bounds(df: pl.DataFrame) -> tuple[str | None, str | None]:
    """daily 帧 trade_date min/max → YYYYMMDD（供 python 物化窗）。"""
    if df.is_empty() or "trade_date" not in df.columns:
        return None, None
    from factorzen.discovery.intraday_expr import _to_yyyymmdd

    try:
        return _to_yyyymmdd(df["trade_date"].min()), _to_yyyymmdd(df["trade_date"].max())
    except Exception:
        return None, None


def _align_panel_trade_date(panel: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """面板 trade_date dtype 对齐到网格（复用日内 attach 同款）。"""
    from factorzen.discovery.intraday_expr import _align_trade_date

    return _align_trade_date(panel, grid)


def _python_panel_aligned(
    r: FactorRecord,
    df: pl.DataFrame,
    *,
    market: str,
    universe: str | None,
    python_materializer: Callable[[str], pl.DataFrame | None] | None,
    start: str | None,
    end: str | None,
) -> pl.DataFrame | None:
    """python 型 → dtype 对齐、键唯一的三列面板；失败/空/重复键 → None。

    重复 (trade_date, ts_code) 是因子作者 bug：join 会行数膨胀（legacy 面板失真、
    compact 列错位），必须响亮跳过而非静默吞。
    """
    name = r.name or _python_name_from_expression(r.expression)
    if not name:
        return None
    if python_materializer is not None:
        raw = python_materializer(name)
    elif universe and start and end:
        from factorzen.discovery.python_factor import materialize_python_panel

        raw = materialize_python_panel(name, start, end, universe, market=market)
    else:
        return None
    if raw is None or raw.is_empty():
        return None
    need = {"trade_date", "ts_code", "factor_value"}
    if not need.issubset(set(raw.columns)):
        return None
    panel = raw.select(["trade_date", "ts_code", "factor_value"])
    n_keys = panel.select(["trade_date", "ts_code"]).unique().height
    if n_keys != panel.height:
        _LOG.warning(
            "python 因子 %r 面板含重复 (trade_date, ts_code)（%d 行 / %d 唯一键），跳过",
            name, panel.height, n_keys,
        )
        return None
    return _align_panel_trade_date(panel, df)


def _materialize_python_on_grid(
    r: FactorRecord,
    df: pl.DataFrame,
    *,
    market: str,
    universe: str | None,
    python_materializer: Callable[[str], pl.DataFrame | None] | None,
    start: str | None,
    end: str | None,
) -> pl.DataFrame | None:
    """python 型 → 对齐 daily 网格的因子面板；失败/空 → None（调用方计入 n_skip）。"""
    panel = _python_panel_aligned(
        r, df, market=market, universe=universe,
        python_materializer=python_materializer, start=start, end=end,
    )
    if panel is None:
        return None
    # 池语义=同一网格：inner-join 限制到 daily 的 (trade_date, ts_code)
    joined = (
        df.select(["trade_date", "ts_code"])
        .join(panel, on=["trade_date", "ts_code"], how="inner")
        .filter(
            pl.col("factor_value").is_not_null()
            & pl.col("factor_value").is_finite()
        )
    )
    return joined if not joined.is_empty() else None


def _python_series_on_grid(
    r: FactorRecord,
    df: pl.DataFrame,
    *,
    market: str,
    universe: str | None,
    python_materializer: Callable[[str], pl.DataFrame | None] | None,
    start: str | None,
    end: str | None,
) -> pl.Series | None:
    """compact 路径：python 面板 left-join 到 df 行序，返回与 df 等长的 factor 列。"""
    panel = _python_panel_aligned(
        r, df, market=market, universe=universe,
        python_materializer=python_materializer, start=start, end=end,
    )
    if panel is None:
        return None
    joined = df.select(["trade_date", "ts_code"]).join(
        panel, on=["trade_date", "ts_code"], how="left",
    )
    if joined.height != df.height:
        # 键已唯一仍长度漂移 = 网格自身异常；列错位比缺列危险，直接跳过
        return None
    return joined["factor_value"]


def _build_library_pool_legacy(
    recs: list,
    df: pl.DataFrame,
    leaf_map: dict[str, str] | None,
    *,
    eval_start=None,
    market: str = "",
    universe: str | None = None,
    python_materializer: Callable[[str], pl.DataFrame | None] | None = None,
) -> dict[str, pl.DataFrame]:
    """旧路径：每因子独立 [trade_date, ts_code, factor_value] 帧（filter null/inf）。"""
    pool: dict[str, pl.DataFrame] = {}
    n_skip = 0
    py_recs = [r for r in recs if _is_python_record(r)]
    can_mat_py = python_materializer is not None or bool(universe)
    if py_recs and not can_mat_py:
        _LOG.warning(
            "build_library_pool(%s): 跳过 %d 条 python 记录"
            "（未提供 universe 或 python_materializer）",
            market, len(py_recs),
        )
    start, end = _pool_date_bounds(df)
    for r in recs:
        try:
            if _is_python_record(r):
                if not can_mat_py:
                    n_skip += 1
                    continue
                panel = _materialize_python_on_grid(
                    r, df, market=market, universe=universe,
                    python_materializer=python_materializer,
                    start=start, end=end,
                )
                if panel is None:
                    n_skip += 1
                    continue
                if eval_start is not None:
                    panel = panel.filter(pl.col("trade_date") >= eval_start)
                if panel.is_empty():
                    n_skip += 1
                    continue
                pool[r.expression] = panel
                continue
            # expression 型：现状路径（逐字节语义）
            node = parse_expr(r.expression, leaf_map)
            series = evaluate_materialized(node, df, leaf_map)
            panel = (
                df.select(["trade_date", "ts_code"])
                .with_columns(series.alias("factor_value"))
                .filter(
                    pl.col("factor_value").is_not_null()
                    & pl.col("factor_value").is_finite()
                )
            )
            if eval_start is not None:
                panel = panel.filter(pl.col("trade_date") >= eval_start)
            if panel.is_empty():
                n_skip += 1
                continue
            pool[r.expression] = panel
        except Exception as exc:
            n_skip += 1
            _LOG.debug(
                "build_library_pool skip %r: %s: %s",
                r.expression, type(exc).__name__, exc,
            )
            continue
    if n_skip:
        _LOG.info(
            "build_library_pool(%s): skipped %d / kept %d",
            market, n_skip, len(pool),
        )
    return pool


def _build_library_pool_compact(
    recs: list,
    df: pl.DataFrame,
    leaf_map: dict[str, str] | None,
    *,
    eval_start=None,
    market: str = "",
    universe: str | None = None,
    python_materializer: Callable[[str], pl.DataFrame | None] | None = None,
) -> CompactLibraryPool | dict:
    """单骨架宽面板：键一份 + 每因子一列 f64；非有限→null（不丢行，消费方 scatter 处理）。

    求值在完整 ``df`` 上（滚动预热），``eval_start`` 只裁最终骨架与值列行。
    """
    # 值列 f32(仅超阈值大池):87 因子×8.75M 行 f64 值列 ~5.8G 蚕食全 A 余量
    # (v14 探针死于 #78 号因子时余量已被累积吃光)。f32 仅存储层——scatter/QR 在
    # numpy 边界升回 f64,__getitem__ 出口升回 f64;csi800 级(估算 <2G)保持 f64
    # 字节级零回归。1e-7 级舍入远低于一切裁决阈值(残差 floor 0.008/lift 0.001)。
    _est_value_bytes = len(recs) * df.height * 8
    _use_f32 = _est_value_bytes >= POOL_VALUE_F32_BYTES_THRESHOLD
    if _use_f32:
        print(
            f"[library-pool] 值列 f32 模式(估算 {_est_value_bytes / (1024**3):.1f}G"
            f"≥阈值 {POOL_VALUE_F32_BYTES_THRESHOLD / (1024**3):.0f}G)", flush=True,
        )
    value_cols: list[pl.Series] = []
    names: list[str] = []
    n_skip = 0
    py_recs = [r for r in recs if _is_python_record(r)]
    can_mat_py = python_materializer is not None or bool(universe)
    if py_recs and not can_mat_py:
        _LOG.warning(
            "build_library_pool(%s) compact: 跳过 %d 条 python 记录"
            "（未提供 universe 或 python_materializer）",
            market, len(py_recs),
        )
    start, end = _pool_date_bounds(df)
    for _fi, r in enumerate(recs, start=1):
        try:
            # 大帧逐因子进度(可观测性:OOM 时最后一行钉死凶手因子;小帧静默)
            if df.height >= 3_000_000:
                print(f"[library-pool] [{_fi}/{len(recs)}] {r.expression[:70]}", flush=True)
            if _is_python_record(r):
                if not can_mat_py:
                    n_skip += 1
                    continue
                series = _python_series_on_grid(
                    r, df, market=market, universe=universe,
                    python_materializer=python_materializer,
                    start=start, end=end,
                )
                if series is None:
                    n_skip += 1
                    continue
            else:
                node = parse_expr(r.expression, leaf_map)
                series = evaluate_materialized(node, df, leaf_map)
            # 非有限 → null（保留行；scatter/__getitem__ 与旧 filter 对齐）
            col = (
                pl.DataFrame({"_v": series})
                .select(
                    pl.when(
                        pl.col("_v").is_not_null() & pl.col("_v").is_finite()
                    )
                    .then(pl.col("_v"))
                    .otherwise(None)
                    .alias(r.expression)
                )[r.expression]
            )
            if eval_start is not None:
                col = (
                    df.select(pl.col("trade_date"))
                    .with_columns(col)
                    .filter(pl.col("trade_date") >= eval_start)[r.expression]
                )
            if col.null_count() >= col.len():
                n_skip += 1
                continue
            if _use_f32:
                col = col.cast(pl.Float32)
            value_cols.append(col.alias(r.expression))
            names.append(r.expression)
        except Exception as exc:
            n_skip += 1
            _LOG.debug(
                "build_library_pool compact skip %r: %s: %s",
                r.expression, type(exc).__name__, exc,
            )
            continue

    if n_skip:
        _LOG.info(
            "build_library_pool(%s) compact: skipped %d / kept %d",
            market, n_skip, len(names),
        )
    if not names:
        return {}

    skeleton = df.select(["trade_date", "ts_code"])
    if eval_start is not None:
        skeleton = skeleton.filter(pl.col("trade_date") >= eval_start)
    wide = skeleton.with_columns(value_cols)
    return CompactLibraryPool(wide, tuple(names))


def _backfill_record_names(records: list[FactorRecord]) -> None:
    """写盘前 name 回填（幂等、确定性）。冲突只 warning，不去重。"""
    for r in records:
        if r.name:
            continue
        if _is_python_record(r):
            peeled = _python_name_from_expression(r.expression)
            if peeled:
                r.name = peeled
            continue
        # expression 型：规范形 hash（与 by_expr 主键同口径）
        r.name = default_name_for_expression(_normalize(r.expression))

    # 同批 name 冲突：不同 expression 撞 hash / 手写名 → 仅告警（registry 侧 Batch 2 处理）
    by_name: dict[str, str] = {}
    for r in records:
        if not r.name:
            continue
        prev = by_name.get(r.name)
        if prev is not None and prev != r.expression:
            _LOG.warning(
                "factor_library name 冲突: name=%r expression_a=%r expression_b=%r"
                "（不去重，registry 冲突策略见 Batch 2）",
                r.name, prev, r.expression,
            )
        else:
            by_name[r.name] = r.expression


def _save_library(market: str, records: list[FactorRecord], root: str = DEFAULT_ROOT) -> None:
    _backfill_record_names(records)
    path = library_path(market, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records)
    path.write_text(payload, encoding="utf-8")


# ── upsert ───────────────────────────────────────────────────────────────────

def _record_from_candidate(
    cand: dict, norm_expr: str, market: str,
    eval_window: tuple[str | None, str | None],
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
    # holdout 覆盖天数：接受 holdout_n_days / n_holdout_days 两种键
    hnd = g("holdout_n_days", "n_holdout_days")
    if hnd is not None:
        try:
            hnd = int(hnd)
        except (TypeError, ValueError):
            hnd = None

    # 日内叶子溯源：表达式 parse 后 ∩ INTRADAY_FEATURES；去重键 by_expr 不动
    from factorzen.core.feature_schema import INTRADAY_FEATURES
    from factorzen.discovery.expression import feature_names

    i_leaves: list[str] | None = None
    i_panel: str | None = None
    try:
        feats = feature_names(parse_expr(norm_expr))
        hit = sorted(feats & INTRADAY_FEATURES)
        if hit:
            i_leaves = hit
            raw_panel = g("intraday_panel")
            if isinstance(raw_panel, dict):
                ver = raw_panel.get("version") or "v1"
                fr = raw_panel.get("freq") or "5min"
                i_panel = f"{ver}@{fr}"
            elif isinstance(raw_panel, str) and raw_panel:
                i_panel = raw_panel
            else:
                i_panel = "v1@5min"
    except Exception:
        pass

    # kind/name/impl：显式键优先于 py:: 推断（expression 型默认；旧调用方零改动）
    inferred_python = is_python_identity(norm_expr)
    kind_raw = g("kind")
    if kind_raw in ("expression", "python"):
        kind = str(kind_raw)
    else:
        kind = "python" if inferred_python else "expression"
    name_raw = g("name")
    impl_raw = g("impl")
    if kind == "python":
        name = name_raw if name_raw is not None else _python_name_from_expression(norm_expr)
        impl = impl_raw if impl_raw is not None else name
    else:
        name = name_raw
        impl = impl_raw

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
        # admission_track 默认 single（单因子裸口径 upsert 不改轨道）
        admission_ic=g("admission_ic"),  # lift 轨方向权威；旧行缺失 → None
        holdout_n_days=hnd,
        # paper forward 确认 provenance：与 added_at 同款从 prev 保留，
        # 避免幂等重写 / 复测静默洗掉 forward_confirmed_at / forward_n_days
        forward_confirmed_at=(
            prev.forward_confirmed_at if prev is not None else None
        ),
        forward_n_days=prev.forward_n_days if prev is not None else None,
        eval_start=eval_start,
        eval_end=eval_end,
        universe=universe,
        horizon=horizon,
        # lift 准入 provenance（缺失→None；threshold 键兼容 row 的 "threshold"）
        admission_start=g("admission_start"),
        admission_end=g("admission_end"),
        scored_start=g("scored_start"),
        scored_end=g("scored_end"),
        block_days=g("block_days"),
        cv_train_days=g("cv_train_days"),
        cv_test_days=g("cv_test_days"),
        lift_threshold=g("lift_threshold", "threshold"),
        lift_se_mult=g("lift_se_mult", "se_mult"),
        baseline_hash=g("baseline_hash"),
        profile_name=g("profile_name"),
        frequency=g("frequency"),
        source_run_id=run_id,
        source_session_dir=session_dir,
        git_sha=git_sha,
        added_at=prev.added_at if prev is not None else now,   # 保留原入库日
        updated_at=now,
        kind=kind,
        name=name,
        impl=impl,
        intraday_leaves=i_leaves,
        intraday_panel=i_panel,
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
        rec.evidence_tier = "v2"  # 新写入路径一律 v2（与 lift 轨对称）
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


def upsert_probation(
    market: str, passed_lift_rows: list[dict], *,
    eval_window: tuple[str, str], universe: str | None, horizon: int | None,
    run_id: str | None, session_dir: str | None, git_sha: str | None, now: str,
    leaf_map: dict[str, str] | None = None, root: str = DEFAULT_ROOT,
) -> UpsertResult:
    """把 lift 实验通过的灰区候选以 ``status="probation"`` 入库。

    **跳过单因子 library gate**——它们已由组合 lift 裁决；本函数不调用
    ``acceptance_reasons``。灰区门已排除 ≥0.7 库相关，此处**不再重复**算库相关
    （注释：省一次物化；若日后灰区门放宽，再考虑 upsert 内二次相关标记）。

    仍做规范形去重：已在库 → 更新指标/lift 字段 + ``status=probation``（保留
    ``added_at``）；新 → 新增。不改变库内既有 active/correlated 记录的 status
    （除非规范形撞到同一表达式——则升级/覆盖为 probation 并写 lift）。
    """
    existing = load_library(market, root=root)
    by_expr: dict[str, FactorRecord] = {r.expression: r for r in existing}

    res = UpsertResult()
    for row in passed_lift_rows:
        raw = row.get("expression")
        if not raw:
            continue
        if not row.get("passed", True):
            res.skipped += 1
            continue
        norm = _normalize(raw, leaf_map)
        prev = by_expr.get(norm)
        rec = _record_from_candidate(
            row, norm, market, eval_window, universe, horizon,
            run_id, session_dir, git_sha, now, prev,
        )
        rec.status = "probation"
        # lift 字段：优先 row 上的 lift / lift_baseline / baseline
        rec.lift = _as_float(row.get("lift"))
        rec.lift_baseline = _as_float(
            row.get("lift_baseline", row.get("baseline"))
        )
        if prev is None:
            res.added += 1
        else:
            res.updated += 1
        by_expr[norm] = rec
        res.records.append(rec)

    _save_library(market, list(by_expr.values()), root=root)
    render_markdown(market, root=root)
    return res


def upsert_lift_admissions(
    rows: list[dict], *, market: str,
    root: str = DEFAULT_ROOT,
    meta: dict | None = None,
    threshold: float = DEFAULT_LIFT_THRESHOLD,
    se_mult: float = 1.0,
    allow_active: bool = False,
) -> dict:
    """把 ``run_lift_tests`` 结果行按 ``lift_admission`` 写入因子库（lift 准入轨道）。

    **与 ``upsert`` 的分工**：
    - ``upsert``：单因子裸口径 library gate + 去相关 → ``admission_track="single"``。
    - 本函数：**不走**单因子 library gate；门就是 lift 本身
      （``lift_admission`` → active / probation / reject）。落盘
      ``admission_track="lift"``。

    **status cap（``allow_active``，默认 False）**：
    校准完成前的运营护栏（审查报告 §14.1），**不是永久语义**。
    ``lift_admission`` 返回 ``"active"`` 且 ``not allow_active`` 时：
    落盘 ``status="probation"``，但 ``admission_decision`` 保留原始裁决 ``"active"``，
    计数进 ``added_probation`` 并累加 ``capped_active``。``allow_active=True`` 时
    decision 即 status（现行为）。reject / 降级路径不受 cap 影响。

    **已 forward-confirmed 的 lift active 短路（状态机单调性）**：
    若 ``prev`` 已是 lift 轨 ``active`` 且 ``forward_confirmed_at`` 非空，复测
    decision 仍为 ``active`` 时**绕过 cap**，保持 ``status="active"`` 并保留
    确认字段（幂等重跑不得撤销已确认状态）。真实失败（decision=probation /
    reject）仍按既有路径降级。

    reject 语义：
    - 已有 lift 轨 ``active``/``probation`` 复测失败 → 降级 ``no_lift``
      （对齐 rebuild preserved_lift 复审），计 ``demoted_no_lift``；
    - 其余（新表达式 / single 轨 / 已是 no_lift 等）→ 跳过不写库，计 ``rejected``。
      **single 轨记录绝不被 reject 改写**。

    已存在同 expression：更新指标与 status（保留 ``added_at``），不重复添加。
    逐行 try/except：一行坏数据不崩整批，进 ``errors`` 列表。

    返回 ``{"added_active", "added_probation", "rejected", "demoted_no_lift",
    "capped_active", "errors"}``（``demoted_no_lift`` / ``capped_active`` 仅在发生时出现）。
    """
    from factorzen.discovery.lift_test import lift_admission

    meta = meta or {}
    eval_start = meta.get("eval_start")
    eval_end = meta.get("eval_end")
    universe = meta.get("universe")
    horizon = meta.get("horizon")
    run_id = meta.get("run_id") or meta.get("source_run_id")
    session_dir = meta.get("session_dir") or meta.get("source_session_dir")
    git_sha = meta.get("git_sha")
    now = meta.get("now")
    if not now:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    existing = load_library(market, root=root)
    by_expr: dict[str, FactorRecord] = {r.expression: r for r in existing}

    out: dict[str, Any] = {
        "added_active": 0,
        "added_probation": 0,
        "rejected": 0,
        "errors": [],
    }
    leaf_map = meta.get("leaf_map")
    dirty = False

    for i, row in enumerate(rows or []):
        try:
            if not isinstance(row, dict):
                out["errors"].append({"index": i, "error": "row is not a dict"})
                continue
            raw = row.get("expression")
            if not raw:
                out["errors"].append({"index": i, "error": "missing expression"})
                continue
            # norm/prev 提前：reject 降级路径也要查既有 lift 轨记录
            norm = _normalize(str(raw), leaf_map if isinstance(leaf_map, dict) else None)
            prev = by_expr.get(norm)
            decision = lift_admission(row, threshold=threshold, se_mult=se_mult)
            if decision == "reject":
                # lift 轨 active/probation 复测失败 → 降级 no_lift（对齐 rebuild）
                if (
                    prev is not None
                    and (prev.admission_track or "single") == "lift"
                    and prev.status in ("active", "probation")
                ):
                    prev.status = "no_lift"
                    prev.lift = _as_float(row.get("lift"))
                    prev.lift_baseline = _as_float(
                        row.get("lift_baseline", row.get("baseline"))
                    )
                    prev.lift_se = _as_float(row.get("lift_se"))
                    prev.lift_first_half = _as_float(row.get("lift_first_half"))
                    prev.lift_second_half = _as_float(row.get("lift_second_half"))
                    if run_id is not None:
                        prev.source_run_id = run_id
                    if session_dir is not None:
                        prev.source_session_dir = session_dir
                    if git_sha is not None:
                        prev.git_sha = git_sha
                    prev.updated_at = now
                    by_expr[norm] = prev
                    dirty = True
                    out["demoted_no_lift"] = out.get("demoted_no_lift", 0) + 1
                else:
                    # 无 prev / single 轨 / 已是 no_lift 等：不改写，计 rejected
                    out["rejected"] += 1
                continue

            # single 轨 active 不被 lift 批次覆盖/降级（与 rebuild 侧守卫同语义:
            # single 成员资格由裸口径 gate 管理,lift 批处理无权改写）。
            if (
                prev is not None
                and (prev.admission_track or "single") != "lift"
                and prev.status == "active"
            ):
                out["skipped_single_track"] = out.get("skipped_single_track", 0) + 1
                continue
            eval_window = (
                row.get("eval_start") or eval_start or (prev.eval_start if prev else None),
                row.get("eval_end") or eval_end or (prev.eval_end if prev else None),
            )
            # row 级 provenance 优先；meta 仅补缺；se_mult/threshold 由 upsert 入参兜底
            cand = dict(row)
            for _pk in (
                "admission_start", "admission_end", "scored_start", "scored_end",
                "block_days", "cv_train_days", "cv_test_days",
                "lift_threshold", "lift_se_mult", "baseline_hash",
                "profile_name", "frequency", "threshold", "se_mult",
            ):
                if cand.get(_pk) is None and meta.get(_pk) is not None:
                    cand[_pk] = meta[_pk]
            if cand.get("lift_threshold") is None and cand.get("threshold") is None:
                cand["lift_threshold"] = threshold
            if cand.get("lift_se_mult") is None and cand.get("se_mult") is None:
                cand["lift_se_mult"] = se_mult
            # 用 row 指标 + meta provenance 建记录（缺字段 None）
            rec = _record_from_candidate(
                cand, norm, market, eval_window,
                row.get("universe") if row.get("universe") is not None else universe,
                row.get("horizon") if row.get("horizon") is not None else horizon,
                run_id, session_dir, git_sha, now, prev,
            )
            rec.admission_track = "lift"
            # 统计裁决原文始终落盘；status 受 allow_active 运营护栏约束
            rec.admission_decision = decision  # "active" | "probation"
            # 已 forward-confirmed 的 lift active：复测 pass 保持 active，绕过 cap
            # （cap 只限制首次自动晋升，不撤销已确认状态；失败降级仍走下方分支）
            prev_confirmed_active = (
                prev is not None
                and (prev.admission_track or "single") == "lift"
                and prev.status == "active"
                and prev.forward_confirmed_at is not None
            )
            if decision == "active":
                if prev_confirmed_active:
                    rec.status = "active"
                    out["added_active"] += 1
                elif not allow_active:
                    # cap：校准前默认最多写 probation（§14.1），provenance 不丢
                    rec.status = "probation"
                    out["added_probation"] += 1
                    out["capped_active"] = out.get("capped_active", 0) + 1
                else:
                    rec.status = "active"
                    out["added_active"] += 1
            else:
                rec.status = decision  # probation
                out["added_probation"] += 1
            rec.evidence_tier = "v2"  # 新写入路径一律 v2
            rec.lift = _as_float(row.get("lift"))
            rec.lift_baseline = _as_float(row.get("lift_baseline", row.get("baseline")))
            rec.lift_se = _as_float(row.get("lift_se"))
            rec.lift_first_half = _as_float(row.get("lift_first_half"))
            rec.lift_second_half = _as_float(row.get("lift_second_half"))
            # 覆盖天数：row 或 meta
            hnd = row.get("holdout_n_days", row.get("n_holdout_days", meta.get("holdout_n_days")))
            if hnd is not None:
                with contextlib.suppress(TypeError, ValueError):
                    rec.holdout_n_days = int(hnd)

            by_expr[norm] = rec
            dirty = True
        except Exception as exc:
            out["errors"].append({
                "index": i,
                "expression": (row.get("expression") if isinstance(row, dict) else None),
                "error": f"{type(exc).__name__}: {exc}",
            })

    if dirty:
        _save_library(market, list(by_expr.values()), root=root)
        render_markdown(market, root=root)
    return out


def tag_legacy_records(market: str, *, root: str = DEFAULT_ROOT) -> dict[str, int]:
    """把库中 ``evidence_tier is None`` 的记录落盘标 ``"legacy"``（幂等，**不改 status**）。

    已有 tier（``legacy``/``v2`` 等）的记录不动。用于区分历史入库与新口径写入，
    打标不降级——residual / lift baseline 仍以 active pool 为基准。
    返回 ``{"tagged": n, "total": len(records)}``。
    """
    records = load_library(market, root=root)
    n_tagged = 0
    for r in records:
        if r.evidence_tier is None:
            r.evidence_tier = "legacy"
            n_tagged += 1
    if n_tagged:
        _save_library(market, records, root=root)
        render_markdown(market, root=root)
    return {"tagged": n_tagged, "total": len(records)}


def _as_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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
    n_prob = sum(1 for r in records if r.status == "probation")
    windows = sorted({(r.eval_start, r.eval_end) for r in records if r.eval_start})
    win_str = ", ".join(f"{s}–{e}" for s, e in windows) if windows else "-"
    updated = max((r.updated_at for r in records if r.updated_at), default="-")

    lines = [
        f"# 因子库 · {market}",
        "",
        f"- active: **{n_active}** · correlated: **{n_corr}** · probation: **{n_prob}** · 合计 {len(records)}",
        f"- 评估窗口: {win_str}",
        f"- 更新时间: {updated}",
        "",
    ]
    if records:
        lines += [
            "| # | expression | market | ic_train | holdout_ic | dsr_pvalue | n_train | status | tier | eval 窗口 | added_at |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(sorted(records, key=_sort_key), 1):
            win = f"{r.eval_start or '-'}–{r.eval_end or '-'}"
            corr = f" (~{r.correlated_with})" if r.status == "correlated" and r.correlated_with else ""
            tier = r.evidence_tier or "-"
            lines.append(
                f"| {i} | `{r.expression}` | {r.market} | {_fmt(r.ic_train)} | "
                f"{_fmt(r.holdout_ic)} | {_fmt(r.dsr_pvalue)} | {_fmt(r.n_train)} | "
                f"{r.status}{corr} | {tier} | {win} | {r.added_at or '-'} |"
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
    lines = ["# 因子库总览", "",
             "| market | active | correlated | probation | 最强因子 (holdout_ic) |",
             "|---|---|---|---|---|"]
    for market in ("ashare", "crypto", "futures", "us"):
        records = load_library(market, root=root)
        n_active = sum(1 for r in records if r.status == "active")
        n_corr = sum(1 for r in records if r.status == "correlated")
        n_prob = sum(1 for r in records if r.status == "probation")
        best = min(records, key=_sort_key, default=None)  # _sort_key 用负号，min=最强
        best_str = (f"`{best.expression}` ({_fmt(best.holdout_ic)})"
                    if best is not None else "-")
        lines.append(f"| {market} | {n_active} | {n_corr} | {n_prob} | {best_str} |")
    md = "\n".join(lines) + "\n"
    out = Path(root) / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return md


# ── rebuild：候选源收集 + 统一窗口评估器 ─────────────────────────────────────

def collect_source_expressions(
    market: str, *,
    mine_team_root: str = str(MINE_TEAM_DIR),
    mining_sessions_root: str = str(MINING_SESSIONS_DIR),
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

    from factorzen.discovery.evaluation import _preprocess_daily, evaluate_expressions
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
    # ── lift 轨复审注入点（测试 mock / 生产默认 run_lift_tests）──────────────
    lift_runner: Callable[..., list[dict]] | None = None,
    combine_fn: Callable | None = None,
    active_factor_dfs: dict[str, pl.DataFrame] | None = None,
    daily: pl.DataFrame | None = None,
    lift_threshold: float = DEFAULT_LIFT_THRESHOLD,
    se_mult: float = 1.0,
    profile=None,
    admission_start: str | None = None,
    admission_end: str | None = None,
) -> UpsertResult:
    """在统一窗口重算历史因子并**从头重建**该市场库（"可比"且"权威"的关键）。

    ``sources``：从历史产物收集的候选表达式（`collect_source_expressions`）。
    ``evaluate``：``list[expr] -> list[候选 dict]``——复用挖掘的数据装配+评估（CLI 注入闭包，
    测试注入 mock）。别在此重写评估。窗口=``eval_window``，holdout 由 evaluate 内部切末段。
    ``compact_materialize``：去相关的紧凑矩阵物化器（`build_library_evaluator` 返回；内存有界）。
    ``fresh``（默认 True）：rebuild 语义 = **从零重算全部历史源**，先清空该市场旧库再写入——
    否则旧库里已失效的记录（如 P0 修复前误收的前视因子、或已从算子库移除的表达式）会残留，
    rebuild 就不再"权威"。清空只针对本 market 文件，不动别的市场。

    **双轨 rebuild**：
    - **single 轨**（``admission_track=="single"`` 或旧记录无字段）：现有裸口径
      library gate + 去相关，**零回归**。
    - **单因子 probation 原样保留**（``upsert_probation`` 遗留路径，非 lift 轨）：
      fresh 清空前抽出，upsert 后写回（不重算）。
    - **lift 轨**（``admission_track=="lift"``）：单轨 rebuild 得到新 active 池后，
      逐个重跑 add-one lift（``lift_runner`` / 默认 ``run_lift_tests``，
      ``active_factor_dfs``=新池，``top_m=None``）→ ``lift_admission``：
      active / probation / reject→``status="no_lift"``（记录保留，不删除）。
      **复审路径不 cap**：``decision=="active"`` 维持/写回 active 是对已入库记录的
      降级/维持判定，不是新晋升（与 ``upsert_lift_admissions`` 的 ``allow_active``
      运营护栏分工不同）。复审整体 try/except：失败不毁单轨结果，manifest 记 error，
      lift 轨保持原状。

    注入点：``lift_runner(cands, *, active_factor_dfs, combine_fn=..., **kw) -> list[dict]``；
    ``combine_fn`` 转给 runner；``active_factor_dfs`` / ``daily`` 供生产默认路径。
    落 ``rebuild_{market}_manifest.json``（窗口/源/git_sha/时间 + lift 复审计数，可复现）。
    """
    from factorzen.discovery.lift_test import lift_admission

    # 保留：single 轨 probation + 全部 lift 轨（fresh 会清库，须先抽出）
    preserved_probation: list[FactorRecord] = []
    preserved_lift: list[FactorRecord] = []
    if fresh:
        existing_pre = load_library(market, root=root)
        for r in existing_pre:
            track = r.admission_track or "single"
            if track == "lift":
                preserved_lift.append(r)
            elif r.status == "probation":
                # single 轨（或旧无字段）probation：原样保留，不重算
                preserved_probation.append(r)
        library_path(market, root).unlink(missing_ok=True)  # 从零重建，清旧库（仅本 market）
    else:
        # 非 fresh：仍对库内 lift 轨做复审（不抽 single probation）
        preserved_lift = [
            r for r in load_library(market, root=root)
            if (r.admission_track or "single") == "lift"
        ]

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

    # 写回 single 轨 probation（规范形已在库则不覆盖——同 expr 被 gate 路径重算进库时
    # 以 gate 为准；仅当库内尚无该 expr 时补回试用记录）
    if preserved_probation:
        lib = load_library(market, root=root)
        by_expr = {r.expression: r for r in lib}
        n_kept = 0
        for pr in preserved_probation:
            if pr.expression not in by_expr:
                by_expr[pr.expression] = pr
                n_kept += 1
        if n_kept:
            _save_library(market, list(by_expr.values()), root=root)
            render_markdown(market, root=root)
        res.records.extend(
            [pr for pr in preserved_probation if pr.expression in by_expr
             and by_expr[pr.expression].status == "probation"]
        )

    # ── lift 轨复审 ──────────────────────────────────────────────────────────
    n_lift_reviewed = 0
    n_lift_active = 0
    n_lift_probation = 0
    n_lift_demoted = 0
    n_lift_evaluated = 0
    lift_review_error: str | None = None

    if preserved_lift:
        try:
            # 新 active 池物化（供 add-one 基线）；优先注入的 active_factor_dfs
            pool_dfs = active_factor_dfs
            if pool_dfs is None and materialize is not None:
                pool_dfs = {}
                for r in load_library(market, root=root):
                    if r.status != "active":
                        continue
                    try:
                        panel = materialize(r.expression)
                    except Exception:
                        panel = None
                    if panel is not None:
                        pool_dfs[r.expression] = panel
            if pool_dfs is None:
                pool_dfs = {}

            runner = lift_runner
            if runner is None:
                # 生产默认：run_lift_tests（需 daily）+ LiftEvalContext
                if daily is None:
                    raise RuntimeError(
                        "lift 轨复审需要 lift_runner 或 daily（默认 run_lift_tests）"
                    )
                from factorzen.discovery.evaluation import _preprocess_daily
                from factorzen.discovery.lift_test import (
                    LiftEvalContext,
                    run_lift_tests,
                )

                # prep 一次；按 horizon 缓存 ctx（ret panel 依赖 horizon，不可共享）
                prepped = _preprocess_daily(daily, profile).sort(
                    ["ts_code", "trade_date"]
                )
                profile_name = (
                    getattr(profile, "name", None) if profile is not None else None
                )
                ctx_by_h: dict[int, LiftEvalContext] = {}
                # 捕获 rebuild 全局 horizon（runner 参数 horizon 会遮蔽外层名）
                _rebuild_h = horizon

                def _ctx_for(h: int) -> LiftEvalContext:
                    if h not in ctx_by_h:
                        ctx_by_h[h] = LiftEvalContext(
                            market=market,
                            prepped=prepped,
                            leaf_map=leaf_map,
                            horizon=int(h),
                            admission_start=admission_start,
                            admission_end=admission_end,
                            library_root=root,
                            profile_name=profile_name,
                        )
                    return ctx_by_h[h]

                def runner(
                    cands, *, active_factor_dfs=None, combine_fn=None,
                    horizon=None, **kw,
                ):
                    # 调用方传 rec.horizon；缺省退回 rebuild 全局 horizon 或 5
                    if horizon is not None:
                        h = int(horizon)
                    elif _rebuild_h is not None:
                        h = int(_rebuild_h)
                    else:
                        h = 5
                    return run_lift_tests(
                        cands,
                        market=market,
                        daily=daily,
                        leaf_map=leaf_map,
                        library_root=root,
                        top_m=None,
                        threshold=lift_threshold,
                        active_factor_dfs=active_factor_dfs,
                        combine_fn=combine_fn,
                        ctx=_ctx_for(h),
                    )

            lib = load_library(market, root=root)
            by_expr = {r.expression: r for r in lib}
            reviewed: list[FactorRecord] = []

            for rec in preserved_lift:
                n_lift_reviewed += 1
                cand_row = {
                    "expression": rec.expression,
                    "ic_train": rec.ic_train,
                    "holdout_ic": rec.holdout_ic,
                    "n_train": rec.n_train,
                }
                # rec.horizon 优先（准入目标）；否则 rebuild 全局 horizon；再否则 5
                rec_h = rec.horizon if rec.horizon is not None else (horizon if horizon is not None else 5)
                rows = runner(
                    [cand_row],
                    active_factor_dfs=pool_dfs,
                    combine_fn=combine_fn,
                    horizon=rec_h,
                )
                n_lift_evaluated += 1  # 每次 add-one 计 1 次 lgbm（多重检验 N）
                if not rows:
                    # 无结果 → 视为无增量
                    decision = "reject"
                    lift_row: dict = {}
                else:
                    lift_row = rows[0]
                    decision = lift_admission(
                        lift_row, threshold=lift_threshold, se_mult=se_mult,
                    )

                # 在原记录上更新 status / lift 字段（保留 provenance / added_at）
                updated = FactorRecord.from_dict(rec.to_dict())
                updated.updated_at = now
                updated.admission_track = "lift"
                if lift_row:
                    if lift_row.get("lift") is not None:
                        updated.lift = _as_float(lift_row.get("lift"))
                    if lift_row.get("lift_baseline", lift_row.get("baseline")) is not None:
                        updated.lift_baseline = _as_float(
                            lift_row.get("lift_baseline", lift_row.get("baseline"))
                        )
                    if lift_row.get("lift_se") is not None:
                        updated.lift_se = _as_float(lift_row.get("lift_se"))
                    if lift_row.get("lift_first_half") is not None:
                        updated.lift_first_half = _as_float(lift_row.get("lift_first_half"))
                    if lift_row.get("lift_second_half") is not None:
                        updated.lift_second_half = _as_float(lift_row.get("lift_second_half"))

                # 复审不 cap：decision 即 status（保留既有 active，非 auto-lift 新晋升）
                if decision == "active":
                    updated.status = "active"
                    updated.admission_decision = "active"
                    n_lift_active += 1
                elif decision == "probation":
                    updated.status = "probation"
                    updated.admission_decision = "probation"
                    n_lift_probation += 1
                else:
                    updated.status = "no_lift"
                    updated.admission_decision = "reject"
                    n_lift_demoted += 1

                # 同 expr 已被 single 轨 gate 收录 → 不覆盖 single 记录（single 零回归）
                existing = by_expr.get(updated.expression)
                if existing is not None and (existing.admission_track or "single") != "lift":
                    continue
                by_expr[updated.expression] = updated
                reviewed.append(updated)

            _save_library(market, list(by_expr.values()), root=root)
            render_markdown(market, root=root)
            res.records.extend(reviewed)
        except Exception as exc:
            # 复审失败：不毁单轨结果；lift 轨写回原状
            lift_review_error = f"{type(exc).__name__}: {exc}"
            _LOG.warning("rebuild lift review failed for %s: %s", market, lift_review_error)
            lib = load_library(market, root=root)
            by_expr = {r.expression: r for r in lib}
            for pr in preserved_lift:
                if pr.expression not in by_expr or (
                    by_expr[pr.expression].admission_track or "single"
                ) == "lift":
                    by_expr[pr.expression] = pr
            _save_library(market, list(by_expr.values()), root=root)
            render_markdown(market, root=root)
            # 失败时计数归零（未成功复审）；n_lift_reviewed 保留尝试意图
            n_lift_active = sum(1 for r in preserved_lift if r.status == "active")
            n_lift_probation = sum(1 for r in preserved_lift if r.status == "probation")
            n_lift_demoted = sum(1 for r in preserved_lift if r.status == "no_lift")
            n_lift_evaluated = 0

    manifest = {
        "market": market, "eval_start": eval_window[0], "eval_end": eval_window[1],
        "universe": universe, "horizon": horizon, "git_sha": git_sha, "rebuilt_at": now,
        "n_sources": len(sources), "n_unique": len(uniq), "n_evaluated": len(cand_dicts),
        "added": res.added, "updated": res.updated, "correlated": res.correlated,
        "skipped": res.skipped,
        "n_probation_preserved": len(preserved_probation),
        "n_lift_reviewed": n_lift_reviewed,
        "n_lift_active": n_lift_active,
        "n_lift_probation": n_lift_probation,
        "n_lift_demoted": n_lift_demoted,
        "n_lift_evaluated": n_lift_evaluated,
        # lift 复审评分窗（与 single 轨 holdout 尾段对齐；None=未裁）
        "lift_admission_start": admission_start,
        "lift_admission_end": admission_end,
    }
    if lift_review_error is not None:
        manifest["lift_review_error"] = lift_review_error
        res.lift_review_error = lift_review_error
    if manifest_extra:
        manifest.update(manifest_extra)
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / f"rebuild_{market}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return res
