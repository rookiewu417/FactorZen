"""因子资产库：meta.json + factor.py + factor.parquet（4 列面板唯一落点）。

**架构裁决**：
- ``workspace/factor_library/{market}.jsonl`` 仍是裁决唯一真相
  （status / lift / admission / forward）。
- 本模块是资产库载体：可 import 的 ``factor.py`` + 元数据 + **数值面板**。
- **因子数值面板唯一落点**是 ``factors/<market>/<name>/factor.parquet``
  （4 列：trade_date / ts_code / factor_value / factor_clean）。
- 评估 run 只留 json/html 等轻量产物，**不写任何 parquet**
  （落 ``factors/<market>/<name>/evaluations/{run_id}/``）。

目录布局::

    workspace/factors/<market>/<name>/
    ├── meta.json
    ├── factor.py
    ├── factor.parquet   # 4 列面板（每因子一份）
    └── evaluations/     # 评估 run 产物（json/html）
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from factorzen.config.settings import FACTOR_LIBRARY_DIR, FACTOR_STORE_DIR
from factorzen.discovery.factor_library import (
    DEFAULT_ROOT as DEFAULT_LIB_ROOT,
)
from factorzen.discovery.factor_library import (
    FactorRecord,
    _is_python_record,
    _normalize,
    default_name_for_expression,
    load_library,
)

_LOG = logging.getLogger(__name__)

DEFAULT_ROOT = str(FACTOR_STORE_DIR)
MATERIALIZE_STATUSES = frozenset({"active", "probation"})
# 因子数值面板唯一落点；False 时跳过写/读（仅测试逃生）。
STORE_FACTOR_PARQUET_ENABLED = True
FACTOR_PANEL_COLUMNS = (
    "trade_date",
    "ts_code",
    "factor_value",
    "factor_clean",
)
_HYPOTHESIS_MAX = 200

# 资产库 parquet 物化口径（与 jsonl 裁决/评估口径分离）：
# ledger 的 eval_start/eval_end/universe 仍是挖掘评估窗；
# factor.parquet 统一 all_a × 2016-01-01 ~ 最新已完结交易日。
STORE_MATERIALIZE_UNIVERSE = "all_a"
STORE_MATERIALIZE_START = "2016-01-01"


def store_materialize_end() -> str:
    """sync 时的最新已完结交易日（``YYYY-MM-DD``）。

    不以「今天」直接当交易日 end：盘中/未收盘 bar 可能不齐。
    统一取 ``date.today()`` 之前的上一个交易日
    （``factorzen.core.calendar.prev_trade_date``）。
    """
    from datetime import date as _date

    from factorzen.core.calendar import prev_trade_date

    return prev_trade_date(_date.today()).isoformat()


def finalize_factor_panel(
    panel: pl.DataFrame,
    *,
    stock_basic: pl.DataFrame | None = None,
    daily_basic: pl.DataFrame | None = None,
    neutralize: bool = False,
) -> pl.DataFrame:
    """规范为 4 列面板；缺 ``factor_clean`` 时跑默认预处理补上。

    dtype 在此单点收口（ts_code=Utf8 等），否则 mining prepped 帧的
    Categorical ts_code 会直接落盘，与 eval 路径写的 Utf8 产生 on-disk 分叉。
    """
    if not {"trade_date", "ts_code"}.issubset(panel.columns):
        raise ValueError(
            f"panel must have trade_date/ts_code, got {list(panel.columns)}"
        )
    if set(FACTOR_PANEL_COLUMNS).issubset(panel.columns):
        return _cast_panel_dtypes(panel.select(list(FACTOR_PANEL_COLUMNS)))
    if "factor_value" not in panel.columns:
        raise ValueError(
            f"panel must have factor_value (or full 4-col), got {list(panel.columns)}"
        )
    from factorzen.daily.preprocessing.pipeline import PreprocessingPipeline

    cleaned = PreprocessingPipeline(
        steps=["outlier", "missing", "normalize"],
        neutralize=bool(
            neutralize and (stock_basic is not None or daily_basic is not None)
        ),
    ).run(
        panel,
        col="factor_value",
        stock_basic=stock_basic,
        daily_basic=daily_basic,
    )
    return _cast_panel_dtypes(cleaned.select(list(FACTOR_PANEL_COLUMNS)))


def _cast_panel_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """4 列面板落盘 dtype 单点：Date / Utf8 / Float64 / Float64。"""
    cols: list[pl.Expr] = []
    if df["trade_date"].dtype != pl.Date:
        if df["trade_date"].dtype == pl.Utf8:
            sample = ""
            s = df["trade_date"].drop_nulls().head(1)
            if s.len():
                sample = str(s.to_list()[0])
            fmt = "%Y%m%d" if "-" not in sample else "%Y-%m-%d"
            cols.append(pl.col("trade_date").str.strptime(pl.Date, fmt, strict=False))
        else:
            cols.append(pl.col("trade_date").cast(pl.Date, strict=False))
    cols.extend(
        [
            pl.col("ts_code").cast(pl.Utf8),
            pl.col("factor_value").cast(pl.Float64),
            pl.col("factor_clean").cast(pl.Float64),
        ]
    )
    return df.with_columns(cols)


def write_factor_panel(
    market: str,
    name: str,
    panel: pl.DataFrame,
    *,
    root: str | None = None,
    expression: str | None = None,
    stock_basic: pl.DataFrame | None = None,
    daily_basic: pl.DataFrame | None = None,
    neutralize: bool = False,
) -> Path | None:
    """把 4 列因子面板写入 ``factors/<market>/<name>/factor.parquet``。

    手动/脚本用的整面板写入口；**生产评估路径已不调用**（评估经
    ``pipelines.factor_panel_cache.ensure_factor_store_panel`` 增量维护）。
    注意：meta.materialization 固定标 store 口径窗（2016-01-01~最新完结日），
    **不看实际面板行**——只用于写入完整口径面板，子集面板勿经此落盘。
    """
    if not STORE_FACTOR_PARQUET_ENABLED:
        return None
    root = DEFAULT_ROOT if root is None else root
    d = asset_dir(market, name, root=root)
    d.mkdir(parents=True, exist_ok=True)
    out = finalize_factor_panel(
        panel,
        stock_basic=stock_basic,
        daily_basic=daily_basic,
        neutralize=neutralize,
    )
    pq_path = d / "factor.parquet"
    out.write_parquet(pq_path)
    meta = _read_json(d / "meta.json") or {"name": name, "market": market}
    meta["materialization"] = {
        "start": STORE_MATERIALIZE_START,
        "end": store_materialize_end(),
        "universe": STORE_MATERIALIZE_UNIVERSE,
        "git_sha": _git_sha(),
        "n_rows": int(out.height),
        "generated_at": _utc_now_iso(),
        "expression": expression if expression is not None else meta.get("expression"),
        "columns": list(FACTOR_PANEL_COLUMNS),
    }
    if expression is not None:
        meta["expression"] = expression
    _write_json(d / "meta.json", meta)
    _LOG.info(
        "factor_store panel written %s/%s n_rows=%s path=%s",
        market,
        name,
        out.height,
        pq_path,
    )
    return pq_path


def _date_key(s: str | None) -> str:
    """日期串归一到 ``YYYYMMDD`` 便于比较（兼容 ``YYYY-MM-DD`` / ``YYYYMMDD``）。"""
    if not s:
        return ""
    return str(s).replace("-", "").replace("/", "")[:8]


def _to_ymd8(s: str) -> str:
    """物化装帧通道用 ``YYYYMMDD``。"""
    return _date_key(s)


# expression 型 factor.py 生成模板（单测锁死：import + compute 与生产求值一致）
_EXPRESSION_FACTOR_PY_TEMPLATE = '''\
"""Expression factor: {name}

Expression
    {expression}

Hypothesis
    {hypothesis}

Ledger snapshot (truth = workspace/factor_library jsonl)
    status={status}  ic_train={ic_train}  holdout_ic={holdout_ic}
    admission_ic={admission_ic}  lift={lift}
"""
from __future__ import annotations

import polars as pl

from factorzen.discovery.expression import evaluate_materialized, parse_expr

EXPRESSION = {expression!r}


def compute(daily: pl.DataFrame) -> pl.DataFrame:
    """Evaluate EXPRESSION on a preprocessed daily panel.

    Parameters
    ----------
    daily:
        Sorted by (ts_code, trade_date); must contain leaf columns referenced
        by the expression (and any derived columns the expression needs).

    Returns
    -------
    pl.DataFrame
        Columns: trade_date, ts_code, factor_value (finite rows only).
    """
    node = parse_expr(EXPRESSION)
    values = evaluate_materialized(node, daily)
    return (
        daily.select(["trade_date", "ts_code"])
        .with_columns(values.alias("factor_value"))
        .filter(
            pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite()
        )
    )
'''


def store_root_for_library(lib_root: str | None = None) -> str:
    """把 factor_library 根目录映射到 factors 资产根。

    - 生产默认库根 → ``workspace/factors``
    - 测试/自定义库根 → ``{lib_root}/factors``（与 jsonl 同树，隔离）
    """
    if lib_root is None:
        return DEFAULT_ROOT
    p = Path(lib_root).resolve()
    if p == Path(DEFAULT_LIB_ROOT).resolve() or p == Path(FACTOR_LIBRARY_DIR).resolve():
        return DEFAULT_ROOT
    return str(p / "factors")


def asset_dir(market: str, name: str, *, root: str = DEFAULT_ROOT) -> Path:
    return Path(root) / market / name


def record_asset_name(rec: FactorRecord) -> str:
    """资产目录名：优先 record.name，否则 mined_{sha}。"""
    name = (rec.name or "").strip()
    if name:
        return name
    if _is_python_record(rec) and rec.expression:
        from factorzen.discovery.factor_library import _python_name_from_expression

        py = _python_name_from_expression(rec.expression)
        if py:
            return py
    expr = (rec.expression or "").strip()
    if expr:
        return default_name_for_expression(_normalize(expr))
    raise ValueError("FactorRecord has no name/expression for asset path")


def _truncate(s: str | None, n: int = _HYPOTHESIS_MAX) -> str:
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n]


def _ledger_truth(market: str) -> str:
    return f"workspace/factor_library/{market}.jsonl"


def build_meta(
    rec: FactorRecord,
    *,
    market: str,
    materialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 meta.json 内容（字段契约见模块 docstring / 任务规格）。"""
    name = record_asset_name(rec)
    kind = "python" if _is_python_record(rec) else "expression"
    description = _truncate(rec.hypothesis) if kind == "expression" else (rec.hypothesis or "")
    # python 型 description 也可来自实现；hypothesis 优先
    if kind == "python" and not description:
        description = ""
    return {
        "name": name,
        "kind": kind,
        "expression": rec.expression,
        "frequency": rec.frequency or "daily",
        "description": description,
        "source_run_id": rec.source_run_id,
        "created_at": rec.added_at or rec.updated_at,
        "ledger_snapshot": {
            "status": rec.status,
            "lift": rec.lift,
            "admission_ic": rec.admission_ic,
            "ic_train": rec.ic_train,
            "holdout_ic": rec.holdout_ic,
            "truth": _ledger_truth(market),
        },
        "materialization": materialization,
    }


def render_expression_factor_py(
    *,
    name: str,
    expression: str,
    hypothesis: str | None = None,
    snapshot: dict[str, Any] | None = None,
) -> str:
    """生成 expression 类 factor.py 源码（可 import + compute）。"""
    snap = snapshot or {}
    return _EXPRESSION_FACTOR_PY_TEMPLATE.format(
        name=name,
        expression=expression,
        hypothesis=hypothesis or "",
        status=snap.get("status"),
        ic_train=snap.get("ic_train"),
        holdout_ic=snap.get("holdout_ic"),
        admission_ic=snap.get("admission_ic"),
        lift=snap.get("lift"),
    )


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _git_sha() -> str:
    try:
        from factorzen.core.experiment import get_git_sha

        return get_git_sha() or "unknown"
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def load_materialized_factor(
    record: FactorRecord,
    *,
    market: str,
    root: str = DEFAULT_ROOT,
    start: str,
    end: str,
    universe: str,
    allow_warmup_head: bool = False,
) -> tuple[pl.DataFrame | None, str | None, dict[str, Any] | None]:
    """读 factor_store 物化 parquet；门控全过才返回帧。

    返回 ``(df, reason, hit_meta)``：

    - 命中：``(frame, None, {git_sha, generated_at})``
    - miss：``(None, reason, None)``；调用方可聚合 reason 分布

    命中条件（全部满足）::

        1. 资产目录与 factor.parquet 存在
        2. meta.expression 与 record.expression 完全一致
        3. materialization 非 null 且 universe 与请求严格相等
        4. 下界：默认要求 parquet 实际 min(trade_date) <= start；
           ``allow_warmup_head=True`` 时放宽为 meta.materialization.start <= start
           （滚动因子预热吃掉头部是正常语义，不视为 miss）
           上界仍要求 materialization.end >= end
        5. 过滤到 [start, end] 后非空

    ``allow_warmup_head`` 默认 False：combine 调用点不传，行为零变化。
    eval 轨显式传 True 以吃预热头。

    返回帧 schema：至少 ``trade_date / ts_code / factor_value``；
    若盘上有 ``factor_clean`` 则一并返回（4 列完整面板）。
    读盘损坏 → miss + warning，不炸。

    ``STORE_FACTOR_PARQUET_ENABLED=False`` 时恒 miss。
    """
    if not STORE_FACTOR_PARQUET_ENABLED:
        return None, "store_parquet_disabled", None

    try:
        name = record_asset_name(record)
    except ValueError:
        return None, "no_asset_name", None

    d = asset_dir(market, name, root=root)
    pq_path = d / "factor.parquet"
    if not d.is_dir():
        return None, "missing_asset", None
    if not pq_path.is_file():
        return None, "missing_parquet", None

    meta = _read_json(d / "meta.json")
    if meta is None:
        return None, "meta_unreadable", None

    if meta.get("expression") != record.expression:
        return None, "expression_mismatch", None

    mat = meta.get("materialization")
    if not isinstance(mat, dict):
        return None, "no_materialization", None

    mat_univ = mat.get("universe")
    if mat_univ != universe:
        return None, "universe_mismatch", None

    start_key = _date_key(start)
    end_key = _date_key(end)
    if not start_key or not end_key:
        return None, "bad_request_window", None

    mat_end_key = _date_key(mat.get("end"))
    if not mat_end_key or mat_end_key < end_key:
        return None, "window_end_uncovered", None

    try:
        df = pl.read_parquet(pq_path)
    except Exception as exc:
        _LOG.warning(
            "factor_store: 读 parquet 失败 %s: %s: %s",
            pq_path,
            type(exc).__name__,
            exc,
        )
        return None, "parquet_corrupt", None

    needed = {"trade_date", "ts_code", "factor_value"}
    if not needed.issubset(set(df.columns)):
        return None, "parquet_schema", None

    # 统一 schema，便于与 combine 重算路径对齐
    try:
        td = df["trade_date"]
        if td.dtype != pl.Date:
            if td.dtype == pl.Utf8:
                sample = (
                    str(td.drop_nulls().head(1).to_list()[0])
                    if td.drop_nulls().len()
                    else ""
                )
                fmt = (
                    "%Y%m%d"
                    if len(sample.replace("-", "")) == 8 and "-" not in sample
                    else "%Y-%m-%d"
                )
                df = df.with_columns(
                    pl.col("trade_date").str.strptime(pl.Date, fmt, strict=False)
                )
            else:
                df = df.with_columns(pl.col("trade_date").cast(pl.Date, strict=False))
        cols = [
            pl.col("trade_date").cast(pl.Date),
            pl.col("ts_code").cast(pl.Utf8),
            pl.col("factor_value").cast(pl.Float64),
        ]
        if "factor_clean" in df.columns:
            cols.append(pl.col("factor_clean").cast(pl.Float64))
        df = df.select(cols)
    except Exception as exc:
        _LOG.warning(
            "factor_store: parquet schema cast 失败 %s: %s: %s",
            pq_path,
            type(exc).__name__,
            exc,
        )
        return None, "parquet_schema", None

    if df.is_empty() or df["trade_date"].null_count() == df.height:
        return None, "parquet_empty", None

    # 下界：默认看 parquet 实际最小日；eval 放宽看 meta.materialization.start
    pq_min = df["trade_date"].min()
    if pq_min is None:
        return None, "parquet_empty", None
    pq_min_key = _date_key(
        pq_min.isoformat() if hasattr(pq_min, "isoformat") else str(pq_min)
    )
    if pq_min_key > start_key:
        if allow_warmup_head:
            # 预热吃头：口径 start 覆盖请求 start 即可，不要求 parquet min
            mat_start_key = _date_key(mat.get("start"))
            if not mat_start_key or mat_start_key > start_key:
                return None, "window_start_uncovered", None
        else:
            return None, "window_start_uncovered", None

    # 过滤到请求窗 [start, end]
    from datetime import date as _date

    start_d = _date(int(start_key[:4]), int(start_key[4:6]), int(start_key[6:8]))
    end_d = _date(int(end_key[:4]), int(end_key[4:6]), int(end_key[6:8]))
    out = df.filter(
        (pl.col("trade_date") >= start_d) & (pl.col("trade_date") <= end_d)
    )
    if out.is_empty():
        return None, "empty_after_filter", None
    hit_meta = {
        "git_sha": mat.get("git_sha"),
        "generated_at": mat.get("generated_at"),
    }
    return out, None, hit_meta


def write_factor_asset(
    record: FactorRecord,
    *,
    market: str,
    root: str = DEFAULT_ROOT,
    materialize: bool = False,
    panel: pl.DataFrame | None = None,
    python_source: str | None = None,
) -> str:
    """写单因子三件套。返回资产目录路径。

    Parameters
    ----------
    materialize:
        True 且 status ∈ {{active, probation}} 时写 parquet。
        ``panel`` 非空则直接落盘；否则不自动拉数（物化由 ``sync_store`` 批处理）。
    python_source:
        kind=python 时的 factor.py 全文；缺省则保留已有 factor.py（若无则写占位）。
    """
    name = record_asset_name(record)
    d = asset_dir(market, name, root=root)
    d.mkdir(parents=True, exist_ok=True)

    kind = "python" if _is_python_record(record) else "expression"
    py_path = d / "factor.py"
    if kind == "expression":
        code = render_expression_factor_py(
            name=name,
            expression=record.expression or "",
            hypothesis=record.hypothesis,
            snapshot={
                "status": record.status,
                "ic_train": record.ic_train,
                "holdout_ic": record.holdout_ic,
                "admission_ic": record.admission_ic,
                "lift": record.lift,
            },
        )
        py_path.write_text(code, encoding="utf-8")
    else:
        if python_source is not None:
            py_path.write_text(python_source, encoding="utf-8")
        elif not py_path.exists():
            # 无源码时写最小占位，避免空目录；真正迁移由 sync/手写拷贝完成
            py_path.write_text(
                f'"""Python factor: {name} (source pending migration)."""\n'
                f"# registry name: {name}\n",
                encoding="utf-8",
            )

    # materialization / factor.parquet：默认废除（大面板只在 factors 资产）
    mat_info: dict[str, Any] | None = None
    pq_path = d / "factor.parquet"
    prev_meta = _read_json(d / "meta.json") or {}
    prev_mat = prev_meta.get("materialization")
    should_mat = (
        STORE_FACTOR_PARQUET_ENABLED
        and materialize
        and (record.status or "") in MATERIALIZE_STATUSES
    )

    if should_mat and panel is not None:
        out = finalize_factor_panel(panel)
        out.write_parquet(pq_path)
        mat_info = {
            "start": STORE_MATERIALIZE_START,
            "end": store_materialize_end(),
            "universe": STORE_MATERIALIZE_UNIVERSE,
            "git_sha": _git_sha(),
            "n_rows": int(out.height),
            "generated_at": _utc_now_iso(),
            "expression": record.expression,
            "columns": list(FACTOR_PANEL_COLUMNS),
        }
    elif (record.status or "") not in MATERIALIZE_STATUSES:
        mat_info = None
    elif isinstance(prev_mat, dict) and pq_path.exists():
        if prev_meta.get("expression") == record.expression and _materialization_window_fresh(
            prev_mat
        ):
            mat_info = prev_mat
        else:
            mat_info = None
    else:
        mat_info = None

    meta = build_meta(record, market=market, materialization=mat_info)
    _write_json(d / "meta.json", meta)
    return str(d)


def _materialization_window_fresh(mat: dict[str, Any]) -> bool:
    """store 物化口径是否仍新鲜。

    增量判据（end 语义 = 物化时最新已完结交易日）::

        start <= STORE_MATERIALIZE_START   # 超集覆盖即新鲜（eval 补头可写更早 start，
                                           # 严格相等会与 ensure 乒乓互踩、丢头部行）
        AND universe == STORE_MATERIALIZE_UNIVERSE
        AND mat.end 已覆盖「上一个已完结交易日」(>= store_materialize_end())

    未覆盖则需重物化（例如日历推进一天、或历史 csi300/csi500 资产）。
    """
    mat_start = mat.get("start")
    if not mat_start or _date_key(mat_start) > _date_key(STORE_MATERIALIZE_START):
        return False
    if mat.get("universe") != STORE_MATERIALIZE_UNIVERSE:
        return False
    target_end = store_materialize_end()
    return _date_key(mat.get("end")) >= _date_key(target_end)


def _materialization_fresh(
    rec: FactorRecord,
    meta: dict[str, Any] | None,
    pq_path: Path,
) -> bool:
    """meta.expression 与 parquet 的 store 物化口径仍新鲜 → 可跳过物化。

    不再对照 ``record.eval_start/eval_end/universe``（那是裁决评估口径）；
    只认 ``STORE_MATERIALIZE_*`` + ``store_materialize_end()`` 覆盖。
    """
    if meta is None or not pq_path.exists():
        return False
    mat = meta.get("materialization")
    if not isinstance(mat, dict):
        return False
    if meta.get("expression") != rec.expression:
        return False
    if (
        mat.get("expression") not in (None, rec.expression)
        and mat.get("expression") != rec.expression
    ):
        return False
    if not _materialization_window_fresh(mat):
        return False
    return bool(mat.get("n_rows"))


def _asset_materialization_fresh(meta: dict[str, Any] | None, pq_path: Path) -> bool:
    """store 资产目录增量：materialization 口径新鲜且 parquet 存在 → 可 skip。

    不经 library / status 门；expression 以 meta 自身为准
    （mat.expression 若与 meta.expression 冲突则重物化）。
    """
    if meta is None or not pq_path.exists():
        return False
    mat = meta.get("materialization")
    if not isinstance(mat, dict):
        return False
    meta_expr = meta.get("expression")
    mat_expr = mat.get("expression")
    if mat_expr not in (None, meta_expr) and mat_expr != meta_expr:
        return False
    if not _materialization_window_fresh(mat):
        return False
    return bool(mat.get("n_rows"))


def _apply_asset_materialization(
    asset_path: Path,
    panel: pl.DataFrame,
    *,
    expression: str,
) -> None:
    """写 factor.parquet 并更新 meta.materialization（保留 ledger 等其余字段）。

    不走 write_factor_asset 的 status 门——correlated / store-only python
    也可落盘物化。

    ``STORE_FACTOR_PARQUET_ENABLED=False`` 时为 no-op。
    """
    if not STORE_FACTOR_PARQUET_ENABLED:
        return

    out = finalize_factor_panel(panel)
    out.write_parquet(asset_path / "factor.parquet")
    meta = _read_json(asset_path / "meta.json") or {}
    meta["materialization"] = {
        "start": STORE_MATERIALIZE_START,
        "end": store_materialize_end(),
        "universe": STORE_MATERIALIZE_UNIVERSE,
        "git_sha": _git_sha(),
        "n_rows": int(out.height),
        "generated_at": _utc_now_iso(),
        "expression": expression,
        "columns": list(FACTOR_PANEL_COLUMNS),
    }
    _write_json(asset_path / "meta.json", meta)


def materialize_assets(
    market: str,
    names: Iterable[str] | None = None,
    root: str = DEFAULT_ROOT,
    *,
    panel_loader: PanelLoader | None = None,
) -> dict[str, Any]:
    """直接遍历 factor_store 资产目录物化 parquet（不经 library / status 门）。

    覆盖 correlated、仅 store 有的 python 手写因子等 sync_store 不会物化的资产。
    物化口径与 sync 一致：``STORE_MATERIALIZE_*`` × ``store_materialize_end()``。
    增量：materialization 新鲜且 parquet 存在 → skip。
    单资产失败记入 errors 继续下一条，不炸整批。

    Parameters
    ----------
    names:
        只物化这些目录名；None = 该 market 下全部有 meta.json 的子目录。
    panel_loader:
        CLI 注入的数据装配；需要物化却未注入 → ``ValueError``。

    Returns
    -------
    dict
        materialized / skipped / errors(list[str]) / total
    """
    if not STORE_FACTOR_PARQUET_ENABLED:
        _LOG.warning("factor_store parquet 物化已关闭（STORE_FACTOR_PARQUET_ENABLED=False）")
        return {
            "materialized": 0,
            "skipped": 0,
            "errors": ["store_parquet_disabled"],
            "total": 0,
        }

    from factorzen.discovery.factor_library import (
        is_python_identity,
        python_identity,
    )
    from factorzen.discovery.lift_test import _materializer_from_prepped
    from factorzen.discovery.preparation import expressions_need_intraday

    market_root = Path(root) / market
    if names is None:
        if not market_root.is_dir():
            return {"materialized": 0, "skipped": 0, "errors": [], "total": 0}
        asset_names = sorted(
            p.name
            for p in market_root.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )
    else:
        asset_names = [str(n).strip() for n in names if str(n).strip()]

    stats: dict[str, Any] = {
        "materialized": 0,
        "skipped": 0,
        "errors": [],
        "total": len(asset_names),
    }

    # name -> (asset_dir, meta, expression)
    to_do: list[tuple[str, Path, dict[str, Any], str]] = []
    for name in asset_names:
        d = asset_dir(market, name, root=root)
        try:
            if not d.is_dir():
                stats["errors"].append(name)
                continue
            meta = _read_json(d / "meta.json")
            if meta is None:
                stats["errors"].append(name)
                continue
            pq = d / "factor.parquet"
            if _asset_materialization_fresh(meta, pq):
                stats["skipped"] += 1
                continue
            kind = meta.get("kind") or "expression"
            expr = meta.get("expression")
            if kind == "python" and (not expr or not is_python_identity(str(expr))):
                expr = python_identity(name)
            if not expr:
                stats["errors"].append(name)
                continue
            to_do.append((name, d, meta, str(expr)))
        except Exception as exc:
            _LOG.warning(
                "factor_store materialize_assets 扫描 %s 失败: %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
            stats["errors"].append(name)

    if not to_do:
        return stats

    if panel_loader is None:
        raise ValueError(
            "materialize_assets 需要 panel_loader（由 CLI 层注入；discovery 层不拉数）"
        )

    start = STORE_MATERIALIZE_START
    end = store_materialize_end()
    univ = STORE_MATERIALIZE_UNIVERSE
    start8 = _to_ymd8(start)
    end8 = _to_ymd8(end)

    # 按 need_intraday 分组装帧（与 _materialize_records 同策略）
    groups: dict[bool, list[tuple[str, Path, dict[str, Any], str]]] = {}
    for item in to_do:
        name, _d, _meta, expr = item
        need_ix = False
        if not is_python_identity(expr):
            try:
                need_ix = bool(expressions_need_intraday([expr]))
            except Exception:
                need_ix = False
        groups.setdefault(need_ix, []).append(item)

    for need_ix, items in groups.items():
        try:
            prepped = panel_loader(
                start=start8,
                end=end8,
                universe=univ,
                market=market,
                intraday_leaves=need_ix,
            )
        except Exception as exc:
            _LOG.warning(
                "factor_store materialize_assets: load panel failed %s–%s u=%s ix=%s: %s: %s",
                start8,
                end8,
                univ,
                need_ix,
                type(exc).__name__,
                exc,
            )
            for name, *_rest in items:
                stats["errors"].append(name)
            continue

        mat = _materializer_from_prepped(
            prepped,
            leaf_map=None,
            python_universe=univ,
            python_market=market,
        )
        for name, d, _meta, expr in items:
            try:
                panel = mat(expr)
                if panel is None or (hasattr(panel, "is_empty") and panel.is_empty()):
                    _LOG.warning("factor_store materialize_assets: empty for %s", name)
                    stats["errors"].append(name)
                    continue
                _apply_asset_materialization(d, panel, expression=expr)
                stats["materialized"] += 1
                _LOG.info(
                    "factor_store materialize_assets: %s n_rows=%s",
                    name,
                    panel.height,
                )
            except Exception as exc:
                _LOG.warning(
                    "factor_store materialize_assets: %s failed: %s: %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
                stats["errors"].append(name)
            panel = None  # type: ignore[assignment]
        del prepped, mat

    return stats


def _materialize_records(
    records: list[FactorRecord],
    *,
    market: str,
    root: str,
    default_universe: str = STORE_MATERIALIZE_UNIVERSE,
    panel_loader: PanelLoader | None = None,
) -> int:
    """串行物化 active/probation 记录（生产通道 _materializer_from_prepped）。

    物化窗口/universe 固定为 store 常量口径
    （``STORE_MATERIALIZE_START`` ~ ``store_materialize_end()`` ×
    ``STORE_MATERIALIZE_UNIVERSE``），与 record.eval_* 无关。
    统一口径后仅按 ``need_intraday`` 分组装帧，组内逐条物化，防 OOM。
    返回成功物化条数。

    ``default_universe`` 保留 API 兼容，装帧实际一律用
    ``STORE_MATERIALIZE_UNIVERSE``（忽略入参）。

    ``panel_loader``：数据帧加载器，由调用方（CLI 层）注入——discovery 不拉数，
    避免 discovery→cli 反向依赖（架构守卫 agents→discovery→cli 环）。
    需要物化却未注入 → ``ValueError``。
    """
    del default_universe  # 固定口径；保留形参以免破坏调用方 kwargs
    from factorzen.discovery.lift_test import _materializer_from_prepped
    from factorzen.discovery.preparation import expressions_need_intraday

    targets = [r for r in records if (r.status or "") in MATERIALIZE_STATUSES and r.expression]
    if not targets:
        return 0
    if panel_loader is None:
        raise ValueError(
            "materialize 需要 panel_loader（由 CLI 层注入生产数据装配；discovery 层不拉数）"
        )

    start = STORE_MATERIALIZE_START
    end = store_materialize_end()
    univ = STORE_MATERIALIZE_UNIVERSE
    start8 = _to_ymd8(start)
    end8 = _to_ymd8(end)

    # 统一口径 → 仅按 need_intraday 分组
    groups: dict[bool, list[FactorRecord]] = {}
    for r in targets:
        need_ix = False
        if not _is_python_record(r) and r.expression:
            try:
                need_ix = bool(expressions_need_intraday([r.expression]))
            except Exception:
                need_ix = False
        groups.setdefault(need_ix, []).append(r)

    n_ok = 0
    for need_ix, recs in groups.items():
        try:
            prepped = panel_loader(
                start=start8,
                end=end8,
                universe=univ,
                market=market,
                intraday_leaves=need_ix,
            )
        except Exception as exc:
            _LOG.warning(
                "factor_store: load panel failed %s–%s u=%s ix=%s: %s: %s",
                start8,
                end8,
                univ,
                need_ix,
                type(exc).__name__,
                exc,
            )
            continue
        mat = _materializer_from_prepped(
            prepped,
            leaf_map=None,
            python_universe=univ,
            python_market=market,
        )
        for r in recs:
            name = record_asset_name(r)
            try:
                panel = mat(r.expression)
                if panel is None or (hasattr(panel, "is_empty") and panel.is_empty()):
                    _LOG.warning("factor_store: materialize empty for %s", name)
                    # 仍刷新 meta/py
                    write_factor_asset(r, market=market, root=root, materialize=False)
                    continue
                write_factor_asset(
                    r,
                    market=market,
                    root=root,
                    materialize=True,
                    panel=panel,
                )
                n_ok += 1
                _LOG.info(
                    "factor_store: materialized %s n_rows=%s",
                    name,
                    panel.height,
                )
            except Exception as exc:
                _LOG.warning(
                    "factor_store: materialize %s failed: %s: %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
            # 释放单因子面板
            panel = None  # type: ignore[assignment]
        del prepped, mat
    return n_ok


class PanelLoader(Protocol):
    """数据帧加载器契约（CLI 层实现并注入；discovery 层不拉数）。"""

    def __call__(
        self,
        *,
        start: str,
        end: str,
        universe: str,
        market: str,
        intraday_leaves: bool = False,
    ) -> pl.DataFrame: ...


def sync_store(
    market: str,
    root: str = DEFAULT_ROOT,
    *,
    only: Iterable[str] | None = None,
    materialize: bool = True,
    lib_root: str = DEFAULT_LIB_ROOT,
    default_universe: str = STORE_MATERIALIZE_UNIVERSE,
    panel_loader: PanelLoader | None = None,
) -> dict[str, Any]:
    """全库同步：逐条写 meta+py；按需物化 parquet。

    物化口径固定 ``STORE_MATERIALIZE_*``（all_a / 2016-01-01 ~ 最新已完结
    交易日），与 jsonl 的 eval_start/eval_end/universe 分离。

    增量：若 meta.expression 一致，且 materialization 的 start/universe 与
    store 常量一致、end 已覆盖 ``store_materialize_end()``，且 parquet 存在
    → 跳过物化；否则重物化。

    Returns
    -------
    dict
        written / materialized / skipped_materialize / errors / total
    """
    records = load_library(market, root=lib_root)
    only_set = set(only) if only is not None else None
    if only_set is not None:
        records = [
            r
            for r in records
            if record_asset_name(r) in only_set
            or (r.expression or "") in only_set
            or (r.name or "") in only_set
        ]

    stats: dict[str, Any] = {
        "written": 0,
        "materialized": 0,
        "skipped_materialize": 0,
        "errors": [],
        "total": len(records),
    }
    to_materialize: list[FactorRecord] = []

    for rec in records:
        try:
            name = record_asset_name(rec)
            d = asset_dir(market, name, root=root)
            prev_meta = _read_json(d / "meta.json")
            # 写 meta + py（始终刷新 ledger 快照）
            # 若即将物化则先不写 mat；若跳过则保留 mat
            will_try_mat = materialize and (rec.status or "") in MATERIALIZE_STATUSES
            fresh = will_try_mat and _materialization_fresh(rec, prev_meta, d / "factor.parquet")
            if will_try_mat and not fresh:
                # 先写 meta/py（mat=null 或保留），再批物化
                write_factor_asset(rec, market=market, root=root, materialize=False)
                to_materialize.append(rec)
            else:
                write_factor_asset(rec, market=market, root=root, materialize=False)
                # 刷新 meta 但保留 materialization
                if fresh and prev_meta and prev_meta.get("materialization"):
                    meta = build_meta(
                        rec,
                        market=market,
                        materialization=prev_meta["materialization"],
                    )
                    _write_json(d / "meta.json", meta)
                    stats["skipped_materialize"] += 1
                elif not will_try_mat:
                    pass
            stats["written"] += 1
        except Exception as exc:
            stats["errors"].append(
                {
                    "name": getattr(rec, "name", None),
                    "expression": getattr(rec, "expression", None),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if to_materialize:
        n = _materialize_records(
            to_materialize,
            market=market,
            root=root,
            default_universe=default_universe,
            panel_loader=panel_loader,
        )
        stats["materialized"] = n

    return stats


def verify_store(
    market: str,
    root: str = DEFAULT_ROOT,
    *,
    lib_root: str = DEFAULT_LIB_ROOT,
) -> dict[str, Any]:
    """一致性校验：meta.expression vs jsonl.expression（全部比对）。

    Returns
    -------
    dict
        ok / drifts / missing_in_store / missing_in_ledger / n_checked
    """
    records = load_library(market, root=lib_root)
    by_name: dict[str, FactorRecord] = {}
    for r in records:
        try:
            by_name[record_asset_name(r)] = r
        except ValueError:
            continue

    drifts: list[dict[str, Any]] = []
    missing_in_store: list[str] = []
    store_names: set[str] = set()
    market_root = Path(root) / market
    if market_root.is_dir():
        for child in market_root.iterdir():
            if child.is_dir() and (child / "meta.json").exists():
                store_names.add(child.name)

    for name, rec in by_name.items():
        meta_path = asset_dir(market, name, root=root) / "meta.json"
        if not meta_path.exists():
            missing_in_store.append(name)
            continue
        meta = _read_json(meta_path)
        if meta is None:
            drifts.append(
                {
                    "name": name,
                    "field": "meta.json",
                    "store": None,
                    "ledger": rec.expression,
                    "error": "unreadable meta.json",
                }
            )
            continue
        store_expr = meta.get("expression")
        ledger_expr = rec.expression
        if store_expr != ledger_expr:
            drifts.append(
                {
                    "name": name,
                    "field": "expression",
                    "store": store_expr,
                    "ledger": ledger_expr,
                }
            )

        # 物化口径漂移（active/probation 且已有 materialization/parquet）
        # 旧 csi300/csi500 资产 → 期望判漂移，下次 sync 重物化为 all_a。
        mat = meta.get("materialization")
        pq_path = asset_dir(market, name, root=root) / "factor.parquet"
        if (
            (rec.status or "") in MATERIALIZE_STATUSES
            and isinstance(mat, dict)
            and (pq_path.exists() or bool(mat.get("n_rows")))
        ):
            target_end = store_materialize_end()
            # 超集覆盖（start 更早）不算漂移，与 _materialization_window_fresh 同口径
            mat_start = mat.get("start")
            if not mat_start or _date_key(mat_start) > _date_key(STORE_MATERIALIZE_START):
                drifts.append(
                    {
                        "name": name,
                        "field": "materialization.start",
                        "store": mat.get("start"),
                        "ledger": STORE_MATERIALIZE_START,
                    }
                )
            if mat.get("universe") != STORE_MATERIALIZE_UNIVERSE:
                drifts.append(
                    {
                        "name": name,
                        "field": "materialization.universe",
                        "store": mat.get("universe"),
                        "ledger": STORE_MATERIALIZE_UNIVERSE,
                    }
                )
            if _date_key(mat.get("end")) < _date_key(target_end):
                drifts.append(
                    {
                        "name": name,
                        "field": "materialization.end",
                        "store": mat.get("end"),
                        "ledger": target_end,
                    }
                )

    missing_in_ledger = sorted(store_names - set(by_name.keys()))
    return {
        "ok": not drifts and not missing_in_store,
        "drifts": drifts,
        "missing_in_store": missing_in_store,
        "missing_in_ledger": missing_in_ledger,
        "n_checked": len(by_name),
    }


def sync_records_after_upsert(
    records: Iterable[FactorRecord],
    *,
    market: str,
    lib_root: str = DEFAULT_LIB_ROOT,
    materialize: bool = False,
) -> None:
    """入库路径钩子：写 meta+py，默认不物化（不拖慢 upsert）。

    失败只 warning，绝不抛（护库主路径）。
    """
    root = store_root_for_library(lib_root)
    for rec in records:
        try:
            write_factor_asset(rec, market=market, root=root, materialize=materialize)
        except Exception as exc:
            _LOG.warning(
                "factor_store sync after upsert failed name=%s: %s: %s",
                getattr(rec, "name", None),
                type(exc).__name__,
                exc,
            )


# ── python 因子发现（factor_store 单路径）──────────────────────────────────────


_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def discover_python_factor_files(
    store_root: str = DEFAULT_ROOT,
    *,
    market: str = "ashare",
) -> list[Path]:
    """扫描 ``{store_root}/{market}/*/factor.py``（kind 由 meta 或文件存在判定）。"""
    base = Path(store_root) / market
    if not base.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        py = child / "factor.py"
        if not py.is_file():
            continue
        meta = _read_json(child / "meta.json")
        if meta is not None and meta.get("kind") not in (None, "python"):
            # expression 生成的 factor.py 不是 DailyFactor，跳过注册
            continue
        # 无 meta、kind 缺省、或 kind=python → 候选
        out.append(py)
    return out


def load_python_factor_module(path: Path, *, mod_name: str | None = None):
    """从路径动态 import factor 模块。"""
    import sys

    name = mod_name or f"factorzen_store_{path.parent.name}"
    if not _SAFE_NAME.match(name.replace(".", "_")):
        name = f"factorzen_store_{abs(hash(str(path))) % 10**8}"
    # 去点
    name = name.replace(".", "_").replace("-", "_")
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def register_python_factors_from_store(
    store_root: str = DEFAULT_ROOT,
    *,
    market: str = "ashare",
    override: bool = False,
) -> int:
    """把 factor_store 下 python factor.py 注册进 daily registry。返回新注册数。"""
    from factorzen.daily.factors.base import DailyFactor
    from factorzen.daily.factors.registry import _registry

    n_ok = 0
    for path in discover_python_factor_files(store_root, market=market):
        try:
            mod = load_python_factor_module(path)
        except Exception as exc:
            _LOG.warning("import store factor %s failed: %s", path, exc)
            continue
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, DailyFactor)
                and attr is not DailyFactor
                and _registry.register(attr, override=override)
            ):
                n_ok += 1
    return n_ok
