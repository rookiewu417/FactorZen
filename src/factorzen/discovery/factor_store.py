"""因子资产库三件套：meta.json + factor.py + factor.parquet。

**架构裁决**（用户拍板）：
- ``workspace/factor_library/{market}.jsonl`` 仍是裁决唯一真相
  （status / lift / admission / forward）。
- 本模块是资产库载体：把库内记录物化为可读、可 import、可复现的磁盘资产。
- 入库/rebuild 写入单点同步两处；``verify_store`` 校验一致。

目录布局（定死）::

    workspace/factor_store/<market>/<name>/
    ├── meta.json
    ├── factor.py
    └── factor.parquet   # 仅 active/probation；correlated 省算力不写
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import warnings
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from factorzen.config.settings import FACTOR_LIBRARY_DIR, WORKSPACE_DIR
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

DEFAULT_ROOT = str(WORKSPACE_DIR / "factor_store")
MATERIALIZE_STATUSES = frozenset({"active", "probation"})
_HYPOTHESIS_MAX = 200

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
    """把 factor_library 根目录映射到 factor_store 根。

    - 生产默认库根 → ``workspace/factor_store``
    - 测试/自定义库根 → ``{lib_root}/factor_store``（与 jsonl 同树，隔离）
    """
    if lib_root is None:
        return DEFAULT_ROOT
    p = Path(lib_root).resolve()
    if p == Path(DEFAULT_LIB_ROOT).resolve() or p == Path(FACTOR_LIBRARY_DIR).resolve():
        return DEFAULT_ROOT
    return str(p / "factor_store")


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
    description = _truncate(rec.hypothesis) if kind == "expression" else (
        rec.hypothesis or ""
    )
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
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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

    # materialization：仅 active/probation 且 materialize=True 时写 parquet
    mat_info: dict[str, Any] | None = None
    pq_path = d / "factor.parquet"
    prev_meta = _read_json(d / "meta.json") or {}
    prev_mat = prev_meta.get("materialization")
    should_mat = materialize and (record.status or "") in MATERIALIZE_STATUSES

    if should_mat and panel is not None:
        out = panel
        needed = {"trade_date", "ts_code", "factor_value"}
        if not needed.issubset(set(out.columns)):
            raise ValueError(
                f"panel must have columns {needed}, got {out.columns}"
            )
        out = out.select(["trade_date", "ts_code", "factor_value"])
        out.write_parquet(pq_path)
        mat_info = {
            "start": record.eval_start,
            "end": record.eval_end,
            "universe": record.universe or "csi300",
            "git_sha": _git_sha(),
            "n_rows": int(out.height),
            "generated_at": _utc_now_iso(),
            "expression": record.expression,
        }
    elif (record.status or "") not in MATERIALIZE_STATUSES:
        # correlated / 其他：规格要求 materialization=null（不强制删 parquet）
        mat_info = None
    elif isinstance(prev_mat, dict) and pq_path.exists():
        # 刷新 meta/py 时：expression + 窗口仍一致则保留既有物化 provenance
        if (
            prev_meta.get("expression") == record.expression
            and prev_mat.get("start") == record.eval_start
            and prev_mat.get("end") == record.eval_end
            and prev_mat.get("universe") == (record.universe or "csi300")
        ):
            mat_info = prev_mat
        else:
            mat_info = None
    else:
        mat_info = None

    meta = build_meta(record, market=market, materialization=mat_info)
    _write_json(d / "meta.json", meta)
    return str(d)


def _materialization_fresh(
    rec: FactorRecord,
    meta: dict[str, Any] | None,
    pq_path: Path,
) -> bool:
    """meta.expression 与 parquet materialization 与当前记录一致 → 可跳过物化。"""
    if meta is None or not pq_path.exists():
        return False
    mat = meta.get("materialization")
    if not isinstance(mat, dict):
        return False
    if meta.get("expression") != rec.expression:
        return False
    if mat.get("expression") not in (None, rec.expression) and mat.get(
        "expression"
    ) != rec.expression:
        return False
    if mat.get("start") != rec.eval_start or mat.get("end") != rec.eval_end:
        return False
    univ = rec.universe or "csi300"
    if mat.get("universe") != univ:
        return False
    return bool(mat.get("n_rows"))


def _materialize_records(
    records: list[FactorRecord],
    *,
    market: str,
    root: str,
    default_universe: str = "csi300",
) -> int:
    """串行物化 active/probation 记录（生产通道 _materializer_from_prepped）。

    按 (eval_start, eval_end, universe, need_intraday) 分组装帧，组内逐条物化，
    防 OOM。返回成功物化条数。
    """
    from factorzen.discovery.lift_test import _materializer_from_prepped
    from factorzen.discovery.preparation import expressions_need_intraday

    targets = [
        r
        for r in records
        if (r.status or "") in MATERIALIZE_STATUSES and r.expression
    ]
    if not targets:
        return 0

    # 分组键
    groups: dict[tuple[str, str, str, bool], list[FactorRecord]] = {}
    for r in targets:
        start = r.eval_start or ""
        end = r.eval_end or ""
        univ = r.universe or default_universe
        need_ix = False
        if not _is_python_record(r) and r.expression:
            try:
                need_ix = bool(expressions_need_intraday([r.expression]))
            except Exception:
                need_ix = False
        key = (start, end, univ, need_ix)
        groups.setdefault(key, []).append(r)

    n_ok = 0
    for (start, end, univ, need_ix), recs in groups.items():
        if not start or not end:
            _LOG.warning(
                "factor_store: skip materialize (missing eval window) names=%s",
                [record_asset_name(r) for r in recs],
            )
            continue
        try:
            prepped = _load_prepped_panel(
                start=start,
                end=end,
                universe=univ,
                market=market,
                intraday_leaves=need_ix,
            )
        except Exception as exc:
            _LOG.warning(
                "factor_store: load panel failed %s–%s u=%s ix=%s: %s: %s",
                start,
                end,
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
                    write_factor_asset(
                        r, market=market, root=root, materialize=False
                    )
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


def _load_prepped_panel(
    *,
    start: str,
    end: str,
    universe: str,
    market: str,
    intraday_leaves: bool = False,
    intraday_freq: str = "5min",
) -> pl.DataFrame:
    """经生产通道拉挖掘帧并 preprocess（与 library_exec_audit 同源）。"""
    import argparse

    from factorzen.cli.main import _prepare_agent_mining_data
    from factorzen.discovery.evaluation import _preprocess_daily

    args = argparse.Namespace(
        market=market,
        start=start,
        end=end,
        universe=universe,
        horizon=1,
        intraday_leaves=bool(intraday_leaves),
        intraday_freq=intraday_freq or "5min",
        intraday_expr_leaves=None,
        top_n=50,
        symbols=None,
        freq="daily",
    )
    daily, profile, _prep_meta = _prepare_agent_mining_data(args)
    if daily is None or daily.is_empty():
        raise RuntimeError(
            f"挖掘帧为空 market={market} {start}–{end} universe={universe}"
        )
    prepped = _preprocess_daily(daily, profile).sort(["ts_code", "trade_date"])
    float_cols = [
        c
        for c, dt in zip(prepped.columns, prepped.dtypes, strict=False)
        if dt in (pl.Float32, pl.Float64)
    ]
    if float_cols:
        prepped = prepped.with_columns(
            [pl.col(c).fill_nan(None) for c in float_cols]
        )
    return prepped


def sync_store(
    market: str,
    root: str = DEFAULT_ROOT,
    *,
    only: Iterable[str] | None = None,
    materialize: bool = True,
    lib_root: str = DEFAULT_LIB_ROOT,
    default_universe: str = "csi300",
) -> dict[str, Any]:
    """全库同步：逐条写 meta+py；按需物化 parquet。

    增量：若 meta.expression 与 materialization 窗口/universe 与当前记录一致
    且 parquet 存在 → 跳过物化。

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
            will_try_mat = (
                materialize
                and (rec.status or "") in MATERIALIZE_STATUSES
            )
            fresh = will_try_mat and _materialization_fresh(
                rec, prev_meta, d / "factor.parquet"
            )
            if will_try_mat and not fresh:
                # 先写 meta/py（mat=null 或保留），再批物化
                write_factor_asset(
                    rec, market=market, root=root, materialize=False
                )
                to_materialize.append(rec)
            else:
                write_factor_asset(
                    rec, market=market, root=root, materialize=False
                )
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
            write_factor_asset(
                rec, market=market, root=root, materialize=materialize
            )
        except Exception as exc:
            _LOG.warning(
                "factor_store sync after upsert failed name=%s: %s: %s",
                getattr(rec, "name", None),
                type(exc).__name__,
                exc,
            )


# ── python 因子发现（factor_store + 旧 workspace/factors 兼容）────────────────


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


def warn_legacy_workspace_factors() -> None:
    """若 ``workspace/factors/daily`` 仍有手写 .py，打 DeprecationWarning。"""
    legacy = WORKSPACE_DIR / "factors" / "daily"
    if not legacy.is_dir():
        return
    py_files = [
        p
        for p in legacy.glob("*.py")
        if p.name not in ("__init__.py",) and not p.name.startswith("TEMPLATE")
    ]
    if not py_files:
        return
    names = ", ".join(p.stem for p in py_files[:8])
    more = f" (+{len(py_files) - 8} more)" if len(py_files) > 8 else ""
    warnings.warn(
        f"workspace/factors/daily is deprecated; migrate python factors to "
        f"workspace/factor_store/<market>/<name>/factor.py. Found: {names}{more}",
        DeprecationWarning,
        stacklevel=2,
    )
