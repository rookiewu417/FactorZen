"""probation → active 的 paper forward 确认机制（记录器 + 裁决器）。

确认窗口用**向前推进的真实时间**（paper forward），天然不可窥视、不可回灌。
本模块交付机制本身；ops/runner 的每日 STAGE 接线为后续工作。

PIT 口径（铁律）：``ic(t) = spearman(factor(t-1 截面), ret(t))``——
因子值只用 as_of **前一交易日**及更早信息（t-1 收盘可得），收益为
as_of 日 close(t-1)→close(t)（复权价优先 close_adj）。与「t 日算 → t+1 执行」对齐。
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from factorzen.core.stats import spearman_avg_rank
from factorzen.discovery.factor_library import DEFAULT_ROOT, load_library, render_markdown
from factorzen.discovery.lift_test import paired_lift_stats

_LOG = logging.getLogger(__name__)

# 预热默认：复用 agent 路 AGENT_WARMUP_LOOKBACK（504 交易日）。
# 理由：库内因子大量来自 LLM/team 路径，窗口无搜索空间上界（可嵌套 ~250 日）；
# search_space_max_lookback(=180) 只覆盖随机搜索 windows≤60，对库内长窗因子欠预热。
# 记录器只评单日截面，装配小窗 + 长前缀即可。
def _default_lookback() -> int:
    from factorzen.pipelines.factor_mine import AGENT_WARMUP_LOOKBACK

    return int(AGENT_WARMUP_LOOKBACK)


def forward_track_path(market: str, root: str = DEFAULT_ROOT) -> Path:
    return Path(root) / "forward_track" / f"{market}.jsonl"


def _date_str(v: Any) -> str:
    """trade_date / as_of → 可比较的 YYYYMMDD 字符串。"""
    if v is None:
        return ""
    if isinstance(v, date) and not isinstance(v, datetime):
        return v.strftime("%Y%m%d")
    if isinstance(v, datetime):
        return v.strftime("%Y%m%d")
    s = str(v).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:4] + s[5:7] + s[8:10]
    if len(s) >= 8 and s[:8].isdigit():
        return s[:8]
    return s.replace("-", "")[:8]


def _updated_at_key(v: Any) -> str:
    """updated_at（常为 YYYY-MM-DD）→ YYYYMMDD，便于与 forward date 比较。"""
    return _date_str(v)


def _load_forward_rows(market: str, root: str) -> list[dict]:
    path = forward_track_path(market, root)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append_forward_rows(market: str, root: str, rows: list[dict]) -> None:
    path = forward_track_path(market, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _existing_keys(rows: list[dict]) -> set[tuple[str, str, str | None]]:
    """幂等键 ``(date, expression, universe)``；legacy 无 universe → None。"""
    keys: set[tuple[str, str, str | None]] = set()
    for r in rows:
        d = _date_str(r.get("date"))
        expr = r.get("expression")
        if d and expr:
            uni = r.get("universe")  # missing → None，与新写入口径对齐
            keys.add((d, str(expr), uni if uni is None else str(uni)))
    return keys


def _assemble_daily(
    market: str, as_of: str, lookback_days: int, universe: str | None = None,
) -> pl.DataFrame:
    """生产装配：小窗 [as_of, as_of] + lookback 预热前缀。

    仅 ashare 接入 ``prepare_mining_daily``；非 A 股 fail closed（勿用 A 股数据
    求值非 A 股因子）。测试应注入 daily，跳过装配。

    ``universe``：forward IC 的截面必须与因子**准入时的 universe 一致**——
    csi300 准入的因子在全 A 截面上算 forward IC 是另一个统计量，不能用于裁决
    （首跑实测 n_stocks=5511 暴露此漂移）。
    """
    if market != "ashare":
        raise ValueError(
            f"forward-track 暂未接入 {market} 的 profile/provider/leaf-map；"
            f"非 A 股入口 fail closed，勿用 A 股数据求值非 A 股因子。"
        )
    from factorzen.pipelines.factor_mine import prepare_mining_daily

    # start=end=as_of：评分只覆盖确认日；FactorDataContext 再往前拉 lookback 交易日预热，
    # 从而 t-1 落在帧内（因子滞后截面 + 收益分母）。
    return prepare_mining_daily(as_of, as_of, universe=universe,
                                lookback_days=lookback_days)


def _effective_universe(rec: Any, force_universe: str | None) -> str | None:
    """显式 ``universe`` 入参覆盖 record.universe；否则取准入口径。"""
    if force_universe is not None:
        return force_universe
    uni = getattr(rec, "universe", None)
    return uni if uni is None else str(uni)


def _preprocess(daily: pl.DataFrame, leaf_map: dict[str, str] | None) -> pl.DataFrame:
    """与 build_library_pool / lift 同款：先 ``_preprocess_daily`` 再物化。"""
    from factorzen.agents.evaluation import _preprocess_daily

    # leaf_map 非 None 时通常已有市场派生列；A 股 profile=None 走默认 prep。
    # 本记录器不持有 profile，统一走 A 股 prep（测试注入帧同样受益）。
    _ = leaf_map
    return _preprocess_daily(daily).sort(["ts_code", "trade_date"])


def _materialize_panel(
    expr: str,
    prepped: pl.DataFrame,
    leaf_map: dict[str, str] | None,
) -> pl.DataFrame | None:
    """复用 factor_library 物化路径（evaluate_materialized + 面板装配），禁止内联新实现。"""
    from factorzen.discovery.expression import evaluate_materialized, parse_expr

    try:
        node = parse_expr(expr, leaf_map)
        series = evaluate_materialized(node, prepped, leaf_map)
        panel = (
            prepped.select(["trade_date", "ts_code"])
            .with_columns(series.alias("factor_value"))
            .filter(pl.col("factor_value").is_not_null() & pl.col("factor_value").is_finite())
        )
        if panel.is_empty():
            return None
        return panel
    except Exception as exc:
        _LOG.debug("forward materialize %r: %s: %s", expr, type(exc).__name__, exc)
        return None


def _trading_dates_sorted(prepped: pl.DataFrame) -> list[str]:
    dates = prepped["trade_date"].unique().to_list()
    return sorted(_date_str(d) for d in dates if d is not None)


def _prev_trade_date(dates: list[str], as_of: str) -> str | None:
    """as_of 之前最近一个交易日；as_of 不在序列中时取严格小于 as_of 的最大日。"""
    as_of_s = _date_str(as_of)
    prev = [d for d in dates if d < as_of_s]
    return prev[-1] if prev else None


def _ret_on_as_of(
    prepped: pl.DataFrame, as_of: str, prev: str,
) -> pl.DataFrame:
    """as_of 日收益 close(t-1)→close(t)；复权价优先 close_adj。

    返回 ``[ts_code, ret]``。
    """
    price_col = "close_adj" if "close_adj" in prepped.columns else "close"
    # 统一比较用字符串键
    df = prepped.select(
        pl.col("trade_date"),
        pl.col("ts_code"),
        pl.col(price_col).alias("px"),
    ).with_columns(
        pl.col("trade_date").map_elements(_date_str, return_dtype=pl.Utf8).alias("dstr")
    )
    p_prev = (
        df.filter(pl.col("dstr") == prev)
        .select(["ts_code", pl.col("px").alias("px_prev")])
    )
    p_asof = (
        df.filter(pl.col("dstr") == _date_str(as_of))
        .select(["ts_code", pl.col("px").alias("px_asof")])
    )
    joined = p_asof.join(p_prev, on="ts_code", how="inner")
    return (
        joined.with_columns(
            (pl.col("px_asof") / pl.col("px_prev") - 1.0).alias("ret")
        )
        .select(["ts_code", "ret"])
        .filter(pl.col("ret").is_not_null() & pl.col("ret").is_finite())
    )


def _ic_as_of(
    factor_panel: pl.DataFrame,
    ret_df: pl.DataFrame,
    prev: str,
) -> tuple[float | None, int]:
    """spearman(factor(t-1), ret(t))；退化 → (None, n)。"""
    f = factor_panel.with_columns(
        pl.col("trade_date").map_elements(_date_str, return_dtype=pl.Utf8).alias("dstr")
    ).filter(pl.col("dstr") == prev)
    if f.is_empty() or ret_df.is_empty():
        return None, 0
    m = f.select(["ts_code", "factor_value"]).join(ret_df, on="ts_code", how="inner")
    if m.height < 2:
        return None, int(m.height)
    fa = m["factor_value"].to_numpy().astype(float)
    ra = m["ret"].to_numpy().astype(float)
    mask = np.isfinite(fa) & np.isfinite(ra)
    fa, ra = fa[mask], ra[mask]
    n = int(fa.size)
    if n < 2:
        return None, n
    return spearman_avg_rank(fa, ra), n


def _eval_forward_on_frame(
    to_eval: list,
    prepped: pl.DataFrame,
    as_of_s: str,
    leaf_map: dict[str, str] | None,
    *,
    force_universe: str | None,
) -> tuple[list[dict], int]:
    """在已预处理的单帧上评估一组因子，返回 (rows, failed_count)。

    注入 daily 路径与生产分组路径共用；每行写入各自 ``universe``。
    """
    dates = _trading_dates_sorted(prepped)
    prev = _prev_trade_date(dates, as_of_s)
    new_rows: list[dict] = []
    failed = 0
    if prev is None:
        _LOG.warning("forward_track: as_of=%s 无前序交易日，本组记 failed", as_of_s)
        for r in to_eval:
            uni = _effective_universe(r, force_universe)
            new_rows.append({
                "date": as_of_s,
                "expression": r.expression,
                "ic": None,
                "n_stocks": 0,
                "status_at_record": r.status,
                "universe": uni,
            })
        return new_rows, len(new_rows)

    ret_df = _ret_on_as_of(prepped, as_of_s, prev)
    for r in to_eval:
        uni = _effective_universe(r, force_universe)
        panel = _materialize_panel(r.expression, prepped, leaf_map)
        if panel is None:
            ic, n_stocks = None, 0
            failed += 1
        else:
            ic, n_stocks = _ic_as_of(panel, ret_df, prev)
            if ic is None:
                failed += 1
        new_rows.append({
            "date": as_of_s,
            "expression": r.expression,
            "ic": float(ic) if ic is not None and math.isfinite(ic) else None,
            "n_stocks": int(n_stocks),
            "status_at_record": r.status,
            "universe": uni,
        })
    return new_rows, failed


def record_forward_ics(
    market: str,
    as_of: str,
    *,
    root: str = DEFAULT_ROOT,
    statuses: tuple[str, ...] = ("probation", "active"),
    daily: pl.DataFrame | None = None,
    leaf_map: dict[str, str] | None = None,
    lookback_days: int | None = None,
    universe: str | None = None,
) -> dict:
    """记录 as_of 日库内因子的 paper forward RankIC（PIT 口径）。

    落盘 ``{root}/forward_track/{market}.jsonl``。
    幂等：同 (date, expression, universe) 已存在 → 跳过不重写。
    返回 ``{recorded, skipped_existing, failed}``。

    ``universe``：显式截面口径（CLI ``--universe`` escape hatch）。非 None 时
    **强制所有记录**用该 universe 装配/记账（覆盖准入口径，打 warning）。
    ``None``（默认）→ 生产路径按各 ``record.universe`` 分组分别装配；注入
    ``daily`` 时保持单帧单组（调用方保证口径）。
    """
    from collections import defaultdict

    as_of_s = _date_str(as_of)
    lb = int(lookback_days) if lookback_days is not None else _default_lookback()
    force_uni = universe  # 显式入参；None = 按 record.universe

    recs = [r for r in load_library(market, root=root) if r.status in statuses]
    existing = _existing_keys(_load_forward_rows(market, root))

    if force_uni is not None and daily is None:
        uniques = {getattr(r, "universe", None) for r in recs}
        if any(u != force_uni for u in uniques):
            _LOG.warning(
                "forward_track: 显式 universe=%r 覆盖各因子准入口径 %s",
                force_uni, sorted(str(u) for u in uniques),
            )

    to_eval: list = []
    skipped = 0
    for r in recs:
        uni = _effective_universe(r, force_uni)
        key = (as_of_s, r.expression, uni)
        if key in existing:
            skipped += 1
            continue
        to_eval.append(r)

    if not to_eval:
        return {"recorded": 0, "skipped_existing": skipped, "failed": 0}

    new_rows: list[dict] = []
    failed = 0

    if daily is not None:
        # 测试/注入路径：单帧单组，不按 universe 分组
        prepped = _preprocess(daily, leaf_map)
        rows, n_fail = _eval_forward_on_frame(
            to_eval, prepped, as_of_s, leaf_map, force_universe=force_uni,
        )
        new_rows.extend(rows)
        failed += n_fail
    else:
        # 生产路径：按有效 universe 分组，每组独立装配截面
        groups: dict[str | None, list] = defaultdict(list)
        for r in to_eval:
            groups[_effective_universe(r, force_uni)].append(r)
        for grp_uni, group_recs in groups.items():
            daily_g = _assemble_daily(market, as_of_s, lb, universe=grp_uni)
            prepped = _preprocess(daily_g, leaf_map)
            rows, n_fail = _eval_forward_on_frame(
                group_recs, prepped, as_of_s, leaf_map, force_universe=force_uni,
            )
            new_rows.extend(rows)
            failed += n_fail

    _append_forward_rows(market, root, new_rows)
    return {
        "recorded": len(new_rows),
        "skipped_existing": skipped,
        "failed": failed,
    }


def _sign_from_ic_train(ic_train: float | None) -> float | None:
    if ic_train is None:
        return None
    try:
        v = float(ic_train)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v == 0.0:
        return None
    return 1.0 if v > 0 else -1.0


def _ics_after_updated(
    forward_rows: list[dict],
    expr: str,
    updated_at: str | None,
    universe: str | None = None,
) -> list[tuple[str, float]]:
    """该因子 updated_at（进入 probation）之后的 (date, ic) 序列；丢弃 ic=None。

    updated_at 缺失 → 用全部（旧记录无进入时刻，宁多勿漏；真实确认应写 updated_at）。
    过滤条件：date > updated_at（严格晚于进入日，进入当日不计入）。

    ``universe``：只取同口径截面 IC。legacy 无 universe 的 row 视为 None，
    仅匹配 ``universe is None`` 的记录——避免混用错误截面 IC。
    """
    cutoff = _updated_at_key(updated_at) if updated_at else None
    # 规范化：与 ledger 写入的 str|None 对齐
    want_uni = universe if universe is None else str(universe)
    out: list[tuple[str, float]] = []
    for r in forward_rows:
        if r.get("expression") != expr:
            continue
        row_uni = r.get("universe")
        row_uni_n = row_uni if row_uni is None else str(row_uni)
        if row_uni_n != want_uni:
            continue
        ic = r.get("ic")
        if ic is None:
            continue
        try:
            ic_f = float(ic)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ic_f):
            continue
        d = _date_str(r.get("date"))
        if cutoff and not (d > cutoff):
            continue
        out.append((d, ic_f))
    out.sort(key=lambda x: x[0])
    return out


def _load_library_raw(market: str, root: str) -> list[dict]:
    """原始 jsonl 行（保留未知字段，如 forward_confirmed_at）。"""
    path = Path(root) / f"{market}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _save_library_raw(market: str, root: str, rows: list[dict]) -> None:
    path = Path(root) / f"{market}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def forward_review(
    market: str,
    *,
    root: str = DEFAULT_ROOT,
    min_days: int = 60,
    se_mult: float = 1.645,
    block_days: int = 20,
    apply: bool = False,
) -> list[dict]:
    """裁决 probation 因子的 paper forward 证据。

    - 只处理 status==\"probation\"（single/active/no_lift/correlated 不动）。
    - adj_ic = ic * sign(方向)；方向优先 ``admission_ic``（单因子 admission 窗 RankIC），
      兜底 ``ic_train``；缺失/0 符号 → hold + reason=missing_sign。
    - 块 SE 复用 ``paired_lift_stats``（adj_ic 当 cand，零序列当 base），禁止重写块 SE。
    - apply=True：promote→active(+forward_confirmed_at/forward_n_days)；demote→no_lift。
    """
    lib = load_library(market, root=root)
    probation = [r for r in lib if r.status == "probation"]
    fwd = _load_forward_rows(market, root)
    today = date.today().isoformat()

    results: list[dict] = []
    apply_map: dict[str, dict] = {}  # expression → patch fields

    for rec in probation:
        expr = rec.expression
        # admission_ic 优先（lift 轨权威方向）；ic_train 兜底（single 轨/旧行）
        sign = _sign_from_ic_train(
            rec.admission_ic if rec.admission_ic is not None else rec.ic_train
        )
        series = _ics_after_updated(fwd, expr, rec.updated_at, rec.universe)

        base_row: dict[str, Any] = {
            "expression": expr,
            "decision": "hold",
            "n_days": len(series),
            "mean": None,
            "se": None,
            "ci_low": None,
            "reason": None,
        }

        if sign is None:
            base_row["reason"] = "missing_sign"
            results.append(base_row)
            continue

        if len(series) < int(min_days):
            base_row["reason"] = "insufficient_days"
            results.append(base_row)
            continue

        adj = [ic * sign for _, ic in series]
        dates = [d for d, _ in series]
        cand_daily = pl.DataFrame(
            {"trade_date": dates, "ic": adj},
            schema={"trade_date": pl.Utf8, "ic": pl.Float64},
        )
        base_daily = pl.DataFrame(
            {"trade_date": dates, "ic": [0.0] * len(dates)},
            schema={"trade_date": pl.Utf8, "ic": pl.Float64},
        )
        stats = paired_lift_stats(cand_daily, base_daily, block_days=block_days)
        mean = stats.get("lift")
        se = stats.get("lift_se")
        n_blocks = int(stats.get("n_blocks") or 0)
        n_days = int(stats.get("n_days") or len(series))

        base_row["n_days"] = n_days
        base_row["mean"] = mean
        base_row["se"] = se
        if mean is not None and se is not None and math.isfinite(se):
            base_row["ci_low"] = float(mean) - float(se_mult) * float(se)
        elif mean is not None:
            base_row["ci_low"] = None

        decision = "hold"
        reason: str | None = None
        if (
            mean is not None
            and se is not None
            and math.isfinite(float(se))
            and n_blocks >= 2
            and float(mean) - float(se_mult) * float(se) > 0
        ):
            decision = "promote"
        elif (
            mean is not None
            and se is not None
            and math.isfinite(float(se))
            and float(mean) + float(se_mult) * float(se) < 0
        ):
            decision = "demote"
        else:
            reason = "inconclusive"

        base_row["decision"] = decision
        base_row["reason"] = reason
        results.append(base_row)

        if apply and decision in ("promote", "demote"):
            if decision == "promote":
                apply_map[expr] = {
                    "status": "active",
                    "forward_confirmed_at": today,
                    "forward_n_days": n_days,
                    "updated_at": today,
                }
            else:
                apply_map[expr] = {
                    "status": "no_lift",
                    "updated_at": today,
                }

    if apply and apply_map:
        raw = _load_library_raw(market, root)
        new_rows: list[dict] = []
        for row in raw:
            row_expr = row.get("expression")
            # 只改 status==probation 且在 apply_map 中的行；其它（active/correlated/no_lift）原样
            if (
                row_expr in apply_map
                and row.get("status") == "probation"
            ):
                patched = dict(row)
                patched.update(apply_map[row_expr])
                new_rows.append(patched)
            else:
                new_rows.append(row)
        _save_library_raw(market, root, new_rows)
        render_markdown(market, root=root)

    return results
