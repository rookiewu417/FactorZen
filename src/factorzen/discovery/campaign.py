# src/factorzen/discovery/campaign.py
"""campaign trial family：跨 session 的 DSR 多重检验 N 累计。

同一「搜索问题」定义（market/universe/start/end/holdout_ratio/objective/horizon/gate）
下的多个 team session 共享一个 family；`campaign_prior` 从 ExperimentIndex jsonl
重建历史唯一表达式 IR 池，供 `node_finalize_guardrails` 与本 session 池做表达式级 union。

git_sha / LLM 模型 / membership_hash 是 provenance，不进 key——代码小改或缓存更新
不应重置 family（诚实取舍；manifest 另行记录 provenance）。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


def _canon(v):
    """字段规范化：None 统一；字符串 strip，空串 → None；其余原样。"""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return v


def campaign_key(
    *,
    market,
    universe,
    start,
    end,
    holdout_ratio,
    objective,
    horizon,
    gate,
) -> str:
    """统计问题定义的规范化 hash（sha256 前 16 hex）。

    进 key 的是「同一搜索问题」的定义；git_sha / LLM 模型 / membership_hash
    是 provenance 不进 key（代码小改/缓存更新不应重置 family——诚实取舍，
    manifest 另行记录）。字段规范化：None 统一、字符串 strip、JSON 排序键。
    """
    payload = {
        "end": _canon(end),
        "gate": _canon(gate),
        "holdout_ratio": _canon(holdout_ratio),
        "horizon": _canon(horizon),
        "market": _canon(market),
        "objective": _canon(objective),
        "start": _canon(start),
        "universe": _canon(universe),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class CampaignPrior:
    campaign_id: str
    n_trials: int  # 历史唯一表达式数（去重后）
    expressions: set[str]  # 历史唯一表达式集合（供本 session 去重）
    irs: list[float]  # 每唯一表达式一个带符号 IR（首次评估值）
    n_sessions: int  # 覆盖的历史 run_id 数
    source_path: str


def campaign_prior(
    index_path,
    *,
    market,
    universe,
    start,
    end,
    exclude_run_ids: set[str] | None = None,
    campaign_id: str | None = None,
) -> CampaignPrior | None:
    """从 ExperimentIndex jsonl 重建同 campaign 的历史 trial 池。

    过滤分两支：
    - ``campaign_id`` 非 None：按行顶层 ``campaign_id`` 精确匹配（已编码
      market/universe/start/end/holdout/objective/horizon/gate）。缺字段的
      legacy 行保守排除（与 index 对缺 data_window 老行一致）。
    - ``campaign_id`` 为 None（legacy / M1）：按 data_window 的
      start/end/universe/market 过滤；返回的 campaign_id 用全-None 算。

    两支共通：compile_ok 为真、ir_train 可转 float 且有限；exclude_run_ids
    里的 run_id 跳过（session 末排除本 run，防双计）。按 expression 去重保首行。
    文件不存在/空 → None。损坏行跳过不崩。
    """
    path = Path(index_path)
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None

    want_cid = _canon(campaign_id) if campaign_id is not None else None
    want_start = _canon(start)
    want_end = _canon(end)
    want_universe = _canon(universe)
    want_market = _canon(market)
    exclude = exclude_run_ids or set()
    strict = want_cid is not None

    # 按 expression 去重保首行；同时收集 run_id
    first_ir: dict[str, float] = {}
    order: list[str] = []
    run_ids: set[str] = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue

        rid = r.get("run_id")
        if rid is not None and rid in exclude:
            continue

        if strict:
            # 精确分族：缺 campaign_id 的旧行保守排除
            row_cid = _canon(r.get("campaign_id"))
            if row_cid is None or row_cid != want_cid:
                continue
        else:
            dw = r.get("data_window")
            if not isinstance(dw, dict):
                continue
            if _canon(dw.get("start")) != want_start:
                continue
            if _canon(dw.get("end")) != want_end:
                continue
            if _canon(dw.get("universe")) != want_universe:
                continue
            if _canon(dw.get("market")) != want_market:
                continue

        if not r.get("compile_ok", True):
            continue

        expr = r.get("expression")
        if not expr or not isinstance(expr, str):
            continue

        try:
            ir = float(r["ir_train"])
        except (TypeError, ValueError, KeyError):
            continue
        if not math.isfinite(ir):
            continue

        if expr in first_ir:
            continue
        first_ir[expr] = ir
        order.append(expr)
        if rid is not None:
            run_ids.add(str(rid))

    if strict:
        # 与 manifest 写入的完整-key campaign_id 一致（可审计重建 basis）
        assert want_cid is not None
        cid: str = want_cid
    else:
        cid = campaign_key(
            market=market, universe=universe, start=start, end=end,
            holdout_ratio=None, objective=None, horizon=None, gate=None,
        )

    if not order:
        # 有文件但无匹配 trial：返回空 prior（调用方仍可写 campaign_id；
        # finalize 侧 prior.n_trials=0 与 prior=None 对 basis 等价）
        return CampaignPrior(
            campaign_id=cid,
            n_trials=0,
            expressions=set(),
            irs=[],
            n_sessions=0,
            source_path=str(path),
        )

    irs = [first_ir[e] for e in order]
    return CampaignPrior(
        campaign_id=cid,
        n_trials=len(order),
        expressions=set(order),
        irs=irs,
        n_sessions=len(run_ids),
        source_path=str(path),
    )


__all__ = ["CampaignPrior", "campaign_key", "campaign_prior"]
