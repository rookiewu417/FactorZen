# src/factorzen/agents/experiment_index.py
"""跨 session 长期记忆：experiment_index.jsonl 读写 + 归一化查重 + 已知有效/无效。

**按数据窗口分族**：一个窗口上「已验证有效」的因子，换个窗口未必成立。若 `recall()` 从整个
index 无差别召回，即便统计上按窗口分族，LLM 也已经在拿跨窗口的提示——信息流的族必须与
统计族对齐。族边界 = `(start, end, universe, market)`：PIT 数据对固定窗口不可变
（`get_universe(end, ...)` 取期末快照），同元组 = 同数据 = 同族。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl  # POSIX 文件锁（Linux 优先；Windows 无此模块时降级为无锁追加）
except ImportError:  # pragma: no cover - 仅非 POSIX 平台
    fcntl = None  # type: ignore[assignment]

from factorzen.discovery.expression import is_lookahead_expr, parse_expr, to_expr_string

_LOG = logging.getLogger(__name__)

# Critic 否决了「方向」的裁决 → 该因子不再作为「可借鉴的已验证有效方向」喂给后续假设生成。
# revise_expr 不在此列：方向对、只是表达式需改，思路仍值得借鉴。
_VETOED_VERDICTS = frozenset({"drop", "revise_hypothesis"})

_warned_legacy_records = False


def _normalize(expr: str) -> str:
    try:
        return to_expr_string(parse_expr(expr))
    except ValueError:
        return expr


def build_lift_reject_record(
    *,
    expression: str,
    data_window: dict | None,
    lift: float | None,
    lift_se: float | None,
    lift_reason: str,
    source: str,
    ic_train: float | None = None,
    residual_ic_train: float | None = None,
    baseline_rank_ic: float | None = None,
    admission_start: str | None = None,
    admission_end: str | None = None,
    ts: str | None = None,
) -> dict:
    """构造 ``lift_rejected`` 附加事件记录（session 钩子 / CLI --apply 共用）。

    只写 reject；active/probation 走既有入库，不经此通道。
    ``data_window`` 必须用**来源 session 的窗口**，以便同族 recall 命中。
    """
    from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_REJECTED

    return {
        "expression": expression,
        "data_window": data_window,
        "reject_category": REJECT_CATEGORY_LIFT_REJECTED,
        "passed": False,
        "compile_ok": True,
        "ic_train": ic_train,
        "residual_ic_train": residual_ic_train,
        "lift": lift,
        "lift_se": lift_se,
        "lift_reason": lift_reason,
        "baseline_rank_ic": baseline_rank_ic,
        "admission_start": admission_start,
        "admission_end": admission_end,
        "source": source,
        "ts": ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }


def window_key(window: dict | None) -> str | None:
    """数据窗口指纹。None 表示「不限定窗口」。"""
    if not window:
        return None
    return "|".join(str(window.get(k)) for k in ("start", "end", "universe", "market"))


class ExperimentIndex:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def append(self, records: list[dict]) -> None:
        # team workers / 并行 session 会并发写同一 jsonl；无锁多次 write 会交错、
        # 产出损坏行。整批组装成单个 payload + POSIX 独占锁一次写入，保证行原子、不交错。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        if not payload:
            return
        with self.path.open("a") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(payload)
                f.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _last_wins(recs: list[dict]) -> list[dict]:
        """同一（归一化）表达式只保留**最新**那条记录。

        index 是 append-only 的**事件日志**，一个表达式的当前状态 = 它最后一次被记录的状态。
        Librarian 每轮写入时 `passed` 取当轮护栏结论，而收尾复核（`node_finalize_guardrails`）
        会用最终 N 把早轮候选降级并补写更正记录。不做后写覆盖的话，旧的 `passed=True`
        与新的 `passed=False` 会同时命中 `known_valid`/`known_invalid`，前者继续把已被
        否掉的因子当「已验证有效」喂给后续 session。

        无 `expression` 字段的记录原样保留（不参与去重）。
        """
        latest: dict[str, dict] = {}
        passthrough: list[dict] = []
        for r in recs:
            expr = r.get("expression")
            if expr is None:
                passthrough.append(r)
            else:
                latest[_normalize(expr)] = r
        return passthrough + list(latest.values())

    def _scoped(self, data_window: dict | None) -> list[dict]:
        """按数据窗口过滤记录 + 同表达式后写覆盖。`data_window=None` → 不过滤（向后兼容）。

        无 `data_window` 字段的老记录不知道来自哪个窗口，过滤时**保守排除**并告警一次
        ——静默丢弃历史会让 LLM 的负例/正例库莫名其妙地变空。

        覆盖发生在**窗口过滤之后**：族边界优先于时间顺序，另一个窗口上的结论不该
        覆盖本窗口（同一表达式在不同数据窗口上本就可能一个有效一个无效）。
        """
        key = window_key(data_window)
        recs = self.load()
        if key is None:
            return self._last_wins(recs)
        global _warned_legacy_records
        kept, legacy = [], 0
        for r in recs:
            rk = window_key(r.get("data_window"))
            if rk is None:
                legacy += 1
            elif rk == key:
                kept.append(r)
        if legacy and not _warned_legacy_records:
            _warned_legacy_records = True
            _LOG.warning(
                "experiment_index 有 %d 条记录缺 data_window 字段（早于按窗口分族的版本），"
                "按窗口查询时已保守排除。它们仍可通过不带 data_window 的查询看到。", legacy
            )
        return self._last_wins(kept)

    def seen_expressions(self, *, data_window: dict | None = None) -> set[str]:
        return {_normalize(r["expression"]) for r in self._scoped(data_window)
                if "expression" in r}

    def known_invalid(self, k: int = 5, *, data_window: dict | None = None) -> list[str]:
        """「已验证无效」= 没过定量护栏。按 |IC| 升序（最没用的优先）喂给 LLM 作负例。

        注意判据是 `not passed` 这个**事实**——被去相关剔除、或被 Critic 否决的因子
        `passed` 仍为 True，它们不是「无效因子」，不该混进负例污染 LLM 的认知。

        排除**编译失败**的记录：它们 `ic_train=None` → 排序键 0.0 → 会占满 top-k，
        把有信息的「能编译但 IC 低」负例全挤出去。语法坑的价值在 `seen_expressions()`
        的跨 session 去重，不在负例库。

        排除 **holdout 覆盖失败**（``reject_category=holdout_coverage``）：那是缺数据，
        不是方向性证据，回灌会把 LLM 推向「北向思路无效」的错误结论。

        排除 **库内高相关**（``reject_category=library_correlated``）：IC 未必低，是「重复
        方向」非「无效」；混进负例会误导「这方向没信号」。

        排除 **灰区 / lift 队列**（``reject_category=gray_zone|lift_queue``）：
        单因子弱但待组合 lift 裁决，不是「已验证无效」——混进负例会误杀试用通道方向。

        排除 **组合层 lift 拒绝**（``reject_category=lift_rejected``）：
        组合层无增量 ≠ 单因子无信号，混入负例会误导 LLM 认为方向没信号；
        lift 拒绝走 ``known_lift_rejects`` 独立通道。

        排除 **弱 IC**（``reject_category=ic_too_weak`` 或 reason 含「太弱」）：
        该窗强度不足 ≠ 方向已验证无效；回灌会把事件式提案压成「方向无效」死循环。

        排除 **无 IC 的行**（``ic_train is None``：预热不足 / duplicate_fingerprint 等
        评估未出值）：零方向信息，排序键 abs(None or 0)=0 会挤占 top-k——与排除
        编译失败同理；它们的价值在 ``seen_expressions()`` 去重，不在负例库。
        """
        from factorzen.discovery.guardrails import (
            REJECT_CATEGORY_GRAY_ZONE,
            REJECT_CATEGORY_HOLDOUT_COVERAGE,
            REJECT_CATEGORY_IC_TOO_WEAK,
            REJECT_CATEGORY_LIBRARY_CORRELATED,
            REJECT_CATEGORY_LIFT_QUEUE,
            REJECT_CATEGORY_LIFT_REJECTED,
        )

        def _is_coverage_fail(r: dict) -> bool:
            if r.get("reject_category") == REJECT_CATEGORY_HOLDOUT_COVERAGE:
                return True
            rr = r.get("reject_reason") or ""
            return "覆盖不足" in rr

        def _is_library_corr(r: dict) -> bool:
            return r.get("reject_category") == REJECT_CATEGORY_LIBRARY_CORRELATED

        def _is_lift_queue_or_gray(r: dict) -> bool:
            cat = r.get("reject_category")
            return cat in (REJECT_CATEGORY_GRAY_ZONE, REJECT_CATEGORY_LIFT_QUEUE)

        def _is_lift_rejected(r: dict) -> bool:
            return r.get("reject_category") == REJECT_CATEGORY_LIFT_REJECTED

        def _is_ic_too_weak(r: dict) -> bool:
            if r.get("reject_category") == REJECT_CATEGORY_IC_TOO_WEAK:
                return True
            rr = r.get("reject_reason") or ""
            if "太弱" not in rr:
                return False
            # 同条 reason 若含反号/无信号 → 方向证据，不按弱 IC 排除
            return not ("反号" in rr or "无信号" in rr)

        recs = [r for r in self._scoped(data_window)
                if not r.get("passed", False) and r.get("compile_ok", True)
                and r.get("ic_train") is not None
                and not is_lookahead_expr(r.get("expression") or "")
                and not _is_coverage_fail(r)
                and not _is_library_corr(r)
                and not _is_lift_queue_or_gray(r)
                and not _is_lift_rejected(r)
                and not _is_ic_too_weak(r)]
        recs.sort(key=lambda r: abs(r.get("ic_train") or 0.0))  # 最没用的优先
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]

    def known_lift_rejects(
        self, k: int = 5, *, data_window: dict | None = None,
    ) -> list[dict]:
        """组合层 lift 拒绝记录（``reject_category=lift_rejected``）。

        走 ``_scoped`` 同语义（窗口分族 + last-wins）；按 ``ts`` 降序取前 k
        （缺 ``ts`` 按文件序末尾优先）。返回
        ``[{"expression", "lift", "lift_reason"}]``。
        """
        from factorzen.discovery.guardrails import REJECT_CATEGORY_LIFT_REJECTED

        recs = [
            r for r in self._scoped(data_window)
            if r.get("reject_category") == REJECT_CATEGORY_LIFT_REJECTED
            and r.get("expression")
        ]
        # 带索引：缺 ts 时文件序末尾优先（_scoped/_last_wins 后序 = 末次写入序）
        indexed = list(enumerate(recs))

        def _key(item: tuple[int, dict]) -> tuple:
            i, r = item
            ts = r.get("ts")
            if ts is None or ts == "":
                # 无 ts：用索引当次序，reverse 后末尾优先
                return (0, "", i)
            return (1, str(ts), i)

        indexed.sort(key=_key, reverse=True)
        out: list[dict] = []
        for _, r in indexed[:k]:
            out.append({
                "expression": r["expression"],
                "lift": r.get("lift"),
                "lift_reason": r.get("lift_reason"),
            })
        return out

    def known_valid(self, k: int = 5, *, data_window: dict | None = None) -> list[str]:
        """「可供借鉴」是一个**决策**，由事实（passed）与两类否决共同推出，此处集中判定。

        - `passed`：过了定量护栏（不可变事实，见 `AttemptRecord.passed_guardrails`）
        - `verdict not in {drop, revise_hypothesis}`：Critic 未否决这个**方向**
          （`revise_expr` = 方向对、表达式需改 → 思路仍值得借鉴，保留）
        - `not decorrelated`：未因与已有候选高度相关而被剔除（重复的思路无需再借鉴）

        排序按 **|holdout_ic|** 降序：护栏明确接纳负 IC 反转因子
        （`guardrail_passed` 的 `same_sign` + `ci_high<0` 分支），带符号排序会把最强的
        反转因子挤到末尾、被 top-k 截断，系统性把 LLM 的借鉴方向偏离反转因子族。
        """
        recs = [
            r for r in self._scoped(data_window)
            if r.get("passed", False)
            and r.get("verdict") not in _VETOED_VERDICTS
            and not r.get("decorrelated", False)
            # 前视因子（负窗口，历史误记 passed）绝不当「已验证有效」喂回 LLM——否则引导它
            # 继续生成前视。parse 层已根治新生成，此处堵历史产物回灌的口子。
            and not is_lookahead_expr(r.get("expression") or "")
        ]
        recs.sort(key=lambda r: abs(r.get("holdout_ic") or 0.0), reverse=True)
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]

    def leaf_stats(
        self,
        leaf_names: list[str],
        data_window: dict | None = None,
    ) -> dict[str, dict]:
        """按叶子聚合历史尝试（词边界匹配，流式读文件）。

        对每个 ``leaf`` 返回：
        - ``n_exprs``：含该 leaf 的唯一表达式数（``compile_ok=False`` 不计）
        - ``n_passed``：其中 ``passed=True`` 的数量
        - ``best_abs_ic``：其中最大 ``|ic_train|``（None 记 0）
        - ``n_coverage_fail``：``reject_category == holdout_coverage`` 的数量
          （缺数据，不算方向失败；挖穿判定用 ``n_exprs - n_coverage_fail``）

        统计口径与 ``known_invalid`` 一致：走窗口分族 + 同表达式后写覆盖。
        匹配用 ``\\b<leaf>\\b``，避免 ``roe`` 误命中 ``grossprofit_margin`` 等子串。
        """
        from factorzen.discovery.guardrails import REJECT_CATEGORY_HOLDOUT_COVERAGE

        empty = {
            "n_exprs": 0,
            "n_passed": 0,
            "best_abs_ic": 0.0,
            "n_coverage_fail": 0,
        }
        if not leaf_names:
            return {}
        # leaf 名都是合法标识符；escape 防意外元字符
        patterns = {
            name: re.compile(rf"\b{re.escape(name)}\b") for name in leaf_names
        }
        key = window_key(data_window)
        # 流式：只保留同表达式最新记录（与 _last_wins 同语义），不全量 list 进内存。
        latest: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open() as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    expr = r.get("expression")
                    if expr is None:
                        continue
                    if key is not None:
                        rk = window_key(r.get("data_window"))
                        if rk is None or rk != key:
                            continue
                    latest[_normalize(expr)] = r

        out: dict[str, dict] = {name: dict(empty) for name in leaf_names}
        for norm_expr, r in latest.items():
            if not r.get("compile_ok", True):
                continue
            # 用原始 expression 匹配叶子（与落盘一致）；归一化串也可，叶名标识符不变。
            text = r.get("expression") or norm_expr
            is_cov = r.get("reject_category") == REJECT_CATEGORY_HOLDOUT_COVERAGE
            if not is_cov:
                rr = r.get("reject_reason") or ""
                is_cov = "覆盖不足" in rr
            passed = bool(r.get("passed", False))
            abs_ic = abs(r.get("ic_train") or 0.0)
            for name, pat in patterns.items():
                if not pat.search(text):
                    continue
                st = out[name]
                st["n_exprs"] += 1
                if passed:
                    st["n_passed"] += 1
                if abs_ic > st["best_abs_ic"]:
                    st["best_abs_ic"] = abs_ic
                if is_cov:
                    st["n_coverage_fail"] += 1
        return out
