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
from pathlib import Path

try:
    import fcntl  # POSIX 文件锁（Linux 优先；Windows 无此模块时降级为无锁追加）
except ImportError:  # pragma: no cover - 仅非 POSIX 平台
    fcntl = None  # type: ignore[assignment]

from factorzen.discovery.expression import parse_expr, to_expr_string

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
        """
        recs = [r for r in self._scoped(data_window)
                if not r.get("passed", False) and r.get("compile_ok", True)]
        recs.sort(key=lambda r: abs(r.get("ic_train") or 0.0))  # 最没用的优先
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]

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
        ]
        recs.sort(key=lambda r: abs(r.get("holdout_ic") or 0.0), reverse=True)
        return [_normalize(r["expression"]) for r in recs[:k] if "expression" in r]
