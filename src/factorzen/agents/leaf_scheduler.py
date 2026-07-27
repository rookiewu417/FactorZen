# src/factorzen/agents/leaf_scheduler.py
"""叶子级 UCB 调度：把「探索/利用」从提示词文案变成**采样决策**。

现状的问题：`leaf_stats` / `library_crowded` 算得很细，但只以自然语言注入 prompt
（`format_leaf_guidance` / `format_library_crowded`）——LLM 看了可听可不听，事后既无法
归因也无法度量。RD-Agent(Q) 的实测结论是调度层本身值钱：bandit 选方向 IC 0.0532 /
ARR 14.21%，LLM 自选 0.0476 / 10.09%，随机 0.0445 / 8.97%（+11.8% IC，不换模型）。

本模块只做一件事：给定历史统计，**确定性**地选出本轮该重点开采的 K 个叶子。
选择结果由调用方作为明确指令注入 Hypothesis prompt，并落 rounds_log 供复现与归因。
"""
from __future__ import annotations

import math

# 探索系数默认值。UCB1 的理论常数是 sqrt(2)≈1.414；这里取 1.4 同量级，
# 语义是「一个从没试过的叶子，探索项约等于 |IC| 0.06~0.09 的利用项」——刻意让冷启动
# 压过中等强度的老叶（挖掘的瓶颈是覆盖面，不是把已知好叶再榨一遍）。
DEFAULT_UCB_C = 1.4
# 每条库内 active 因子对该叶的降权。0.15 的标度参照探索项：库里已有 ~10 条的叶
# 会被扣掉约 1.5，足以把冷启动优势抹平——「已充分开采」不该再当探索目标。
DEFAULT_CROWD_PENALTY = 0.15
DEFAULT_FOCUS_K = 3
# 本 session 每被选为焦点一次，该叶的「利用+探索」乘一次这个因子（0.7 → 三轮后剩 34%）。
# 定这个量级的依据：UCB 探索项 sqrt(ln N/(1+n)) 只随**跨 session 累计**的尝试数衰减，
# 单 session 内 leaf_stats 每叶增量很小（一轮往往只 +1~2 条表达式），分数排序几十轮不动
# ——20 轮 A/B 里 vwap 被选 16 轮、log_vol 14 轮、high 13 轮，"探索" 退化成锁定同 3 叶连挖。
# 0.7 让一个领跑叶大约 3~4 轮后被后排追上（0.7^4≈0.24），既能轮换又不至于一轮就丢掉
# 真正强的方向；`pick_decay=1.0` 即关闭衰减，回到纯 UCB。
DEFAULT_PICK_DECAY = 0.7


def _direction_tries(st: dict | None) -> int:
    """该叶的**方向尝试**数 = ``n_exprs - n_coverage_fail``。

    与 `librarian.raw_exhausted_leaves` / `build_leaf_guidance` 的挖穿口径**同源**：
    holdout 覆盖不足是缺数据、不是方向失败，不该计入「这个方向试过多少次」。
    口径漂移会让「已经试到枯竭」和「还值得探索」两套判断互相打架。
    """
    if not st:
        return 0
    n_exprs = int(st.get("n_exprs") or 0)
    n_cov = int(st.get("n_coverage_fail") or 0)
    return max(0, n_exprs - n_cov)


def ucb_focus_leaves(
    stats: dict[str, dict],
    leaf_names: list[str],
    *,
    exhausted: set[str] | None = None,
    crowded: dict[str, int] | None = None,
    k: int = DEFAULT_FOCUS_K,
    c: float = DEFAULT_UCB_C,
    crowd_penalty: float = DEFAULT_CROWD_PENALTY,
    session_picks: dict[str, int] | None = None,
    pick_decay: float = DEFAULT_PICK_DECAY,
) -> list[tuple[str, float]]:
    """按 UCB 分数选出本轮的 K 个焦点叶子，返回 ``[(leaf, score), ...]`` 降序。

    分数 = (利用 + 探索) × 本 session 选择衰减 - 拥挤惩罚::

        score_i = ( best_abs_ic_i
                  + c * sqrt(ln(1 + N_total) / (1 + n_i)) ) * pick_decay ** picks_i
                - crowd_penalty * crowded[i]

    - **利用项** ``best_abs_ic``：该叶历史最好的 |IC|（`leaf_stats` 产出，缺省 0）。
    - **探索项** ``n_i`` = 该叶方向尝试数（见 `_direction_tries`），``N_total`` =
      **存活叶**（已排除 ``exhausted``）方向尝试之和。排除项不进分母也不进 N_total：
      它们已经出局，让它们抬高别人的探索项等于用死叶给活叶发奖金。
    - **拥挤惩罚** ``crowded``：叶 → 库内 active 因子数（`library_covered_by_family` 产出）。
      库里已经很多的方向，再挖出来的东西大概率过不了 lift 增量闸。``crowded`` 里
      不在 ``leaf_names`` 的键被忽略。

    - **本 session 选择衰减** ``session_picks``：叶 → 本 session 已被选为焦点的**轮数**，
      每一轮乘一次 ``pick_decay``。``None`` / 空 dict → 不衰减（旧调用者逐值零回归）；
      ``pick_decay=1.0`` 同样等于关闭。不在 ``leaf_names`` 的键被忽略，负数按 0 处理。

    ``exhausted``：挖穿叶，**硬排除**。必须与 `filter_exhausted_expressions` 用的同一份
    ——被调度器选中、又被硬闸过滤掉是自相矛盾，会让整轮空转。

    **为什么需要 session 内衰减**（参照 AlphaPROBE 的 retrieval penalty ``(1-ω)^k``：
    同一条被检索 k 次的记忆，其权重按 ``(1-ω)^k`` 递减，防止检索塌缩到少数几条）：
    UCB 的探索项只随**跨 session 累计**的 ``n_i`` 衰减，而 ``leaf_stats`` 在单 session 内
    每叶增量很小（一轮往往只 +1~2 条表达式），``sqrt(ln N/(1+n))`` 的变化被 ``ln`` 压平，
    分数排序几十轮基本不动 —— 实测 20 轮 session 里 vwap 被选 16 轮、log_vol 14 轮、
    high 13 轮，"探索" 退化成锁定同 3 叶连挖。按本 session 已选轮数额外乘一个衰减因子，
    几轮后领跑叶被后排追上，自动轮换到下一批方向。

    **衰减不作用于拥挤惩罚**（这是个反直觉但要命的取舍）：惩罚是**负项**，若把整个分数
    （含惩罚）一起乘 ``pick_decay``，重复选择反而会把惩罚一起缩小 —— 拥挤叶越挖越便宜，
    与惩罚项的语义正好反了。惩罚衡量的是「库里已有多少同族因子」，这是 session 外的客观
    事实，不该随本 session 挑了几次而变。因此衰减只乘正项（利用+探索），惩罚原样减。

    **设计取舍**：衰减是**乘性**而非加性（加性罚需要与 IC/探索项同量纲，一旦 ``c`` 或
    ``crowd_penalty`` 改了标度就得跟着重调；乘性对标度免疫）；且**不进 leaf_stats**
    ——它是本 session 的调度状态，不是该叶的历史事实，落库会污染跨 session 的 UCB 统计。
    计数由调用方（`run_team_agent` 的 session 级 dict）持有并逐轮透传，本函数保持纯函数。

    **冷启动天然优先是设计意图**：``n_i = 0`` 时探索项取到最大值
    ``c * sqrt(ln(1 + N_total))``，从没试过的叶子必然排在前面。挖掘的瓶颈是覆盖面，
    冷启动叶正是唯一还有未知信息的方向；`best_abs_ic` 只在尝试数拉平后才起区分作用。

    **确定性**：无随机源；分数并列按叶名字典序 tie-break。同参两次调用逐值相同——
    调度决策要落 rounds_log/manifest，不可复现的选择等于没记。

    ``k <= 0`` 或无存活叶 → 空列表（调用方据此不注入，零回归）。
    """
    if k <= 0 or not leaf_names:
        return []
    exhausted = exhausted or set()
    crowded = crowded or {}
    session_picks = session_picks or {}

    survivors = [name for name in leaf_names if name not in exhausted]
    if not survivors:
        return []

    tries = {name: _direction_tries(stats.get(name)) for name in survivors}
    n_total = sum(tries.values())
    # ln(1 + N)：N=0 时为 0，全新 session 里所有叶探索项同为 0、退化成纯字典序，
    # 而非 log(0) 崩掉。
    log_term = math.log(1.0 + n_total)

    scored: list[tuple[str, float]] = []
    for name in survivors:
        st = stats.get(name)
        best = float((st or {}).get("best_abs_ic") or 0.0)
        explore = c * math.sqrt(log_term / (1 + tries[name]))
        penalty = crowd_penalty * float(crowded.get(name, 0) or 0)
        picks = max(0, int(session_picks.get(name, 0) or 0))
        # picks=0 → 因子恰为 1.0，乘法是精确的 → 无 session_picks 时逐值零回归。
        # 衰减**只乘正项**：penalty 是负项，一起乘会让重复选择反而减轻拥挤惩罚（见 docstring）。
        scored.append((name, (best + explore) * pick_decay**picks - penalty))

    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored[:k]
