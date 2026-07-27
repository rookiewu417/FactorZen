"""叶子 UCB 调度器：纯函数判别用例 + team 轨接线（含 leaf_scheduler=False 零回归）。

期望分数是**离线独立手算**后写死的字面量（见 assert 上方注释的算式），不在测试里
复用被测实现的公式——否则断言恒真，改错了公式也照样绿。
"""
from __future__ import annotations

import json

import pytest

from factorzen.agents.leaf_scheduler import ucb_focus_leaves
from factorzen.llm.prompt_fragments import format_focus_leaves

# ── 共用 stats 夹具 ───────────────────────────────────────────────────────────
# n_i = n_exprs - n_coverage_fail（与 librarian 挖穿口径同源）
#   alpha_leaf: n_i = 10 - 0 = 10, best|IC| = 0.06   老叶、试得多、信号还行
#   beta_leaf : n_i =  4 - 2 =  2, best|IC| = 0.01   中间态；2 条 coverage 失败不算方向尝试
#   gamma_leaf: n_i =  0 - 0 =  0, best|IC| = 0.0    冷启动，从没试过
# N_total = 10 + 2 + 0 = 12，ln(1+12) = ln(13) = 2.5649493574615367
_STATS = {
    "alpha_leaf": {"n_exprs": 10, "n_passed": 1, "best_abs_ic": 0.06, "n_coverage_fail": 0},
    "beta_leaf": {"n_exprs": 4, "n_passed": 0, "best_abs_ic": 0.01, "n_coverage_fail": 2},
    "gamma_leaf": {"n_exprs": 0, "n_passed": 0, "best_abs_ic": 0.0, "n_coverage_fail": 0},
}
_LEAVES = ["alpha_leaf", "beta_leaf", "gamma_leaf"]


def test_ucb_scores_match_hand_computation():
    """手算判别：三叶分数逐个对字面量，且排序 = 冷启动 > 中间 > 老叶。

    c=1.4, crowd_penalty 不生效（无 crowded）：
      alpha = 0.06 + 1.4*sqrt(ln13/11) = 0.06 + 0.6760379723482867
      beta  = 0.01 + 1.4*sqrt(ln13/3)  = 0.01 + 1.2945141611977589
      gamma = 0.00 + 1.4*sqrt(ln13/1)  = 2.242164298311926
    """
    out = ucb_focus_leaves(_STATS, _LEAVES, k=3)

    assert [name for name, _s in out] == ["gamma_leaf", "beta_leaf", "alpha_leaf"]
    got = dict(out)
    assert got["alpha_leaf"] == pytest.approx(0.7360379723482866, abs=1e-12)
    assert got["beta_leaf"] == pytest.approx(1.3045141611977589, abs=1e-12)
    assert got["gamma_leaf"] == pytest.approx(2.242164298311926, abs=1e-12)


def test_ucb_top_k_truncates_by_score():
    """k 截断取分数最高的前 k 个（不是输入序前 k 个）。"""
    out = ucb_focus_leaves(_STATS, _LEAVES, k=2)
    assert [n for n, _s in out] == ["gamma_leaf", "beta_leaf"]
    assert len(ucb_focus_leaves(_STATS, _LEAVES, k=1)) == 1
    # k 超过可选叶数 → 全给，不补空
    assert len(ucb_focus_leaves(_STATS, _LEAVES, k=99)) == 3
    # k<=0 → 空（不是「全给」）
    assert ucb_focus_leaves(_STATS, _LEAVES, k=0) == []


def test_ucb_cold_start_leaf_wins_over_proven_leaf():
    """n_i=0 的叶子探索项最大 → 冷启动天然排第一，即便 best|IC| 是 0。

    判别点：alpha_leaf 的 best|IC|=0.06 远高于 gamma_leaf 的 0.0，
    若只按利用项排序 alpha 会第一；UCB 必须把 gamma 顶上去。
    """
    out = ucb_focus_leaves(_STATS, _LEAVES, k=1)
    assert out[0][0] == "gamma_leaf"
    # 纯利用（c=0）时结论反转 —— 证明上面的第一名确实来自探索项而非巧合
    exploit_only = ucb_focus_leaves(_STATS, _LEAVES, k=1, c=0.0)
    assert exploit_only[0][0] == "alpha_leaf"
    assert exploit_only[0][1] == pytest.approx(0.06, abs=1e-12)


def test_ucb_exhausted_is_hard_excluded():
    """exhausted 叶硬排除：不出现在结果里，且不参与 N_total。

    exhausted={alpha_leaf}（n_i=10）→ 存活叶 n_i 之和 = 2，ln(1+2)=1.0986122886681098：
      beta  = 0.01 + 1.4*sqrt(ln3/3) = 0.8572071934282064
      gamma = 0.00 + 1.4*sqrt(ln3/1) = 1.467405903555487
    若 N_total 误把 exhausted 叶也算进去（=12），beta 会是 1.3045…、gamma 2.2421…，
    两个字面量都对不上 —— 这条断言因此能抓住 N_total 口径漂移。
    """
    out = ucb_focus_leaves(_STATS, _LEAVES, exhausted={"alpha_leaf"}, k=3)

    assert [n for n, _s in out] == ["gamma_leaf", "beta_leaf"]
    got = dict(out)
    assert "alpha_leaf" not in got
    assert got["beta_leaf"] == pytest.approx(0.8572071934282064, abs=1e-12)
    assert got["gamma_leaf"] == pytest.approx(1.467405903555487, abs=1e-12)

    # 全部挖穿 → 空列表（不许兜底放行，否则与 filter_exhausted_expressions 硬闸自相矛盾）
    assert ucb_focus_leaves(_STATS, _LEAVES, exhausted=set(_LEAVES), k=3) == []


def test_ucb_crowd_penalty_changes_ranking():
    """库内 active 已多的叶降权，且降权足以改变排序。

    crowded={gamma_leaf: 10}, crowd_penalty=0.15 → gamma 扣 1.5：
      gamma = 2.242164298311926 - 1.5 = 0.7421642983119261
      beta  = 1.3045141611977589（未扣）
      alpha = 0.7360379723482866（未扣）
    排序由 gamma>beta>alpha 变为 beta>gamma>alpha —— gamma 掉到第二但仍险胜 alpha，
    差值仅 0.0061，写死字面量才能锁住惩罚系数被改动时的失败。
    """
    out = ucb_focus_leaves(_STATS, _LEAVES, crowded={"gamma_leaf": 10}, k=3)

    assert [n for n, _s in out] == ["beta_leaf", "gamma_leaf", "alpha_leaf"]
    got = dict(out)
    assert got["gamma_leaf"] == pytest.approx(0.7421642983119261, abs=1e-12)
    assert got["alpha_leaf"] == pytest.approx(0.7360379723482866, abs=1e-12)

    # crowd_penalty=0 → 惩罚关闭，排序回到无 crowded 的基线
    off = ucb_focus_leaves(_STATS, _LEAVES, crowded={"gamma_leaf": 10}, crowd_penalty=0.0, k=3)
    assert [n for n, _s in off] == ["gamma_leaf", "beta_leaf", "alpha_leaf"]
    # crowded 里出现不在 leaf_names 的叶 → 忽略，不炸也不影响任何叶的分数
    irrelevant = ucb_focus_leaves(_STATS, _LEAVES, crowded={"no_such_leaf": 99}, k=3)
    assert [n for n, _s in irrelevant] == ["gamma_leaf", "beta_leaf", "alpha_leaf"]
    assert dict(irrelevant)["gamma_leaf"] == pytest.approx(2.242164298311926, abs=1e-12)


def test_ucb_is_deterministic_with_name_tiebreak():
    """分数并列按叶名字典序；两次调用逐值相同（无随机源）。"""
    tied = {
        "zeta": {"n_exprs": 0, "n_passed": 0, "best_abs_ic": 0.0, "n_coverage_fail": 0},
        "alpha": {"n_exprs": 0, "n_passed": 0, "best_abs_ic": 0.0, "n_coverage_fail": 0},
        "mid": {"n_exprs": 0, "n_passed": 0, "best_abs_ic": 0.0, "n_coverage_fail": 0},
    }
    names = ["zeta", "mid", "alpha"]          # 输入序刻意与字典序相反
    first = ucb_focus_leaves(tied, names, k=3)
    second = ucb_focus_leaves(tied, names, k=3)

    assert first == second                     # 逐值（含分数）相同
    assert [n for n, _s in first] == ["alpha", "mid", "zeta"]
    # 三者分数确实相等（否则排序是被分数而非 tie-break 决定的，用例失去判别力）
    assert len({round(s, 12) for _n, s in first}) == 1
    # k 截断在并列时也确定：取字典序最小的那个
    assert [n for n, _s in ucb_focus_leaves(tied, names, k=1)] == ["alpha"]


def test_ucb_missing_and_degenerate_inputs():
    """stats 缺条目按全 0 处理；空输入不炸。"""
    # leaf_names 有、stats 没有 → 视作从未尝试（与 build_leaf_guidance 的缺省同款）
    out = ucb_focus_leaves({}, ["a", "b"], k=2)
    assert [n for n, _s in out] == ["a", "b"]
    assert out[0][1] == pytest.approx(out[1][1], abs=1e-12)
    assert ucb_focus_leaves({}, [], k=3) == []
    assert ucb_focus_leaves(_STATS, [], k=3) == []
    # None 值不许穿透成 TypeError
    sloppy = {"a": {"n_exprs": None, "best_abs_ic": None, "n_coverage_fail": None}}
    assert len(ucb_focus_leaves(sloppy, ["a"], k=1)) == 1


# ── prompt 片段 ──────────────────────────────────────────────────────────────


def test_format_focus_leaves_is_directive_not_background():
    text = format_focus_leaves([("gamma_leaf", 2.2422), ("beta_leaf", 1.3045)])
    assert "gamma_leaf" in text and "beta_leaf" in text
    assert "焦点" in text
    # 明确指令而非背景信息
    assert "请" in text
    # None / 空 → 空串（零回归闸）
    assert format_focus_leaves(None) == ""
    assert format_focus_leaves([]) == ""


def test_propose_injects_focus_leaves_and_none_is_zero_regression():
    """两个 propose 入口都注入焦点叶；focus_leaves=None 时 prompt 逐字节不变。"""
    from factorzen.agents.roles.hypothesis import propose_hypotheses, propose_structured

    class FakeLLM:
        def __init__(self, payload):
            self.calls: list = []
            self.payload = payload

        def __call__(self, messages):
            self.calls.append(messages)
            return self.payload

    focus = [("holder_num_chg", 1.9), ("i_rv", 1.2)]

    plain_payload = json.dumps({"hypotheses": ["dir"]})
    llm = FakeLLM(plain_payload)
    propose_hypotheses(llm, known_invalid=[], known_valid=[], n=1, focus_leaves=focus)
    blob = "\n".join(m["content"] for m in llm.calls[0])
    assert "holder_num_chg" in blob and "i_rv" in blob and "焦点" in blob

    llm_off = FakeLLM(plain_payload)
    propose_hypotheses(llm_off, known_invalid=[], known_valid=[], n=1)
    llm_none = FakeLLM(plain_payload)
    propose_hypotheses(llm_none, known_invalid=[], known_valid=[], n=1, focus_leaves=None)
    assert llm_off.calls[0] == llm_none.calls[0]        # 显式 None == 不传
    assert "焦点" not in "\n".join(m["content"] for m in llm_off.calls[0])

    struct_payload = json.dumps({"hypotheses": [
        {"direction": "x", "mechanism": "m", "expected_sign": 1, "falsification": "f"},
    ]})
    sllm = FakeLLM(struct_payload)
    propose_structured(sllm, known_invalid=[], known_valid=[], n=1, focus_leaves=focus)
    sblob = "\n".join(m["content"] for m in sllm.calls[0])
    assert "holder_num_chg" in sblob and "焦点" in sblob

    sllm_off = FakeLLM(struct_payload)
    propose_structured(sllm_off, known_invalid=[], known_valid=[], n=1)
    assert "焦点" not in "\n".join(m["content"] for m in sllm_off.calls[0])


# ── team 轨接线 ──────────────────────────────────────────────────────────────


def _toy_daily():
    import datetime as dt

    import numpy as np
    import polars as pl

    days, d = [], dt.date(2022, 1, 3)
    while len(days) < 80:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    rows = []
    rng = np.random.default_rng(0)
    for c in [f"{i:06d}.SZ" for i in range(20)]:
        px = 10.0
        for dd in days:
            px *= 1 + rng.standard_normal() * 0.01
            rows.append({
                "trade_date": dd, "ts_code": c, "close": px,
                "open": px, "high": px, "low": px,
                "vol": 1e6, "amount": 1e7,
            })
    return pl.DataFrame(rows)


_DW = {"start": "20220101", "end": "20220630", "universe": "csi300", "market": "ashare"}


def _run_round(tmp_path, *, leaf_scheduler: bool, captured: list[str]):
    from factorzen.agents.team_orchestrator import run_team_agent

    def llm_fn(messages):
        blob = "\n".join(m["content"] for m in messages)
        captured.append(blob)
        if "风控审计员" in blob:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in blob:
            return json.dumps({"expressions": ["rank(close)"]})
        return json.dumps({"hypotheses": ["动量"]})

    return run_team_agent(
        _toy_daily(), llm_fn,
        n_rounds=1, seed=1, top_k=3,
        index_path=str(tmp_path / "experiment_index.jsonl"),
        data_window=dict(_DW),
        library_orthogonal=False,
        auto_lift=False,
        heal_rounds=0,
        leaf_scheduler=leaf_scheduler,
    )


def test_leaf_scheduler_on_injects_prompt_and_logs_focus(tmp_path):
    """开关打开：焦点叶文案进 prompt，且 rounds_log 落 focus_leaves（名+分数）。"""
    captured: list[str] = []
    result = _run_round(tmp_path / "on", leaf_scheduler=True, captured=captured)

    all_text = "\n".join(captured)
    assert "焦点" in all_text

    assert result.rounds_log
    focus = result.rounds_log[0].get("focus_leaves")
    assert focus, "leaf_scheduler=True 必须落 focus_leaves"
    assert all(isinstance(row, list) and len(row) == 2 for row in focus)
    names = [row[0] for row in focus]
    assert len(names) == len(set(names))                     # 无重复叶
    scores = [row[1] for row in focus]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)            # 按分数降序
    # 落盘的分数是 round(.,4) —— 可复现要求分数与 prompt 里的选择同源
    assert all(s == round(s, 4) for s in scores)
    # 焦点叶必须是本 session 的**真实存活叶**，不是调度器凭空造的名字。
    # 拿 AgentContext 的叶集合对照（而非 prompt 文本——焦点叶自己就被写进了 prompt，
    # 用 prompt 校验等于自证，恒真）。
    from factorzen.agents.nodes import AgentContext

    # session 内的存活叶是默认 A 股叶集的子集（leaf_health 只摘不加），
    # 所以焦点叶必须 ⊆ 默认叶集。
    live = set(AgentContext().leaf_names)
    assert live, "夹具前提：默认 A 股上下文必须有叶子"
    assert set(names) <= live, f"焦点叶越出存活叶集: {set(names) - live}"
    # 默认 k=3；叶集合远大于 3 → 必须做了截断而不是全给
    assert len(names) == 3

    # 可 JSON 序列化（rounds_log 要进 manifest）
    json.dumps(result.rounds_log)


def test_leaf_scheduler_off_is_zero_regression(tmp_path):
    """开关关闭（默认）：prompt 与 rounds_log 与「没有这个特性」时逐字节一致。"""
    cap_off: list[str] = []
    res_off = _run_round(tmp_path / "off", leaf_scheduler=False, captured=cap_off)

    assert "焦点" not in "\n".join(cap_off)
    assert res_off.rounds_log
    assert "focus_leaves" not in res_off.rounds_log[0], (
        "关闭时不得新增 rounds_log 键——否则 manifest 形状对旧 run 漂移"
    )

    # 默认值必须是关：不传参数 == 显式 False
    cap_default: list[str] = []
    from factorzen.agents.team_orchestrator import run_team_agent

    def llm_fn(messages):
        blob = "\n".join(m["content"] for m in messages)
        cap_default.append(blob)
        if "风控审计员" in blob:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in blob:
            return json.dumps({"expressions": ["rank(close)"]})
        return json.dumps({"hypotheses": ["动量"]})

    res_default = run_team_agent(
        _toy_daily(), llm_fn,
        n_rounds=1, seed=1, top_k=3,
        index_path=str(tmp_path / "default" / "experiment_index.jsonl"),
        data_window=dict(_DW),
        library_orthogonal=False,
        auto_lift=False,
        heal_rounds=0,
    )
    assert cap_default == cap_off
    assert res_default.rounds_log[0] == res_off.rounds_log[0]


def test_recall_carries_leaf_stats_same_params(tmp_path):
    """recall 把它算过的 leaf_stats 挂在 Recall 上——调度器与挖穿判定同一份，构造上同参。"""
    from factorzen.agents.experiment_index import ExperimentIndex
    from factorzen.agents.roles.librarian import recall

    idx = ExperimentIndex(str(tmp_path / "e.jsonl"))
    idx.append([{
        "expression": "rank(holder_num_chg)", "data_window": dict(_DW),
        "passed": False, "compile_ok": True, "ic_train": 0.02,
    }])

    r = recall(idx, k=5, data_window=dict(_DW), leaf_names=["holder_num_chg", "roe"])
    assert r.leaf_stats is not None
    assert r.leaf_stats == idx.leaf_stats(["holder_num_chg", "roe"], data_window=dict(_DW))
    assert r.leaf_stats["holder_num_chg"]["n_exprs"] == 1

    # leaf_names 未传 → 未算过 → None（零回归）
    assert recall(idx, k=5, data_window=dict(_DW)).leaf_stats is None


def test_cli_mine_team_has_leaf_scheduler_flag():
    """CLI 旗标存在、默认关。"""
    from factorzen.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "mine", "team", "--start", "20220101", "--end", "20220630",
    ])
    assert args.leaf_scheduler is False
    args_on = parser.parse_args([
        "mine", "team", "--start", "20220101", "--end", "20220630", "--leaf-scheduler",
    ])
    assert args_on.leaf_scheduler is True
