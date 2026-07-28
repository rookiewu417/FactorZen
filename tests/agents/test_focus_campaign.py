"""定向 campaign 模式：focus override + exhausted 豁免 + horizon 透传。

背景：稀疏事件叶（fc_surprise / assets_yoy / express_yoy …）在**稠密**单因子护栏下
结构性失败，被 `raw_exhausted_leaves` 判「试≥15 次 0 过关」打成挖穿 → UCB 硬排除 +
`filter_exhausted_expressions` 直接丢弃含它们的表达式，LLM 再也碰不到。
定向 campaign 的语义是「我知道它标了挖穿，我要再攻一次」：用户显式指定本 session 主攻
叶集，跳过 UCB、只对这些叶豁免 exhausted。
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

# ── 纯函数：override 解析 ────────────────────────────────────────────────────


def test_resolve_focus_override_parses_and_intersects():
    """逗号分隔 str / list 都吃；保序去重；与存活叶求交。"""
    from factorzen.agents.leaf_scheduler import resolve_focus_override

    live = ["fc_surprise", "assets_yoy", "express_yoy", "close"]

    assert resolve_focus_override("fc_surprise,assets_yoy", live) == [
        "fc_surprise", "assets_yoy"]
    # 顺序按用户给定，不按 live 顺序（用 live 的逆序验证：若实现照抄 live 序会失败）
    assert resolve_focus_override("assets_yoy,fc_surprise", live) == [
        "assets_yoy", "fc_surprise"]
    # 空白容忍 + 保序去重
    assert resolve_focus_override(" express_yoy , close ,express_yoy ", live) == [
        "express_yoy", "close"]
    # list 形态
    assert resolve_focus_override(["close", "assets_yoy"], live) == [
        "close", "assets_yoy"]
    # None / 空串 / 全空白 → 不启用（零回归）
    assert resolve_focus_override(None, live) == []
    assert resolve_focus_override("", live) == []
    assert resolve_focus_override("  ,  ", live) == []
    assert resolve_focus_override([], live) == []


def test_resolve_focus_override_partial_invalid_warns_and_drops(caplog):
    """部分叶名无效 → 警告 + 忽略，有效的照常返回。"""
    import logging

    from factorzen.agents.leaf_scheduler import resolve_focus_override

    live = ["fc_surprise", "close"]
    with caplog.at_level(logging.WARNING):
        out = resolve_focus_override("fc_surprise,not_a_leaf,typo_leaf", live)
    assert out == ["fc_surprise"]
    blob = caplog.text
    assert "not_a_leaf" in blob and "typo_leaf" in blob
    # 有效叶不该被当成错误报出来
    assert blob.count("fc_surprise") == 0


def test_resolve_focus_override_all_invalid_fails_loudly():
    """全部无效 → ValueError（静默降级成"跑普通 UCB"会让用户以为定向跑过了）。"""
    from factorzen.agents.leaf_scheduler import resolve_focus_override

    with pytest.raises(ValueError) as exc:
        resolve_focus_override("nope,also_nope", ["fc_surprise", "close"])
    assert "nope" in str(exc.value)


# ── prompt 文案：override 模式的事件式引导 ─────────────────────────────────


def test_format_focus_leaves_override_adds_event_guidance():
    """override 文案含叶名 + 事件式构造引导；非 override 文案逐字节不变。"""
    from factorzen.llm.prompt_fragments import format_focus_leaves

    ucb = format_focus_leaves([("fc_surprise", 1.2), ("assets_yoy", 0.9)])
    ov = format_focus_leaves(
        [("fc_surprise", None), ("assets_yoy", None)], override=True)

    assert "fc_surprise" in ucb and "fc_surprise" in ov
    assert ucb != ov, "override 必须换文案，否则等于没加引导"
    # 事件式引导的判别锚点（措辞可变，但这两个语义必须在）
    assert "事件" in ov
    assert "ts_max" in ov or "ts_sum" in ov or "窗口" in ov
    # 非 override 路径逐字节零回归（显式 False == 不传）
    assert format_focus_leaves(
        [("fc_surprise", 1.2), ("assets_yoy", 0.9)], override=False) == ucb
    assert format_focus_leaves(None, override=True) == ""
    assert format_focus_leaves([], override=True) == ""


# ── team 轨接线 ──────────────────────────────────────────────────────────────


_DW = {"start": "20220101", "end": "20220630", "universe": "csi300", "market": "ashare"}


def _toy_daily():
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


def _scripted_llm(exprs, captured=None):
    """scripted llm：固定假设 + 固定表达式集；把 prompt 原文抄进 captured。"""
    def llm_fn(messages):
        blob = "\n".join(m["content"] for m in messages)
        if captured is not None:
            captured.append(blob)
        if "风控审计员" in blob:
            return json.dumps({"verdict": "keep", "reason": "ok"})
        if "翻译成" in blob:
            return json.dumps({"expressions": list(exprs)})
        return json.dumps({"hypotheses": ["事件驱动"]})
    return llm_fn


def _run(tmp_path, *, n_rounds=1, focus_leaves=None, exprs=("rank(close)",),
         captured=None, index_path=None, horizon=None):
    from factorzen.agents.team_orchestrator import run_team_agent

    kw = {}
    if focus_leaves is not None:
        kw["focus_leaves"] = focus_leaves
    if horizon is not None:
        kw["horizon"] = horizon
    return run_team_agent(
        _toy_daily(), _scripted_llm(exprs, captured),
        n_rounds=n_rounds, seed=1, top_k=3,
        index_path=index_path or str(tmp_path / "experiment_index.jsonl"),
        data_window=dict(_DW),
        library_orthogonal=False,
        auto_lift=False,
        heal_rounds=0,
        leaf_scheduler=True,
        **kw,
    )


def _spy_ucb(monkeypatch):
    """给 `ucb_focus_leaves` 挂计数 spy；返回 calls 列表。"""
    from factorzen.agents import leaf_scheduler as _ls

    calls: list[tuple] = []
    orig = _ls.ucb_focus_leaves

    def spy(*a, **kw):
        calls.append((a, kw))
        return orig(*a, **kw)

    monkeypatch.setattr(_ls, "ucb_focus_leaves", spy)
    return calls


def test_focus_override_pins_focus_every_round_and_skips_ucb(tmp_path, monkeypatch):
    """override：每轮焦点恒为指定叶集（无 pick-decay 轮换），且 UCB 完全不被调用。"""
    calls = _spy_ucb(monkeypatch)
    target = ["fc_surprise", "assets_yoy", "express_yoy"]
    captured: list[str] = []
    res = _run(tmp_path, n_rounds=3, focus_leaves=",".join(target),
               captured=captured)

    assert len(res.rounds_log) == 3
    for entry in res.rounds_log:
        focus = entry.get("focus_leaves")
        assert focus, "override 每轮都必须落 focus_leaves"
        assert [row[0] for row in focus] == target, "override 焦点必须恒定且保序"
        assert all(row[1] is None for row in focus), "override 无 UCB 分数 → null"
    # 落盘可 JSON 序列化（进 manifest）
    json.dumps(res.rounds_log)

    assert calls == [], "override 模式不得调 ucb_focus_leaves（定向就是跳过调度）"

    # 对照：同一夹具不给 override → UCB 被调用且焦点逐轮轮换（证明本用例的
    # "恒定" 不是夹具天然如此）
    calls.clear()
    res_ucb = _run(tmp_path / "ucb", n_rounds=3)
    assert len(calls) == 3, "无 override 时 UCB 必须每轮被调"
    ucb_focus = [[row[0] for row in e["focus_leaves"]] for e in res_ucb.rounds_log]
    assert ucb_focus[0] != ucb_focus[1], "对照组必须轮换，否则本用例无判别力"


def test_focus_override_prompt_uses_event_guidance(tmp_path):
    """override 叶名 + 事件式引导进 Hypothesis prompt。"""
    captured: list[str] = []
    _run(tmp_path, focus_leaves="fc_surprise,assets_yoy", captured=captured)
    blob = "\n".join(captured)
    assert "fc_surprise" in blob and "assets_yoy" in blob
    assert "事件" in blob


def _seed_exhausted_index(path, leaves, n=16):
    """给每个叶灌 n 条 passed=False 的历史记录 → 触发挖穿判定（阈值 15）。"""
    from factorzen.agents.experiment_index import ExperimentIndex

    idx = ExperimentIndex(str(path))
    idx.append([
        {
            "expression": f"add({leaf}, {i})",
            "data_window": dict(_DW),
            "passed": False, "compile_ok": True, "ic_train": 0.001,
        }
        for leaf in leaves for i in range(n)
    ])
    return idx


def test_focus_override_exempts_only_its_own_exhausted_leaves(tmp_path):
    """override 叶的表达式不再被 exhausted 硬闸丢弃；其它 exhausted 叶照常丢。"""
    from factorzen.agents.roles.librarian import raw_exhausted_leaves

    idx_path = tmp_path / "experiment_index.jsonl"
    idx = _seed_exhausted_index(idx_path, ["fc_surprise", "express_yoy"])
    # 夹具前提：两个叶确实被判挖穿（否则整个用例恒真）
    stats = idx.leaf_stats(["fc_surprise", "express_yoy"], data_window=dict(_DW))
    assert set(raw_exhausted_leaves(stats, ["fc_surprise", "express_yoy"])) == {
        "fc_surprise", "express_yoy"}

    exprs = ["rank(fc_surprise)", "rank(express_yoy)"]

    # ── 对照：无 override → 两条都被丢
    base = _run(tmp_path / "base", exprs=exprs, index_path=str(idx_path))
    b0 = base.rounds_log[0]
    assert b0["n_exhausted_filtered"] == 2, "对照组必须两条全丢，否则本用例无判别力"
    assert b0["expressions"] == []

    # ── override 只豁免 fc_surprise
    ov = _run(tmp_path / "ov", exprs=exprs, index_path=str(idx_path),
              focus_leaves="fc_surprise")
    o0 = ov.rounds_log[0]
    assert "rank(fc_surprise)" in o0["expressions"], "override 叶必须被放行"
    assert "rank(express_yoy)" not in o0["expressions"], "豁免不许扩散到其它挖穿叶"
    assert o0["n_exhausted_filtered"] == 1


def test_focus_override_exempts_prompt_side_too(tmp_path):
    """豁免同参：override 叶不得再出现在 prompt 的「已挖穿(避开)」清单里。

    `librarian.recall` 的 docstring 明写两个消费方（提示词 / 硬过滤）必须同参；
    只豁免硬闸会让 prompt 同时说「fc_surprise 已挖穿避开」和「fc_surprise 是本轮焦点」。
    """
    idx_path = tmp_path / "experiment_index.jsonl"
    _seed_exhausted_index(idx_path, ["fc_surprise", "express_yoy"])

    cap_base: list[str] = []
    _run(tmp_path / "base", index_path=str(idx_path), captured=cap_base)
    base_blob = "\n".join(cap_base)
    assert "已挖穿" in base_blob
    assert "fc_surprise" in base_blob, "对照前提：无 override 时它出现在挖穿清单"

    cap_ov: list[str] = []
    res = _run(tmp_path / "ov", index_path=str(idx_path),
               focus_leaves="fc_surprise", captured=cap_ov)
    # 挖穿清单里不得再有它（focus 段会提到它，故按 rounds_log 的结构化摘要断言，
    # 不用 prompt 文本——那会被 focus 段自证）
    lg = res.rounds_log[0]["leaf_guidance"]
    assert lg is not None
    exhausted_names = [s.split("(")[0] for s in lg["exhausted"]]
    assert "fc_surprise" not in exhausted_names
    assert "express_yoy" in exhausted_names, "非 override 叶仍须留在挖穿清单"


def test_focus_leaves_none_is_zero_regression(tmp_path, monkeypatch):
    """focus_leaves=None：UCB 照跑、prompt 与 rounds_log 与不传该参数逐值一致。"""
    calls = _spy_ucb(monkeypatch)
    cap_a: list[str] = []
    res_a = _run(tmp_path / "a", captured=cap_a)              # 不传 focus_leaves
    n_a = len(calls)
    cap_b: list[str] = []
    res_b = _run(tmp_path / "b", focus_leaves=None, captured=cap_b)  # 显式 None
    assert n_a == 1 and len(calls) == 2, "两次都必须走 UCB"
    assert cap_a == cap_b
    assert res_a.rounds_log[0] == res_b.rounds_log[0]
    # UCB 分数是真 float，不是 override 的 null
    assert all(isinstance(row[1], float)
               for row in res_a.rounds_log[0]["focus_leaves"])


def test_focus_override_all_invalid_raises_from_session(tmp_path):
    """session 级 fail loudly：全无效叶名直接抛，不静默退化成普通 UCB。"""
    with pytest.raises(ValueError):
        _run(tmp_path, focus_leaves="not_a_leaf,also_not")


# ── CLI / pipeline 接线 ─────────────────────────────────────────────────────


def test_mine_team_set_accepts_focus_leaves_and_horizon():
    """`--set focus_leaves=...` / `--set horizon=5` 经 `_MINE_TEAM_SET` 落 args。"""
    from factorzen.cli import main as cli_main

    schema = cli_main._MINE_TEAM_SET
    assert schema["focus_leaves"][0] is str
    assert schema["focus_leaves"][1] is None
    assert schema["horizon"] == (int, 1, None)

    parser = cli_main.build_parser()
    args = parser.parse_args([
        "mine", "team", "--start", "20220101", "--end", "20220630",
        "--set", "focus_leaves=fc_surprise,assets_yoy",
        "--set", "horizon=5",
    ])
    assert cli_main._apply_set_overrides(args, schema) is None
    assert args.focus_leaves == "fc_surprise,assets_yoy"
    assert args.horizon == 5

    # 缺省：focus_leaves=None、horizon=1（零回归）
    args_d = parser.parse_args([
        "mine", "team", "--start", "20220101", "--end", "20220630",
    ])
    assert cli_main._apply_set_overrides(args_d, schema) is None
    assert args_d.focus_leaves is None
    assert args_d.horizon == 1


def test_cmd_mine_team_threads_focus_and_horizon(monkeypatch, tmp_path):
    """`_cmd_mine_team` → `run_team_mine` 实参含 focus_leaves / horizon。"""
    import factorzen.pipelines.factor_mine_team as pmt
    from factorzen.cli import main as cli_main

    seen: dict = {}

    def fake_run_team_mine(daily, **kw):
        seen.update(kw)
        return {"run_dir": str(tmp_path), "n_candidates": 0, "n_trials": 0,
                "candidates": []}

    monkeypatch.setattr(pmt, "run_team_mine", fake_run_team_mine)
    monkeypatch.setattr(
        cli_main, "_prepare_agent_mining_data",
        lambda args: (_toy_daily(), None, {}))

    parser = cli_main.build_parser()
    args = parser.parse_args([
        "mine", "team", "--start", "20220101", "--end", "20220630",
        "--set", "focus_leaves=fc_surprise,assets_yoy",
        "--set", "horizon=5",
    ])
    assert cli_main._cmd_mine_team(args) == 0
    assert seen["focus_leaves"] == "fc_surprise,assets_yoy"
    assert seen["horizon"] == 5


def test_run_team_mine_threads_to_run_team_agent_and_manifest(monkeypatch, tmp_path):
    """`run_team_mine` → `run_team_agent` 实参 + manifest.params 记 focus_leaves/horizon。"""
    import factorzen.pipelines.factor_mine_team as pmt

    class _Stop(Exception):
        pass

    seen: dict = {}

    def fake_run_team_agent(daily, llm_fn, **kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(pmt, "run_team_agent", fake_run_team_agent)
    with pytest.raises(_Stop):
        pmt.run_team_mine(
            _toy_daily(), n_rounds=1, seed=1,
            index_path=str(tmp_path / "idx.jsonl"),
            out_dir=str(tmp_path / "out"),
            llm_fn=_scripted_llm(["rank(close)"]),
            focus_leaves="fc_surprise",
            horizon=5,
        )
    assert seen["focus_leaves"] == "fc_surprise"
    assert seen["horizon"] == 5

    # manifest params：跑真实 run_team_agent 太重，改在 write_team_manifest 处截获
    captured_params: dict = {}

    def fake_manifest(result, *, out_dir, run_id, params, partial=False):
        captured_params.update(params)
        raise _Stop()

    monkeypatch.setattr(pmt, "run_team_agent",
                        lambda daily, llm_fn, **kw: object())
    monkeypatch.setattr(pmt, "write_team_manifest", fake_manifest)
    with pytest.raises(_Stop):
        pmt.run_team_mine(
            _toy_daily(), n_rounds=1, seed=1,
            index_path=str(tmp_path / "idx2.jsonl"),
            out_dir=str(tmp_path / "out2"),
            llm_fn=_scripted_llm(["rank(close)"]),
            focus_leaves="fc_surprise,assets_yoy",
            horizon=5,
        )
    assert captured_params["focus_leaves"] == "fc_surprise,assets_yoy"
    assert captured_params["horizon"] == 5

    # 缺省：focus_leaves=None、horizon=1（旧调用方零回归）
    captured_params.clear()
    with pytest.raises(_Stop):
        pmt.run_team_mine(
            _toy_daily(), n_rounds=1, seed=1,
            index_path=str(tmp_path / "idx3.jsonl"),
            out_dir=str(tmp_path / "out3"),
            llm_fn=_scripted_llm(["rank(close)"]),
        )
    assert captured_params["focus_leaves"] is None
    assert captured_params["horizon"] == 1
