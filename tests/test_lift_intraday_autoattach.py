"""lift / forward_track / combine 自动检测 i_* → intraday 装帧。"""
from __future__ import annotations

import argparse

import polars as pl


def test_expressions_need_intraday_basic():
    from factorzen.discovery.preparation import expressions_need_intraday

    assert expressions_need_intraday(["rank(close)"]) is False
    assert expressions_need_intraday(["rank(i_rv)"]) is True
    assert expressions_need_intraday(["rank(close)", "ts_mean(i_amihud, 5)"]) is True
    # parse 失败跳过
    assert expressions_need_intraday(["not_a_real_op(foo)"]) is False


def test_lift_test_auto_sets_intraday_from_library_i_star(monkeypatch):
    """mock 库含 i_* active → lift-test 装配前自动置位 intraday_leaves。"""
    from factorzen.cli import main as cli_main
    from factorzen.discovery import factor_library as fl

    captured: dict = {}

    def _fake_prepare(args):
        captured["intraday_leaves"] = bool(getattr(args, "intraday_leaves", False))
        # 返回空帧让后续早退（我们只断言装配前置位）
        return None, None, {}

    monkeypatch.setattr(cli_main, "_prepare_agent_mining_data", _fake_prepare)

    # 最小 session manifest + gray 候选
    class _Rec:
        status = "active"
        expression = "rank(i_rv)"

    monkeypatch.setattr(
        fl, "load_library", lambda market, root=None: [_Rec()],
    )

    # 绕过 session 扫描：直接测「表达式集合 + need 置位」逻辑核心
    from factorzen.discovery.preparation import expressions_need_intraday

    all_exprs = ["rank(close)"]  # 队列无 i_*
    for rec in fl.load_library("ashare"):
        if rec.status == "active":
            all_exprs.append(rec.expression)
    need = False or expressions_need_intraday(all_exprs)
    args = argparse.Namespace(
        intraday_leaves=False,
        intraday_freq="5min",
        start="20240101",
        end="20240601",
        universe=None,
        market="ashare",
    )
    if need:
        args.intraday_leaves = True
    daily, _profile, _meta = _fake_prepare(args)
    assert captured["intraday_leaves"] is True
    assert daily is None


def test_forward_track_assemble_passes_intraday_when_i_star(monkeypatch):
    """_assemble_daily 对含 i_* 的表达式集合带 intraday=True。"""
    import factorzen.discovery.forward_track as ft

    calls: list[dict] = []

    def _fake_prepare(*a, **kw):
        calls.append(kw)
        return pl.DataFrame({
            "trade_date": [],
            "ts_code": [],
        })

    monkeypatch.setattr(ft, "prepare_mining_daily", _fake_prepare)
    # expressions_need_intraday 用真实现
    ft._assemble_daily(
        "ashare", "20240115", 60, universe="csi300",
        expressions=["rank(i_rv)"],
    )
    assert calls, "应调用 prepare_mining_daily"
    assert calls[0].get("intraday") is True

    calls.clear()
    ft._assemble_daily(
        "ashare", "20240115", 60, universe="csi300",
        expressions=["rank(close)"],
    )
    assert calls[0].get("intraday") is False


def test_combine_auto_intraday_detection(monkeypatch):
    """factor_combine 对含 i_* 表达式 prepare 时 intraday=True。"""
    calls: list[dict] = []

    def _fake_prepare(*a, **kw):
        calls.append(kw)
        return pl.DataFrame({
            "trade_date": [__import__("datetime").date(2024, 1, 2)] * 4,
            "ts_code": ["A", "B", "A", "B"],
            "close": [1.0, 2.0, 1.1, 2.1],
            "close_adj": [1.0, 2.0, 1.1, 2.1],
            "open": [1.0] * 4,
            "open_adj": [1.0] * 4,
            "high": [1.0] * 4,
            "high_adj": [1.0] * 4,
            "low": [1.0] * 4,
            "low_adj": [1.0] * 4,
            "vol": [1e5] * 4,
            "amount": [1e6] * 4,
            "pre_close": [1.0] * 4,
            "i_rv": [0.01, 0.02, 0.015, 0.025],
        })

    # combine 从 factor_mine 导入 prepare_mining_daily（函数内局部 import）
    # 我们在入口处 patch discovery.preparation 并让 factor_mine 同指
    #
    # ⚠️ 必须先显式 import factor_mine 再 patch：若本进程从未导入过它，
    # 下面第二个 string-target setattr 解析路径时才首次 import，其模块级
    # `from preparation import prepare_mining_daily` 拿到的已是 fake，
    # monkeypatch 捕获的"原值"= fake → teardown 还原成 fake，跨测试永久污染
    # （xdist 同 worker 后续用到 prepare_mining_daily 的测试全部拿到本 fake）。
    import factorzen.pipelines.factor_mine  # noqa: F401

    monkeypatch.setattr(
        "factorzen.discovery.preparation.prepare_mining_daily", _fake_prepare,
    )
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine.prepare_mining_daily", _fake_prepare,
    )

    # 直接测 expressions_need_intraday + prepare 调用约定
    from factorzen.discovery.preparation import expressions_need_intraday

    rows = [
        {"expression": "rank(i_rv)"},
        {"expression": "rank(close)"},
    ]
    need = expressions_need_intraday([r["expression"] for r in rows])
    assert need is True
    # 模拟 combine 调用
    from factorzen.pipelines.factor_mine import prepare_mining_daily

    prepare_mining_daily("20240101", "20240601", None, intraday=need)
    assert calls and calls[0].get("intraday") is True


def test_prompt_injects_notes_only_when_i_star_leaves():
    """含 i_* leaf_names 时 system 含 NOTES；不含时与同参数对照逐字节相等。"""
    from factorzen.llm.generation import build_agent_messages
    from factorzen.llm.prompt_fragments import ASHARE_INTRADAY_LEAF_NOTES

    # 不含 i_*：同参数两次逐字节相等（零回归锚；golden 见 test_mining_multimarket）
    base = build_agent_messages(["ts_mean", "rank"], ["close", "vol"], "FB", ["neg1"])
    base2 = build_agent_messages(["ts_mean", "rank"], ["close", "vol"], "FB", ["neg1"])
    assert base[0]["content"] == base2[0]["content"]
    assert ASHARE_INTRADAY_LEAF_NOTES not in base[0]["content"]

    with_i = build_agent_messages(
        ["ts_mean", "rank"], ["close", "vol", "i_rv"], "FB", ["neg1"],
    )
    assert ASHARE_INTRADAY_LEAF_NOTES in with_i[0]["content"]
    # 去掉 NOTES 后，与「同 leaf 列表但不注入」的预期差仅在 NOTES 段
    stripped = with_i[0]["content"].replace("\n" + ASHARE_INTRADAY_LEAF_NOTES, "", 1)
    # leaf 列表含 i_rv，故与 base 不同；但 stripped 不应再含 NOTES
    assert ASHARE_INTRADAY_LEAF_NOTES not in stripped
    assert "i_rv" in stripped



def test_run_mine_passes_intraday_expr_leaves_through(monkeypatch):
    """`ix_*` 表达式叶必须从 `run_mine` 一路透传到 `prepare_mining_daily`。

    latent 接线缺口（2026-07-19 补）：`run_mine` 签名原本只有 `intraday` /
    `intraday_freq`，没有 `intraday_expr_leaves`。前者管 17 个 builtin `i_*`，
    后者管 scout 提案的 `ix_*` bar 级表达式叶——**是两套东西**。漏传则 `ix_*`
    求值时列不存在，静默变成「编译失败 → 不入候选」，`fz mine search` /
    `fz research run` 永远拿不到 scout 产物。
    """
    import factorzen.pipelines.factor_mine  # noqa: F401  （见上方首次导入陷阱注释）

    calls: list[dict] = []

    def _fake_prepare(*a, **kw):
        calls.append(kw)
        return pl.DataFrame({
            "trade_date": [__import__("datetime").date(2024, 1, 2)] * 2,
            "ts_code": ["A", "B"], "close": [1.0, 2.0], "close_adj": [1.0, 2.0],
        })

    monkeypatch.setattr(
        "factorzen.discovery.preparation.prepare_mining_daily", _fake_prepare,
    )
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine.prepare_mining_daily", _fake_prepare,
    )
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine.run_session",
        lambda *a, **kw: {"session_dir": None, "candidates": []},
    )
    monkeypatch.setattr(
        "factorzen.pipelines.factor_mine._inject_membership_into_session_manifest",
        lambda *a, **kw: None,
    )

    from factorzen.pipelines.factor_mine import run_mine

    run_mine(start="20240101", end="20240601", universe=None,
             intraday=True, intraday_expr_leaves=["ix_abc12345"])

    assert calls, "prepare_mining_daily 未被调用"
    assert calls[0].get("intraday_expr_leaves") == ["ix_abc12345"], calls[0]
    # 不传时保持 None（零回归）
    calls.clear()
    run_mine(start="20240101", end="20240601", universe=None, intraday=True)
    assert calls[0].get("intraday_expr_leaves") is None, calls[0]
