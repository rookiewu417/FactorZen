"""ops 阶段级幂等状态 OpsState 的测试(原子落盘,重入跳过已完成阶段)。"""
from __future__ import annotations

from datetime import date

from factorzen.ops.state import OpsState


def test_mark_done_persists(tmp_path):
    """标记 done 后 is_done 为真,且落盘——新实例重读仍为真(支持跨进程重入)。"""
    st = OpsState(tmp_path, date(2026, 1, 5))
    assert st.is_done("data") is False
    st.mark_done("data", detail="补齐 60 日")
    assert st.is_done("data") is True
    # 新实例(模拟重跑进程)从磁盘恢复
    st2 = OpsState(tmp_path, date(2026, 1, 5))
    assert st2.is_done("data") is True


def test_mark_failed_not_done(tmp_path):
    st = OpsState(tmp_path, date(2026, 1, 5))
    st.mark_failed("audit", detail="daily 有缺口")
    assert st.is_done("audit") is False
    st2 = OpsState(tmp_path, date(2026, 1, 5))
    assert st2.is_done("audit") is False


def test_failed_then_done_overrides(tmp_path):
    """先失败后重跑成功:done 覆盖 failed(重入修复语义)。"""
    st = OpsState(tmp_path, date(2026, 1, 5))
    st.mark_failed("data", detail="超时")
    st.mark_done("data", detail="重跑成功")
    assert st.is_done("data") is True


def test_summary_contains_stages(tmp_path):
    st = OpsState(tmp_path, date(2026, 1, 5))
    st.mark_done("data")
    st.mark_failed("audit", detail="x")
    s = st.summary()
    assert s["data"]["status"] == "done"
    assert s["audit"]["status"] == "failed"
    assert s["audit"]["detail"] == "x"


def test_different_dates_isolated(tmp_path):
    """不同交易日的状态互不干扰(各自一个 json 文件)。"""
    a = OpsState(tmp_path, date(2026, 1, 5))
    a.mark_done("data")
    b = OpsState(tmp_path, date(2026, 1, 6))
    assert b.is_done("data") is False


def test_no_tmp_residue(tmp_path):
    """原子写不留 .tmp 残留文件。"""
    st = OpsState(tmp_path, date(2026, 1, 5))
    st.mark_done("data")
    assert list(tmp_path.glob("*.tmp")) == []
