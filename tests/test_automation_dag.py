"""automation/dag.py 单元测试。

验证 build_daily_dag 构建结果和流水线跳过逻辑。
"""

from unittest.mock import patch

import pytest
from apscheduler.schedulers.background import BackgroundScheduler

# ────────────────────────────────────────────────────────────────────────────────
# build_daily_dag
# ────────────────────────────────────────────────────────────────────────────────


def test_build_daily_dag_returns_background_scheduler():
    """build_daily_dag 应返回 BackgroundScheduler 实例。"""
    from factorzen.automation.dag import build_daily_dag

    scheduler = build_daily_dag(factor_list=[])
    assert isinstance(scheduler, BackgroundScheduler)


def test_build_daily_dag_has_at_least_one_job():
    """build_daily_dag 返回的 scheduler 应含有至少 1 个 job。"""
    from factorzen.automation.dag import build_daily_dag

    scheduler = build_daily_dag(factor_list=["momentum_20d"])
    jobs = scheduler.get_jobs()
    assert len(jobs) >= 1


def test_build_daily_dag_job_id():
    """daily_pipeline job 应以 'daily_pipeline' 作为 id。"""
    from factorzen.automation.dag import build_daily_dag

    scheduler = build_daily_dag(factor_list=[])
    job_ids = [j.id for j in scheduler.get_jobs()]
    assert "daily_pipeline" in job_ids


def test_build_daily_dag_not_started():
    """build_daily_dag 返回的 scheduler 默认未启动（调用方负责 start）。"""
    from factorzen.automation.dag import build_daily_dag

    scheduler = build_daily_dag(factor_list=[])
    assert not scheduler.running


# ────────────────────────────────────────────────────────────────────────────────
# run_daily_pipeline — 非交易日跳过
# ────────────────────────────────────────────────────────────────────────────────


def test_pipeline_skips_on_non_trade_day():
    """is_trade_date 返回 False 时，流水线不应调用任何 job。"""
    with (
        patch("factorzen.automation.dag.is_trade_date", return_value=False),
        patch("factorzen.automation.dag.job_fetch_daily") as mock_fd,
        patch("factorzen.automation.dag.job_fetch_index") as mock_fi,
        patch("factorzen.automation.dag.job_compute_factors") as mock_cf,
        patch("factorzen.automation.dag.job_evaluate") as mock_ev,
        patch("factorzen.automation.dag.job_generate_report") as mock_gr,
    ):
        from factorzen.automation.dag import run_daily_pipeline

        run_daily_pipeline("20250101", ["momentum_20d"], "000300.SH")

        mock_fd.assert_not_called()
        mock_fi.assert_not_called()
        mock_cf.assert_not_called()
        mock_ev.assert_not_called()
        mock_gr.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────────
# run_daily_pipeline — 交易日执行
# ────────────────────────────────────────────────────────────────────────────────


def test_pipeline_runs_all_jobs_on_trade_day():
    """is_trade_date 返回 True 时，流水线应依次调用所有 job。"""
    with (
        patch("factorzen.automation.dag.is_trade_date", return_value=True),
        patch("factorzen.automation.dag.job_fetch_daily") as mock_fd,
        patch("factorzen.automation.dag.job_fetch_index") as mock_fi,
        patch("factorzen.automation.dag.job_compute_factors") as mock_cf,
        patch("factorzen.automation.dag.job_evaluate") as mock_ev,
        patch("factorzen.automation.dag.job_generate_report") as mock_gr,
    ):
        from factorzen.automation.dag import run_daily_pipeline

        run_daily_pipeline("20250513", ["momentum_20d"], "000300.SH")

        mock_fd.assert_called_once_with("20250513")
        mock_fi.assert_called_once_with("20250513")
        mock_cf.assert_called_once_with("20250513", ["momentum_20d"])
        mock_ev.assert_called_once_with("20250513", "momentum_20d")
        mock_gr.assert_called_once_with("20250513", "momentum_20d")


def test_pipeline_runs_per_factor_jobs_for_each_factor():
    """每个因子都应被单独评估和报告。"""
    factors = ["factor_a", "factor_b", "factor_c"]

    with (
        patch("factorzen.automation.dag.is_trade_date", return_value=True),
        patch("factorzen.automation.dag.job_fetch_daily"),
        patch("factorzen.automation.dag.job_fetch_index"),
        patch("factorzen.automation.dag.job_compute_factors"),
        patch("factorzen.automation.dag.job_evaluate") as mock_ev,
        patch("factorzen.automation.dag.job_generate_report") as mock_gr,
    ):
        from factorzen.automation.dag import run_daily_pipeline

        run_daily_pipeline("20250513", factors, "000300.SH")

        assert mock_ev.call_count == len(factors)
        assert mock_gr.call_count == len(factors)
        for f in factors:
            mock_ev.assert_any_call("20250513", f)
            mock_gr.assert_any_call("20250513", f)


def test_pipeline_empty_factor_list():
    """空因子列表时，evaluate/generate_report 不应被调用。"""
    with (
        patch("factorzen.automation.dag.is_trade_date", return_value=True),
        patch("factorzen.automation.dag.job_fetch_daily"),
        patch("factorzen.automation.dag.job_fetch_index"),
        patch("factorzen.automation.dag.job_compute_factors"),
        patch("factorzen.automation.dag.job_evaluate") as mock_ev,
        patch("factorzen.automation.dag.job_generate_report") as mock_gr,
    ):
        from factorzen.automation.dag import run_daily_pipeline

        run_daily_pipeline("20250513", [], "000300.SH")
        mock_ev.assert_not_called()
        mock_gr.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────────
# _with_retry — 重试逻辑
# ────────────────────────────────────────────────────────────────────────────────


def test_with_retry_succeeds_first_try():
    """函数第一次成功时，_with_retry 不等待。"""
    from factorzen.automation.dag import _with_retry

    call_count = 0

    def ok():
        nonlocal call_count
        call_count += 1

    _with_retry(ok, max_retries=3, base_seconds=0)
    assert call_count == 1


def test_with_retry_retries_on_failure():
    """函数失败时，_with_retry 应重试最多 max_retries 次。"""
    from factorzen.automation.dag import _with_retry

    call_count = 0

    def always_fail():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("fail")

    with patch("factorzen.automation.dag.time.sleep"), pytest.raises(RuntimeError):  # 不真正等待
        _with_retry(always_fail, max_retries=2, base_seconds=0)

    assert call_count == 3  # 1 次首次 + 2 次重试


def test_with_retry_succeeds_on_second_attempt():
    """函数第二次成功时，_with_retry 应返回而不继续重试。"""
    from factorzen.automation.dag import _with_retry

    call_count = 0

    def fail_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first fail")

    with patch("factorzen.automation.dag.time.sleep"):
        _with_retry(fail_once, max_retries=3, base_seconds=0)

    assert call_count == 2
