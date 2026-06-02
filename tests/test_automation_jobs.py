"""automation/jobs.py 单元测试。

所有外部依赖（loader、subprocess）均通过 mock 替换，保证测试速度和隔离性。
"""

from unittest.mock import MagicMock, patch

import pytest

# ────────────────────────────────────────────────────────────────────────────────
# job_fetch_daily
# ────────────────────────────────────────────────────────────────────────────────


def test_job_fetch_daily_calls_loader():
    """job_fetch_daily 应调用 fetch_daily(start=date, end=date)。"""
    with patch("factorzen.automation.jobs.fetch_daily") as mock_fetch:
        from factorzen.automation.jobs import job_fetch_daily

        job_fetch_daily("20250513")
        mock_fetch.assert_called_once_with(start="20250513", end="20250513")


def test_job_fetch_daily_reraises_on_error():
    """loader 抛出异常时，job_fetch_daily 应记录并重新抛出。"""
    with patch("factorzen.automation.jobs.fetch_daily", side_effect=RuntimeError("network error")):
        from factorzen.automation.jobs import job_fetch_daily

        with pytest.raises(RuntimeError, match="network error"):
            job_fetch_daily("20250513")


# ────────────────────────────────────────────────────────────────────────────────
# job_fetch_index
# ────────────────────────────────────────────────────────────────────────────────


def test_job_fetch_index_calls_each_benchmark():
    """job_fetch_index 应为 BENCHMARK_INDICES 中的每个 key 调用 fetch_index_daily。"""
    from factorzen.config.constants import BENCHMARK_INDICES

    with patch("factorzen.automation.jobs.fetch_index_daily") as mock_idx:
        from factorzen.automation.jobs import job_fetch_index

        job_fetch_index("20250513")
        assert mock_idx.call_count == len(BENCHMARK_INDICES)
        called_codes = {call.kwargs["index_code"] for call in mock_idx.call_args_list}
        assert called_codes == set(BENCHMARK_INDICES.keys())


def test_job_fetch_index_reraises_on_error():
    """fetch_index_daily 抛出异常时，job_fetch_index 应重新抛出。"""
    with patch("factorzen.automation.jobs.fetch_index_daily", side_effect=ValueError("bad code")):
        from factorzen.automation.jobs import job_fetch_index

        with pytest.raises(ValueError, match="bad code"):
            job_fetch_index("20250513")


# ────────────────────────────────────────────────────────────────────────────────
# job_compute_factors
# ────────────────────────────────────────────────────────────────────────────────


def test_job_compute_factors_calls_registry():
    """job_compute_factors 应通过 get_factor 查找每个因子并实例化。

    get_factor 是在函数体内懒加载的，需要在注册模块处 patch。
    """
    mock_factor_cls = MagicMock()
    mock_factor_cls.return_value = MagicMock()

    # get_factor 在 job_compute_factors 内部做延迟导入，patch 其实际模块
    with patch("factorzen.daily.factors.registry.get_factor", return_value=mock_factor_cls) as mock_get:
        from factorzen.automation.jobs import job_compute_factors

        job_compute_factors("20250513", ["momentum_20d", "reversal_5d"])
        assert mock_get.call_count == 2
        mock_get.assert_any_call("momentum_20d")
        mock_get.assert_any_call("reversal_5d")


def test_job_compute_factors_empty_list():
    """空因子列表时，job_compute_factors 应正常完成不报错。"""
    # 空列表不进入 for 循环，get_factor 永远不会被调用
    from factorzen.automation.jobs import job_compute_factors

    # 不应抛出异常
    job_compute_factors("20250513", [])


def test_job_compute_factors_reraises_on_error():
    """get_factor 抛出 KeyError 时，job_compute_factors 应重新抛出。"""
    with patch("factorzen.daily.factors.registry.get_factor", side_effect=KeyError("unknown_factor")):
        from factorzen.automation.jobs import job_compute_factors

        with pytest.raises(KeyError):
            job_compute_factors("20250513", ["unknown_factor"])


# ────────────────────────────────────────────────────────────────────────────────
# job_evaluate
# ────────────────────────────────────────────────────────────────────────────────


def test_job_evaluate_calls_subprocess():
    """job_evaluate 应通过 subprocess.run 调用 run_daily_single.py。"""
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("factorzen.automation.jobs.subprocess.run", return_value=mock_result) as mock_run:
        from factorzen.automation.jobs import job_evaluate

        job_evaluate("20250513", "momentum_20d")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "factorzen.pipelines.daily_single" in cmd
        assert "momentum_20d" in cmd
        assert "20250513" in cmd


def test_job_evaluate_raises_on_nonzero_returncode():
    """subprocess 返回非 0 时，job_evaluate 应抛出 RuntimeError。"""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = "some output"
    mock_result.stderr = "some error"

    with patch("factorzen.automation.jobs.subprocess.run", return_value=mock_result):
        from factorzen.automation.jobs import job_evaluate

        with pytest.raises(RuntimeError, match="退出码=1"):
            job_evaluate("20250513", "momentum_20d")


def test_job_evaluate_reraises_subprocess_exception():
    """subprocess.run 本身抛出异常时，job_evaluate 应重新抛出。"""
    with patch("factorzen.automation.jobs.subprocess.run", side_effect=OSError("pixi not found")):
        from factorzen.automation.jobs import job_evaluate

        with pytest.raises(OSError, match="pixi not found"):
            job_evaluate("20250513", "momentum_20d")


# ────────────────────────────────────────────────────────────────────────────────
# job_generate_report
# ────────────────────────────────────────────────────────────────────────────────


def test_job_generate_report_calls_subprocess():
    """job_generate_report 应通过 subprocess.run 调用 generate_report.py。"""
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("factorzen.automation.jobs.subprocess.run", return_value=mock_result) as mock_run:
        from factorzen.automation.jobs import job_generate_report

        job_generate_report("20250513", "momentum_20d")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "factorzen.pipelines.generate_report" in cmd
        assert "momentum_20d" in cmd
        assert "20250513" in cmd


def test_job_generate_report_raises_on_nonzero_returncode():
    """subprocess 返回非 0 时，job_generate_report 应抛出 RuntimeError。"""
    mock_result = MagicMock()
    mock_result.returncode = 2
    mock_result.stdout = ""
    mock_result.stderr = "template not found"

    with patch("factorzen.automation.jobs.subprocess.run", return_value=mock_result):
        from factorzen.automation.jobs import job_generate_report

        with pytest.raises(RuntimeError, match="退出码=2"):
            job_generate_report("20250513", "momentum_20d")


def test_job_generate_report_reraises_subprocess_exception():
    """subprocess.run 本身抛出异常时，job_generate_report 应重新抛出。"""
    with patch("factorzen.automation.jobs.subprocess.run", side_effect=FileNotFoundError("no pixi")):
        from factorzen.automation.jobs import job_generate_report

        with pytest.raises(FileNotFoundError):
            job_generate_report("20250513", "momentum_20d")
