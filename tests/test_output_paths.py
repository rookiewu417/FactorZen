from factorzen.config import settings
from factorzen.config.settings import (
    OUTPUT_DAILY_FACTORS,
    OUTPUT_DAILY_REPORTS,
    OUTPUT_DAILY_RESULTS,
    daily_factor_output_dir,
    daily_report_output_dir,
    daily_result_output_dir,
)


def test_qlib_alpha158_outputs_go_to_qlib158_bucket():
    factor = "qlib_alpha158_kmid"

    assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS / "qlib158"
    assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS / "qlib158"
    assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS / "qlib158"


def test_qlib_alpha360_outputs_go_to_qlib360_bucket():
    factor = "qlib_alpha360_close0"

    assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS / "qlib360"
    assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS / "qlib360"
    assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS / "qlib360"


def test_personal_factor_outputs_stay_in_daily_roots():
    factor = "momentum_20d"

    assert daily_factor_output_dir(factor) == OUTPUT_DAILY_FACTORS
    assert daily_result_output_dir(factor) == OUTPUT_DAILY_RESULTS
    assert daily_report_output_dir(factor) == OUTPUT_DAILY_REPORTS


def test_evaluations_are_sibling_to_runs_and_artifacts_stay_in_runs():
    assert settings.FACTOR_EVALUATIONS_DIR == settings.WORKSPACE_DIR / "factor_evaluations"
    assert settings.WORKSPACE_RUNS == settings.WORKSPACE_DIR / "runs"
    assert settings.OUTPUT_DIR == settings.WORKSPACE_RUNS / "artifacts"


def test_default_log_dir_stays_in_runs():
    from factorzen.core.logger import default_log_dir

    assert default_log_dir() == settings.WORKSPACE_RUNS / "logs"
