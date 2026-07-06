"""无人值守运营配置模型 OpsConfig 的测试。"""
from __future__ import annotations

import pytest

from factorzen.ops.config import OpsConfig, load_ops_config


def test_load_ops_config_roundtrip(tmp_path):
    """从 YAML 读取显式字段 + 未写字段取默认值。"""
    p = tmp_path / "ops.yaml"
    p.write_text(
        "session_dir: workspace/execution/prod-001\n"
        "portfolio_run_dirs_glob: 'workspace/portfolios/prod-*'\n"
        "lookback_days: 60\n",
        encoding="utf-8",
    )
    cfg = load_ops_config(p)
    assert cfg.session_dir == "workspace/execution/prod-001"
    assert cfg.portfolio_run_dirs_glob == "workspace/portfolios/prod-*"
    assert cfg.lookback_days == 60
    # 未写字段取默认值
    assert cfg.audit_fail_on == "error"
    assert cfg.benchmark == "000300.SH"
    assert cfg.initial_cash == 1_000_000.0
    assert cfg.notify_kind == "stdout"
    assert cfg.signal_command is None
    assert cfg.audit_types == ["daily", "daily_basic"]


def test_load_ops_config_accepts_str_path(tmp_path):
    """load_ops_config 接受 str 路径(非仅 Path)。"""
    p = tmp_path / "ops.yaml"
    p.write_text("session_dir: s\nportfolio_run_dirs_glob: g\n", encoding="utf-8")
    cfg = load_ops_config(str(p))
    assert cfg.session_dir == "s"


def test_load_ops_config_rejects_bad_fail_on(tmp_path):
    """audit_fail_on 只接受 error/warning,非法值报错。"""
    p = tmp_path / "ops.yaml"
    p.write_text(
        "session_dir: s\nportfolio_run_dirs_glob: g\naudit_fail_on: nonsense\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_ops_config(p)


def test_load_ops_config_rejects_unknown_field(tmp_path):
    """extra='forbid':未知字段(拼写错误)必须报错,而非静默忽略。"""
    p = tmp_path / "ops.yaml"
    p.write_text(
        "session_dir: s\nportfolio_run_dirs_glob: g\nlookback_dayz: 30\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_ops_config(p)


def test_load_ops_config_missing_required(tmp_path):
    """缺必填字段 session_dir 报错。"""
    p = tmp_path / "ops.yaml"
    p.write_text("portfolio_run_dirs_glob: g\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ops_config(p)


def test_load_ops_config_missing_file_raises(tmp_path):
    """文件不存在时抛错并带路径信息。"""
    missing = tmp_path / "nope.yaml"
    with pytest.raises((FileNotFoundError, ValueError)):
        load_ops_config(missing)


@pytest.mark.parametrize("bad", [0, -1, -90])
def test_ops_config_rejects_nonpositive_lookback_days(bad):
    """lookback_days 必须 > 0：零/负窗口无法取数，须在配置层拒绝而非跑到中途才崩。"""
    with pytest.raises(ValueError):
        OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", lookback_days=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_ops_config_rejects_nonpositive_initial_cash(bad):
    """initial_cash 必须 > 0：零/负本金无法纸面执行。"""
    with pytest.raises(ValueError):
        OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", initial_cash=bad)


def test_ops_config_rejects_negative_slippage():
    """slippage_bps 必须 >= 0：负滑点无经济意义（0 允许，表示零滑点对照）。"""
    with pytest.raises(ValueError):
        OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", slippage_bps=-1.0)


def test_ops_config_accepts_zero_slippage():
    """slippage_bps=0.0 合法（零滑点对照），不应被 >=0 约束误伤。"""
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g", slippage_bps=0.0)
    assert cfg.slippage_bps == 0.0


def test_ops_config_defaults_directly():
    """直接构造(仅两个必填)时全部默认值就位。"""
    cfg = OpsConfig(session_dir="s", portfolio_run_dirs_glob="g")
    assert cfg.lookback_days == 90
    assert cfg.universe is None
    assert cfg.slippage_bps == 0.0
    assert cfg.notify_url_env == "FACTORZEN_NOTIFY_WEBHOOK"
    assert cfg.publish_enabled is False
    assert cfg.publish_site_dir == "workspace/ops/site"
    assert cfg.state_dir == "workspace/ops/state"
