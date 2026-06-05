"""daily_single 的 --set 接线测试：经 --dry-run 路径（纯配置流，无需数据）验证生效。"""

from __future__ import annotations

import json

import pytest

from factorzen.pipelines import daily_single


def _run_dry(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["daily_single", *argv, "--dry-run"])
    daily_single.main()


def test_set_override_no_config_bakes_topn(monkeypatch, capsys):
    """无 YAML + --set backtest.top_n=30 → 单一 topn_30 策略（不套用 4 策略默认套件）。"""
    _run_dry(
        monkeypatch,
        ["--factor", "f", "--start", "20230101", "--end", "20231231", "--set", "backtest.top_n=30"],
    )
    bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
    assert bt["top_n"] == 30
    assert len(bt["strategies"]) == 1
    assert bt["strategies"][0]["name"] == "topn_30"
    assert bt["strategies"][0]["params"] == {"top_n": 30}


def test_set_override_preprocessing(monkeypatch, capsys):
    _run_dry(
        monkeypatch,
        [
            "--factor",
            "f",
            "--start",
            "20230101",
            "--end",
            "20231231",
            "--set",
            "preprocessing.neutralize=true",
            "--set",
            "preprocessing.normalizer=rank_normal",
        ],
    )
    pp = json.loads(capsys.readouterr().out)["config"]["preprocessing"]
    assert pp["neutralize"] is True
    assert pp["normalizer"] == "rank_normal"


def test_no_set_no_config_keeps_default_suite(monkeypatch, capsys):
    """对照：无 --set 无 --config 时维持现状（4 策略默认套件）。"""
    _run_dry(monkeypatch, ["--factor", "f", "--start", "20230101", "--end", "20231231"])
    bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
    assert len(bt["strategies"]) == 4


def test_set_override_with_config(monkeypatch, capsys, tmp_path):
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        "factor: f\nstart: '20230101'\nend: '20231231'\nbacktest:\n  top_n: 50\n",
        encoding="utf-8",
    )
    _run_dry(monkeypatch, ["--config", str(cfg), "--set", "backtest.top_n=20"])
    bt = json.loads(capsys.readouterr().out)["config"]["backtest"]
    assert bt["top_n"] == 20
    assert bt["strategies"][0]["name"] == "topn_20"


def test_set_override_invalid_value_exits(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily_single",
            "--factor",
            "f",
            "--start",
            "20230101",
            "--end",
            "20231231",
            "--set",
            "preprocessing.normalizer=bogus",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        daily_single.main()
    assert exc.value.code == 2


def test_set_override_malformed_exits(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily_single",
            "--factor",
            "f",
            "--start",
            "20230101",
            "--end",
            "20231231",
            "--set",
            "backtest.top_n",  # 缺 '='
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        daily_single.main()
    assert exc.value.code == 2
