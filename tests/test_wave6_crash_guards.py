"""Wave6 crash-P2：sim 跳过半成品目录（无 manifest）+ validate overfit 缺参友好报错。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl


def test_load_weights_skips_dir_without_manifest(tmp_path: Path):
    """含 weights.parquet 无 manifest.json 的半成品目录应被跳过，不 FileNotFoundError。"""
    from factorzen.sim.engine import _load_weights_by_date

    good = tmp_path / "20240102"
    good.mkdir()
    pl.DataFrame({"ts_code": ["A.SZ"], "target_weight": [1.0]}).write_parquet(good / "weights.parquet")
    import json
    (good / "manifest.json").write_text(json.dumps({"signal_date": "2024-01-02", "status": "optimal"}))

    half = tmp_path / "20240103"  # 半成品：只有 weights，无 manifest
    half.mkdir()
    pl.DataFrame({"ts_code": ["A.SZ"], "target_weight": [1.0]}).write_parquet(half / "weights.parquet")

    out = _load_weights_by_date([str(good), str(half)])  # 不应抛异常
    assert date(2024, 1, 2) in out
    assert len(out) == 1  # 半成品目录被跳过


def test_validate_overfit_missing_factor_friendly_error(capsys):
    """fz validate overfit 不给 factor → 返回 2 + 友好提示，而非裸 KeyError traceback。"""
    from factorzen.cli.main import main

    rc = main(["validate", "overfit", "--start", "20240101", "--end", "20241231"])
    assert rc == 2
    assert "缺少因子名" in capsys.readouterr().err
