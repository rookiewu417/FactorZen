"""Agent/Team candidates.csv 须含 rank + passed，供 fz mine export-alpha 消费。"""
from __future__ import annotations

from pathlib import Path


def test_agent_candidates_csv_df_has_rank_passed():
    from factorzen.discovery.export import agent_candidates_csv_df

    df = agent_candidates_csv_df([{"expression": "rank(close)", "holdout_ic": 0.1, "dsr": 0.6}])
    assert "rank" in df.columns and "passed" in df.columns and "expression" in df.columns
    assert df["rank"].to_list() == [1]
    assert df["passed"].to_list() == [True]


def test_export_alpha_reads_agent_candidates(tmp_path: Path):
    """read_candidate_expression（export-alpha 用）能读 Agent candidates.csv，不再报缺 rank。"""
    from factorzen.discovery.export import agent_candidates_csv_df, read_candidate_expression

    cands = [{"expression": "rank(close)", "holdout_ic": 0.1, "dsr": 0.6},
             {"expression": "ts_mean(vol, 5)", "holdout_ic": 0.05, "dsr": 0.4}]
    agent_candidates_csv_df(cands).write_csv(tmp_path / "candidates.csv")

    assert read_candidate_expression(str(tmp_path), rank=1, require_passed=True) == "rank(close)"
    assert read_candidate_expression(str(tmp_path), rank=2, require_passed=True) == "ts_mean(vol, 5)"
