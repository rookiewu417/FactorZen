import json
from pathlib import Path

from factorzen.agents.manifest import write_session_manifest
from factorzen.agents.orchestrator import AgentResult
from factorzen.agents.state import AgentState, AttemptRecord


def test_manifest_written_with_audit_trail(tmp_path: Path):
    s = AgentState(seed=42)
    s.attempts = [AttemptRecord(0, "h", "rank(close)", True, 0.04, True, "keep", None)]
    s.candidates = [{"expression": "rank(close)", "holdout_ic": 0.03, "dsr": 0.8}]
    res = AgentResult(state=s, candidates=s.candidates, n_trials=5)
    p = write_session_manifest(res, out_dir=str(tmp_path), run_id="t1",
                               params={"n_rounds": 3, "seed": 42})
    m = json.loads(Path(p).read_text())
    assert m["seed"] == 42 and m["n_trials"] == 5
    assert m["params"]["n_rounds"] == 3
    assert m["attempts"][0]["expression"] == "rank(close)"   # 全程尝试可审计
    assert m["candidates"][0]["dsr"] == 0.8
    assert "git_sha" in m
