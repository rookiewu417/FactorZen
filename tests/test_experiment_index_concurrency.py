"""experiment_index 并发写：多进程并发 append 不得交错/损坏/丢行。"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

_PER_PROC = 40


def _worker(args: tuple[str, int]) -> None:
    path, k = args
    from factorzen.agents.experiment_index import ExperimentIndex
    idx = ExperimentIndex(path)
    idx.append([{"expression": f"ts_mean(close, {k})_{i}", "passed": bool(i % 2)}
                for i in range(_PER_PROC)])


def test_concurrent_process_appends_no_corruption(tmp_path):
    path = str(tmp_path / "experiment_index.jsonl")
    n_proc = 8
    ctx = mp.get_context("fork")
    with ctx.Pool(n_proc) as pool:
        pool.map(_worker, [(path, k) for k in range(n_proc)])

    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    assert len(lines) == n_proc * _PER_PROC, "并发写丢行/多行"
    for line in lines:            # 每行都是完整合法 JSON（无交错截断）
        rec = json.loads(line)
        assert "expression" in rec

    from factorzen.agents.experiment_index import ExperimentIndex
    assert len(ExperimentIndex(path).load()) == n_proc * _PER_PROC


def test_append_empty_is_noop(tmp_path):
    from factorzen.agents.experiment_index import ExperimentIndex
    path = tmp_path / "idx.jsonl"
    idx = ExperimentIndex(str(path))
    idx.append([])
    assert not path.exists() or path.read_text() == ""
