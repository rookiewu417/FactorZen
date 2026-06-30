# tests/test_agent_memory.py
from factorzen.agents.memory import family_groups, negative_recall


def test_negative_recall_picks_low_ic():
    seen = [("rank(close)", 0.001), ("ts_mean(vol,5)", 0.08), ("div(close,open)", -0.002)]
    neg = negative_recall(seen, k=2, ic_threshold=0.01)
    # 只召回 IC < 阈值的，按 |IC| 升序（最没用的优先），不含高 IC 的
    assert "ts_mean(vol,5)" not in neg
    assert "rank(close)" in neg
    assert len(neg) <= 2


def test_negative_recall_empty_when_all_good():
    seen = [("a", 0.1), ("b", 0.2)]
    assert negative_recall(seen, ic_threshold=0.01) == []


def test_family_groups_union_find():
    names = ["f1", "f2", "f3", "f4"]
    # f1-f2 高相关, f3-f4 高相关, 两组互不相关
    pairs = {("f1", "f2"): 0.9, ("f1", "f3"): 0.1, ("f3", "f4"): 0.85, ("f2", "f3"): 0.2}
    groups = family_groups(pairs, names, threshold=0.7)
    # 应分成两族 {f1,f2} 和 {f3,f4}
    assert {frozenset(g) for g in groups} == {frozenset({"f1", "f2"}), frozenset({"f3", "f4"})}


def test_family_groups_all_singletons_when_low_corr():
    names = ["a", "b", "c"]
    pairs = {("a", "b"): 0.1, ("b", "c"): 0.2, ("a", "c"): 0.05}
    groups = family_groups(pairs, names, threshold=0.7)
    assert len(groups) == 3  # 全独立
