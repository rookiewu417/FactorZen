from factorzen.agents.state import AgentState, AttemptRecord


def test_attempt_record_fields():
    r = AttemptRecord(iteration=1, hypothesis="低换手反转", expression="rank(close)",
                      compile_ok=True, ic_train=0.03, passed_guardrails=False,
                      critic_verdict=None, error=None)
    assert r.iteration == 1 and r.compile_ok is True


def test_agent_state_defaults_and_serializable():
    import json
    s = AgentState(seed=42)
    assert s.iteration == 0 and s.attempts == [] and s.candidates == []
    assert s.seen_expressions == set() and s.negative_examples == []
    s.attempts.append(AttemptRecord(iteration=0, hypothesis="h", expression="rank(close)",
                                    compile_ok=True, ic_train=0.05, passed_guardrails=True,
                                    critic_verdict="keep", error=None))
    s.seen_expressions.add("rank(close)")
    d = s.to_dict()  # set 转 list，dataclass 转 dict
    assert json.dumps(d)  # 不抛 = JSON 可序列化
    assert d["attempts"][0]["expression"] == "rank(close)"
    assert "rank(close)" in d["seen_expressions"]
