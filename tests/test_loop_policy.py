"""Agent policies that don't need a browser: vision fallback + site origin."""

from agent.llm import ScriptedLLMClient
from agent.loop import Agent, AgentConfig, _origin
from agent.memory import Memory
from agent.observation import Observation
from agent.types import Action


def _agent(**cfg):
    return Agent(ScriptedLLMClient([]), AgentConfig(**cfg))


def _obs(n_elements):
    return Observation(url="u", title="t",
                       elements=[{"ref": f"@e{i}", "role": "link", "name": "x"}
                                 for i in range(n_elements)])


def test_vision_off_never_uses_vision():
    a = _agent(vision_fallback=False)
    assert a._should_use_vision(_obs(0), Memory(goal="g")) is False


def test_vision_fires_on_sparse_page():
    a = _agent(vision_fallback=True)
    assert a._should_use_vision(_obs(0), Memory(goal="g")) is True
    assert a._should_use_vision(_obs(5), Memory(goal="g")) is False


def test_vision_fires_when_stuck():
    a = _agent(vision_fallback=True)
    mem = Memory(goal="g")
    for i in range(2):  # two consecutive failures
        mem.record(i, Action(type="click", ref="@e1"), ok=False)
    assert a._should_use_vision(_obs(5), mem) is True


def test_origin_helper():
    assert _origin("http://localhost:7770/x/y?z=1") == "http://localhost:7770"
    assert _origin("https://a.com/p") == "https://a.com"
    assert _origin("about:blank") == ""   # off-site / blank
    assert _origin("") == ""
