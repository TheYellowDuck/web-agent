"""Agent policies that don't need a browser: vision fallback + site origin."""

from agent.llm import ScriptedLLMClient
from agent.loop import Agent, AgentConfig, _origin
from agent.memory import Memory
from agent.observation import Observation
from agent.prompts import build_reflection_messages
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


def test_reflection_messages_include_before_and_after():
    # The judge must see BOTH states so it can detect "nothing changed" — a no-op
    # click otherwise reads as success just because the after-page is non-empty.
    before = Observation(url="http://s/a", title="t",
                         elements=[{"ref": "@e1", "role": "link", "name": "Login"}])
    after = Observation(url="http://s/a", title="t",
                        elements=[{"ref": "@e1", "role": "link", "name": "Login"}])
    msg = build_reflection_messages("goal", "click(@e1)", before, after)[0]["content"]
    assert "PAGE BEFORE" in msg and "PAGE AFTER" in msg
    assert "(unchanged)" in msg          # URL did not change → flagged

    after2 = Observation(url="http://s/b", title="t", elements=[])
    msg2 = build_reflection_messages("goal", "click(@e1)", before, after2)[0]["content"]
    assert "(unchanged)" not in msg2     # URL changed → not flagged
