"""Lightweight planning + completeness-gate helpers (browser-free)."""

from agent import actions as A
from agent.loop import _looks_like_list
from agent.memory import Memory
from agent.observation import Observation
from agent.prompts import build_planner_messages


def test_schema_has_optional_plan():
    s = A.action_output_schema()
    assert "plan" in s["properties"]
    assert "plan" not in s["required"]  # optional


def test_looks_like_list():
    assert _looks_like_list("Catso, Dibbins, Anglebert Dinkherhump")
    assert _looks_like_list("1. Alpha\n2. Beta\n3. Gamma")
    assert _looks_like_list("- a\n- b")
    assert not _looks_like_list("It costs $19.99")
    assert not _looks_like_list("")
    assert not _looks_like_list(None)


def test_planner_shows_plan_when_planning_on():
    obs = Observation(url="u", title="t", elements=[])
    msg = build_planner_messages(
        "goal", obs, Memory(goal="g"), planning=True, plan="1. open reviews [done]\n2. read all",
    )[0]["content"]
    assert "PLAN SO FAR:" in msg and "read all" in msg
    assert '"plan"' in msg  # asks the model to maintain it


def test_planner_omits_plan_when_off():
    obs = Observation(url="u", title="t", elements=[])
    msg = build_planner_messages("goal", obs, Memory(goal="g"), planning=False)[0]["content"]
    assert "PLAN SO FAR:" not in msg
