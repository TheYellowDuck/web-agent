"""The note scratchpad (memory + prompt surfacing)."""

from agent.memory import Memory
from agent.observation import Observation
from agent.prompts import build_planner_messages


def test_render_notes():
    m = Memory(goal="g")
    assert m.render_notes() == ""
    m.notes += ["Catso mentions small ear cups", "Dibbins mentions small ear cups"]
    r = m.render_notes()
    assert "1. Catso" in r and "2. Dibbins" in r


def test_notes_surfaced_in_planner():
    m = Memory(goal="g")
    m.notes.append("found: SONY WH1000XM3")
    msg = build_planner_messages("goal", Observation(url="u", title="t", elements=[]), m)[0]["content"]
    assert "NOTES (your scratchpad" in msg and "SONY WH1000XM3" in msg


def test_no_notes_block_when_empty():
    msg = build_planner_messages(
        "goal", Observation(url="u", title="t", elements=[]), Memory(goal="g")
    )[0]["content"]
    assert "NOTES (your scratchpad" not in msg
