"""Action parsing + validation against a snapshot."""

import pytest

from agent import actions as A
from agent.observation import Observation
from agent.types import Action


@pytest.fixture
def obs():
    return Observation(
        url="https://example.com",
        title="Example",
        elements=[
            {"ref": "@e1", "role": "link", "name": "Products"},
            {"ref": "@e2", "role": "search", "name": "Search", "input_type": "search"},
            {"ref": "@e3", "role": "select", "name": "Size", "options": ["S", "M", "L"]},
        ],
    )


def test_schema_shape():
    s = A.action_output_schema()
    assert {"thought", "action"} <= set(s["properties"])  # plan is optional/extra
    assert s["required"] == ["thought", "action"]
    assert s["properties"]["action"]["required"] == ["type"]


def test_parse_normalizes_ref():
    a = A.parse_action({"thought": "x", "action": {"type": "click", "ref": "e1"}})
    assert a.ref == "@e1"


def test_parse_bare_action():
    a = A.parse_action({"type": "done", "answer": "hi"})
    assert a.type == "done" and a.answer == "hi"


def test_parse_missing_type_raises():
    with pytest.raises(ValueError):
        A.parse_action({"thought": "x", "action": {"ref": "@e1"}})


def test_valid_click(obs):
    assert A.validate_action(Action(type="click", ref="@e1"), obs) is None


def test_click_missing_ref(obs):
    assert A.validate_action(Action(type="click"), obs)


def test_click_unknown_ref(obs):
    assert "not found" in A.validate_action(Action(type="click", ref="@e9"), obs)


def test_type_requires_text(obs):
    assert A.validate_action(Action(type="type", ref="@e2"), obs)
    assert A.validate_action(Action(type="type", ref="@e2", text="hi"), obs) is None


def test_select_requires_option(obs):
    assert A.validate_action(Action(type="select", ref="@e3"), obs)
    assert A.validate_action(Action(type="select", ref="@e3", option="M"), obs) is None


def test_navigate_requires_target(obs):
    assert A.validate_action(Action(type="navigate"), obs)
    assert A.validate_action(Action(type="navigate", target="back"), obs) is None


def test_note_requires_text(obs):
    assert A.validate_action(Action(type="note"), obs)            # empty -> error
    assert A.validate_action(Action(type="note", text="  "), obs) # whitespace -> error
    assert A.validate_action(Action(type="note", text="found: Catso"), obs) is None
    assert "note" in A.action_output_schema()["properties"]["action"]["properties"]["type"]["enum"]


def test_action_describe_roundtrip():
    a = Action(type="type", ref="@e2", text="hello")
    assert "type(" in a.describe() and "hello" in a.describe()
    assert Action.from_dict(a.to_dict()) == a
