"""Agent Workflow Memory: induction, retrieval, prompt injection."""

from agent.prompts import build_planner_messages
from agent.observation import Observation
from agent.memory import Memory
from agent import workflow_memory as wm


def _traj(task_id, goal, success, actions):
    return {
        "task_id": task_id, "goal": goal, "success": success,
        "n_steps": len(actions),
        "steps": [{"action_ok": True, "action": a} for a in actions],
    }


def test_induce_only_from_successes_and_collapses_repeats():
    rows = [
        _traj("t1", "Search for usb wifi", True,
              [{"type": "type", "text": "usb wifi"}, {"type": "scroll"},
               {"type": "scroll"}, {"type": "click", "ref": "@e8"},
               {"type": "done", "answer": "ok"}]),
        _traj("t2", "A failed task", False, [{"type": "click", "ref": "@e1"}]),
    ]
    wfs = wm.induce_workflows(rows)
    assert len(wfs) == 1                                   # the failure is excluded
    w = wfs[0]
    assert w.steps == ['type the query "usb wifi"', "scroll for more",
                       "click the relevant control"]       # two scrolls collapsed
    assert "done" not in " ".join(w.steps)


def test_induce_keeps_shortest_per_goal():
    rows = [
        _traj("g#r1", "same goal", True, [{"type": "click"}, {"type": "click"},
                                          {"type": "navigate", "target": "x"}]),
        _traj("g#r2", "same goal", True, [{"type": "click"}]),   # shorter
    ]
    wfs = wm.induce_workflows(rows)
    assert len(wfs) == 1 and wfs[0].n_steps == 1


def test_select_relevant_by_keyword_overlap():
    wfs = wm.induce_workflows([
        _traj("a", "Search for chairs by price", True, [{"type": "type", "text": "chairs"}]),
        _traj("b", "Read the population of Canada", True, [{"type": "type", "text": "Canada"}]),
    ])
    got = wm.select_relevant(wfs, "Show me the laptops listed by price", k=1)
    assert len(got) == 1 and "chairs" in got[0].goal       # the price/listing routine


def test_select_excludes_own_task():
    wfs = wm.induce_workflows([
        _traj("task-5#r1", "list the reviews of widget", True, [{"type": "click"}]),
    ])
    # The task must not retrieve its own solution (any seed of it).
    assert wm.select_relevant(wfs, "list the reviews of widget",
                              exclude_task_id="task-5#r2") == []


def test_format_and_inject_into_planner():
    wfs = wm.induce_workflows([
        _traj("a", "Search for usb wifi", True,
              [{"type": "type", "text": "usb wifi"}, {"type": "click"}]),
    ])
    hint = wm.format_for_prompt(wfs)
    assert "Reusable routines" in hint and "usb wifi" in hint
    msg = build_planner_messages(
        "Search for bluetooth speaker", Observation(url="u", title="t"),
        Memory(goal="g"), workflows=hint,
    )[0]["content"]
    assert "Reusable routines" in msg                       # injected into the prompt


def test_workflow_roundtrip():
    w = wm.Workflow(goal="g", steps=["click the relevant control"], n_steps=3,
                    source_task_id="t", keywords={"g"})
    back = wm.Workflow.from_dict(w.to_dict())
    assert back.goal == "g" and back.steps == w.steps and back.n_steps == 3
