"""WebArena adapter: URL templating, task loading, and skip-when-unconfigured."""

import json

from agent.types import Step, Trajectory
from eval.webarena import config
from eval.webarena import scorer as wa
from eval.webarena.tasks import load_tasks

SAMPLE = [
    {
        "task_id": 1,
        "intent": "What is the price of the Blue Widget?",
        "start_url": "__HOMEPAGE__/products.html",
        "sites": ["homepage"],
        "reference_length": 2,
        "eval": {"eval_types": ["string_match"],
                 "reference_answers": {"must_include": ["19.99"]}},
    }
]


def test_substitute(monkeypatch):
    monkeypatch.setenv("WA_HOMEPAGE", "http://site:7770/")
    assert config.substitute("__HOMEPAGE__/x") == "http://site:7770/x"


def test_missing_sites_reported(monkeypatch):
    monkeypatch.delenv("WA_HOMEPAGE", raising=False)
    assert config.missing_sites("__HOMEPAGE__/x") == ["__HOMEPAGE__"]


def test_load_templates_and_converts(tmp_path, monkeypatch):
    monkeypatch.setenv("WA_HOMEPAGE", "http://site:7770")
    f = tmp_path / "wa.json"
    f.write_text(json.dumps(SAMPLE))
    tasks = load_tasks(str(f))
    assert len(tasks) == 1
    t = tasks[0]
    assert t.start_url == "http://site:7770/products.html"
    assert t.benchmark == "webarena"
    assert t.difficulty == "easy"  # reference_length 2 -> easy
    assert t.eval_spec["reference_answers"]["must_include"] == ["19.99"]


def test_load_skips_unconfigured_site(tmp_path, monkeypatch):
    # No WA_HOMEPAGE -> the task's site is unresolved, so it's skipped (not 404'd).
    monkeypatch.delenv("WA_HOMEPAGE", raising=False)
    f = tmp_path / "wa.json"
    f.write_text(json.dumps(SAMPLE))
    assert load_tasks(str(f)) == []


def test_scorer_must_include(monkeypatch):
    monkeypatch.setenv("WA_HOMEPAGE", "http://site:7770")
    f_task = load_tasks  # noqa: F841 (clarity)
    from agent.types import Task
    task = Task(task_id="webarena-1", goal="g", start_url="http://site:7770/products.html",
                benchmark="webarena",
                eval_spec={"eval_types": ["string_match"],
                           "reference_answers": {"must_include": ["19.99"]}})
    traj = Trajectory(task_id="t", goal="g", model="m", config={}, answer="It is $19.99")
    ok, _ = wa.score(task, traj)
    assert ok is True
    traj.answer = "It is $42.00"
    assert wa.score(task, traj)[0] is False


def test_scorer_url_match():
    from agent.types import Task
    task = Task(task_id="webarena-2", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": ["url_match"],
                           "reference_url": "http://site/contact"})
    traj = Trajectory(task_id="t", goal="g", model="m", config={})
    traj.steps.append(Step(index=0, url="http://site/contact", observation_hash="h",
                          thought="", action={"type": "navigate"}, action_ok=True))
    assert wa.score(task, traj)[0] is True
