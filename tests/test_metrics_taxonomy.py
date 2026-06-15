"""Metrics aggregation (incl. CIs / multi-run) + failure taxonomy."""

from agent.types import Step, Trajectory
from eval import failure_taxonomy as ft
from eval import metrics


def _row(tid, *, success, status="done", steps=1, err=None, action_error=None,
         used_vision=False, difficulty="easy"):
    t = Trajectory(task_id=tid, goal="g", model="m", config={}, status=status, error=err)
    for i in range(steps):
        t.steps.append(Step(index=i, url="u", observation_hash="h", thought="",
                            action={"type": "click"}, action_ok=(action_error is None),
                            action_error=action_error, used_vision=used_vision))
    t.success = success
    t.score_detail = {"difficulty": difficulty, "benchmark": "local"}
    t.end_time = t.start_time + 1.0
    return t.to_dict()


# --- metrics ---------------------------------------------------------------

def test_success_rate_and_ci():
    rows = [_row("a", success=True), _row("b", success=False),
            _row("c", success=True), _row("d", success=False)]
    s = metrics.summarize(rows)
    assert s["success_rate"] == 0.5
    lo, hi = s["success_rate_ci95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_multi_run_grouping():
    rows = [_row("a#r1", success=True), _row("a#r2", success=False),
            _row("b#r1", success=False), _row("b#r2", success=False)]
    s = metrics.summarize(rows)
    assert s["n_unique_tasks"] == 2
    assert s["runs_per_task"] == 2.0
    assert s["pass_any_rate"] == 0.5  # task 'a' solved at least once


def test_difficulty_breakdown():
    rows = [_row("a", success=True, difficulty="easy"),
            _row("b", success=False, difficulty="hard")]
    s = metrics.summarize(rows)
    assert s["success_rate_by_difficulty"]["easy"]["success_rate"] == 1.0
    assert s["success_rate_by_difficulty"]["hard"]["success_rate"] == 0.0


def test_wilson_bounds():
    assert metrics.wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = metrics.wilson_interval(10, 10)
    assert hi == 1.0 and lo < 1.0


# --- taxonomy --------------------------------------------------------------

def test_premature_done():
    assert ft.classify(_row("a", success=False, status="done")) == "premature_done"


def test_infinite_loop():
    r = _row("a", success=False, status="error", err="stuck in a loop")
    assert ft.classify(r) == "infinite_loop"


def test_hallucinated_action():
    r = _row("a", success=False, status="budget_exceeded", steps=4,
             action_error="ref @e9 not found in current snapshot")
    assert ft.classify(r) == "hallucinated_action"


def test_connection_error_not_llm_error():
    # Site/infra unreachable must be flagged distinctly, not as an agent failure.
    r = _row("a", success=False, status="error",
             err="Error: Page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:7770/")
    assert ft.classify(r) == "connection_error"


def test_max_steps():
    assert ft.classify(_row("a", success=False, status="budget_exceeded", steps=15)) \
        == "max_steps"


def test_vision_fallback_failure():
    r = _row("a", success=False, status="error", err="too many consecutive failures",
             steps=3, action_error="click failed: timeout", used_vision=True)
    assert ft.classify(r) == "vision_fallback_failure"


def test_breakdown_skips_success():
    rows = [_row("a", success=True), _row("b", success=False, status="done")]
    bd = ft.breakdown(rows)
    assert bd == {"premature_done": 1}
