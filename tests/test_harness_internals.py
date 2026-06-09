"""Harness internals: multi-run expansion + rate-limit detection."""

from agent.types import Task
from eval.harness import _expand_jobs, _is_rate_limit


def _tasks(n=2):
    return [Task(task_id=f"t{i}", goal="g", start_url="x") for i in range(n)]


def test_expand_single_run_is_identity():
    ts = _tasks(2)
    out = _expand_jobs(ts, 1)
    assert [t.task_id for t in out] == ["t0", "t1"]


def test_expand_multi_run_suffixes():
    out = _expand_jobs(_tasks(2), 3)
    ids = [t.task_id for t in out]
    assert ids == ["t0#r1", "t0#r2", "t0#r3", "t1#r1", "t1#r2", "t1#r3"]
    # base fields preserved
    assert all(t.goal == "g" for t in out)


def test_is_rate_limit():
    assert _is_rate_limit("RateLimitError: 429 ...")
    assert _is_rate_limit("Error code: 429")
    assert not _is_rate_limit("BadRequestError: 400")
    assert not _is_rate_limit(None)
