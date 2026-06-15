"""Scorers: generic (local/custom) + WebArena deterministic + 3-valued logic."""

from agent.types import Step, Task, Trajectory
from eval.harness import _generic_score
from eval.webarena import scorer as wa


def _traj(answer=None, urls=()):
    t = Trajectory(task_id="t", goal="g", model="m", config={}, answer=answer)
    for i, u in enumerate(urls):
        t.steps.append(Step(index=i, url=u, observation_hash="h", thought="",
                            action={"type": "navigate"}, action_ok=True))
    return t


# --- generic ---------------------------------------------------------------

def test_generic_answer_contains():
    task = Task(task_id="t", goal="g", start_url="x",
                eval_spec={"type": "answer_match", "answer": "19.99", "mode": "contains"})
    ok, _ = _generic_score(task, _traj(answer="It is $19.99"))
    assert ok is True


def test_generic_answer_exact_fail():
    task = Task(task_id="t", goal="g", start_url="x",
                eval_spec={"type": "answer_match", "answer": "yes", "mode": "exact"})
    ok, _ = _generic_score(task, _traj(answer="yes indeed"))
    assert ok is False


def test_generic_url_contains():
    task = Task(task_id="t", goal="g", start_url="x",
                eval_spec={"type": "url_contains", "value": "products.html"})
    ok, _ = _generic_score(task, _traj(urls=["a/index.html", "a/products.html"]))
    assert ok is True


# --- webarena --------------------------------------------------------------

def test_wa_exact_match():
    task = Task(task_id="t", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": ["string_match"],
                           "reference_answers": {"exact_match": "42"}})
    ok, _ = wa.score(task, _traj(answer="42"))
    assert ok is True
    ok, _ = wa.score(task, _traj(answer="43"))
    assert ok is False


def test_wa_must_include():
    task = Task(task_id="t", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": ["string_match"],
                           "reference_answers": {"must_include": ["red", "large"]}})
    assert wa.score(task, _traj(answer="a large red shirt"))[0] is True
    assert wa.score(task, _traj(answer="a large shirt"))[0] is False


def test_wa_url_match():
    task = Task(task_id="t", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": ["url_match"],
                           "reference_url": "http://site/order/5"})
    assert wa.score(task, _traj(urls=["http://site/order/5"]))[0] is True
    assert wa.score(task, _traj(urls=["http://site/order/9"]))[0] is False


def test_wa_fuzzy_degraded_without_judge():
    task = Task(task_id="t", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": ["fuzzy_match"],
                           "reference_answers": {"fuzzy_match": ["sunny"]}})
    ok, detail = wa.score(task, _traj(answer="it is sunny today"))
    assert ok is True and detail["fuzzy_match"]["degraded"] is True


def test_wa_fuzzy_reference_not_dragged_down_by_string_match():
    # WebArena lists eval_types ["string_match"] with a fuzzy_match reference.
    # Fuzzy must REPLACE string_match (not be AND-ed with an empty string check),
    # else a correct fuzzy answer can never pass. (Regression test.)
    task = Task(task_id="t", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": ["string_match"],
                           "reference_answers": {"fuzzy_match": ["sunny"]}})
    ok, detail = wa.score(task, _traj(answer="the weather is sunny"))
    assert ok is True
    assert "string_match" not in detail  # only fuzzy ran


def test_wa_program_html_unscored_without_content():
    # No captured page content -> must be unscored (None), never a silent verdict.
    task = Task(task_id="t", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": ["program_html"],
                           "program_html": [{"required_contents": {"must_include": ["X"]}}]})
    ok, _ = wa.score(task, _traj(answer=""))
    assert ok is None


def test_string_match_normalizes_typographic_punctuation():
    # Ground truth has a curly apostrophe; answer has a straight one — must pass.
    task = Task(task_id="webarena-x", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": ["string_match"],
                           "reference_answers": {"must_include": ["wasn’t able to format"]}})
    traj = Trajectory(task_id="t", goal="g", model="m", config={},
                      answer="The user wasn't able to format the card")
    assert wa.score(task, traj)[0] is True


def test_combine_three_valued():
    assert wa._combine([True, True]) is True
    assert wa._combine([True, False]) is False
    assert wa._combine([True, None]) is None
    assert wa._combine([False, None]) is False
