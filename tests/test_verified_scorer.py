"""Tests for the type-aware deterministic scorer (eval.webarena.verified).

These pin the specific false-negative classes the scorer is meant to recover —
the |OR| operator, unicode/number folding, and deterministic N/A — while
confirming it never starts passing genuinely wrong answers (no over-counting).
"""

from agent.types import Step, Task, Trajectory
from eval.webarena import verified


def _task(refs, eval_types=("string_match",)):
    return Task(task_id="webarena-x", goal="g", start_url="x", benchmark="webarena",
                eval_spec={"eval_types": list(eval_types), "reference_answers": refs})


def _traj(answer=None, urls=()):
    t = Trajectory(task_id="t", goal="g", model="m", config={}, answer=answer)
    for i, u in enumerate(urls):
        t.steps.append(Step(index=i, url=u, observation_hash="h", thought="",
                            action={"type": "navigate"}, action_ok=True))
    return t


# --- the |OR| operator (task 386: ref "65 |OR| 3", agent answered "3") --------

def test_or_operator_passes_either_alternative():
    task = _task({"must_include": ["65 |OR| 3"]})
    assert verified.score(task, _traj(answer="3"))[0] is True
    assert verified.score(task, _traj(answer="65"))[0] is True
    assert verified.score(task, _traj(answer="the rating is 7"))[0] is False


def test_baseline_substring_misses_the_or_operator():
    # Regression contrast: the shipped scorer can't parse |OR|, so it rejects a
    # correct answer — exactly the false-negative the verified scorer recovers.
    from eval.webarena import scorer as baseline
    task = _task({"must_include": ["65 |OR| 3"]})
    assert baseline.score(task, _traj(answer="3"))[0] is False
    assert verified.score(task, _traj(answer="3"))[0] is True


# --- unicode / number folding (task 146: "16x24") -----------------------------

def test_unicode_multiplication_sign_folds_to_x():
    task = _task({"must_include": ["16x24"]})
    assert verified.score(task, _traj(answer="the size is 16×24"))[0] is True


def test_currency_and_thousands_separators_normalized():
    task = _task({"must_include": ["745.00"]})
    assert verified.score(task, _traj(answer="$745.00"))[0] is True
    task2 = _task({"must_include": ["17774.32"]})
    assert verified.score(task2, _traj(answer="up to $17,774.32"))[0] is True


def test_plain_must_include_still_works():
    task = _task({"must_include": ["red", "large"]})
    assert verified.score(task, _traj(answer="a large red shirt"))[0] is True
    assert verified.score(task, _traj(answer="a large shirt"))[0] is False


# --- deterministic N/A (tasks 225/313/376) ------------------------------------

def test_na_absence_detected_when_aware():
    task = _task({"fuzzy_match": "N/A"})
    assert verified.score(task, _traj(answer="There are no customer reviews yet."))[0] is True
    assert verified.score(task, _traj(answer="No phone number is listed."))[0] is True


def test_na_unscored_when_not_aware():
    # Without the heuristic we can't verify 'none' deterministically -> unscored.
    task = _task({"fuzzy_match": "N/A"})
    assert verified.score(task, _traj(answer="There are no reviews."), na_aware=False)[0] is None


def test_na_does_not_pass_a_login_failure():
    # An N/A task answered with a crash/login failure is NOT a valid 'none'.
    task = _task({"fuzzy_match": "N/A"})
    ans = "I was unable to determine this because I could not log in (incorrect credentials)."
    assert verified.score(task, _traj(answer=ans))[0] is False


def test_na_does_not_pass_a_real_answer():
    # Agent that confidently lists items is not asserting absence -> stays fail.
    task = _task({"fuzzy_match": "N/A"})
    assert verified.score(task, _traj(answer="The discounted items are A, B and C."))[0] is False


# --- no over-counting: wrong answers still fail -------------------------------

def test_wrong_price_range_still_fails():
    task = _task({"must_include": ["0.14", "745.00"]})
    assert verified.score(task, _traj(answer="from $4.99 to $17,774.32"))[0] is False


def test_concrete_fuzzy_target_requires_the_value():
    task = _task({"fuzzy_match": ["March 11th 2023"]})
    assert verified.score(task, _traj(answer="ordered on March 11th 2023"))[0] is True
    assert verified.score(task, _traj(answer="I could not find the order"))[0] is False


# --- exact_match + url_match -------------------------------------------------

def test_exact_match_normalized():
    task = _task({"exact_match": "16x24"})
    assert verified.score(task, _traj(answer="16×24"))[0] is True
    assert verified.score(task, _traj(answer="16x25"))[0] is False


def test_url_match():
    task = _task({}, eval_types=("url_match",))
    task.eval_spec["reference_url"] = "http://site/order/5"
    assert verified.score(task, _traj(urls=["http://site/order/5"]))[0] is True
    assert verified.score(task, _traj(urls=["http://site/order/9"]))[0] is False


# --- Trajectory.from_dict round-trip (the offline re-score depends on it) -----

def test_trajectory_from_dict_roundtrip():
    t = _traj(answer="hello", urls=["http://s/a", "http://s/b"])
    t.success = False
    t.answer_grounding = 0.7
    back = Trajectory.from_dict(t.to_dict())
    assert back.answer == "hello"
    assert back.answer_grounding == 0.7
    assert [s.url for s in back.steps] == ["http://s/a", "http://s/b"]
    # The verified scorer must produce the same verdict on the reconstructed obj.
    task = _task({}, eval_types=("url_match",))
    task.eval_spec["reference_url"] = "http://s/b"
    assert verified.score(task, back)[0] is True
