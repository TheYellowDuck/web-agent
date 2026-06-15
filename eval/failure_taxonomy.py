"""Classify failed trajectories into a small, interpretable taxonomy.

Categories (from the plan):
- wrong_element          — acted on the wrong element / action didn't help
- hallucinated_action    — emitted a ref that wasn't on the page, or malformed
- premature_done         — called done before the goal was met
- infinite_loop          — repeated the same action with no progress
- vision_fallback_failure— failed even after the screenshot fallback fired
- navigation_error       — page/navigation errors dominated
- max_steps              — ran out of step budget
- llm_error              — the model call itself failed
- other
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def classify(traj: dict[str, Any]) -> str:
    """Return a single category for a (presumed failed) trajectory dict."""
    if traj.get("success"):
        return "success"

    status = traj.get("status")
    steps = traj.get("steps", []) or []
    err = (traj.get("error") or "").lower()

    if status == "error" and "loop" in err:
        return "infinite_loop"
    if status == "error" and "consecutive failures" in err:
        # Why did it keep failing? Look at the dominant step error.
        return _dominant_step_failure(steps)
    if status == "error" and (
        "connection" in err or "net::" in err or "err_connection" in err
        or "goto" in err or "dns" in err
    ):
        # Site/infra unreachable (e.g. the sandbox container is down) — an
        # environment failure, not an agent failure. Flagged distinctly so these
        # aren't mistaken for capability problems.
        return "connection_error"
    if status == "error" and err:
        return "llm_error"

    if status == "done":
        # It thought it was finished but the scorer disagreed.
        return "premature_done"

    if status == "budget_exceeded":
        # Distinguish "kept making invalid moves" from "just slow".
        invalid = _count(steps, lambda s: s.get("action_error"))
        if invalid >= max(2, len(steps) // 2):
            return _dominant_step_failure(steps)
        return "max_steps"

    return "other"


def _dominant_step_failure(steps: list[dict[str, Any]]) -> str:
    errs = [s.get("action_error", "") for s in steps if s.get("action_error")]
    text = " ".join(errs).lower()
    used_vision = any(s.get("used_vision") for s in steps)
    if "not found in current snapshot" in text or "parse error" in text:
        return "hallucinated_action"
    if "failed:" in text or "timeout" in text:
        cat = "wrong_element"
    elif "navigate" in text:
        cat = "navigation_error"
    else:
        cat = "wrong_element"
    if used_vision:
        return "vision_fallback_failure"
    return cat


def _count(steps: list[dict[str, Any]], pred) -> int:
    return sum(1 for s in steps if pred(s))


def breakdown(trajectories: list[dict[str, Any]]) -> dict[str, int]:
    """Counts per category over the failed trajectories in a run."""
    counts: Counter[str] = Counter()
    for t in trajectories:
        if t.get("success"):
            continue
        counts[classify(t)] += 1
    return dict(counts)
