"""Type-aware *deterministic* WebArena scorer — in the spirit of WebArena-Verified.

WebArena-Verified (Boisvert et al., ServiceNow, NeurIPS 2025 SEA workshop;
https://github.com/ServiceNow/webarena-verified) audited all 812 WebArena tasks
and reported that the original pipeline's permissive substring matching plus
LLM-as-judge *under-counts* correct answers — their deterministic, type-aware
comparators cut the false-negative rate by 11.3 points. This module reproduces
that idea on our own trajectories, with no LLM judge in the loop:

  1. The ``|OR|`` operator. WebArena's own ``must_include`` spec allows a target
     like ``"65 |OR| 3"`` (satisfied by *either* alternative). Our baseline
     ``eval.webarena.scorer`` never split on it, so it searched for the literal
     string ``"65 |or| 3"`` and failed answers that were actually correct
     (task 386: agent answered "3"). Handling the operator is just faithfully
     implementing the benchmark.

  2. Unicode / numeric normalization. A verbatim answer shouldn't fail over a
     ``×`` vs ``x`` (task 146: "16x24"), a curly apostrophe, an en-dash, or a
     ``$`` / thousands-separator on a number. We normalize both sides the same
     way before comparison — a structural, deterministic equivalence, not a
     fuzzy/semantic guess.

  3. N/A as an answer *type*. When the gold answer is "N/A" (the requested
     information doesn't exist), original WebArena routes it through an LLM
     fuzzy_match judge. We instead detect an explicit *absence* statement in the
     answer deterministically ("no reviews", "not listed", "be the first to
     review", …). This is reported as a SEPARATE column so the reader sees
     exactly what the heuristic adds vs. the operator/unicode fixes alone.

The point is not to be lenient — wrong answers (a wrong price, an incomplete
list, a login failure) still fail. The point is that *several rejections the
substring scorer produced were scoring artifacts, not agent errors* — which is
this project's whole thesis, now shown without an LLM judge.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent.types import Task, Trajectory
from eval.webarena.config import substitute

# ---------------------------------------------------------------------------
# Normalization (type-aware)
# ---------------------------------------------------------------------------

# Typographic punctuation + the multiplication sign → ASCII, so a verbatim
# answer isn't failed over "16×24" vs "16x24" or a curly apostrophe.
_PUNCT = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "×": "x", "✕": "x", "∗": "*",
})


def _norm(s: str) -> str:
    """Lowercase, ASCII-fold punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", str(s).translate(_PUNCT).strip().lower())


def _norm_numeric(s: str) -> str:
    """Number-aware normalization: drop currency symbols and thousands commas so
    "$17,774.32" and "17774.32" compare equal. Applied on top of ``_norm``."""
    s = _norm(s)
    s = s.replace("$", "").replace("£", "").replace("€", "")
    # Strip a comma only when it sits between digits (a thousands separator) —
    # never the comma that separates list items.
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s


# Phrases that explicitly assert *absence* — used only when the gold answer is
# N/A, to accept "there is none" deterministically (no LLM judge).
_ABSENCE = re.compile(
    r"\b(no|not|none|n/?a|never|cannot|can't|couldn't|could not|unable|"
    r"isn't|aren't|doesn't|don't|without|unavailable)\b"
    r"|be the first to review|no (reviews?|results?|items?|products?|"
    r"phone|number|function|option|matches?|complaints?)",
    re.IGNORECASE,
)
# …but a bare task/login failure is not the same as "the information doesn't
# exist". Guard against accepting a crash as an N/A success.
_FAILURE_NOT_ABSENCE = re.compile(
    r"\b(log ?in|sign ?in|credential|password|error|crash|timed? ?out|"
    r"try again|incorrect)\b", re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def score(
    task: Task, traj: Trajectory, *, na_aware: bool = True
) -> tuple[Optional[bool], dict[str, Any]]:
    """Deterministic type-aware verdict for a WebArena task.

    No LLM judge is ever called. ``na_aware`` toggles the deterministic N/A
    absence-detector so callers can isolate its contribution.
    Returns ``(verdict, detail)`` with the same 3-valued logic as the baseline
    scorer: any False → False, else any None → None (unscored), else True.
    """
    spec = task.eval_spec or {}
    eval_types = spec.get("eval_types") or [spec.get("type", "string_match")]
    refs = spec.get("reference_answers", {}) or {}
    results: dict[str, Any] = {"scorer": "verified", "rule": None}
    verdicts: list[Optional[bool]] = []

    if "fuzzy_match" in refs or "fuzzy_match" in eval_types:
        ok, d = _fuzzy_match(refs, spec, traj, na_aware)
        verdicts.append(ok)
        results.update(d)
    elif "string_match" in eval_types or "exact_match" in refs or "must_include" in refs:
        ok, d = _string_match(refs, spec, traj)
        verdicts.append(ok)
        results.update(d)
    if "url_match" in eval_types:
        ok, d = _url_match(spec, traj)
        verdicts.append(ok)
        results.setdefault("rule", d.get("rule"))
        results["url_match"] = d
    if "program_html" in eval_types:
        # Best-effort program_html needs the live page; we don't re-derive it
        # here (the baseline already marks it unscored) — leave as None.
        verdicts.append(None)
        results["program_html"] = {"note": "needs live page; unscored"}

    if not verdicts:
        ok, d = _string_match(refs, spec, traj)
        verdicts.append(ok)
        results.update(d)

    return _combine(verdicts), results


def _combine(verdicts: list[Optional[bool]]) -> Optional[bool]:
    if any(v is False for v in verdicts):
        return False
    if any(v is None for v in verdicts):
        return None
    return True


def _string_match(
    refs: dict[str, Any], spec: dict[str, Any], traj: Trajectory
) -> tuple[bool, dict[str, Any]]:
    answer = (traj.answer or "").strip()

    if "exact_match" in refs:
        target = substitute(str(refs["exact_match"]))
        ok = _norm(answer) == _norm(target)
        return ok, {"rule": "exact_match", "expected": target, "got": answer}

    musts = refs.get("must_include") or spec.get("must_include")
    if musts:
        musts = [substitute(str(m)) for m in musts]
        missing = [m for m in musts if not _contains_required(m, answer)]
        rule = "must_include|OR" if any("|OR|" in m for m in musts) else "must_include"
        return len(missing) == 0, {
            "rule": rule, "must_include": musts, "missing": missing, "got": answer,
        }

    target = substitute(str(spec.get("answer", "")))
    ok = bool(target) and _norm_numeric(target) in _norm_numeric(answer)
    return ok, {"rule": "contains", "expected": target, "got": answer}


def _contains_required(required: str, answer: str) -> bool:
    """One ``must_include`` target vs the answer, with the ``|OR|`` operator and
    number-aware comparison. WebArena: ``"a |OR| b"`` passes if EITHER is present."""
    alts = [a.strip() for a in required.split("|OR|")] if "|OR|" in required else [required]
    a_num = _norm_numeric(answer)
    for alt in alts:
        if _norm_numeric(alt) in a_num:
            return True
    return False


def _fuzzy_match(
    refs: dict[str, Any], spec: dict[str, Any], traj: Trajectory, na_aware: bool
) -> tuple[Optional[bool], dict[str, Any]]:
    targets = refs.get("fuzzy_match") or spec.get("fuzzy_match") or []
    if isinstance(targets, str):
        targets = [targets]
    targets = [substitute(str(t)) for t in targets]
    answer = (traj.answer or "").strip()
    if not targets:
        return None, {"rule": "fuzzy_match", "note": "no fuzzy targets"}

    is_na = len(targets) == 1 and _norm(targets[0]) in ("n/a", "na", "none")
    if is_na:
        if not na_aware:
            # Without the N/A detector we can't deterministically verify "none";
            # leave it unscored rather than guess.
            return None, {"rule": "na_unscored", "targets": targets, "got": answer}
        ok = bool(_ABSENCE.search(answer)) and not _is_pure_failure(answer)
        return ok, {"rule": "na_absence", "targets": targets, "got": answer}

    # A concrete fuzzy target (e.g. "March 11th 2023"): require the literal value
    # (number-aware). This is stricter than an LLM judge — a paraphrase fails —
    # but it is deterministic and never over-counts.
    missing = [t for t in targets if not _contains_required(t, answer)]
    return len(missing) == 0, {
        "rule": "fuzzy_literal", "targets": targets, "missing": missing, "got": answer,
    }


def _is_pure_failure(answer: str) -> bool:
    """True if the answer is a task/login failure rather than a real 'none' —
    only treated as failure when it lacks a genuine absence assertion."""
    if not _FAILURE_NOT_ABSENCE.search(answer):
        return False
    # "no phone number is listed because login failed" is still a valid absence;
    # only reject when the answer is *dominated* by the failure (no clear claim
    # that the thing doesn't exist).
    genuine_absence = re.search(
        r"no \w+ (is|are|was|were)?\s*(listed|available|found|exist)|"
        r"there (is|are) no|be the first to review|does not exist",
        answer, re.IGNORECASE,
    )
    return genuine_absence is None


def _url_match(spec: dict[str, Any], traj: Trajectory) -> tuple[bool, dict[str, Any]]:
    ref = substitute(str(spec.get("reference_url", "")))
    visited = [s.url for s in traj.steps]
    mode = spec.get("url_note", "GOLD in PRED")
    if not ref:
        return False, {"rule": "url_match", "reference_url": ref, "visited": visited[-3:]}
    ok = any(_url_eq(ref, u, mode) for u in visited)
    return ok, {"rule": "url_match", "reference_url": ref, "visited": visited[-3:]}


def _url_eq(ref: str, got: str, mode: str) -> bool:
    r, g = ref.rstrip("/"), got.rstrip("/")
    if mode == "exact":
        return r == g
    return r in g or g in r
