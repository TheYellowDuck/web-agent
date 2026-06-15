"""Deterministic + assisted WebArena scorers.

WebArena ships ground-truth eval configs. We support its three families:

  * string_match  — exact_match / must_include over the agent's answer (deterministic)
  * fuzzy_match   — semantic inclusion, judged by an LLM (needs a judge model)
  * url_match      — the visited URL must match a reference URL
  * program_html  — DOM/content assertions on specific pages (best-effort here;
                    true evaluation needs the live page, so when we lack the
                    captured content we return *unscored* rather than guess)

Scoring uses 3-valued logic: a run is success only if every required check is
True; if any check is False it's a failure; if a required check can't be
evaluated (None) and nothing failed, the whole task is **unscored** (None) — we
never silently mark an unverifiable task as pass or fail.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent.llm import BaseLLMClient
from agent.types import Task, Trajectory
from eval.webarena.config import substitute


def score(
    task: Task, traj: Trajectory, *, judge_llm: Optional[BaseLLMClient] = None
) -> tuple[Optional[bool], dict[str, Any]]:
    spec = task.eval_spec or {}
    eval_types = spec.get("eval_types") or [spec.get("type", "string_match")]
    results: dict[str, Any] = {"eval_types": eval_types}
    verdicts: list[Optional[bool]] = []

    refs = spec.get("reference_answers", {}) or {}

    # String family: WebArena puts a fuzzy_match reference under eval_types
    # ["string_match"], and fuzzy_match REPLACES exact/must_include (it's the
    # task's reference type) — don't run both and AND them together.
    if "fuzzy_match" in refs or "fuzzy_match" in eval_types:
        ok, d = _fuzzy_match(spec, traj, judge_llm)
        verdicts.append(ok)
        results["fuzzy_match"] = d
    elif "string_match" in eval_types or "exact_match" in refs or "must_include" in refs:
        ok, d = _string_match(spec, traj)
        verdicts.append(ok)
        results["string_match"] = d
    if "url_match" in eval_types:
        ok, d = _url_match(spec, traj)
        verdicts.append(ok)
        results["url_match"] = d
    if "program_html" in eval_types:
        ok, d = _program_html(spec, traj)
        verdicts.append(ok)
        results["program_html"] = d

    if not verdicts:  # fall back to a plain answer comparison
        ok, d = _string_match(spec, traj)
        verdicts.append(ok)
        results["string_match"] = d

    return _combine(verdicts), results


def _combine(verdicts: list[Optional[bool]]) -> Optional[bool]:
    """3-valued AND: any False -> False; else any None -> None; else True."""
    if any(v is False for v in verdicts):
        return False
    if any(v is None for v in verdicts):
        return None
    return True


def _string_match(spec: dict[str, Any], traj: Trajectory) -> tuple[bool, dict[str, Any]]:
    answer = (traj.answer or "").strip()
    refs = spec.get("reference_answers", {}) or {}

    if "exact_match" in refs:
        target = substitute(str(refs["exact_match"]))
        return _norm(answer) == _norm(target), {
            "mode": "exact_match", "expected": target, "got": answer
        }

    musts = refs.get("must_include") or spec.get("must_include")
    if musts:
        musts = [substitute(str(m)) for m in musts]
        missing = [m for m in musts if _norm(m) not in _norm(answer)]
        return len(missing) == 0, {
            "mode": "must_include", "must_include": musts, "missing": missing, "got": answer
        }

    target = substitute(str(spec.get("answer", "")))
    return (bool(target) and _norm(target) in _norm(answer)), {
        "mode": "contains", "expected": target, "got": answer
    }


def _fuzzy_match(
    spec: dict[str, Any], traj: Trajectory, judge_llm: Optional[BaseLLMClient]
) -> tuple[Optional[bool], dict[str, Any]]:
    refs = spec.get("reference_answers", {}) or {}
    targets = refs.get("fuzzy_match") or spec.get("fuzzy_match") or []
    if isinstance(targets, str):
        targets = [targets]
    targets = [substitute(str(t)) for t in targets]
    answer = (traj.answer or "").strip()
    if not targets:
        return None, {"mode": "fuzzy_match", "note": "no fuzzy targets"}

    # WebArena uses "N/A" as the reference when the correct response is that the
    # requested information doesn't exist / the task is unachievable.
    is_na = len(targets) == 1 and _norm(targets[0]) in ("n/a", "na", "none")

    if judge_llm is None:
        # Degraded deterministic fallback so we still return *something* honest.
        missing = [t for t in targets if _norm(t) not in _norm(answer)]
        return (len(missing) == 0), {
            "mode": "fuzzy_match", "judge": None, "degraded": True,
            "targets": targets, "missing": missing, "got": answer,
        }

    schema = {
        "type": "object",
        "properties": {"satisfied": {"type": "boolean"}, "why": {"type": "string"}},
        "required": ["satisfied"],
    }
    if is_na:
        instruction = (
            "The correct answer is that the requested information does not exist / "
            "is not available / there is none (N/A). Does the answer below correctly "
            "convey that (e.g. 'no reviews', 'no phone number listed', 'none found')?"
        )
    else:
        instruction = f"Required meanings (all must be conveyed): {targets}\nDoes the answer satisfy ALL of them?"
    msg = [{
        "role": "user",
        "content": (
            f"Answer: {answer!r}\n\n{instruction}\n"
            'Respond JSON {"satisfied": bool, "why": str}.'
        ),
    }]
    resp = judge_llm.complete(
        system="You judge whether a free-text answer conveys required information.",
        messages=msg, json_schema=schema,
    )
    parsed = resp.parsed or {}
    return bool(parsed.get("satisfied", False)), {
        "mode": "fuzzy_match", "judge": judge_llm.model,
        "targets": targets, "why": parsed.get("why", ""),
    }


def _url_match(spec: dict[str, Any], traj: Trajectory) -> tuple[bool, dict[str, Any]]:
    ref = substitute(str(spec.get("reference_url", "")))
    visited = [s.url for s in traj.steps]
    mode = spec.get("url_note", "GOLD in PRED")
    if not ref:
        return False, {"reference_url": ref, "visited": visited[-3:]}
    ok = any(_url_eq(ref, u, mode) for u in visited)
    return ok, {"reference_url": ref, "visited": visited[-3:]}


def _program_html(spec: dict[str, Any], traj: Trajectory) -> tuple[Optional[bool], dict[str, Any]]:
    """Best-effort program_html: check required_contents against captured page text.

    True WebArena program_html runs locators/JS on specific URLs at scoring time
    (needs the live site). When we have captured page text we approximate the
    common ``required_contents`` check; otherwise we return None (unscored) so a
    task we can't verify isn't silently passed or failed.
    """
    targets = spec.get("program_html") or []
    # Gather whatever page content we captured during the run.
    captured = " ".join(
        (s.observation_text or "") for s in traj.steps if s.observation_text
    )
    captured = (captured + " " + (traj.answer or "")).strip()
    if not captured:
        return None, {"mode": "program_html", "note": "no captured page content; "
                      "needs live-page scoring", "unscored": True}

    required: list[str] = []
    for entry in targets:
        rc = (entry or {}).get("required_contents", {}) if isinstance(entry, dict) else {}
        required += [substitute(str(x)) for x in rc.get("must_include", [])]
    if not required:
        return None, {"mode": "program_html", "note": "no required_contents to check"}

    missing = [r for r in required if _norm(r) not in _norm(captured)]
    return (len(missing) == 0), {
        "mode": "program_html", "approximated": True,
        "required": required, "missing": missing,
    }


def _url_eq(ref: str, got: str, mode: str) -> bool:
    r, g = ref.rstrip("/"), got.rstrip("/")
    if mode == "exact":
        return r == g
    return r in g or g in r


# Map typographic punctuation to ASCII so a verbatim answer isn't failed over a
# curly vs straight quote (a real gotcha: WebArena ground truth uses “ ” ’ – —).
_PUNCT = str.maketrans({
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).translate(_PUNCT).strip().lower())
