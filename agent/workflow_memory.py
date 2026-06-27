"""Agent Workflow Memory (AWM) — induce reusable routines from past runs.

After Wang et al., "Agent Workflow Memory" (ICML 2025, arXiv:2409.07429): extract
the action sequence that solved a task into a compact, reusable *workflow*, then
inject the most relevant ones into the planner prompt for a new task. The paper
reports +51.1% relative success on WebArena *while reducing steps to solve*.

Two scopes (same as the paper):
  • offline — induce from a corpus of already-solved trajectories (this module's
    `induce_workflows`), then run with that memory fixed.
  • online — induce on the fly from tasks solved earlier in the same run.

Honest limitation here: our saved steps record the action and its portable args
(query text, select option, nav target, key) but NOT the element name behind a
ref (refs aren't portable across pages), so induced workflows capture the action
*shape* of a solution, not click-by-click targets. That's still a useful routine
("to search: type the query, then submit") and matches AWM's abstraction spirit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Actions that carry portable (page-independent) intent worth recording.
_PORTABLE = {"type", "select", "navigate", "press", "scroll", "note", "click"}
_STOP = {"the", "a", "an", "of", "for", "in", "to", "and", "or", "me", "my", "i",
         "is", "are", "what", "show", "find", "list", "all", "from", "with", "on"}


@dataclass
class Workflow:
    """A reusable routine induced from one solved task."""
    goal: str
    steps: list[str]                       # compact, portable action descriptions
    n_steps: int
    source_task_id: str
    keywords: set[str] = field(default_factory=set)

    def render(self) -> str:
        body = " → ".join(self.steps) if self.steps else "(no actions)"
        return f"- For \"{self.goal}\" ({self.n_steps} steps): {body}"

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "steps": self.steps, "n_steps": self.n_steps,
                "source_task_id": self.source_task_id, "keywords": sorted(self.keywords)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Workflow":
        return cls(goal=d["goal"], steps=list(d.get("steps", [])),
                   n_steps=int(d.get("n_steps", 0)),
                   source_task_id=d.get("source_task_id", "?"),
                   keywords=set(d.get("keywords") or _keywords(d.get("goal", ""))))


def _keywords(goal: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", goal.lower())
            if len(w) > 2 and w not in _STOP}


def _portable_action(action: dict[str, Any]) -> Optional[str]:
    """Compact, page-independent description of a step's action (or None to skip)."""
    t = action.get("type")
    if t not in _PORTABLE:
        return None
    if t == "type":
        return f'type the query "{action.get("text", "")}"'
    if t == "select":
        return f'select "{action.get("option", "")}"'
    if t == "navigate":
        tgt = action.get("target", "")
        return "go back" if tgt in ("back", "forward") else "navigate to a url"
    if t == "press":
        return f'press {action.get("key", "")}'
    if t == "scroll":
        return "scroll for more"
    if t == "note":
        return "record a finding"
    if t == "click":
        return "click the relevant control"
    return None


def induce_workflows(trajectories: list[dict[str, Any]]) -> list[Workflow]:
    """Build one workflow per *successful* trajectory (deduping repeated routines).

    Consecutive identical actions are collapsed (e.g. paging) so the routine reads
    as a pattern, not a transcript.
    """
    # Keep the *shortest* solving routine per goal — the canonical workflow, not
    # every seed's transcript (consolidation, in AWM's spirit).
    best: dict[str, Workflow] = {}
    for r in trajectories:
        if not r.get("success"):
            continue
        steps: list[str] = []
        for s in r.get("steps", []):
            if not s.get("action_ok"):
                continue
            desc = _portable_action(s.get("action", {}))
            if desc and (not steps or steps[-1] != desc):  # collapse repeats
                steps.append(desc)
        if not steps:
            continue
        goal = r.get("goal", "")
        n = int(r.get("n_steps", len(steps)))
        wf = Workflow(goal=goal, steps=steps, n_steps=n,
                      source_task_id=r.get("task_id", "?"), keywords=_keywords(goal))
        prev = best.get(goal)
        if prev is None or (n, len(steps)) < (prev.n_steps, len(prev.steps)):
            best[goal] = wf
    return list(best.values())


def select_relevant(
    workflows: list[Workflow], goal: str, *, k: int = 3, exclude_task_id: str = ""
) -> list[Workflow]:
    """Top-k workflows by keyword overlap with the goal (Jaccard), excluding the
    task's own workflow so we never leak its solution back to it."""
    gk = _keywords(goal)
    base = re.sub(r"#r\d+$", "", exclude_task_id)
    scored = []
    for w in workflows:
        if re.sub(r"#r\d+$", "", w.source_task_id) == base and base:
            continue
        union = gk | w.keywords
        sim = len(gk & w.keywords) / len(union) if union else 0.0
        if sim > 0:
            scored.append((sim, w))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:k]]


def format_for_prompt(workflows: list[Workflow]) -> str:
    if not workflows:
        return ""
    lines = ["Reusable routines from similar tasks solved before (use as a guide, "
             "adapt to the current page):"]
    lines += [w.render() for w in workflows]
    return "\n".join(lines)
