"""WebJudge-style LLM-as-judge for the realistic slice.

Open-web tasks have no ground-truth string to match, so an LLM judges whether
the trajectory accomplished the goal. The judge sees the goal, the agent's
extracted answer, its action sequence, the URLs it visited, and (if captured)
the final-page element summary — then returns a structured verdict.

This is deliberately a *separate* model call from the agent so the judge can be
a different (often stronger) model, and so scoring is re-runnable offline.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Optional

from agent.llm import BaseLLMClient, make_llm_client, resolve_tier
from agent.types import Task, Trajectory

JUDGE_SYSTEM = """\
You are a strict evaluator of web-agent task completion. Given a user's goal and
a transcript of what an agent did, decide whether the goal was actually achieved.
Be skeptical: an agent claiming success is not evidence of success. Reward only
trajectories whose actions plausibly accomplish the goal and whose final answer
(if the task asks for information) is correct and specific. Penalize premature
stops, wrong pages, and vague or unsupported answers."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "confidence": {"type": "number", "description": "0.0–1.0"},
        "reasoning": {"type": "string"},
    },
    "required": ["success", "reasoning"],
}


def judge(
    task: Task,
    traj: Trajectory,
    *,
    judge_llm: Optional[BaseLLMClient] = None,
    max_screenshots: int = 3,
) -> tuple[bool, dict[str, Any]]:
    judge_llm = judge_llm or _default_judge()
    transcript = _render_transcript(traj)
    images = _final_screenshots(traj, max_screenshots)
    vision_note = (
        f"\n\n{len(images)} screenshot(s) of the agent's final page(s) are "
        "attached — rely on them over the agent's self-report."
        if images
        else ""
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"GOAL:\n{task.goal}\n\n"
                f"AGENT FINAL ANSWER:\n{traj.answer or '(none)'}\n\n"
                f"TRANSCRIPT:\n{transcript}{vision_note}\n\n"
                "Did the agent accomplish the goal? Respond with JSON "
                '{"success": bool, "confidence": number, "reasoning": str}.'
            ),
        }
    ]
    resp = judge_llm.complete(
        system=JUDGE_SYSTEM, messages=messages, json_schema=_SCHEMA, images_b64=images
    )
    parsed = resp.parsed or {}
    success = bool(parsed.get("success", False))
    return success, {
        "scorer": "webjudge",
        "judge_model": judge_llm.model,
        "n_screenshots": len(images),
        "confidence": parsed.get("confidence"),
        "reasoning": parsed.get("reasoning", ""),
        "judge_cost_usd": resp.cost_usd,
    }


def _final_screenshots(traj: Trajectory, k: int) -> list[str]:
    """Base64 of the last ``k`` captured screenshots, if any were saved."""
    paths = [s.screenshot_path for s in traj.steps if s.screenshot_path]
    out: list[str] = []
    for p in paths[-k:]:
        try:
            out.append(base64.b64encode(Path(p).read_bytes()).decode("ascii"))
        except Exception:
            pass
    return out


def _render_transcript(traj: Trajectory, max_steps: int = 20) -> str:
    lines: list[str] = []
    for s in traj.steps[:max_steps]:
        a = s.action or {}
        atype = a.get("type", "?")
        detail = {k: v for k, v in a.items() if k != "type"}
        status = "ok" if s.action_ok else f"FAILED({s.action_error})"
        lines.append(f"  {s.index}. [{s.url}] {atype} {detail} -> {status}")
    lines.append(f"  final status: {traj.status}")
    return "\n".join(lines)


def _default_judge() -> BaseLLMClient:
    model = os.environ.get("WEBJUDGE_MODEL")
    model = resolve_tier(model) if model else resolve_tier("frontier")
    return make_llm_client(model)
