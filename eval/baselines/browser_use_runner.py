"""Run a task with vanilla `browser-use` and adapt its output to our Trajectory.

This lets browser-use sit as a third line next to our scaffold (reflection
OFF / ON). It's intentionally thin: we drive browser-use with its own defaults
and only translate the result into the common record so the *same* scorers and
metrics apply.

browser-use's API shifts between releases; this adapter is defensive and will
raise a clear message if the installed version doesn't match. Pin a version in
the `baseline` extra and adjust here if needed.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agent.types import Step, Task, Trajectory


def available() -> bool:
    try:
        import browser_use  # noqa: F401

        return True
    except Exception:
        return False


def run_baseline(
    task: Task, *, model: str = "gpt-4o-mini", max_steps: int = 15
) -> Trajectory:
    """Execute one task with browser-use; return a Trajectory we can score."""
    import asyncio

    traj = Trajectory(
        task_id=f"baseline-{task.task_id}",
        goal=task.goal,
        model=f"browser-use:{model}",
        config={"baseline": "browser-use", "max_steps": max_steps},
    )
    try:
        history = asyncio.run(_run(task, model, max_steps))
        _ingest(history, traj)
        traj.status = "done"
    except Exception as e:
        traj.status = "error"
        traj.error = f"{type(e).__name__}: {e}"
    finally:
        traj.end_time = time.time()
    return traj


async def _run(task: Task, model: str, max_steps: int) -> Any:
    from browser_use import Agent  # type: ignore

    llm = _make_llm(model)
    agent = Agent(task=f"{task.goal}\nStart at: {task.start_url}", llm=llm)
    return await agent.run(max_steps=max_steps)


def _make_llm(model: str) -> Any:
    """Build whatever chat-model object the installed browser-use expects."""
    lower = model.lower()
    if lower.startswith("claude"):
        from browser_use import ChatAnthropic  # type: ignore

        return ChatAnthropic(model=model)
    if lower.startswith("gemini"):
        from browser_use import ChatGoogle  # type: ignore

        return ChatGoogle(model=model)
    from browser_use import ChatOpenAI  # type: ignore

    return ChatOpenAI(model=model)


def _ingest(history: Any, traj: Trajectory) -> None:
    """Best-effort extraction of answer / steps / urls from a browser-use run."""
    # Final answer
    for getter in ("final_result", "final_answer"):
        fn = getattr(history, getter, None)
        if callable(fn):
            try:
                traj.answer = fn()
                break
            except Exception:
                pass

    urls = _safe_call(history, "urls") or []
    actions = _safe_call(history, "action_names") or []
    n = max(len(urls), len(actions))
    for i in range(n):
        traj.steps.append(
            Step(
                index=i,
                url=str(urls[i]) if i < len(urls) else "",
                observation_hash="",
                thought="",
                action={"type": str(actions[i]) if i < len(actions) else "unknown"},
                action_ok=True,
            )
        )


def _safe_call(obj: Any, name: str) -> Optional[Any]:
    fn = getattr(obj, name, None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return None
    return None
