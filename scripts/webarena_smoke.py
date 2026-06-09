"""WebArena integration smoke — proves the adapter end-to-end with no real sites.

Points the __HOMEPAGE__ site at the bundled local static site, loads the
WebArena-schema sample tasks, drives a scripted agent over them, and scores with
the WebArena deterministic scorers (string_match / url_match). If this passes,
the WebArena code path (templating → loader → agent → scorer) works; bringing up
the real Docker sites + a real task file is then just configuration.

    python -m scripts.webarena_smoke
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SITE_DIR = Path(__file__).resolve().parent.parent / "eval" / "local_site"
# Wire the stand-in site BEFORE importing the loader (config reads env at call time).
os.environ["WA_HOMEPAGE"] = SITE_DIR.resolve().as_uri()

from agent.loop import Agent, AgentConfig  # noqa: E402
from agent.llm import ScriptedLLMClient  # noqa: E402
from eval.harness import score_trajectory  # noqa: E402
from eval.webarena.tasks import load_tasks  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "eval" / "webarena" / "sample_tasks.json"


def main() -> int:
    home = os.environ["WA_HOMEPAGE"]
    tasks = load_tasks(str(SAMPLE))
    print(f"Loaded {len(tasks)} WebArena sample task(s); __HOMEPAGE__ -> {home}\n")
    if not tasks:
        print("No tasks loaded (templating/skip logic failed).")
        return 1

    plans = {
        "webarena-9001": [{"type": "done", "answer": "The Blue Widget is priced at $19.99."}],
        "webarena-9002": [
            {"type": "navigate", "target": f"{home}/contact.html"},
            {"type": "done", "answer": "Opened the contact page."},
        ],
    }

    ok_all = True
    for task in tasks:
        agent = Agent(ScriptedLLMClient(plans[task.task_id]), AgentConfig(max_steps=4))
        traj = agent.run(task)
        score_trajectory(task, traj)
        ok = bool(traj.success)
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {task.task_id} "
              f"start={task.start_url.split('/')[-1]} -> success={traj.success} "
              f"({traj.score_detail.get('string_match') or traj.score_detail.get('url_match')})")

    print("\nWEBARENA INTEGRATION OK" if ok_all else "\nWEBARENA INTEGRATION FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
