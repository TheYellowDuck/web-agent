"""Load the frozen Online-Mind2Web slice.

The slice lives in ``eval/tasks/mind2web_slice.json`` — a curated, stratified
~30-task set (easy 1–5 steps / medium 6–10 / hard 11+). Freezing it keeps the
realistic number reproducible across runs.

A tiny starter slice is bundled; drop in the official Online-Mind2Web tasks
(same schema) to expand it.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.types import Task

SLICE_PATH = Path(__file__).resolve().parent.parent / "tasks" / "mind2web_slice.json"


def load_slice(path: str | Path | None = None) -> list[Task]:
    p = Path(path) if path else SLICE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Mind2Web slice not found at {p}. See README → Online-Mind2Web."
        )
    rows = json.loads(p.read_text(encoding="utf-8"))
    rows = rows["tasks"] if isinstance(rows, dict) else rows
    tasks: list[Task] = []
    for r in rows:
        r.setdefault("benchmark", "mind2web")
        r.setdefault("eval_spec", {"type": "webjudge"})
        tasks.append(Task.from_dict(r))
    return tasks
