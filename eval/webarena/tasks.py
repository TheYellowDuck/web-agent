"""Load WebArena tasks from standard WebArena task JSON.

Point ``WEBARENA_TASKS`` (or pass a path) at a WebArena config file — either the
upstream ``test.raw.json`` array or a per-task file. URLs and reference answers
are templated against your self-hosted site base URLs (see config.py).

We don't bundle the WebArena dataset (it's large and lives upstream); this is
the adapter that turns it into our ``Task`` objects with deterministic scoring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from agent.types import Task
from eval.webarena.config import missing_sites, substitute


def load_tasks(path: Optional[str] = None) -> list[Task]:
    path = path or os.environ.get("WEBARENA_TASKS")
    if not path:
        raise RuntimeError(
            "Set WEBARENA_TASKS to a WebArena task JSON file (or pass --tasks "
            "<path>). The dataset is not bundled; see README → WebArena setup."
        )
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else [raw]
    return [t for t in (_convert(r) for r in rows) if t is not None]


def _convert(row: dict[str, Any]) -> Optional[Task]:
    start_url = substitute(row.get("start_url", ""))
    if not start_url or missing_sites(row.get("start_url", "")):
        # A required site base URL isn't configured — skip rather than 404.
        return None

    eval_block = row.get("eval", {}) or {}
    eval_spec = {
        "eval_types": eval_block.get("eval_types", ["string_match"]),
        "reference_answers": eval_block.get("reference_answers", {}),
        "reference_url": eval_block.get("reference_url", ""),
        "url_note": eval_block.get("url_note", "GOLD in PRED"),
    }
    intent = row.get("intent", "")
    return Task(
        task_id=f"webarena-{row.get('task_id', row.get('id', 'na'))}",
        goal=intent,
        start_url=start_url,
        benchmark="webarena",
        difficulty=_difficulty_from(row),
        reference_steps=row.get("reference_length"),
        eval_spec=eval_spec,
        metadata={
            "sites": row.get("sites", []),
            "require_login": bool(row.get("require_login", False)),
            "require_reset": bool(row.get("require_reset", False)),
        },
    )


def _difficulty_from(row: dict[str, Any]) -> str:
    # WebArena doesn't ship a tier; use reference length if present as a proxy.
    n = row.get("reference_length")
    if isinstance(n, int):
        if n <= 5:
            return "easy"
        if n <= 10:
            return "medium"
        return "hard"
    return "unknown"
