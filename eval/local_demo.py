"""A tiny, fully-offline demo benchmark.

Three deterministic tasks against the bundled static site (served via
``file://`` — no Docker, no network, no API key needed for the *site*). Use it
to prove the whole pipeline works end-to-end and as a CI smoke target.
"""

from __future__ import annotations

from pathlib import Path

from agent.types import Task

SITE_DIR = Path(__file__).resolve().parent / "local_site"


def _url(page: str) -> str:
    return (SITE_DIR / page).resolve().as_uri()


def load_local_tasks() -> list[Task]:
    return [
        Task(
            task_id="local-price",
            goal="Find the price of the Blue Widget and report it.",
            start_url=_url("index.html"),
            benchmark="local",
            difficulty="easy",
            reference_steps=2,
            eval_spec={"type": "answer_match", "answer": "19.99", "mode": "contains"},
        ),
        Task(
            task_id="local-email",
            goal="What is Acme's support email address?",
            start_url=_url("index.html"),
            benchmark="local",
            difficulty="easy",
            reference_steps=2,
            eval_spec={
                "type": "answer_match",
                "answer": "support@acme.example",
                "mode": "contains",
            },
        ),
        Task(
            task_id="local-nav",
            goal="Open the Products page.",
            start_url=_url("index.html"),
            benchmark="local",
            difficulty="easy",
            reference_steps=1,
            eval_spec={"type": "url_contains", "value": "products.html"},
        ),
    ]
