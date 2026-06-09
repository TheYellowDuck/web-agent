"""Shared dataclasses used across the agent and the eval harness.

Kept dependency-free so both `agent.*` and `eval.*` can import these without
pulling in Playwright or any LLM SDK.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

ActionType = Literal[
    "click",
    "type",
    "select",
    "scroll",
    "navigate",
    "wait",
    "done",
]

ALL_ACTION_TYPES: tuple[str, ...] = (
    "click",
    "type",
    "select",
    "scroll",
    "navigate",
    "wait",
    "done",
)


@dataclass
class Action:
    """A single typed action the agent can take.

    Only the fields relevant to ``type`` are populated; the rest stay ``None``.
    Validation against the live snapshot happens in ``agent.actions``.
    """

    type: ActionType
    ref: Optional[str] = None          # element ref, e.g. "@e12" (click/type/select)
    text: Optional[str] = None         # text to type
    option: Optional[str] = None       # option label/value for select
    direction: Optional[str] = None    # "up" | "down" | "left" | "right" for scroll
    target: Optional[str] = None       # url | "back" | "forward" for navigate
    ms: Optional[int] = None           # wait duration in milliseconds
    answer: Optional[str] = None       # extracted answer for done

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Action":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def describe(self) -> str:
        d = self.to_dict()
        t = d.pop("type")
        if not d:
            return t
        args = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"{t}({args})"


# ---------------------------------------------------------------------------
# Steps & trajectories
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One iteration of the agent loop, fully logged for offline re-scoring."""

    index: int
    url: str
    observation_hash: str
    thought: str
    action: dict[str, Any]
    action_ok: bool
    action_error: Optional[str] = None
    # Reflection (only present when reflection is enabled)
    reflection_ok: Optional[bool] = None
    reflection_note: Optional[str] = None
    # Debug/scoring aids (populated on demand to keep trajectories small):
    #   observation_text — serialized page the model saw (kept on failed steps,
    #     or every step when capture is forced); makes failures re-readable.
    #   screenshot_path — PNG written to disk (for WebJudge + demo GIFs).
    observation_text: Optional[str] = None
    screenshot_path: Optional[str] = None
    # Bookkeeping
    used_vision: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Trajectory:
    """The full record of one task attempt."""

    task_id: str
    goal: str
    model: str
    config: dict[str, Any]
    steps: list[Step] = field(default_factory=list)
    status: str = "running"            # running | done | budget_exceeded | error
    answer: Optional[str] = None
    answer_grounding: Optional[float] = None  # 0–1: answer tokens seen on a page
    success: Optional[bool] = None     # filled in by a scorer
    score_detail: dict[str, Any] = field(default_factory=dict)
    failure_category: Optional[str] = None
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # -- derived metrics -------------------------------------------------
    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.steps), 6)

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    @property
    def vision_fallbacks(self) -> int:
        return sum(1 for s in self.steps if s.used_vision)

    @property
    def latency_s(self) -> float:
        if self.end_time is None:
            return 0.0
        return round(self.end_time - self.start_time, 3)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "goal": self.goal,
            "model": self.model,
            "config": self.config,
            "status": self.status,
            "answer": self.answer,
            "answer_grounding": self.answer_grounding,
            "success": self.success,
            "score_detail": self.score_detail,
            "failure_category": self.failure_category,
            "error": self.error,
            "n_steps": self.n_steps,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "vision_fallbacks": self.vision_fallbacks,
            "latency_s": self.latency_s,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps": [s.to_dict() for s in self.steps],
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

Difficulty = Literal["easy", "medium", "hard", "unknown"]


@dataclass
class Task:
    """A benchmark task: a goal, a start URL, and how to score success."""

    task_id: str
    goal: str
    start_url: str
    benchmark: str = "custom"          # webarena | mind2web | local | custom
    difficulty: Difficulty = "unknown"
    reference_steps: Optional[int] = None   # human reference step count
    max_steps: Optional[int] = None
    # Scoring spec — interpreted by the benchmark's scorer.
    #   webarena: {"type": "string_match", "answer": "..."} / {"type": "url_match", ...}
    #   mind2web: {"type": "webjudge"}
    #   local:    {"type": "url_contains"/"text_contains"/"answer_match", ...}
    eval_spec: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
