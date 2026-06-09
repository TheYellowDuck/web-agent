"""Aggregate metrics over a set of trajectories.

Works on trajectory dicts (as written to JSONL) so runs can be re-scored and
re-aggregated entirely offline.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

TIERS = ("easy", "medium", "hard", "unknown")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion — honest error bars for small N."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def _base_task_id(task_id: str) -> str:
    """Strip a multi-run suffix: 'foo#r3' -> 'foo'."""
    return re.sub(r"#r\d+$", "", task_id)


def load_trajectories(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return round(statistics.fmean(xs), 4) if xs else 0.0


def _difficulty(t: dict[str, Any]) -> str:
    d = (t.get("config", {}) or {}).get("difficulty") or t.get("difficulty")
    if d in TIERS:
        return d
    # difficulty is carried on the task; harness stamps it into score_detail.
    return (t.get("score_detail", {}) or {}).get("difficulty", "unknown")


def summarize(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """Top-level metrics block for a single (model, config) run."""
    n = len(trajectories)
    scored = [t for t in trajectories if t.get("success") is not None]
    successes = [t for t in scored if t.get("success")]

    by_tier: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        tt = [t for t in scored if _difficulty(t) == tier]
        if tt:
            by_tier[tier] = {
                "n": len(tt),
                "success_rate": _mean(1.0 if t.get("success") else 0.0 for t in tt),
            }

    # Step efficiency = steps taken / human reference steps (lower is better).
    effs = []
    for t in trajectories:
        ref = (t.get("score_detail", {}) or {}).get("reference_steps")
        if ref:
            effs.append(t["n_steps"] / ref)

    n_success = sum(1 for t in scored if t.get("success"))
    ci_lo, ci_hi = wilson_interval(n_success, len(scored))

    # Multi-run aggregation: how many *distinct* tasks were solved at least once.
    base_ids = {_base_task_id(t["task_id"]) for t in scored}
    solved_base = {
        _base_task_id(t["task_id"]) for t in scored if t.get("success")
    }
    runs_per_task = round(len(scored) / len(base_ids), 2) if base_ids else 0

    return {
        "n_tasks": n,
        "n_scored": len(scored),
        "n_unique_tasks": len(base_ids),
        "runs_per_task": runs_per_task,
        "success_rate": _mean(1.0 if t.get("success") else 0.0 for t in scored),
        "success_rate_ci95": [ci_lo, ci_hi],
        "pass_any_rate": round(len(solved_base) / len(base_ids), 4) if base_ids else 0.0,
        "success_rate_by_difficulty": by_tier,
        "mean_steps": _mean(t["n_steps"] for t in trajectories),
        "step_efficiency": _mean(effs) if effs else None,
        "mean_cost_usd": _mean(t["total_cost_usd"] for t in trajectories),
        "total_cost_usd": round(sum(t["total_cost_usd"] for t in trajectories), 4),
        "mean_latency_s": _mean(t["latency_s"] for t in trajectories),
        "mean_tokens": _mean(t["total_tokens"] for t in trajectories),
        "vision_fallback_rate": _mean(
            1.0 if t.get("vision_fallbacks", 0) else 0.0 for t in trajectories
        ),
        "mean_answer_grounding": _grounding_mean(trajectories),
        "n_successes": len(successes),
    }


def _grounding_mean(trajectories: list[dict[str, Any]]):
    vals = [t["answer_grounding"] for t in trajectories if t.get("answer_grounding") is not None]
    return _mean(vals) if vals else None


def compare(runs: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Summaries keyed by run label (e.g. 'reflect_on', 'reflect_off')."""
    return {label: summarize(trajs) for label, trajs in runs.items()}
