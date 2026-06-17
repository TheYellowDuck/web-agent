"""Aggregate metrics over a set of trajectories.

Works on trajectory dicts (as written to JSONL) so runs can be re-scored and
re-aggregated entirely offline.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

# wilson_interval / pass_at_k live in eval.stats (the single home for the
# statistics primitives); re-exported here so existing callers keep working.
from eval.stats import pass_at_k, wilson_interval

__all__ = ["wilson_interval", "pass_at_k", "summarize", "compare",
           "load_trajectories"]

TIERS = ("easy", "medium", "hard", "unknown")


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

    # Multi-run aggregation: per-base-task attempt/success counts.
    attempts: dict[str, list[int]] = {}
    for t in scored:
        attempts.setdefault(_base_task_id(t["task_id"]), []).append(
            1 if t.get("success") else 0
        )
    base_ids = set(attempts)
    solved_base = {tid for tid, v in attempts.items() if any(v)}
    runs_per_task = round(len(scored) / len(base_ids), 2) if base_ids else 0

    return {
        "n_tasks": n,
        "n_scored": len(scored),
        "n_unique_tasks": len(base_ids),
        "runs_per_task": runs_per_task,
        "success_rate": _mean(1.0 if t.get("success") else 0.0 for t in scored),
        "success_rate_ci95": [ci_lo, ci_hi],
        "pass_any_rate": round(len(solved_base) / len(base_ids), 4) if base_ids else 0.0,
        "pass_at_k": _pass_at_k_block(attempts),
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


def _pass_at_k_block(attempts: dict[str, list[int]]) -> dict[str, float]:
    """Unbiased pass@k for k = 1 … max attempts, averaged across base tasks.

    For each k, only tasks with at least k attempts contribute (so pass@3 isn't
    diluted by tasks that were only run once). pass@1 equals the per-attempt
    success rate; the dict has a single entry for single-run suites.
    """
    if not attempts:
        return {}
    max_k = max(len(v) for v in attempts.values())
    out: dict[str, float] = {}
    for k in range(1, max_k + 1):
        vals = [pass_at_k(len(v), sum(v), k) for v in attempts.values() if len(v) >= k]
        if vals:
            out[str(k)] = round(statistics.fmean(vals), 4)
    return out


def compare(runs: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Summaries keyed by run label (e.g. 'reflect_on', 'reflect_off')."""
    return {label: summarize(trajs) for label, trajs in runs.items()}
