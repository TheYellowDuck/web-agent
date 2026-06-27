"""Offline re-score: original substring scorer vs. type-aware deterministic scorer.

Reproduces WebArena-Verified's central finding — that the original WebArena
pipeline's substring matching *under-counts* correct answers — on this project's
own saved trajectories, using NO LLM judge. We re-score a saved run two ways:

    baseline   eval.webarena.scorer  (substring / degraded fuzzy — what shipped)
    type-aware eval.webarena.verified (|OR| operator, unicode/number folding,
                                       deterministic N/A absence detection)

and report which rejections were scoring artifacts vs. genuine agent errors.

    python -m scripts.rescore_webarena \
        --trajectories results/trajectories/webarena-scale__claude-sonnet-4-6__reflect-on.jsonl \
        --tasks eval/webarena/shopping_scale.json

Fully offline and deterministic — no API key, no Docker, no network.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional
from urllib.parse import urlsplit

from agent.types import Trajectory
from eval.metrics import load_trajectories
from eval.stats import wilson_interval


def _base_url(rows: list[dict]) -> str:
    for r in rows:
        for s in r.get("steps", []):
            u = s.get("url", "")
            p = urlsplit(u)
            if p.scheme in ("http", "https") and p.netloc:
                return f"{p.scheme}://{p.netloc}"
    return ""


def _sr(verdicts: list[Optional[bool]]) -> tuple[float, int, int, tuple[float, float]]:
    scored = [v for v in verdicts if v is not None]
    n_succ = sum(1 for v in scored if v)
    rate = n_succ / len(scored) if scored else 0.0
    return rate, n_succ, len(scored), wilson_interval(n_succ, len(scored))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--tasks", required=True, help="WebArena task JSON for these trajectories")
    args = ap.parse_args(argv)

    rows = load_trajectories(args.trajectories)
    # Make URL templating faithful offline: point __SHOPPING__ at the host the
    # agent actually ran against (so url_match reproduces the shipped verdicts).
    base = _base_url(rows)
    if base and not os.environ.get("WA_SHOPPING"):
        os.environ["WA_SHOPPING"] = base

    from eval.webarena.scorer import score as baseline_score
    from eval.webarena.tasks import load_tasks as wa_load
    from eval.webarena import verified

    tasks = {t.task_id: t for t in wa_load(args.tasks)}
    by_id = {r["task_id"]: r for r in rows}

    base_v: list[Optional[bool]] = []
    strict_v: list[Optional[bool]] = []
    full_v: list[Optional[bool]] = []
    recovered: list[tuple[str, str, str]] = []
    saved_mismatch = 0

    print(f"{'task':<14} {'saved':>6} {'base':>6} {'+ops':>6} {'+na':>6}  rule / note")
    print("-" * 84)
    for tid, task in tasks.items():
        r = by_id.get(tid)
        if r is None:
            continue
        traj = Trajectory.from_dict(r)
        b, _ = baseline_score(task, traj, judge_llm=None)
        s, _ = verified.score(task, traj, na_aware=False)
        f, fd = verified.score(task, traj, na_aware=True)
        base_v.append(b)
        strict_v.append(s)
        full_v.append(f)

        saved = r.get("success")
        if saved != b:
            saved_mismatch += 1
        # A recovery: baseline rejected (False) but type-aware accepts (True).
        if b is False and f is True:
            recovered.append((tid, fd.get("rule", "?"), (traj.answer or "")[:46]))

        def m(v: Optional[bool]) -> str:
            return "✓" if v else ("·" if v is None else "✗")

        rule = fd.get("rule") or "-"
        flag = "  ⬅ RECOVERED" if (b is False and f is True) else ""
        print(f"{tid:<14} {m(saved):>6} {m(b):>6} {m(s):>6} {m(f):>6}  {rule}{flag}")

    print("-" * 84)
    for label, vv in (("baseline (substring)", base_v),
                      ("type-aware: +operators/unicode", strict_v),
                      ("type-aware: +N/A absence", full_v)):
        rate, ns, nsc, (lo, hi) = _sr(vv)
        print(f"  {label:<34} {ns}/{nsc} = {rate:.3f}  (95% CI {lo:.2f}–{hi:.2f})")
    print("  (strict denominator is smaller: without the N/A heuristic, 'none' "
          "answers are honestly\n   left UNSCORED rather than failed — 3-valued, "
          "not guessed.)")

    if recovered:
        print(f"\nFalse-negatives recovered by type-aware scoring ({len(recovered)}):")
        for tid, rule, ans in recovered:
            print(f"  • {tid:<14} via {rule:<12} answer={ans!r}")
    if saved_mismatch:
        print(f"\n[warn] {saved_mismatch} task(s) differ from the saved verdict — "
              "re-score env may not match the original run.")
    else:
        print("\n[ok] baseline re-score reproduces every saved verdict exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
