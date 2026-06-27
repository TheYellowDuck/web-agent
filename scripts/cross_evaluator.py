"""Cross-evaluator disagreement matrix — how much does the *scorer* decide the score?

The project's thesis is that headline web-agent numbers are governed by scoring
methodology. This makes that quantitative: run several evaluator families over
ONE fixed set of trajectories and report where they disagree, by how much, and
in which direction. No surveyed paper runs this many evaluator families
head-to-head on a single fixed trajectory set.

The 2024–25 literature documents disagreement in BOTH directions, which this
tool is built to expose on one unchanged agent:
  • Rule-based scoring UNDER-counts vs. experts — AgentRewardBench, Lù et al.
    2025 (arXiv 2504.08942).
  • Lenient LLM-judge scoring OVER-counts (agent-hallucination false positives) —
    "An Illusion of Progress?", Online-Mind2Web, 2025 (arXiv 2504.01382).

Evaluator families (each maps a trajectory → pass / fail / unscored):
  WebArena trajectories:
    substring     eval.webarena.scorer (the shipped, permissive substring scorer)
    type_aware    eval.webarena.verified (|OR|, unicode/number folding, N/A)
    llm_judge     substring scorer + an LLM fuzzy_match judge   [--judge-model]
    grounding@T   answer-grounding ≥ T  (an answer-integrity lens, not task success)
  Mind2Web trajectories:
    webjudge      the recorded WebJudge verdict (a lenient LLM judge)
    grounding@T   answer-grounding ≥ T
    nonempty      did the agent commit any answer at all

Reports per-family success rate, a pairwise agreement-% and Cohen's-κ matrix, and
the directional bias of each disagreement. Fully offline unless --judge-model is
passed.

    python -m scripts.cross_evaluator \
        --trajectories results/trajectories/webarena-scale__claude-sonnet-4-6__reflect-on.jsonl \
        --tasks eval/webarena/shopping_scale.json
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Callable, Optional
from urllib.parse import urlsplit

from agent.types import Task, Trajectory
from eval.metrics import load_trajectories
from eval.stats import cohen_kappa, wilson_interval

Verdict = Optional[bool]
Evaluator = Callable[[Task, dict, Trajectory], Verdict]


def _base_url(rows: list[dict]) -> str:
    for r in rows:
        for s in r.get("steps", []):
            p = urlsplit(s.get("url", ""))
            if p.scheme in ("http", "https") and p.netloc:
                return f"{p.scheme}://{p.netloc}"
    return ""


def _grounding(threshold: float) -> Evaluator:
    def f(task: Task, row: dict, traj: Trajectory) -> Verdict:
        g = row.get("answer_grounding")
        return None if g is None else (g >= threshold)
    return f


def _build_webarena_evaluators(judge_model: Optional[str]) -> dict[str, Evaluator]:
    from eval.webarena.scorer import score as baseline_score
    from eval.webarena import verified

    evals: dict[str, Evaluator] = {
        "substring": lambda t, r, tr: baseline_score(t, tr, judge_llm=None)[0],
        "type_aware": lambda t, r, tr: verified.score(t, tr, na_aware=True)[0],
        "grounding@0.5": _grounding(0.5),
    }
    if judge_model:
        from agent.llm import make_llm_client, resolve_tier
        judge = make_llm_client(resolve_tier(judge_model))
        evals["llm_judge"] = lambda t, r, tr: baseline_score(t, tr, judge_llm=judge)[0]
    return evals


def _build_mind2web_evaluators() -> dict[str, Evaluator]:
    return {
        "webjudge": lambda t, r, tr: r.get("success"),  # recorded lenient judge
        "grounding@0.5": _grounding(0.5),
        "nonempty": lambda t, r, tr: bool((tr.answer or "").strip()),
    }


def _load_tasks(spec: str, benchmark: str) -> dict[str, Task]:
    if benchmark == "webarena":
        from eval.webarena.tasks import load_tasks as wa_load
        return {t.task_id: t for t in wa_load(spec)}
    if benchmark == "mind2web":
        from eval.harness import _load_task_file
        return {t.task_id: t for t in _load_task_file(spec)}
    from eval.harness import _load_task_file
    return {t.task_id: t for t in _load_task_file(spec)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--benchmark", default=None,
                    help="webarena | mind2web (default: infer from trajectories)")
    ap.add_argument("--judge-model", default=None,
                    help="enable the llm_judge family (WebArena fuzzy_match); needs an API key")
    args = ap.parse_args(argv)

    rows = load_trajectories(args.trajectories)
    prefix = (rows[0].get("task_id", "") if rows else "").lower()
    benchmark = args.benchmark or (
        "webarena" if prefix.startswith("webarena")
        else "mind2web" if prefix.startswith(("mind2web", "m2w"))
        else "webarena")

    if benchmark == "webarena":
        base = _base_url(rows)
        if base and not os.environ.get("WA_SHOPPING"):
            os.environ["WA_SHOPPING"] = base
        evaluators = _build_webarena_evaluators(args.judge_model)
    elif benchmark == "mind2web":
        evaluators = _build_mind2web_evaluators()
    else:
        print(f"unsupported benchmark: {benchmark}")
        return 1

    tasks = _load_tasks(args.tasks, benchmark)
    names = list(evaluators)

    # verdicts[name] = {task_id: pass/fail/None}
    verdicts: dict[str, dict[str, Verdict]] = {n: {} for n in names}
    order: list[str] = []
    for row in rows:
        tid = row["task_id"]
        # tasks are keyed without the multi-run '#rN' suffix.
        task = tasks.get(tid) or tasks.get(re.sub(r"#r\d+$", "", tid))
        if task is None:
            continue
        traj = Trajectory.from_dict(row)
        order.append(tid)
        for name, ev in evaluators.items():
            try:
                verdicts[name][tid] = ev(task, row, traj)
            except Exception:
                verdicts[name][tid] = None

    # --- per-evaluator success rate ---
    print(f"benchmark = {benchmark} | {len(order)} trajectories | "
          f"families = {', '.join(names)}\n")
    print("Per-evaluator success rate (scored only):")
    for n in names:
        vals = [v for v in verdicts[n].values() if v is not None]
        ns = sum(1 for v in vals if v)
        lo, hi = wilson_interval(ns, len(vals))
        rate = ns / len(vals) if vals else 0.0
        print(f"  {n:<14} {ns}/{len(vals)} = {rate:.3f}  (95% CI {lo:.2f}–{hi:.2f})")

    # --- pairwise agreement % and Cohen's κ ---
    print("\nPairwise agreement %  /  Cohen's κ   (over tasks both families scored):")
    header = "".join(f"{n[:11]:>13}" for n in names)
    print(f"{'':<14}{header}")
    for a in names:
        cells = []
        for b in names:
            tids = [t for t in order
                    if verdicts[a].get(t) is not None and verdicts[b].get(t) is not None]
            if a == b or not tids:
                cells.append(f"{'—':>13}")
                continue
            n11 = sum(1 for t in tids if verdicts[a][t] and verdicts[b][t])
            n00 = sum(1 for t in tids if not verdicts[a][t] and not verdicts[b][t])
            n10 = sum(1 for t in tids if verdicts[a][t] and not verdicts[b][t])
            n01 = sum(1 for t in tids if not verdicts[a][t] and verdicts[b][t])
            agree = (n11 + n00) / len(tids)
            k = cohen_kappa(n11, n10, n01, n00)
            cells.append(f"{agree*100:>5.0f}%/{(f'{k:+.2f}' if k is not None else ' n/a'):>6}")
        print(f"{a:<14}{''.join(cells)}")

    # --- directional disagreement (who passes when they differ) ---
    print("\nDirectional disagreements (A passes, B fails):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            tids = [t for t in order
                    if verdicts[a].get(t) is not None and verdicts[b].get(t) is not None]
            a_only = [t for t in tids if verdicts[a][t] and not verdicts[b][t]]
            b_only = [t for t in tids if not verdicts[a][t] and verdicts[b][t]]
            if not a_only and not b_only:
                continue
            print(f"  {a} vs {b}: {a}+only={len(a_only)} {b}+only={len(b_only)}"
                  + (f"  → {a} is more lenient" if len(a_only) > len(b_only)
                     else f"  → {b} is more lenient" if len(b_only) > len(a_only)
                     else "  → symmetric"))
            for t in a_only:
                print(f"      [{a}] {t}")
            for t in b_only:
                print(f"      [{b}] {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
