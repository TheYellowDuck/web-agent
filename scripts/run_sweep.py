"""Orchestrate the project's experiments, then render charts.

Covers:
  * reflection ablation (ON vs OFF) on a benchmark
  * model sweep across tiers (frontier / mid / cheap)
  * (re-run with --tasks mind2web for the sandbox-vs-realistic comparison)

Each (model, reflect) combination writes its own JSONL run so the charts can
group them. This is a thin convenience wrapper over `eval.harness.run_suite`.

    # Reflection ablation on the local demo with the dev model:
    python -m scripts.run_sweep --tasks local-demo --models mid --ablate-reflection

    # Three-model cost/success sweep on WebArena:
    python -m scripts.run_sweep --tasks webarena --models frontier mid cheap

    # Realistic slice for the best config:
    python -m scripts.run_sweep --tasks mind2web --models frontier --reflect
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.env import load_dotenv
from agent.llm import resolve_tier
from agent.loop import AgentConfig
from eval import metrics
from eval.harness import TRAJ_DIR, load_tasks, run_suite


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run experiment sweeps + render charts.")
    p.add_argument("--tasks", required=True,
                   help="local-demo | webarena | mind2web | path")
    p.add_argument("--models", nargs="+", default=["mid"],
                   help="tier names (frontier/mid/cheap) or model ids")
    p.add_argument("--reflect", action="store_true",
                   help="run with reflection ON (single setting)")
    p.add_argument("--ablate-reflection", action="store_true",
                   help="run each model both ON and OFF")
    p.add_argument("--vision", action="store_true")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--runs", type=int, default=1, help="runs per task (CIs / pass@k)")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    p.add_argument("--headed", action="store_true")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--charts", action="store_true", help="render charts when done")
    args = p.parse_args(argv)
    load_dotenv()

    tasks = load_tasks(args.tasks, limit=args.limit)
    print(f"Loaded {len(tasks)} task(s) from {args.tasks}")

    reflect_settings = [False, True] if args.ablate_reflection else [args.reflect]
    judge_model = resolve_tier(args.judge_model) if args.judge_model else None

    base = Path(args.tasks).stem if args.tasks.endswith(".json") else args.tasks
    runs: list[Path] = []
    for tier in args.models:
        model = resolve_tier(tier)
        for reflect in reflect_settings:
            cfg = AgentConfig(
                reflect=reflect, vision_fallback=args.vision, max_steps=args.max_steps,
                confirm_irreversible=(args.tasks == "mind2web"),
                capture_screenshots=(args.tasks == "mind2web"),
                confine_to_site=(args.tasks == "webarena"),
            )
            slug = model.replace("/", "-")
            refl = "reflect-on" if reflect else "reflect-off"
            out = TRAJ_DIR / f"{base}__{slug}__{refl}.jsonl"
            print(f"\n>>> {base} | {model} | {refl}")
            results = run_suite(
                tasks, model=model, config=cfg, out_path=out,
                runs=args.runs, workers=args.workers, headless=not args.headed,
                judge_model=judge_model,
                on_progress=lambda r: print(
                    f"   {'✓' if r['success'] else '✗'} {r['task_id']} "
                    f"({r['status']}, {r['n_steps']} steps, ${r['total_cost_usd']:.4f})"
                ),
            )
            summ = metrics.summarize(results)
            ci = summ["success_rate_ci95"]
            print(f"   SR={summ['success_rate']:.2f} (95% CI {ci[0]:.2f}–{ci[1]:.2f}) "
                  f"mean_cost=${summ['mean_cost_usd']:.4f} "
                  f"mean_steps={summ['mean_steps']:.1f}")
            runs.append(out)

    print(f"\nWrote {len(runs)} run file(s) under {TRAJ_DIR}")
    if args.charts:
        from scripts.make_charts import main as charts_main

        charts_main([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
