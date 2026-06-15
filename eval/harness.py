"""The eval runner: task -> trajectory -> score, with reproducible logging.

Each run writes one JSONL file under results/trajectories/. Everything needed
to re-score offline (observation hashes, thoughts, actions, results, costs,
timestamps) is in there.

CLI:
    python -m eval.harness --tasks local-demo --model claude-sonnet-4-6 --reflect
    python -m eval.harness --tasks eval/tasks/my_tasks.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from agent.env import load_dotenv
from agent.llm import BaseLLMClient, make_llm_client, resolve_tier
from agent.loop import Agent, AgentConfig
from agent.types import Task, Trajectory
from eval import failure_taxonomy, metrics

ROOT = Path(__file__).resolve().parent.parent
TRAJ_DIR = ROOT / "results" / "trajectories"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_trajectory(
    task: Task, traj: Trajectory, *, judge_llm: Optional[BaseLLMClient] = None
) -> None:
    """Set ``success`` / ``score_detail`` / ``failure_category`` on ``traj``."""
    traj.score_detail.setdefault("difficulty", task.difficulty)
    traj.score_detail.setdefault("benchmark", task.benchmark)
    if task.reference_steps:
        traj.score_detail.setdefault("reference_steps", task.reference_steps)

    benchmark = task.benchmark
    try:
        if benchmark == "webarena":
            from eval.webarena.scorer import score as wa_score

            ok, detail = wa_score(task, traj, judge_llm=judge_llm)
        elif benchmark == "mind2web":
            from eval.mind2web.webjudge import judge

            ok, detail = judge(task, traj, judge_llm=judge_llm)
        else:
            ok, detail = _generic_score(task, traj)
    except Exception as e:  # a scorer failure shouldn't lose the trajectory
        ok, detail = None, {"scorer_error": f"{type(e).__name__}: {e}"}

    traj.success = ok
    traj.score_detail.update(detail)
    traj.failure_category = failure_taxonomy.classify(traj.to_dict())


def _generic_score(task: Task, traj: Trajectory) -> tuple[Optional[bool], dict[str, Any]]:
    """Deterministic scorer for local/custom tasks via ``task.eval_spec``."""
    spec = task.eval_spec or {}
    kind = spec.get("type", "answer_match")

    if kind == "answer_match":
        target = spec.get("answer", "")
        mode = spec.get("mode", "contains")
        got = (traj.answer or "").strip()
        ok = _text_match(got, target, mode)
        return ok, {"scorer": "answer_match", "expected": target, "got": got}

    if kind in ("url_contains", "url_match"):
        value = spec.get("value", "")
        urls = [s.url for s in traj.steps]
        if traj.answer:
            urls.append(traj.answer)
        if kind == "url_match":
            ok = any(u.rstrip("/") == value.rstrip("/") for u in urls)
        else:
            ok = any(value in u for u in urls)
        return ok, {"scorer": kind, "value": value, "visited": urls[-3:]}

    return None, {"scorer": "none", "note": f"unknown eval_spec type {kind!r}"}


def _text_match(got: str, target: str, mode: str) -> bool:
    g, t = got.lower(), str(target).lower()
    if mode == "exact":
        return g.strip() == t.strip()
    if mode == "regex":
        return re.search(target, got, re.IGNORECASE) is not None
    return t in g  # contains (default)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(spec: str, *, limit: Optional[int] = None) -> list[Task]:
    if spec == "local-demo":
        from eval.local_demo import load_local_tasks

        tasks = load_local_tasks()
    elif spec == "webarena":
        from eval.webarena.tasks import load_tasks as wa_load

        tasks = wa_load()
    elif spec == "mind2web":
        from eval.mind2web.tasks import load_slice

        tasks = load_slice()
    else:
        tasks = _load_task_file(spec)

    if limit:
        tasks = tasks[:limit]
    return tasks


def _load_task_file(path: str) -> list[Task]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tasks not found: {path}")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        rows = data["tasks"] if isinstance(data, dict) else data
    return [Task.from_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _expand_jobs(tasks: list[Task], runs: int) -> list[Task]:
    """One Task per (task, run). Multi-run gets a '#rN' suffix so trajectories
    stay distinct while metrics can group back by base id."""
    if runs <= 1:
        return list(tasks)
    jobs: list[Task] = []
    for t in tasks:
        for j in range(1, runs + 1):
            jobs.append(dataclasses.replace(t, task_id=f"{t.task_id}#r{j}"))
    return jobs


def _webarena_auth(task: Task) -> Optional[str]:
    """Storage-state path for an authenticated WebArena task, else None."""
    if task.benchmark != "webarena" or not task.metadata.get("require_login"):
        return None
    try:
        from eval.webarena.auth import auth_state_for

        return auth_state_for(task.metadata.get("sites", []))
    except Exception:
        return None


def run_one(
    task: Task,
    *,
    model: str,
    config: AgentConfig,
    headless: bool = True,
    provider: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> dict[str, Any]:
    """Run + score a single task. Self-contained so it can run in a subprocess."""
    import time

    from agent.browser import BrowserSession

    load_dotenv()  # workers are separate processes; ensure keys are present
    llm = make_llm_client(model, provider=provider)
    judge_llm = make_llm_client(judge_model) if judge_model else None
    _maybe_reset_site(task)
    storage_state = _webarena_auth(task)

    # Retry the whole task if it died purely from a rate limit (the SDK already
    # backs off per-call; this catches the rare case it exhausts its budget).
    traj = None
    for attempt in range(3):
        browser = BrowserSession(headless=headless, storage_state=storage_state).start()
        try:
            traj = Agent(llm, config).run(task, browser=browser)
        finally:
            browser.close()
        if not (traj.status == "error" and _is_rate_limit(traj.error)):
            break
        time.sleep(5 * (attempt + 1))  # linear backoff before re-running

    score_trajectory(task, traj, judge_llm=judge_llm)
    return traj.to_dict()


def _is_rate_limit(error: Optional[str]) -> bool:
    e = (error or "").lower()
    return "rate_limit" in e or "ratelimit" in e or "429" in e


_RESET_WARNED = False


def _maybe_reset_site(task: Task) -> None:
    """Reset stateful sites before a `require_reset` WebArena task.

    A faithful reset is environment-specific (restore DB / recreate container),
    so the command is supplied via $WA_RESET_CMD. We run it, then drop cached
    auth states so the next session logs in fresh. If a task needs a reset but
    none is configured, warn once — its result may be contaminated by prior state.
    """
    global _RESET_WARNED
    if task.benchmark != "webarena" or not task.metadata.get("require_reset"):
        return
    cmd = os.environ.get("WA_RESET_CMD")
    if not cmd:
        if not _RESET_WARNED:
            print("[warn] task(s) set require_reset but WA_RESET_CMD is unset — "
                  "state may carry over between tasks.", file=sys.stderr)
            _RESET_WARNED = True
        return
    import subprocess

    subprocess.run(cmd, shell=True, check=False)
    from eval.webarena.auth import state_path  # invalidate cached logins
    for site in task.metadata.get("sites", []):
        try:
            state_path(site).unlink()
        except OSError:
            pass


def run_suite(
    tasks: list[Task],
    *,
    model: str,
    config: AgentConfig,
    out_path: Path,
    runs: int = 1,
    workers: int = 1,
    headless: bool = True,
    provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> list[dict[str, Any]]:
    """Run every (task × run), score, stream results to JSONL, return the dicts.

    ``workers > 1`` runs tasks across processes (each gets its own browser +
    LLM client — sidesteps Playwright's thread constraints)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs = _expand_jobs(tasks, runs)
    results: list[dict[str, Any]] = []

    with open(out_path, "w", encoding="utf-8") as f:
        def emit(row: dict[str, Any]) -> None:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            results.append(row)
            if on_progress:
                on_progress(row)

        if workers <= 1:
            for task in jobs:
                emit(run_one(task, model=model, config=config, headless=headless,
                             provider=provider, judge_model=judge_model))
        else:
            kw = dict(model=model, config=config, headless=headless,
                      provider=provider, judge_model=judge_model)
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_one, task, **kw): task for task in jobs}
                for fut in as_completed(futs):
                    emit(fut.result())
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_out(tasks_spec: str, model: str, cfg: AgentConfig, label: str | None) -> Path:
    base = label or Path(tasks_spec).stem
    model_slug = re.sub(r"[^a-zA-Z0-9.]+", "-", model)
    refl = "reflect-on" if cfg.reflect else "reflect-off"
    return TRAJ_DIR / f"{base}__{model_slug}__{refl}.jsonl"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the web-agent eval harness.")
    parser.add_argument(
        "--tasks", required=True,
        help="local-demo | webarena | mind2web | path to a .json/.jsonl task file",
    )
    parser.add_argument(
        "--model", default=os.environ.get("WEBAGENT_MODEL", "claude-sonnet-4-6"),
        help="model id, or a tier name: frontier | mid | cheap",
    )
    parser.add_argument("--provider", default=None, help="force a provider adapter")
    parser.add_argument("--reflect", action="store_true", help="enable reflection")
    parser.add_argument("--vision", action="store_true", help="enable vision fallback")
    parser.add_argument("--set-of-marks", action="store_true",
                        help="attach a numbered-box screenshot every step (multimodal)")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--headed", action="store_true", help="run with a visible browser")
    parser.add_argument("--guardrails", action="store_true",
                        help="block irreversible actions (open-web slice)")
    parser.add_argument("--runs", type=int, default=1,
                        help="runs per task (enables confidence intervals / pass@k)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes")
    parser.add_argument("--capture-screenshots", action="store_true",
                        help="save per-step screenshots (auto-on for mind2web)")
    parser.add_argument("--verify-answers", action="store_true",
                        help="bounce an ungrounded final answer back once to verify")
    parser.add_argument("--judge-model", default=None,
                        help="model for Mind2Web WebJudge scoring")
    parser.add_argument("--label", default=None, help="output filename label")
    parser.add_argument("--out", default=None, help="explicit output jsonl path")
    args = parser.parse_args(argv)
    load_dotenv()  # populate os.environ from .env if present

    model = resolve_tier(args.model)
    judge_model = resolve_tier(args.judge_model) if args.judge_model else None
    capture = args.capture_screenshots or args.tasks == "mind2web"
    config = AgentConfig(
        reflect=args.reflect,
        vision_fallback=args.vision,
        set_of_marks=args.set_of_marks,
        max_steps=args.max_steps,
        confirm_irreversible=args.guardrails,
        capture_screenshots=capture,
        verify_before_done=args.verify_answers,
        confine_to_site=(args.tasks == "webarena"),
    )

    tasks = load_tasks(args.tasks, limit=args.limit)
    if not tasks:
        print("No tasks loaded.", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else _default_out(args.tasks, model, config, args.label)

    print(f"Running {len(tasks)} task(s) × {args.runs} run(s) | model={model} "
          f"| reflect={config.reflect} | vision={config.vision_fallback} "
          f"| workers={args.workers}")
    print(f"Writing trajectories -> {out_path}\n")

    def progress(row: dict[str, Any]) -> None:
        ok = row.get("success")
        mark = "✓" if ok else ("·" if ok is None else "✗")
        print(f"  {mark} {row['task_id']:<24} status={row['status']:<15} "
              f"steps={row['n_steps']:<3} cost=${row['total_cost_usd']:.4f} "
              f"cat={row.get('failure_category') or '-'}")

    results = run_suite(
        tasks, model=model, config=config, out_path=out_path,
        runs=args.runs, workers=args.workers, headless=not args.headed,
        provider=args.provider, judge_model=judge_model, on_progress=progress,
    )

    summary = metrics.summarize(results)
    fails = failure_taxonomy.breakdown(results)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    if fails:
        print("\n=== Failure taxonomy ===")
        print(json.dumps(fails, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
