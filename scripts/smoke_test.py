"""Offline smoke test — verifies the pipeline with no API key and no network.

Two layers:
  1. Component checks: schemas, action parse/validate, scoring, metrics, taxonomy,
     and the ScriptedLLMClient. These run anywhere (no third-party deps).
  2. End-to-end (best effort): if Playwright + a browser are installed, drive a
     scripted agent over the bundled local site and assert the tasks pass.

    python -m scripts.smoke_test
"""

from __future__ import annotations

import sys

from agent import actions as A
from agent.llm import ScriptedLLMClient, extract_json
from agent.observation import Observation
from agent.types import Action, Step, Task, Trajectory
from eval import failure_taxonomy, metrics
from eval.harness import score_trajectory


def check(name: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"smoke check failed: {name}")


def component_checks() -> None:
    print("Component checks:")

    # Action schema + parse + validate against a fake observation.
    schema = A.action_output_schema()
    check("action schema has thought+action", set(schema["properties"]) == {"thought", "action"})

    obs = Observation(
        url="https://example.com",
        title="Example",
        elements=[
            {"ref": "@e1", "role": "link", "name": "Products"},
            {"ref": "@e2", "role": "search", "name": "", "input_type": "search"},
        ],
    )
    good = A.parse_action({"thought": "go", "action": {"type": "click", "ref": "e1"}})
    check("ref normalized to @e1", good.ref == "@e1")
    check("valid click passes validation", A.validate_action(good, obs) is None)

    bad = Action(type="click", ref="@e99")
    check("missing ref rejected", A.validate_action(bad, obs) is not None)
    check("type without text rejected", A.validate_action(Action(type="type", ref="@e2"), obs))

    # Scripted client returns a parsed action.
    sc = ScriptedLLMClient([{"type": "done", "answer": "hi"}])
    r = sc.complete(system="s", messages=[{"role": "user", "content": "x"}],
                    json_schema=schema)
    check("scripted client parses", r.parsed["action"]["type"] == "done")

    # JSON extraction from fenced text.
    check("extract_json from fence", extract_json('```json\n{"a": 1}\n```') == {"a": 1})

    # Generic scorer.
    task = Task(task_id="t", goal="g", start_url="x", benchmark="local",
                eval_spec={"type": "answer_match", "answer": "19.99", "mode": "contains"})
    traj = Trajectory(task_id="t", goal="g", model="m", config={},
                      answer="It costs $19.99")
    score_trajectory(task, traj)
    check("answer_match scores success", traj.success is True)

    # Metrics + taxonomy on a synthetic mix.
    rows = [
        _fake_traj("a", success=True, steps=2, cost=0.01, difficulty="easy"),
        _fake_traj("b", success=False, status="done", steps=3, cost=0.02, difficulty="hard"),
        _fake_traj("c", success=False, status="budget_exceeded", steps=15, cost=0.05,
                   difficulty="hard"),
    ]
    summ = metrics.summarize(rows)
    check("summary success_rate ~0.33", abs(summ["success_rate"] - 1 / 3) < 1e-3)
    bd = failure_taxonomy.breakdown(rows)
    check("taxonomy flags premature_done", bd.get("premature_done", 0) == 1)
    check("taxonomy flags max_steps", bd.get("max_steps", 0) == 1)


def _fake_traj(tid, *, success, steps, cost, difficulty, status="done") -> dict:
    t = Trajectory(task_id=tid, goal="g", model="m", config={}, status=status)
    for i in range(steps):
        t.steps.append(Step(index=i, url="u", observation_hash="h", thought="",
                            action={"type": "click"}, action_ok=True, cost_usd=cost / steps))
    t.success = success
    t.score_detail = {"difficulty": difficulty, "benchmark": "local"}
    t.end_time = t.start_time + 1.0
    return t.to_dict()


def e2e_check() -> bool:
    print("\nEnd-to-end check (scripted agent on local site):")
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception:
        print("  [SKIP] Playwright not installed (pip install playwright && playwright install chromium)")
        return True

    from agent.browser import BrowserSession
    from agent.loop import Agent, AgentConfig
    from eval.harness import score_trajectory
    from eval.local_demo import SITE_DIR, load_local_tasks

    products = (SITE_DIR / "products.html").resolve().as_uri()
    contact = (SITE_DIR / "contact.html").resolve().as_uri()
    plans = {
        "local-price": [{"type": "navigate", "target": products},
                        {"type": "done", "answer": "The Blue Widget costs $19.99."}],
        "local-email": [{"type": "navigate", "target": contact},
                        {"type": "done", "answer": "support@acme.example"}],
        "local-nav": [{"type": "navigate", "target": products},
                      {"type": "done", "answer": "Opened products."}],
    }

    try:
        BrowserSession(headless=True).start().close()
    except Exception as e:
        print(f"  [SKIP] browser launch failed ({e}); run `playwright install chromium`")
        return True

    ok_all = True
    for task in load_local_tasks():
        agent = Agent(ScriptedLLMClient(plans[task.task_id]), AgentConfig(max_steps=5))
        traj = agent.run(task)
        score_trajectory(task, traj)
        ok = bool(traj.success)
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {task.task_id} "
              f"(status={traj.status}, steps={traj.n_steps})")
    return ok_all


def main() -> int:
    component_checks()
    ok = e2e_check()
    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
