"""The agent loop: ReAct + optional reflection / self-correction.

    loop until done or step budget exceeded:
        observation = perceive()                 # a11y snapshot (+ vision if needed)
        thought, action = plan(goal, history, observation)
        result = act(action)
        if reflect_enabled:
            ok = reflect(action, observation, observation')
            if not ok: note the failure so the next plan can recover
        history.append(step)

Reflection is a toggle so it can be ablated — it's the headline engineering
contribution, and the harness compares ON vs OFF on the same tasks.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from agent import actions as A
from agent.browser import ActionError, BrowserSession
from agent.llm import BaseLLMClient
from agent.memory import Memory
from agent.observation import Observation
from agent.prompts import (
    SYSTEM_PROMPT,
    build_planner_messages,
    build_reflection_messages,
)
from agent.types import Action, Step, Task, Trajectory

# Words that suggest an irreversible side effect — gated unless allowed.
_IRREVERSIBLE = re.compile(
    r"\b(buy|purchase|place order|checkout|pay|delete|remove|send|submit|"
    r"confirm|sign up|register|log ?in|delete account)\b",
    re.IGNORECASE,
)


@dataclass
class AgentConfig:
    reflect: bool = False
    vision_fallback: bool = False
    # Set-of-Marks: attach a screenshot with numbered boxes (matching @e refs)
    # every step — first-class multimodal grounding, not just a fallback.
    set_of_marks: bool = False
    max_steps: int = 15
    max_consecutive_failures: int = 3
    loop_window: int = 4
    # Guardrail: block irreversible actions unless explicitly allowed (the
    # open-web slice runs with this on; WebArena is safe by construction).
    confirm_irreversible: bool = False
    allow_irreversible: bool = False
    # Keep the agent on the task's own site (origin of start_url). Prevents the
    # "wandered off to the real web and hallucinated" failure; off for open-web
    # tasks (Mind2Web) that legitimately cross domains, on for sandbox sites.
    confine_to_site: bool = False
    # Artifacts:
    #   capture_screenshots — save a PNG of the resulting page each step (for
    #     WebJudge scoring and demo GIFs).
    #   persist_observations — "never" | "on_failure" | "always": store the
    #     serialized page the model saw, so failures stay re-readable offline.
    capture_screenshots: bool = False
    persist_observations: str = "on_failure"
    artifacts_dir: Optional[str] = None
    # Anti-hallucination: if the final answer's content isn't supported by text
    # the agent actually saw, send it back once to verify on the page before
    # accepting `done`. The grounding score is always recorded either way.
    verify_before_done: bool = False
    grounding_threshold: float = 0.5
    # Lightweight planning: the agent maintains a running checklist (persisted
    # across steps), and for "list all" answers we bounce once if pages remain
    # unvisited — targets multi-step + incomplete-extraction failures.
    planning: bool = False
    check_completeness: bool = False

    def as_dict(self) -> dict:
        return {
            "reflect": self.reflect,
            "vision_fallback": self.vision_fallback,
            "max_steps": self.max_steps,
            "confirm_irreversible": self.confirm_irreversible,
            "capture_screenshots": self.capture_screenshots,
        }


class Agent:
    def __init__(
        self,
        llm: BaseLLMClient,
        config: Optional[AgentConfig] = None,
        *,
        on_step: Optional[Callable[[Step], None]] = None,
    ):
        self.llm = llm
        self.config = config or AgentConfig()
        self.on_step = on_step
        self._art_dir: Optional[Path] = None
        self._allowed_origin: Optional[str] = None

    # -- public ----------------------------------------------------------
    def run(self, task: Task, *, browser: Optional[BrowserSession] = None) -> Trajectory:
        owns_browser = browser is None
        browser = browser or BrowserSession(headless=True).start()
        traj = Trajectory(
            task_id=task.task_id,
            goal=task.goal,
            model=self.llm.model,
            config=self.config.as_dict(),
        )
        memory = Memory(goal=task.goal)
        max_steps = task.max_steps or self.config.max_steps
        self._allowed_origin = _origin(task.start_url) if self.config.confine_to_site else None
        self._art_dir = None
        if self.config.capture_screenshots:
            base = Path(self.config.artifacts_dir or "results/screenshots")
            self._art_dir = base / _safe_name(task.task_id)
            self._art_dir.mkdir(parents=True, exist_ok=True)
        try:
            browser.goto(task.start_url)
            self._loop(task, browser, memory, traj, max_steps)
        except Exception as e:  # never let a crash lose the partial trajectory
            traj.status = "error"
            traj.error = f"{type(e).__name__}: {e}"
        finally:
            traj.end_time = _now()
            if owns_browser:
                browser.close()
        return traj

    # -- internals -------------------------------------------------------
    def _loop(
        self,
        task: Task,
        browser: BrowserSession,
        memory: Memory,
        traj: Trajectory,
        max_steps: int,
    ) -> None:
        pending_note: Optional[str] = None
        verified_once = False
        completeness_checked = False
        current_plan = ""
        seen_text: list[str] = []  # everything the agent has actually observed
        seen_lines: set[str] = set()
        stale_scrolls = 0
        for i in range(max_steps):
            obs = browser.snapshot()
            # Site confinement: if a click / "back" left the site (or hit a blank
            # page), return to the task page instead of drifting off and guessing.
            if self._allowed_origin and _origin(obs.url) != self._allowed_origin:
                browser.goto(task.start_url)
                obs = browser.snapshot()
                pending_note = (
                    f"You left {self._allowed_origin} (or hit a blank page) — "
                    "returned to the task site. Stay on this site."
                )
            memory.current_url = obs.url
            seen_text.append(" ".join(obs.texts))
            seen_text.append(" ".join(e.get("name", "") for e in obs.elements))
            seen_text.append(obs.url)

            # Detect "scrolling that reveals nothing new" and nudge a decision —
            # the dominant over-exploration failure (agent won't commit done()).
            prev_n = len(seen_lines)
            seen_lines.update(obs.texts)
            grew = len(seen_lines) > prev_n
            last_action = memory.entries[-1].action_desc if memory.entries else ""
            stale_scrolls = stale_scrolls + 1 if (last_action.startswith("scroll") and not grew) else 0
            if stale_scrolls >= 2 and not pending_note:
                pending_note = (
                    "Scrolling has revealed no new content — you have already seen "
                    "what's available here. Navigate elsewhere or call done() with "
                    "your best answer now."
                )
                stale_scrolls = 0

            som = self.config.set_of_marks
            use_vision = som or self._should_use_vision(obs, memory)
            images = None
            if use_vision:
                shot_b64 = browser.screenshot()
                if som:
                    from agent.marks import render_marked_screenshot
                    images = [render_marked_screenshot(
                        base64.b64decode(shot_b64), obs.elements)]
                else:
                    images = [shot_b64]

            # --- plan ---
            plan_resp = self.llm.complete(
                system=SYSTEM_PROMPT,
                messages=build_planner_messages(
                    task.goal, obs, memory, extra_note=pending_note, vision=use_vision,
                    set_of_marks=som, planning=self.config.planning, plan=current_plan,
                    step_index=i, max_steps=max_steps,
                ),
                json_schema=A.action_output_schema(),
                images_b64=images,
            )
            pending_note = None
            payload = plan_resp.parsed or {}
            thought = str(payload.get("thought", "")).strip()
            if self.config.planning and payload.get("plan"):
                current_plan = str(payload["plan"]).strip()

            step = Step(
                index=i,
                url=obs.url,
                observation_hash=obs.hash(),
                thought=thought,
                action={},
                action_ok=False,
                used_vision=use_vision,
                observation_text=obs.serialize(),  # retained per policy in _commit
                prompt_tokens=plan_resp.input_tokens,
                completion_tokens=plan_resp.output_tokens,
                cost_usd=plan_resp.cost_usd,
                latency_s=plan_resp.latency_s,
            )

            # --- parse + validate ---
            try:
                action = A.parse_action(payload)
            except Exception as e:
                step.action_error = f"parse error: {e}"
                self._commit(step, traj, memory, Action(type="wait"),
                             browser=browser, task=task)
                pending_note = step.action_error
                if self._should_abort(memory, traj):
                    return
                continue

            step.action = action.to_dict()
            err = A.validate_action(action, obs)
            if err:
                step.action_error = err
                self._commit(step, traj, memory, action, browser=browser)
                pending_note = f"Previous action invalid: {err}"
                if self._should_abort(memory, traj):
                    return
                continue

            # --- guardrail: block navigating off the task site ---
            if (
                self._allowed_origin
                and action.type == "navigate"
                and (action.target or "").startswith(("http://", "https://"))
                and _origin(action.target) != self._allowed_origin
            ):
                step.action_error = f"blocked: off-site navigation to {_origin(action.target)}"
                self._commit(step, traj, memory, action, browser=browser)
                pending_note = (
                    f"Do not navigate to other websites — stay on {self._allowed_origin}. "
                    "The information you need is on this site."
                )
                continue

            # --- guardrail ---
            if self._blocked(action, obs):
                step.action_error = "blocked: irreversible action not allowed"
                self._commit(step, traj, memory, action, browser=browser)
                pending_note = "That action is irreversible and was blocked."
                continue

            # --- terminal action ---
            if action.type == "done":
                step.action_ok = True
                # Completeness gate: don't accept a list answer while more pages
                # of the list remain unvisited — bounce once to gather them.
                if (
                    self.config.check_completeness
                    and not completeness_checked
                    and obs.pagination.get("next_ref")
                    and _looks_like_list(action.answer)
                ):
                    completeness_checked = True
                    step.reflection_note = "list answer with unvisited pages; gathering more"
                    pending_note = (
                        f"Your answer is a list, but more pages exist (use "
                        f"{obs.pagination['next_ref']}). Visit the remaining pages and "
                        "add every matching item before finishing."
                    )
                    self._commit(step, traj, memory, action, browser=browser)
                    continue
                grounding = _answer_grounding(action.answer or "", " ".join(seen_text))
                # Optional anti-hallucination gate: bounce an ungrounded answer
                # back once to verify on the page before accepting it.
                if (
                    self.config.verify_before_done
                    and not verified_once
                    and action.answer
                    and grounding is not None
                    and grounding < self.config.grounding_threshold
                ):
                    verified_once = True
                    step.reflection_note = (
                        f"answer grounding {grounding:.2f} < "
                        f"{self.config.grounding_threshold}; verifying"
                    )
                    pending_note = (
                        "Your proposed answer is not clearly supported by the page "
                        "content you have seen. Re-read the relevant page to confirm "
                        "the exact value, then call done again."
                    )
                    self._commit(step, traj, memory, action, browser=browser)
                    continue
                traj.answer = action.answer
                traj.answer_grounding = grounding
                traj.status = "done"
                self._commit(step, traj, memory, action, browser=browser)
                return

            # --- act ---
            try:
                browser.act(action)
                step.action_ok = True
            except ActionError as e:
                step.action_error = str(e)
                step.action_ok = False

            # --- reflect (optional) ---
            if self.config.reflect and step.action_ok:
                after = browser.snapshot()
                ok, note, rresp = self._reflect(task.goal, action, obs, after)
                step.reflection_ok = ok
                step.reflection_note = note
                step.prompt_tokens += rresp.input_tokens
                step.completion_tokens += rresp.output_tokens
                step.cost_usd = round(step.cost_usd + rresp.cost_usd, 6)
                step.latency_s = round(step.latency_s + rresp.latency_s, 3)
                if not ok and note:
                    pending_note = f"Reflection: {note}"

            self._commit(step, traj, memory, action, browser=browser)

            if not step.action_ok:
                pending_note = f"Previous action failed: {step.action_error}"
            if self._should_abort(memory, traj):
                return

        traj.status = "budget_exceeded"

    def _reflect(
        self, goal: str, action: Action, before: Observation, after: Observation
    ):
        resp = self.llm.complete(
            system=SYSTEM_PROMPT,
            messages=build_reflection_messages(goal, action.describe(), before, after),
            json_schema=A.reflection_output_schema(),
        )
        parsed = resp.parsed or {}
        ok = bool(parsed.get("ok", True))
        note = str(parsed.get("note", "")).strip() or None
        return ok, note, resp

    def _commit(
        self,
        step: Step,
        traj: Trajectory,
        memory: Memory,
        action: Action,
        *,
        browser: Optional[BrowserSession] = None,
    ) -> None:
        self._finalize_artifacts(step, browser)
        traj.steps.append(step)
        note = step.reflection_note if step.reflection_ok is False else step.action_error
        memory.record(step.index, action, step.action_ok, note=note,
                      obs_hash=step.observation_hash)
        if self.on_step:
            self.on_step(step)

    def _finalize_artifacts(self, step: Step, browser: Optional[BrowserSession]) -> None:
        # Observation retention: keep failures readable, drop the rest to stay small.
        failed = (not step.action_ok) or bool(step.action_error) or (
            step.reflection_ok is False
        )
        policy = self.config.persist_observations
        if policy == "never" or (policy == "on_failure" and not failed):
            step.observation_text = None
        # Screenshot of the resulting page (for WebJudge / demo GIFs).
        if self.config.capture_screenshots and browser is not None and self._art_dir:
            try:
                png = base64.b64decode(browser.screenshot())
                path = self._art_dir / f"step{step.index:02d}.png"
                path.write_bytes(png)
                step.screenshot_path = str(path)
            except Exception:
                pass

    # -- policies --------------------------------------------------------
    def _should_use_vision(self, obs: Observation, memory: Memory) -> bool:
        if not self.config.vision_fallback:
            return False
        # Fall back to the screenshot when the a11y tree is empty/sparse or the
        # agent appears stuck.
        if len(obs.elements) <= 1:
            return True
        if memory.consecutive_failures >= 2:
            return True
        return False

    def _blocked(self, action: Action, obs: Observation) -> bool:
        if self.config.allow_irreversible or not self.config.confirm_irreversible:
            return False
        if action.type != "click" or not action.ref:
            return False
        name = next((e.get("name", "") for e in obs.elements if e["ref"] == action.ref), "")
        return bool(_IRREVERSIBLE.search(name))

    def _should_abort(self, memory: Memory, traj: Trajectory) -> bool:
        if memory.consecutive_failures >= self.config.max_consecutive_failures:
            traj.status = "error"
            traj.error = "too many consecutive failures"
            return True
        if memory.looping(self.config.loop_window):
            traj.status = "error"
            traj.error = "stuck in a loop"
            return True
        return False


def _now() -> float:
    import time

    return time.time()


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:80] or "task"


def _looks_like_list(answer: Optional[str]) -> bool:
    """Heuristic: does the answer enumerate multiple items (so coverage matters)?"""
    a = (answer or "").strip()
    if not a:
        return False
    # comma-separated (≥2), or multiple numbered/bulleted lines.
    if a.count(",") >= 1 and len(a) > 8:
        return True
    lines = [ln for ln in a.splitlines() if ln.strip()]
    bulletish = sum(1 for ln in lines if re.match(r"\s*(\d+[.)]|[-*•])", ln))
    return bulletish >= 2


def _origin(url: str) -> str:
    """scheme://host:port — the navigation boundary for site confinement.
    Returns '' for about:blank / non-http so those count as 'off-site'."""
    p = urlsplit(url or "")
    if p.scheme not in ("http", "https") or not p.netloc:
        return ""
    return f"{p.scheme}://{p.netloc}"


# Stopwords stripped before grounding so filler words don't inflate the score.
_GROUNDING_STOP = {
    "the", "and", "are", "was", "were", "for", "that", "this", "with", "its",
    "approximately", "about", "than", "report", "one", "sentence", "based",
    "while", "which", "from", "into", "have", "has", "what", "does",
}


def _content_tokens(s: str) -> list[str]:
    """Content words/numbers worth grounding (drops short/filler tokens)."""
    out: list[str] = []
    for w in re.findall(r"[a-z0-9][a-z0-9.,%/_-]*", s.lower()):
        w = w.strip(".,")
        if len(w) < 3 or w in _GROUNDING_STOP:
            continue
        out.append(w)
    return out


def _answer_grounding(answer: str, corpus: str) -> Optional[float]:
    """Fraction of the answer's content tokens that appear in observed text.

    A cheap signal for answer hallucination: a number or name the agent reports
    that never appeared on any page it saw drags the score down. Heuristic — a
    paraphrase scores below 1.0 — so it's used as a soft gate / metric, not a
    hard correctness check. Returns None when the answer has no content tokens.
    """
    toks = _content_tokens(answer)
    if not toks:
        return None
    corpus_l = corpus.lower()
    # Numbers are the most checkable and most-hallucinated content, so weight a
    # token containing a digit higher — an invented figure can't hide behind
    # correctly-grounded surrounding words.
    num = den = 0.0
    for t in toks:
        w = 3.0 if any(c.isdigit() for c in t) else 1.0
        den += w
        if t in corpus_l:
            num += w
    return round(num / den, 3) if den else None
