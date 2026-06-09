# Eval-Driven Autonomous Web Agent — Project Plan

> **One-line pitch:** A from-scratch LLM web agent benchmarked honestly — it does well on a deterministic sandbox (WebArena) and visibly struggles on realistic open-web tasks (Online-Mind2Web), quantifying the gap that inflated headline numbers hide.

---

## 1. Thesis & Positioning

The web-agent ecosystem is mature enough that wrapping an existing framework signals nothing in an interview. The differentiated angle is **evaluation honesty**:

- Reported success rates (~90% on WebVoyager) are inflated by easy, shortcut-solvable tasks and weak LLM-as-judge scoring.
- On harder, realistic benchmarks the same class of agent drops to roughly 40–60%.
- **The project's headline result is that gap, measured on your own agent**, plus the engineering that narrows it (reflection / self-correction).

This mirrors the eval-harness rigor approach: the scaffold is table stakes, the *measurement* is the signal.

---

## 2. Goals & Non-Goals

**Goals**
- Build a minimal, well-understood agent scaffold from scratch (you can explain every component).
- Build a reproducible eval harness with deterministic + realistic benchmarks.
- Produce defensible findings: reflection ablation, observation-modality ablation, model cost/success Pareto, sandbox-vs-realistic gap.
- Ship a polished writeup + charts + demo.

**Non-Goals**
- Not a production scraper or a CAPTCHA/anti-bot bypass tool.
- Not a browser-use wrapper (browser-use is a *baseline*, not the agent).
- No side effects without confirmation (no purchases, no form submissions, no logins on live sites outside sandbox).

---

## 3. Architecture

### 3.1 Browser Control Layer
- **Playwright** (Python). Single source of truth for page interaction.
- Headless for eval runs; headed for debugging.
- One `BrowserSession` wrapper exposing: `goto`, `snapshot`, `act(action)`, `screenshot`, `close`.

### 3.2 Observation Layer
- **Primary: accessibility tree + element refs.** Serialize the a11y tree into a compact, numbered list (`@e1`, `@e2`, …) the LLM can address directly. Deterministic, token-cheap, robust to markup changes.
- **Fallback: screenshot + vision.** Only invoked when the a11y tree is insufficient (canvas, custom widgets, ambiguous layout). Track when fallback fires — it's a metric.
- Trim aggressively: hidden/offscreen elements dropped, long text truncated, stable ordering.

### 3.3 Action Space (typed, small)
- `click(ref)`
- `type(ref, text)`
- `select(ref, option)`
- `scroll(direction | ref)`
- `navigate(url | back | forward)`
- `wait(ms | condition)`
- `done(answer?)` — terminal action with optional extracted answer
- Each action validated against the current snapshot before execution (ref must exist).

### 3.4 Agent Loop (ReAct + reflection)
```
loop until done or step_budget exceeded:
  observation = perceive()          # a11y snapshot (+ vision if needed)
  thought, action = plan(goal, history, observation)
  result = act(action)
  observation' = perceive()
  if reflect_enabled:
      ok = reflect(goal, action, observation, observation')   # did it do what we intended?
      if not ok: replan with the failure noted
  history.append(step)
```
- **Reflection / self-correction is the headline engineering contribution** — build it as a toggle so you can ablate it.

### 3.5 State & Memory
- Compact running state: goal, completed subgoals, last N steps (summarized), current URL.
- Avoid re-feeding full history every step (cost + context bloat) — summarize older steps.

### 3.6 Guardrails
- Confirmation gate before any irreversible action (submit, purchase, send, login on live sites).
- In sandbox (WebArena) actions are safe by construction; gate matters for the open-web slice.

### 3.7 LLM Interface (model-agnostic adapter)
- Single `LLMClient` interface so swapping models is a config change, not a rewrite.
- Tracks tokens + cost per call (feeds the cost-per-task metric).
- Supports the 3-model sweep at eval time without touching agent logic.

---

## 4. Repo Structure

```
web-agent/
├── PLAN.md
├── README.md                  # final writeup, charts, demo GIF
├── pyproject.toml
├── agent/
│   ├── browser.py             # Playwright session wrapper
│   ├── observation.py         # a11y tree serialization + refs
│   ├── actions.py             # typed action schema + executor
│   ├── loop.py                # ReAct + reflection loop
│   ├── memory.py              # state tracking / summarization
│   ├── llm.py                 # model-agnostic client + cost tracking
│   └── prompts.py
├── eval/
│   ├── harness.py             # runner: task -> trajectory -> score
│   ├── webarena/              # setup + deterministic scorers
│   ├── mind2web/              # frozen task slice + WebJudge
│   ├── metrics.py             # SR, step efficiency, cost, latency
│   ├── failure_taxonomy.py    # classify failed trajectories
│   └── baselines/
│       └── browser_use_runner.py
├── results/
│   ├── trajectories/          # full logs (jsonl) for every run
│   └── reports/               # generated charts + tables
└── scripts/
    └── make_charts.py
```

---

## 5. Eval Harness (the spine)

### 5.1 WebArena (deterministic backbone)
- Dockerized self-hosted sites (shopping CMS, forum, GitLab, map, wiki).
- Deterministic string/URL-match scoring — ground truth, no LLM judge needed.
- Free, reproducible, no CAPTCHA/ToS issues. **Run the full suite.**

### 5.2 Online-Mind2Web (realistic slice)
- **Frozen ~30-task slice**, stratified across easy (1–5 steps) / medium (6–10) / hard (11+).
- Scored with a **WebJudge-style LLM-as-judge** pipeline.
- Purpose: demonstrate the sandbox-vs-realistic success-rate collapse. Keep it small — full 300-task runs are slow, flaky, and expensive.

### 5.3 Metrics (per task + aggregated)
- **Success rate** (overall + by difficulty tier).
- **Step efficiency** — steps taken vs. human reference steps.
- **Cost per task** (USD, from token tracking).
- **Latency** per task.
- **Failure taxonomy** — wrong element, hallucinated action, premature `done`, infinite loop, vision-fallback failure.

### 5.4 Baselines to compare against
1. Your scaffold, **reflection OFF**.
2. Your scaffold, **reflection ON**.
3. Vanilla **browser-use**.

### 5.5 Logging
- Every run writes a full trajectory (`jsonl`): observation hashes, thoughts, actions, results, costs, timestamps. Everything reproducible and re-scorable offline.

---

## 6. Experiments (the findings)

1. **Reflection ablation** — ON vs OFF on WebArena. Expected: the biggest single delta. *This is your headline engineering result.*
2. **Observation modality** — a11y-only vs a11y + vision fallback. Measure success delta vs cost delta.
3. **Model sweep** — 3 models (frontier / mid-tier / cheap-fast) → **cost-vs-success Pareto chart**. The interesting question: how much does the cheap model give up, and is the frontier model worth its cost on hard tasks?
4. **Sandbox vs realistic gap** — WebArena SR vs Online-Mind2Web SR for your best config. *This is the "illusion of progress" punchline.*

---

## 7. Phased Milestones (evenings/weekends around the co-op)

| Phase | Weeks | Deliverable | Definition of Done |
|-------|-------|-------------|--------------------|
| 0. Setup | 1 | Playwright + a11y observation layer + action executor | Can scripts open a page, snapshot it, and execute a typed `click`/`type` end-to-end |
| 1. Core loop | 2–3 | ReAct loop, no reflection, single dev model | A handful of WebArena tasks pass start-to-finish |
| 2. Harness | 3–5 | Full WebArena runner + metrics + trajectory logging + failure taxonomy | Full WebArena suite runs unattended and produces a metrics table |
| 3. Reflection | 5–6 | Reflection/self-correction toggle + ablation | Reflection ON vs OFF numbers on WebArena, plus a browser-use baseline line |
| 4. Realism | 6–7 | Online-Mind2Web slice + WebJudge | Frozen 30-task slice scored; sandbox-vs-realistic gap quantified |
| 5. Ship | 7–8 | README writeup + charts + demo GIF + "next steps" | A reader understands the thesis, the numbers, and the engineering in 5 minutes |

**Dev discipline:** build everything in phases 0–3 against **one** capable-but-cheap model. Run the 3-model sweep only in phase 4+, once the harness is stable.

---

## 8. Tech Stack (pinned)

- **Language:** Python 3.11+
- **Browser:** Playwright
- **Agent:** custom scaffold (no agent framework dependency)
- **Baseline:** browser-use (latest)
- **Benchmarks:** WebArena (Docker, self-hosted) + Online-Mind2Web slice
- **Models:** pick current top SKU per tier at build time from the Claude / GPT / Gemini families (lineup shifts every month or two — don't lock in early)
- **Data/plots:** pandas + matplotlib (`scripts/make_charts.py`)
- **Logs:** JSONL trajectories

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Open-web flakiness | Lean on WebArena (deterministic) as the backbone; keep the live slice small and frozen |
| API cost blowup | Single dev model during build; sweep only at the end; track cost per call from day one |
| Scope creep | Reflection + the four experiments are the project; everything else is stretch |
| Becomes a wrapper | Scaffold is custom; browser-use is strictly a comparison line |
| Dev-model lock-in | Model-agnostic `LLMClient` from the start so the sweep is a config change |

---

## 10. Deliverables / Interview Artifacts

- **README** that opens with the thesis and the headline chart.
- **Cost-vs-success Pareto chart** (the quant/infra-flavored visual).
- **Sandbox-vs-realistic bar chart** (the illusion-of-progress visual).
- **Reflection ablation table.**
- **Failure taxonomy breakdown.**
- **30–60s demo GIF** of a task running.
- **Interview narrative:** "I built a web agent from scratch, then measured it honestly — here's why the field's headline numbers are misleading, what actually moved the needle (reflection), and what it costs."

---

## 11. Stretch Goals

- Trajectory caching / replay to cut re-run cost.
- RL or rejection-sampling fine-tune of a small open model on successful trajectories.
- Multimodal grounding improvements (better vision fallback).
- Self-hosted open model in the cost sweep (pushes the Pareto frontier on the cheap end).
- Parallelized eval runner for faster full-suite passes.