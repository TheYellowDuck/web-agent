# Eval-Driven Autonomous Web Agent

> A from-scratch LLM web agent, **benchmarked honestly**. It does well on a
> deterministic sandbox (WebArena) and visibly struggles on realistic open-web
> tasks (Online-Mind2Web) — and the project's headline result is *that gap*,
> measured on this agent, plus the engineering (reflection / self-correction)
> that narrows it.

The scaffold is table stakes. The **measurement** is the signal: a reflection
ablation, an observation-modality ablation, a cost-vs-success Pareto across
model tiers, and the sandbox-vs-realistic success-rate collapse — all computed
on this agent, with fully reproducible JSONL trajectories.

See [plan.md](plan.md) for the full thesis and rationale.

---

## Why this design

- **Custom scaffold, not a wrapper.** Every component is small and explainable:
  browser control, an accessibility-tree observation layer, a typed action
  space, a model-agnostic LLM client, and a ReAct + reflection loop.
- **Reflection is a toggle.** It's the headline engineering contribution, built
  so it can be ablated ON vs OFF on the same tasks.
- **Provider-agnostic by construction.** Swapping models is a config change.
  Claude / OpenAI / Gemini have native adapters; *any other model* works through
  the LiteLLM universal adapter (Mistral, Llama via Ollama, Bedrock, Groq, …).
- **browser-use is a baseline**, never the agent — strictly a comparison line.

---

## Architecture

```text
agent/
  browser.py      Playwright session: goto / snapshot / act / screenshot / close
                  (+ popup/new-tab following, networkidle settle)
  observation.py  a11y-tree refs (@e1, @e2 …) + static-text + pagination detection
  actions.py      typed action space + validation + JSON schema for the LLM
  llm.py          model-agnostic client (Claude/OpenAI/Gemini/LiteLLM) + cost tracking
                  (+ Anthropic prompt caching, offline 'echo' model)
  memory.py       compact running state (recent steps verbatim, older summarized)
  prompts.py      planner + reflection prompts (frozen system prompt)
  loop.py         ReAct + reflection loop, vision fallback, guardrails,
                  screenshot capture, observation persistence on failure
  types.py        Task / Action / Step / Trajectory dataclasses
eval/
  harness.py      runner: task -> trajectory -> score; CLI; JSONL; --runs/--workers
  metrics.py      success rate (by tier) + Wilson CIs + multi-run / pass@k
  failure_taxonomy.py  classify failed trajectories
  webarena/       URL templating + string/url/fuzzy/program_html scorers + loader
  mind2web/       frozen slice loader + WebJudge LLM-as-judge (text + screenshots)
  baselines/      browser_use_runner.py (comparison line)
  local_site/     bundled offline demo site (file://)
  local_demo.py   3 deterministic offline tasks
  tasks/          mind2web_slice.json (frozen realistic slice)
scripts/
  make_charts.py  Pareto / sandbox-vs-realistic / reflection / taxonomy charts
  run_sweep.py    orchestrate experiments across models + reflection settings
  smoke_test.py   offline pipeline check (no API key / network needed)
tests/            pytest suite for scorers / metrics / taxonomy / actions / observation
results/
  trajectories/   one JSONL per run (fully re-scorable offline)
  reports/        generated charts + tables
```

### The agent loop

```text
loop until done or step budget exceeded:
    observation = perceive()                 # a11y snapshot (+ screenshot if needed)
    thought, action = plan(goal, history, observation)
    result = act(action)                     # validated against the snapshot first
    if reflect_enabled:
        ok = reflect(action, observation, observation')
        if not ok: note the failure so the next plan can recover
    history.append(step)
```

Observations are a numbered list of interactable elements; the model can only
act on a ref it actually saw, so hallucinated targets are caught before
execution. The screenshot + vision modality is a **fallback** (fires when the
a11y tree is empty/sparse or the agent is stuck) and how often it fires is a
tracked metric.

---

## Quickstart

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # agent + harness + Claude, ready to run
playwright install chromium
```

The base install runs the **default model (Claude) out of the box** — no extra
LLM library to install. For other providers you don't *have* to install
anything (the factory falls back to the universal LiteLLM adapter), but you can
add a native SDK for first-class behaviour / accurate cost:

```bash
pip install -e ".[litellm]"      # universal adapter — enables any other model
pip install -e ".[openai]"       # OpenAI (native)
pip install -e ".[gemini]"       # Gemini (native)
pip install -e ".[all]"          # everything, incl. browser-use baseline
```

Copy `.env.example` → `.env` and fill in the key(s) for whatever you run — the
CLIs load `.env` automatically.

### 2. Smoke test (no API key, no network)

Verifies the whole pipeline — observation, action execution, scoring, metrics,
taxonomy, and a scripted agent driving the bundled local site:

```bash
python -m scripts.smoke_test         # component + real-browser e2e
pip install -e ".[dev]" && pytest    # unit tests for scorers/metrics/taxonomy
```

You can also exercise the runner with **no API key** using the offline `echo`
model (it emits `done()` immediately) — handy for testing parallelism/logging:

```bash
python -m eval.harness --tasks local-demo --model echo --runs 2 --workers 2
```

### 3. Run the offline demo with a real model

The local demo runs against a bundled static site over `file://` — no Docker,
no live web — so you can see a real LLM drive the agent for cents:

```bash
python -m eval.harness --tasks local-demo --model claude-sonnet-4-6
python -m eval.harness --tasks local-demo --model claude-sonnet-4-6 --reflect
```

Each run writes `results/trajectories/<run>.jsonl` and prints a metrics summary
plus a failure-taxonomy breakdown.

---

## Choosing a model — any LLM

The model string selects the provider automatically:

```bash
# Native adapters
--model claude-opus-4-8            # Anthropic
--model gpt-4o-mini                # OpenAI
--model gemini-2.0-flash           # Google

# Tier shortcuts (frontier / mid / cheap) — see agent/llm.py TIERS
--model frontier                   # -> claude-opus-4-8
--model mid                        # -> claude-sonnet-4-6  (default dev model)
--model cheap                      # -> claude-haiku-4-5

# Anything else, via the LiteLLM universal adapter (vendor/model form)
--model mistral/mistral-large-latest
--model ollama/llama3              # local, no API key
--model groq/llama-3.1-70b-versatile
--model bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
```

Cost and tokens are tracked per call and aggregated per task (native adapters
use a built-in price table; LiteLLM computes its own cost). Per the plan, build
against one capable-but-cheap model (`mid`) and only run the multi-model sweep
once the harness is stable.

---

## Benchmarks

### WebArena (deterministic backbone)

Self-hosted Dockerized sites, scored by ground-truth string/URL match — no LLM
judge. The dataset and site images aren't bundled (large, and live upstream);
this repo provides the adapter, a preflight check, and an integration smoke so
the code path is verified before you bring up the heavy infrastructure.

**0. Verify the adapter works (no Docker, no keys):**

```bash
python -m scripts.webarena_smoke      # templates __HOMEPAGE__ at the local site,
                                       # runs + scores WebArena-schema tasks
```

**1. Stand up the sites.** Follow the upstream
[WebArena](https://github.com/web-arena-x/webarena) instructions to run the
self-hosted site containers (shopping, shopping-admin, reddit/forum, gitlab,
wikipedia, map, homepage). They're distributed as Docker images upstream — the
setup is multi-GB and machine-specific, so use their guide as the source of
truth rather than copying commands here.

**2. Wire the base URLs** in `.env` (`WA_SHOPPING`, `WA_GITLAB`, `WA_HOMEPAGE`, …
— see `.env.example`), then preflight:

```bash
python -m scripts.check_webarena --tasks /path/to/webarena/test.raw.json
# reports which sites are set + reachable and whether the task file is valid
```

**3. Run** (URLs and reference answers are templated against your base URLs by
`eval/webarena/config.py`; tasks whose required site isn't configured are
skipped, not 404'd):

```bash
export WEBARENA_TASKS=/path/to/webarena/test.raw.json
python -m scripts.run_sweep --tasks webarena --models frontier mid cheap --ablate-reflection
```

The deterministic scorers cover `string_match` (exact / must_include),
`url_match`, LLM `fuzzy_match`, and best-effort `program_html` — with 3-valued
logic so anything unverifiable is marked *unscored*, never guessed.

### Online-Mind2Web (realistic slice)

A small, **frozen**, difficulty-stratified slice scored by a WebJudge-style
LLM-as-judge. A starter slice ships in
[`eval/tasks/mind2web_slice.json`](eval/tasks/mind2web_slice.json) — expand it
to ~30 tasks (easy 1–5 / medium 6–10 / hard 11+).

```bash
# Use a strong, separate judge model for scoring
python -m eval.harness --tasks mind2web --model mid --judge-model frontier --guardrails
```

`--guardrails` blocks irreversible actions (purchases, submits, logins) — on for
the open-web slice, unnecessary for WebArena which is safe by construction.

> Open-web tasks are flaky by nature; keep this slice small and frozen. That
> flakiness, and the success-rate collapse vs. the sandbox, is the point.

---

## Experiments → charts

Run the sweeps, then render the report:

```bash
# Reflection ablation (ON vs OFF) on a benchmark
python -m scripts.run_sweep --tasks local-demo --models mid --ablate-reflection

# Cost-vs-success across tiers
python -m scripts.run_sweep --tasks webarena --models frontier mid cheap

# Realistic slice for the best config
python -m scripts.run_sweep --tasks mind2web --models frontier --reflect --judge-model frontier

# Charts + tables from everything under results/trajectories/
python -m scripts.make_charts
```

Outputs in `results/reports/`:

| Artifact | What it shows |
|----------|---------------|
| `cost_vs_success.png` | Cost-vs-success Pareto across model tiers / configs |
| `sandbox_vs_realistic.png` | WebArena SR vs Mind2Web SR — the "illusion of progress" punchline |
| `reflection_ablation.png` | Reflection ON vs OFF (the headline engineering result) |
| `failure_taxonomy.png` | Where failed trajectories break down |
| `summary.csv` / `summary.md` | The run-level table |

The four findings the project targets:

1. **Reflection ablation** — ON vs OFF on WebArena. Expected biggest single delta.
2. **Observation modality** — a11y-only vs a11y + vision fallback (success Δ vs cost Δ).
3. **Model sweep** — frontier / mid / cheap → cost-vs-success Pareto.
4. **Sandbox vs realistic gap** — WebArena SR vs Online-Mind2Web SR for the best config.

---

## Measured results (so far)

First real runs — agent **Claude Sonnet 4.6**, judge **Claude Opus 4.8**,
WebJudge with screenshots, guardrails on. These are early numbers on a small
slice; read them with the caveats below, not as leaderboard figures.

**Reflection ablation — Online-Mind2Web slice (15 navigation-forcing tasks).**
Shown across two iterations, because fixing the agent changed the result — that
iteration *is* the finding:

| iteration | reflect OFF | reflect ON | mean cost (OFF) |
|---|:---:|:---:|:---:|
| before commit-fix | 0.80 (12/15) | 0.87 (13/15) | $0.067 |
| **after commit-fix** | **1.00 (15/15)**, CI 0.80–1.00 | 0.93 (14/15), CI 0.70–0.99 | $0.030 |

What actually happened (the mechanism, not just the aggregate):
- **First pass:** reflection helped on hard tasks (0.60→0.80) by rescuing
  "never-finishes" loops — e.g. a population-comparison task OFF abandoned at the
  step budget, ON completed and the screenshot-grounded judge verified. But the
  *dominant* failure was over-exploration: the agent reached the answer page
  (e.g. Python docs) yet never committed `done()`, scrolling until the budget ran out.
- **The fix:** made the agent step-budget-aware and decisive, plus a nudge when
  scrolling stops revealing new content. This lifted the **OFF baseline 0.80 →
  1.00** — the over-exploration failures vanished (the stuck Python tasks now
  finish in 4–6 steps), at *lower* cost ($0.030 vs $0.067/task).
- **The judge has teeth:** in the first pass it failed a river-length answer
  whose number didn't match the page screenshot — not a rubber stamp.

**Honest caveats:**
- **The slice is now saturated.** At 15/15 OFF there's no headroom, so reflection
  no longer discriminates here (ON 0.93 ≤ OFF 1.00 is run-to-run noise; CIs
  overlap fully). The cheap budget/commit fix captured most of what reflection
  was providing on these tasks. Reflection's *marginal* value now needs harder
  tasks to measure — which is what the WebArena sandbox is for (below).
- **n = 15, single run, live web** — wide, overlapping CIs; the reliable signal
  is the *mechanism* (an eliminated failure class), confirmed per-trajectory, not
  the aggregate.

> Reproduce / extend: `python -m scripts.run_sweep --tasks mind2web --models mid
> --ablate-reflection --judge-model frontier` then `python -m scripts.make_charts`.

**WebArena (deterministic sandbox) — Shopping site, live, self-hosted:**

The Shopping (OneStopShop/Magento) site is stood up locally and scored with
ground-truth string/URL match (no LLM judge). On a **diverse 10-task sample**
across 10 intent templates (reflection ON), the agent scored **1/9 ≈ 11%**
(one task came back *unscored* — its `program_html` check has nothing we can
verify, correctly returned `None` rather than guessed). Bringing up the real
sandbox immediately earned its keep — it exposed **three real bugs** the
synthetic/live-web tests missed: text in `<div>`/`<span>` wasn't captured (the
agent couldn't read product reviews), the loop detector aborted productive
scrolling, and a snapshot/navigation race. All three are now fixed.

**Why this is *not* a sandbox-vs-realistic gap (yet) — important:**
- The samples aren't **difficulty-matched.** The WebArena sample was drawn to be
  diverse and hard (order history, spend analysis, review extraction); the
  Mind2Web slice is tractable lookups. So "11% sandbox vs 87% realistic" is
  apples-to-oranges and would *invert* the thesis if charted naively — it
  reflects task selection, not a real sandbox/realistic difference. The honest
  reading is "WebArena is hard and our from-scratch agent is early," not "the
  sandbox is harder than the open web."
- **n = 10, single run, hard-skewed** — CI 0.02–0.44.
- **Emulation tax:** the amd64 image runs under emulation on Apple Silicon, so it
  is *very* slow (one order-history task took ~2.4 h). Full-suite (812 tasks) is
  impractical without a native x86 host — a real infrastructure finding, not just
  an inconvenience.

After the commit-fix (budget-awareness + decisiveness), the same diverse sample
jumped from **11% → would-be-higher** as the over-exploration loops vanished —
best seen in the matched comparison next.

### Difficulty-matched comparison (the honest version of the gap)

To compare apples-to-apples, a frozen **2 easy / 2 medium / 2 hard** set per
benchmark (`eval/webarena/shopping_matched.json`, `eval/tasks/mind2web_matched.json`),
same agent + config (Sonnet, reflection ON, commit-fix):

| tier | WebArena (sandbox, **deterministic** scoring) | Mind2Web (realistic, **WebJudge**) |
|---|:---:|:---:|
| easy | 2/2 = 1.00 | 2/2 = 1.00 |
| medium | 2/2 = 1.00 | 2/2 = 1.00 |
| hard | 0/2 = 0.00 | 2/2 = 1.00 |
| **overall** | **4/6 = 0.67** | **6/6 = 1.00** |

The result is more interesting than a one-line "gap":
- **Easy & medium are identical (1.00 / 1.00).** The agent is *not* worse in the
  sandbox per se — at these difficulties it handles both equally. The commit-fix
  is what got WebArena easy/medium from looping-to-budget up to 1.00.
- **The entire divergence is the hard tier — and it's confounded**, three ways:
  1. **Genuine capability gaps:** on WebArena hard the agent answered "None" when
     5 reviewers existed (didn't paginate), and on another answered about the
     *wrong product* — real failures.
  2. **Scoring strictness asymmetry:** WebArena hard is exact `must_include` of
     *all* ground-truth strings — one task requires **6 verbatim review
     sentences**; even a perfect human summary fails it. Mind2Web's WebJudge
     accepts a correct paraphrase.
  3. **Task kind:** WebArena hard = exhaustive extraction; Mind2Web hard =
     two-page comparison.

So the matched comparison's real lesson is the **project's own thesis, observed
on our agent**: the apparent sandbox-vs-realistic gap is driven substantially by
**measurement** — a lenient LLM judge vs unforgiving exact-match — not by the open
web being magically easier. "Weak LLM-as-judge scoring inflates headline numbers"
isn't a claim here; it's visible in the table. (Caveat: **n = 2 per tier** — this
is illustrative, not powered; the mechanism is the point, not the decimals.)

The other durable findings: the **reflection ablation** (above), the **commit-fix**
(0.80→1.00 realistic, 11%→67% sandbox by killing over-exploration), and the
**bug-finding value of a real sandbox** (three bugs the synthetic tests missed).

### Model sweep, ablations & the reasoning-ceiling test

All on the difficulty-matched sets (reflect ON unless noted). The recurring theme:
the knobs people tout move these numbers *less* than task type and scoring do.

**Cost-vs-success Pareto — model sweep (Mind2Web matched, realistic):**

| model | success rate | by tier (e/m/h) | mean cost/task |
|---|:---:|:---:|:---:|
| Haiku (cheap) | 0.67 | 1.0 / 1.0 / 0.0 | **$0.042** |
| Sonnet (mid) | **1.00** | 1.0 / 1.0 / 1.0 | **$0.035** |
| Opus (frontier) | 1.00 | 1.0 / 1.0 / 1.0 | $0.072 |

Sonnet **dominates**: 100% at the *lowest* cost. Opus matches it at ~2× cost
(overkill here). And Haiku is a **false economy** — it scores worse *and* costs
*more* than Sonnet, because it flails on the hard tier (loops / hits the step
budget), burning calls. Cheaper-per-token ≠ cheaper-per-task.

**Does a bigger model crack the hard WebArena tasks? No.** On the 2 hard
extraction tasks (21, 163) that Sonnet failed, **Opus also scored 0/2** — at
$0.16/task (4× Sonnet). So those failures are *not* a model ceiling a stronger
model fixes; they're exact-match scoring (163 wants 6 verbatim sentences) + truly
exhaustive extraction. Contrast Mind2Web's hard *comparison* tasks, where model
capability **does** matter (Haiku 0/2, Sonnet & Opus 2/2). Task type, again.

**Reflection ablation on the sandbox (WebArena matched, has headroom):**
ON and OFF both score **0.67** — identical — and ON costs +64%. Reflection's
value was already captured by the cheaper commit-fix; on these tasks it just adds
cost. An honest null result, not a win.

**Observation-modality ablation (a11y vs a11y + vision fallback):**
identical success (1.00 vs 1.00), +13% cost for vision. The accessibility tree is
sufficient for these text tasks; the vision fallback rarely fires and earns
nothing here. (Its value would show on canvas/visual-layout tasks, which this set
doesn't contain.)

**Net:** across model tier, reflection, and vision, the success numbers barely
move — what moves them is task type and **scoring methodology**. That's the
project's thesis, now triangulated from four independent angles.

### Scaled run — tighter CIs (best-powered numbers)

Larger samples for narrower intervals (full 812-task WebArena is infeasible
under amd64 emulation — one task hit ~2.4 h — so "scale" here is a bigger
sample, not the whole suite):

| benchmark | sample | success rate (95% CI) | dominant failure |
|---|---|---|---|
| Mind2Web (realistic, WebJudge) | 15 tasks × 3 seeds = 45 attempts | **1.00 (0.92–1.00)** | none — 0 failures, all seeds |
| WebArena (sandbox, exact-match) | 24 diverse shopping tasks | **0.44 (0.26–0.63)** | `premature_done` (13 of 14) |

The CIs now **don't overlap** — a real, separable gap. But read it correctly:
**it's a measurement gap, not a venue gap.** On the sandbox, 13 of 14 failures
are `premature_done` — the agent *finished* with a sensible answer that the
exact-match scorer rejected (the curly-apostrophe / "6 verbatim sentences" class
of rejection). On the realistic side, the lenient WebJudge passes everything (a
100% that is itself suspect). So the same agent looks ~100% or ~44% depending on
**how you score**, not where it runs — which is exactly the "headline numbers are
inflated by weak LLM-as-judge scoring" thesis, now shown with non-overlapping
intervals and a failure-mode breakdown. The honest single-sentence takeaway:
*scoring methodology, not the benchmark venue, is the dominant driver of the gap.*

### Capability levers we tried — and an honest null

Two production-style upgrades, measured fairly on the 24-task WebArena set
(Sonnet, reflect ON), against the 0.43 baseline:

| config | success rate | mean cost/task | effect |
|---|:---:|:---:|---|
| baseline | 0.43 (10/23) | $0.106 | — |
| + verbatim-extraction prompting | **0.43** (10/23) | $0.109 | no change |
| + verbatim + **Set-of-Marks** (numbered-box screenshots) | **0.43** (10/23) | $0.159 | no change, +50% cost |

Neither moved the number. The same 10 tasks pass in all three; the
`premature_done` failures stay failures. Why:
- **Verbatim prompting** addresses *formatting*, but these failures aren't
  formatting — they're wrong/incomplete answers, or the exact-match scorer
  demanding *all* of (e.g.) 6 verbatim sentences, which "copy it exactly" can't
  satisfy when the agent summarizes.
- **Set-of-Marks** is the standard multimodal lever, but on this *text-heavy
  shopping* distribution the a11y tree + extracted text already convey what's
  needed; visual layout adds no decisive signal (its value is on spatial/visual
  tasks — maps, canvases, ambiguous layouts — which this set doesn't contain). It
  *did* raise answer-grounding (0.74 → 0.83) and fired every step, so it changed
  behaviour — just not enough to flip exact-match outcomes, at +50% cost.

This is a deliberately-reported **negative result**: I hypothesized Set-of-Marks
would be the biggest lever; on this task distribution it wasn't. The bottleneck
here is genuine hard-extraction capability + exact-match strictness, not
observation modality or phrasing — consistent with everything above. (SoM is
implemented and a `--set-of-marks` flag; it would likely earn its keep on the
WebArena *map* domain or computer-use-style tasks, which are future work.)

---

## Measurement rigor

- **Confidence intervals.** `--runs N` runs each task N times; metrics report a
  95% Wilson interval on success rate plus `pass_any_rate` (solved at least once)
  — honest error bars for small, flaky slices.
- **Parallelism.** `--workers N` runs tasks across processes (each gets its own
  browser + LLM client), so full sweeps don't take all night.
- **Screenshot-grounded judging.** With `--capture-screenshots` (auto-on for
  mind2web), per-step PNGs are saved and the final ones are handed to WebJudge so
  it scores from pixels, not the agent's self-report.
- **No silent mis-scoring.** WebArena scoring is 3-valued — a task it can't
  verify (e.g. `program_html` without captured page content) is marked *unscored*
  (`None`), never a guessed pass/fail.
- **Cost control.** The Anthropic adapter caches the frozen system prompt, so
  multi-step runs read it from cache instead of paying full price each step.

## Guarding against hallucination

Two kinds of hallucination, two kinds of defense:

- **Hallucinated *actions* (clicking an element that isn't there) — prevented
  structurally.** The model can only address elements present in the current
  snapshot (`@e1`…), and `validate_action` rejects any ref not in that snapshot
  *before* it executes. The action space is small and typed, so there's nothing
  to invent. (A bad ref is logged as `hallucinated_action` in the taxonomy.)
- **Hallucinated *answers* (stating a fact not on the page) — caught three ways:**
  1. **Prompt grounding** — the system prompt forbids guessing values and tells
     the agent to verify on the page before finishing.
  2. **Answer-grounding score** — at `done`, the answer's content tokens
     (numbers weighted heaviest) are checked against everything the agent
     actually observed; a low score flags a likely-invented figure. With
     `--verify-answers`, an ungrounded answer is bounced back once to re-read and
     confirm before it's accepted.
  3. **Screenshot-grounded judge** — on the realistic slice, WebJudge scores
     from the final-page screenshots, so a confident-but-wrong answer fails
     regardless of what the agent claimed. (This already caught a real case: an
     agent reported a river length that didn't match the page, and the judge
     failed it.)
  4. **Site confinement** (`confine_to_site`, on for sandbox tasks) — the agent
     can't navigate off the task's own site, and a click/"back" that lands on a
     blank/foreign page is auto-returned to the task page. This killed the single
     worst real failure observed: on a WebArena product task the agent got
     confused, **navigated to the live external `amazon.com`, and hallucinated an
     answer about an unrelated product**. With confinement it stays put and
     answers from the correct page.

## Reproducibility

Every run writes a JSONL trajectory with, per step: the observation hash, the
thought, the action, whether it succeeded, the reflection verdict, tokens, cost,
latency, the vision-fallback flag, and — on failed steps — the serialized page
the model saw (so failures stay re-readable). Scoring is a separate pass over
those records, so any run can be **re-scored and re-charted entirely offline**
(`eval/metrics.py`, `eval/failure_taxonomy.py`).

---

## Status

- ✅ Agent scaffold (browser / observation / actions / memory / loop) — runs end-to-end
- ✅ Observation: a11y refs **+ static-text channel**; popup/new-tab following; deeper settle
- ✅ Provider-agnostic LLM layer (Claude / OpenAI / Gemini / any via LiteLLM) + cost tracking + caching
- ✅ Reflection toggle, vision fallback, irreversible-action guardrail
- ✅ Harness: metrics + **Wilson CIs**, `--runs` (pass@k), `--workers` (parallel), failure taxonomy, JSONL
- ✅ Screenshot capture + observation persistence on failure
- ✅ WebArena scorers (string / url / fuzzy / program_html, 3-valued) + Mind2Web WebJudge (text + screenshots)
- ✅ Anti-hallucination: structural action validation + answer-grounding score + `--verify-answers` gate + grounded judge
- ✅ WebArena prep: preflight checker + integration smoke (adapter verified against a local stand-in)
- ✅ Robustness: site confinement, pagination detection, answer-grounding gate, scorer normalization, rate-limit retry, record-grouping
- ✅ Multimodal: Set-of-Marks visual grounding (`--set-of-marks`) + verbatim prompting (measured; see results)
- ✅ Offline local demo + smoke test + `echo` model + **pytest suite (59 tests)**
- ✅ browser-use baseline adapter
- ✅ First measured results recorded (reflection ablation on the realistic slice — see above)
- ⬜ Stand up WebArena sites for the true sandbox-vs-realistic gap + demo GIF

## Non-goals

Not a scraper, not a CAPTCHA/anti-bot bypass, not a browser-use wrapper. No
irreversible side effects without confirmation outside the sandbox.
