"""Prompt construction for planning and reflection.

Kept in one place so prompts are easy to read, diff, and ablate. The system
prompt is frozen (no per-request interpolation) to stay cache-friendly.
"""

from __future__ import annotations

from agent.memory import Memory
from agent.observation import Observation

SYSTEM_PROMPT = """\
You are a precise web-browsing agent. You complete a user's goal by issuing one \
typed action at a time against a live web page.

You perceive the page as a numbered list of interactable elements, each with a \
ref like @e12. You may ONLY act on elements present in the current list.

Action space (emit exactly one per turn):
- click(ref)                — click an element
- type(ref, text)           — focus an input and type text (replaces existing value)
- select(ref, option)       — choose an option in a <select> by visible label
- scroll(direction|ref)     — scroll "up"/"down" or bring a ref into view
- hover(ref)                — hover an element to reveal a menu/tooltip it controls
- press(key, ref?)          — press a key like "Enter" (submit a focused field),
                              "Escape" (dismiss a dialog), "ArrowDown" (move in a menu)
- upload(ref, path)         — set a file input to a local file path
- navigate(target)          — go to a url, or "back"/"forward"
- wait(ms)                  — wait for the page to update
- note(text)                — record a finding to your scratchpad (no page change)
- done(answer)              — finish; include the extracted answer if the task asks for one

Rules:
- For "list all / find every" or extraction tasks, use note(text) to record each \
matching item the moment you see it (one per item, copied exactly), across pages \
— don't try to hold the whole list in your head. Then assemble done's answer from \
your accumulated NOTES so nothing is dropped.
- Choose the single best next action toward the goal.
- Prefer the accessibility list; only rely on the screenshot if one is provided.
- The ELEMENTS and PAGE TEXT you are given already cover the WHOLE page, not just
  the visible part — scrolling does NOT reveal more text. Read what you already
  have. Only scroll to trigger lazy-loaded content or bring a specific element
  into view; never scroll just to "see the rest" of text that's already listed.
- Do not invent refs. If nothing useful is available, scroll or navigate.
- When the goal is achieved (or the answer is found), emit done with the answer.
- Ground every claim: only state facts you can actually see in the current \
ELEMENTS or PAGE TEXT. If you have not seen the information needed, navigate or \
scroll to find it — never guess a value (a price, number, name, date) you have \
not read on a page.
- Before calling done with a factual answer, make sure that exact answer is \
supported by text you observed. If you are not certain, take one more step to \
re-read the relevant page and verify, then finish.
- Be decisive and mind your step budget. As soon as you have enough information \
to answer, call done — do NOT keep scrolling or re-reading a page that reveals \
no new information; that wastes your limited steps. Scrolling the same page \
back and forth is never useful.
- If the answer is that nothing matches (e.g. no items/reviews qualify), that is \
a valid answer — call done and say so (e.g. "None"). But for "list all / find \
every" tasks, first make sure you've seen all the items: if a list or reviews \
span multiple pages, use the pagination controls (Next, page numbers) to go \
through them before concluding — don't answer "none" from only the first page.
- Quote the answer VERBATIM from the page. When the task asks for a specific \
value, name, title, price, or quote, copy it exactly as written (same spelling, \
casing, punctuation, units) — do NOT paraphrase, round, summarize, or reformat. \
For "list" tasks, give the exact names/strings as they appear, separated by \
commas. Exact wording is graded.
- Stay on the task's own website. Never navigate to a different site to look for \
the answer; the information is on the current site.
- Never submit irreversible actions (purchases, sending messages, account \
changes) unless the goal explicitly requires it.
- Keep the "thought" short — one or two sentences."""


def build_planner_messages(
    goal: str,
    obs: Observation,
    memory: Memory,
    *,
    extra_note: str | None = None,
    vision: bool = False,
    set_of_marks: bool = False,
    planning: bool = False,
    plan: str = "",
    step_index: int = 0,
    max_steps: int = 15,
    workflows: str = "",
) -> list[dict[str, str]]:
    """User-turn content for an action decision."""
    remaining = max_steps - step_index
    blocks = [
        f"GOAL: {goal}",
        f"STEP: {step_index + 1} of {max_steps} ({remaining} step(s) left)",
    ]
    if workflows:
        blocks += ["", workflows]
    if planning:
        blocks += ["", "PLAN SO FAR:",
                   plan.strip() or "(none yet — create one in the 'plan' field)"]
    blocks += ["", "HISTORY:", memory.render()]
    if memory.notes:
        blocks += ["", "NOTES (your scratchpad — assemble the final answer from these):",
                   memory.render_notes()]
    blocks += ["", "CURRENT PAGE:", obs.serialize()]
    if set_of_marks:
        blocks.append("")
        blocks.append(
            "A screenshot with NUMBERED boxes is attached: box N marks element @eN. "
            "Use the visual layout to pick the right element, and address it by its "
            "@e ref in your action."
        )
    elif vision:
        blocks.append("")
        blocks.append(
            "A screenshot of the current page is also attached for disambiguation."
        )
    # Escalating urgency so the agent commits rather than burning the budget.
    if remaining <= 1:
        blocks += ["", "This is your LAST step — call done now with your best answer."]
    elif remaining <= max(2, max_steps // 4):
        blocks += [
            "",
            "You are low on steps. If you already have enough to answer, call done "
            "now instead of exploring further.",
        ]
    if extra_note:
        blocks.append("")
        blocks.append(f"NOTE: {extra_note}")
    blocks.append("")
    if planning:
        blocks.append(
            "Maintain your checklist in the 'plan' field every step (mark subgoals "
            "done, keep what remains). For 'list all' tasks, the plan must include "
            "visiting every page/section before answering. Respond with JSON: "
            "{\"thought\": str, \"plan\": str, \"action\": {\"type\": ..., ...}}."
        )
    else:
        blocks.append(
            "Respond with JSON: {\"thought\": str, \"action\": {\"type\": ..., ...}}."
        )
    return [{"role": "user", "content": "\n".join(blocks)}]


REFLECTION_INSTRUCTION = (
    "You just executed an action. Using the BEFORE and AFTER snapshots above, "
    "decide whether the action achieved its intended effect (made progress "
    "toward the goal). Look at what actually CHANGED: did the URL change, did new "
    "elements/text appear, did a value get entered? If the page is effectively "
    "unchanged (same URL, same elements) the action most likely did nothing — "
    "say so. Respond with JSON: {\"ok\": bool, \"note\": str}. If not ok, the note "
    "should say what went wrong so the next step can recover."
)


def build_reflection_messages(
    goal: str,
    action_desc: str,
    before: Observation,
    after: Observation,
) -> list[dict[str, str]]:
    # Give the judge BOTH states so it can detect "nothing changed" — a no-op
    # click/scroll otherwise reads as success just because the page is non-empty.
    url_changed = before.url != after.url
    content = "\n".join(
        [
            f"GOAL: {goal}",
            f"ACTION TAKEN: {action_desc}",
            "",
            f"URL BEFORE: {before.url}",
            f"URL AFTER:  {after.url}"
            + ("" if url_changed else "   (unchanged)"),
            "",
            "PAGE BEFORE (elements):",
            before.serialize(max_chars=1500),
            "",
            "PAGE AFTER (elements):",
            after.serialize(max_chars=1500),
            "",
            REFLECTION_INSTRUCTION,
        ]
    )
    return [{"role": "user", "content": content}]
