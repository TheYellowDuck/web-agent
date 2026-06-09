"""Compact running state for the agent.

Re-feeding the full step-by-step history every turn is expensive and bloats
context. Instead we keep the goal, the current URL, and a rolling window of the
most recent steps verbatim, with older steps collapsed into a one-line summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent.types import Action


@dataclass
class MemoryEntry:
    index: int
    action_desc: str
    ok: bool
    note: Optional[str] = None  # reflection / error note
    obs_hash: str = ""          # page state when the action was chosen


@dataclass
class Memory:
    goal: str
    recent_window: int = 6
    entries: list[MemoryEntry] = field(default_factory=list)
    current_url: str = ""

    def record(
        self,
        index: int,
        action: Action,
        ok: bool,
        note: Optional[str] = None,
        obs_hash: str = "",
    ) -> None:
        self.entries.append(
            MemoryEntry(index=index, action_desc=action.describe(), ok=ok,
                        note=note, obs_hash=obs_hash)
        )

    def render(self) -> str:
        """History block for the planner prompt."""
        if not self.entries:
            return "(no actions taken yet)"
        lines: list[str] = []
        older = self.entries[: -self.recent_window]
        recent = self.entries[-self.recent_window :]
        if older:
            ok = sum(1 for e in older if e.ok)
            lines.append(
                f"[{len(older)} earlier steps summarized: {ok} ok, "
                f"{len(older) - ok} failed]"
            )
        for e in recent:
            status = "ok" if e.ok else "FAILED"
            line = f"  step {e.index}: {e.action_desc} -> {status}"
            if e.note:
                line += f" ({e.note})"
            lines.append(line)
        return "\n".join(lines)

    @property
    def consecutive_failures(self) -> int:
        n = 0
        for e in reversed(self.entries):
            if e.ok:
                break
            n += 1
        return n

    def recent_actions(self, k: int = 3) -> list[str]:
        return [e.action_desc for e in self.entries[-k:]]

    def looping(self, window: int = 4) -> bool:
        """Stuck = the same action repeated AND the page never changed.

        Requiring an unchanged observation hash means productive scrolling /
        repeated navigation that *does* reveal new content isn't mistaken for a
        loop — only genuinely no-progress repetition aborts the run.
        """
        recent = self.entries[-window:]
        if len(recent) < window:
            return False
        same_action = len({e.action_desc for e in recent}) == 1
        same_page = len({e.obs_hash for e in recent}) == 1 and recent[0].obs_hash != ""
        return same_action and same_page
