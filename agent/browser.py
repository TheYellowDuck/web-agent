"""Playwright browser session — the single source of truth for page interaction.

Exposes a tiny surface: ``goto``, ``snapshot``, ``act``, ``screenshot``,
``close``. Headless for eval runs, headed for debugging. Actions are executed by
locating the ``data-webagent-ref`` stamped during the last snapshot, so an
action can only touch an element the model actually saw.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Optional

from agent.observation import Observation, snapshot
from agent.types import Action


class ActionError(Exception):
    """Raised when an action cannot be executed against the page."""


class BrowserSession:
    def __init__(
        self,
        *,
        headless: bool = True,
        viewport: tuple[int, int] = (1280, 1100),
        default_timeout_ms: int = 10_000,
        user_agent: Optional[str] = None,
        storage_state: Optional[str] = None,
    ):
        self.headless = headless
        self.viewport = viewport
        self.default_timeout_ms = default_timeout_ms
        self.user_agent = user_agent
        # Playwright storage_state (cookies + localStorage) — used to start a
        # session already logged in (e.g. WebArena authenticated tasks).
        self.storage_state = storage_state
        self._pw = None
        self._browser = None
        self._context = None
        self.page = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "BrowserSession":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        ctx_kwargs: dict[str, Any] = {
            "viewport": {"width": self.viewport[0], "height": self.viewport[1]},
            "user_agent": self.user_agent,
        }
        if self.storage_state and os.path.exists(self.storage_state):
            ctx_kwargs["storage_state"] = self.storage_state
        self._context = self._browser.new_context(**ctx_kwargs)
        self._context.set_default_timeout(self.default_timeout_ms)
        self.page = self._context.new_page()
        return self

    def __enter__(self) -> "BrowserSession":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        for closer in (self._context, self._browser):
            try:
                if closer is not None:
                    closer.close()
            except Exception:
                pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._pw = self._browser = self._context = self.page = None

    # -- perception ------------------------------------------------------
    def goto(self, url: str, *, wait_until: str = "domcontentloaded") -> None:
        self._require_page()
        self.page.goto(url, wait_until=wait_until)

    def snapshot(self, *, max_elements: int = 120) -> Observation:
        self._require_page()
        self._settle()
        try:
            return snapshot(self.page, max_elements=max_elements)
        except Exception:
            # "Execution context was destroyed" — a navigation was in flight when
            # the snapshot script ran. Let it finish and retry once.
            self._settle()
            self.page.wait_for_timeout(400)
            return snapshot(self.page, max_elements=max_elements)

    def screenshot(self, *, full_page: bool = False) -> str:
        """Return a base64-encoded PNG (for the vision fallback)."""
        self._require_page()
        png = self.page.screenshot(full_page=full_page)
        return base64.b64encode(png).decode("ascii")

    @property
    def url(self) -> str:
        return self.page.url if self.page else ""

    # -- action execution ------------------------------------------------
    def act(self, action: Action) -> None:
        """Execute a validated action. Raises ActionError on failure."""
        self._require_page()
        handler = {
            "click": self._click,
            "type": self._type,
            "select": self._select,
            "scroll": self._scroll,
            "hover": self._hover,
            "press": self._press,
            "upload": self._upload,
            "navigate": self._navigate,
            "wait": self._wait,
            "note": lambda a: None,    # scratchpad only — no page interaction
            "done": lambda a: None,
        }.get(action.type)
        if handler is None:
            raise ActionError(f"no executor for action type '{action.type}'")
        try:
            handler(action)
            self._settle()            # also pumps the loop so popup events fire
            if self._sync_active_page():  # a click may have opened a new tab
                self._settle()        # let the newly-active page load
        except ActionError:
            raise
        except Exception as e:  # surface Playwright errors as ActionError
            raise ActionError(f"{action.type} failed: {e}") from e

    def _locator(self, ref: str):
        sel = f'[data-webagent-ref="{ref.lstrip("@")}"]'
        return self.page.locator(sel)

    def _click(self, a: Action) -> None:
        self._locator(a.ref).click(timeout=self.default_timeout_ms)

    def _type(self, a: Action) -> None:
        loc = self._locator(a.ref)
        loc.click(timeout=self.default_timeout_ms)
        loc.fill("")  # clear existing value
        loc.type(a.text or "", delay=15)

    def _select(self, a: Action) -> None:
        loc = self._locator(a.ref)
        try:
            loc.select_option(label=a.option)
        except Exception:
            loc.select_option(value=a.option)

    def _hover(self, a: Action) -> None:
        # Reveal hover-triggered menus / tooltips (then the next snapshot can see
        # the elements they expose).
        self._locator(a.ref).hover(timeout=self.default_timeout_ms)

    def _press(self, a: Action) -> None:
        # Focus the ref first if given (e.g. press Enter in a specific search
        # box); otherwise press at the page level (e.g. Escape to close a modal).
        if a.ref:
            self._locator(a.ref).press(a.key or "", timeout=self.default_timeout_ms)
        else:
            self.page.keyboard.press(a.key or "")

    def _upload(self, a: Action) -> None:
        if not os.path.exists(a.path or ""):
            raise ActionError(f"upload path does not exist: {a.path!r}")
        self._locator(a.ref).set_input_files(a.path, timeout=self.default_timeout_ms)

    def _scroll(self, a: Action) -> None:
        direction = a.direction or "down"
        dy = {"down": 700, "up": -700}.get(direction, 0)
        dx = {"right": 700, "left": -700}.get(direction, 0)
        if a.ref:  # scroll a specific element into view
            try:
                self._locator(a.ref).scroll_into_view_if_needed()
                return
            except Exception:
                pass
        self.page.mouse.wheel(dx, dy)

    def _navigate(self, a: Action) -> None:
        target = (a.target or "").strip()
        if target == "back":
            self.page.go_back()
        elif target == "forward":
            self.page.go_forward()
        elif target:
            self.page.goto(target, wait_until="domcontentloaded")
        else:
            raise ActionError("navigate target missing")

    def _wait(self, a: Action) -> None:
        self.page.wait_for_timeout(min(a.ms or 1000, 10_000))

    # -- helpers ---------------------------------------------------------
    def _sync_active_page(self) -> bool:
        """Follow popups/new tabs and recover if the current page was closed.

        A click with target=_blank (or window.open) opens a new page in the
        context asynchronously; without this the agent keeps acting on the stale
        page. Returns True if the active page changed.
        """
        if self._context is None:
            return False
        # Give a just-triggered popup event a moment to register.
        try:
            (self.page or self._context.pages[-1]).wait_for_timeout(120)
        except Exception:
            pass
        open_pages = [p for p in self._context.pages if not p.is_closed()]
        if not open_pages:
            return False
        if self.page is None or self.page.is_closed() or open_pages[-1] is not self.page:
            self.page = open_pages[-1]
            try:
                self.page.bring_to_front()
            except Exception:
                pass
            return True
        return False

    def _settle(self) -> None:
        """Best-effort wait for the page to quiesce after an action."""
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=4000)
        except Exception:
            pass
        # SPA content often arrives after DOMContentLoaded; wait briefly for the
        # network to go idle, but don't hang on long-poll/streaming connections.
        try:
            self.page.wait_for_load_state("networkidle", timeout=2500)
        except Exception:
            pass

    def _require_page(self) -> None:
        if self.page is None:
            raise ActionError("browser session not started; call start() first")
