"""WebArena authentication — generate + reuse Playwright storage states.

Most WebArena tasks (`require_login: true`) need a logged-in session. The
official harness logs in once per site with fixed test accounts and saves the
cookies/localStorage as a Playwright ``storage_state`` JSON; we do the same and
hand that state to ``BrowserSession`` so authenticated tasks start logged in.

Credentials and login flows mirror upstream
(web-arena-x/webarena `browser_env/env_config.py` + `auto_login.py`). Site base
URLs are read from our ``WA_*`` env vars.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Test accounts published by WebArena (these are throwaway sandbox creds).
ACCOUNTS: dict[str, dict[str, str]] = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
}

# WebArena site key -> our env var holding its base URL.
SITE_BASE_ENV = {
    "shopping": "WA_SHOPPING",
    "shopping_admin": "WA_SHOPPING_ADMIN",
    "reddit": "WA_REDDIT",
    "gitlab": "WA_GITLAB",
}


def _auth_dir() -> Path:
    d = Path(os.environ.get("WEBARENA_AUTH_DIR", ".auth"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(site: str) -> Path:
    return _auth_dir() / f"{site}_state.json"


def _base(site: str) -> str:
    return os.environ.get(SITE_BASE_ENV.get(site, ""), "").rstrip("/")


def _login(site: str, page: Any) -> None:
    """Drive the site's login form (flows mirror upstream auto_login.py)."""
    base = _base(site)
    acct = ACCOUNTS[site]
    if site == "shopping":
        page.goto(f"{base}/customer/account/login/")
        page.get_by_label("Email", exact=True).fill(acct["username"])
        page.get_by_label("Password", exact=True).fill(acct["password"])
        page.get_by_role("button", name="Sign In").click()
    elif site == "reddit":
        page.goto(f"{base}/login")
        page.get_by_label("Username").fill(acct["username"])
        page.get_by_label("Password").fill(acct["password"])
        page.get_by_role("button", name="Log in").click()
    elif site == "shopping_admin":
        page.goto(f"{base}/admin")
        page.get_by_placeholder("user name").fill(acct["username"])
        page.get_by_placeholder("password").fill(acct["password"])
        page.get_by_role("button", name="Sign in").click()
    elif site == "gitlab":
        page.goto(f"{base}/users/sign_in")
        page.get_by_label("Username or email").fill(acct["username"])
        page.get_by_label("Password").fill(acct["password"])
        page.get_by_role("button", name="Sign in").click()
    else:
        raise ValueError(f"no login flow for site {site!r}")
    page.wait_for_load_state("networkidle", timeout=15_000)


def generate(site: str) -> Path:
    """Log in and write the storage state for ``site``. Requires the site up."""
    if not _base(site):
        raise RuntimeError(f"{SITE_BASE_ENV.get(site)} is not set; can't authenticate {site}")
    from playwright.sync_api import sync_playwright

    out = state_path(site)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        try:
            _login(site, page)
            context.storage_state(path=str(out))
        finally:
            context.close()
            browser.close()
    return out


def ensure_auth(site: str, *, regenerate: bool = False) -> Optional[str]:
    """Return a storage-state path for ``site``, generating it if needed.

    Returns None (rather than raising) if the site isn't configured, so the
    caller can run unauthenticated and let the task fail naturally.
    """
    if site not in SITE_BASE_ENV or not _base(site):
        return None
    p = state_path(site)
    if regenerate or not p.exists():
        try:
            generate(site)
        except Exception:
            return None
    return str(p) if p.exists() else None


def auth_state_for(sites: list[str]) -> Optional[str]:
    """Storage state for the first authenticatable site in ``sites``."""
    for site in sites:
        path = ensure_auth(site)
        if path:
            return path
    return None
