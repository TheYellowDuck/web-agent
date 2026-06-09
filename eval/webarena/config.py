"""WebArena site configuration + URL templating.

WebArena task files contain placeholders like ``__SHOPPING__`` that must be
substituted with the base URL of *your* self-hosted instance. Set these via env
(see .env.example). This mirrors the official harness's templating so standard
WebArena task JSON can be loaded unchanged.
"""

from __future__ import annotations

import os

# Placeholder -> environment variable holding that site's base URL.
SITE_ENV = {
    "__SHOPPING__": "WA_SHOPPING",
    "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
    "__REDDIT__": "WA_REDDIT",
    "__GITLAB__": "WA_GITLAB",
    "__MAP__": "WA_MAP",
    "__WIKIPEDIA__": "WA_WIKIPEDIA",
    "__HOMEPAGE__": "WA_HOMEPAGE",
}


def site_urls() -> dict[str, str]:
    return {ph: os.environ.get(env, "") for ph, env in SITE_ENV.items()}


def substitute(text: str) -> str:
    """Replace all WebArena placeholders in ``text`` with configured base URLs."""
    if not text:
        return text
    for ph, url in site_urls().items():
        if ph in text:
            text = text.replace(ph, url.rstrip("/"))
    return text


def missing_sites(text: str) -> list[str]:
    """Placeholders present in ``text`` that have no configured base URL."""
    urls = site_urls()
    return [ph for ph in SITE_ENV if ph in (text or "") and not urls[ph]]
