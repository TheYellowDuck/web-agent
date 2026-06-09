"""Minimal .env loader (no dependency).

The provider SDKs read credentials from ``os.environ``; this populates it from a
``.env`` file so the documented workflow (drop keys in .env, run) just works.
Existing environment variables always win — we never overwrite what's already set.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> int:
    """Load KEY=VALUE lines from ``path`` into os.environ (without overwriting).

    Returns the number of variables set. Silently does nothing if the file is
    absent. Supports ``#`` comments, blank lines, optional ``export`` prefix,
    and surrounding single/double quotes on values.
    """
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val
            n += 1
    return n


def providers_configured() -> list[str]:
    """Which provider credentials are present in the environment (names only)."""
    out = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        out.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        out.append("openai")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        out.append("gemini")
    return out
