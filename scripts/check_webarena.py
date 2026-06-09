"""WebArena preflight: is the sandbox wired up and reachable?

Run this before a WebArena eval to catch the common setup mistakes (a site base
URL not set, a container not up, a stale task file) before burning a run.

    python -m scripts.check_webarena
    python -m scripts.check_webarena --tasks /path/to/test.raw.json
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

from agent.env import load_dotenv
from eval.webarena.config import SITE_ENV, site_urls


def _ping(url: str, timeout: float = 5.0) -> str:
    if url.startswith("file://"):
        return "ok (file)" if Path(url[len("file://"):]).exists() else "MISSING (file)"
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "webagent-check"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return f"ok (HTTP {r.status})"
    except Exception as e:
        return f"UNREACHABLE ({type(e).__name__})"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check WebArena configuration + connectivity.")
    p.add_argument("--tasks", default=os.environ.get("WEBARENA_TASKS"),
                   help="path to a WebArena task JSON (or set WEBARENA_TASKS)")
    args = p.parse_args(argv)
    load_dotenv()

    urls = site_urls()
    configured = {ph: u for ph, u in urls.items() if u}
    print("WebArena site configuration:")
    any_bad = False
    for ph, env in SITE_ENV.items():
        url = urls[ph]
        if not url:
            print(f"  - {env:<18} (unset)")
            continue
        status = _ping(url)
        if "ok" not in status:
            any_bad = True
        print(f"  - {env:<18} {url}  ->  {status}")

    if not configured:
        print("\nNo site base URLs set. Stand up the WebArena Docker sites and set "
              "WA_* in .env (see README → WebArena setup).")

    print("\nTask file:")
    if not args.tasks:
        print("  WEBARENA_TASKS not set (pass --tasks or set the env var).")
    elif not Path(args.tasks).exists():
        print(f"  MISSING: {args.tasks}")
        any_bad = True
    else:
        import json
        try:
            data = json.loads(Path(args.tasks).read_text())
            n = len(data) if isinstance(data, list) else 1
            print(f"  ok: {args.tasks} ({n} task(s))")
        except Exception as e:
            print(f"  INVALID JSON: {e}")
            any_bad = True

    print("\n" + ("Some checks failed — fix the above before running."
                  if any_bad else "Looks good. Run: python -m eval.harness --tasks webarena --model mid"))
    return 1 if any_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
