"""Measure AgentOccam-style observation pruning — offline, on real saved pages.

Two measurements, both reproducible without the WebArena sandbox:

  1. OFFLINE on real Magento/WebArena pages: parse the serialized observations
     saved in the trajectory JSONL (131 real snapshots), apply the page-text
     substring-dedup that `agent.observation.prune_observation` performs, and
     report the token/char reduction. This is the redundancy that actually
     occurs on complex e-commerce pages.

  2. LIVE on the bundled local site (no API, just a browser): snapshot each page,
     serialize with and without pruning, report the delta end-to-end.

    python -m scripts.measure_pruning \
        --trajectories results/trajectories/webarena-scale__claude-sonnet-4-6__reflect-on.jsonl
"""

from __future__ import annotations

import argparse
import glob
import statistics
from typing import Optional

from agent.observation import _dedup_substrings
from eval.metrics import load_trajectories

# A rough chars→tokens factor for reporting (≈4 chars/token for English+markup).
CHARS_PER_TOKEN = 4.0


def _page_text_lines(serialized: str) -> tuple[list[str], int, int]:
    """Split a serialized observation into (page_text_items, head_chars, total)."""
    lines = serialized.splitlines()
    try:
        idx = lines.index("PAGE TEXT:")
    except ValueError:
        return [], len(serialized), len(serialized)
    head = "\n".join(lines[:idx])
    items = [ln[4:] for ln in lines[idx + 1:] if ln.startswith("  - ")]
    return items, len(head), len(serialized)


def measure_offline(paths: list[str]) -> None:
    snaps = []
    for p in paths:
        for row in load_trajectories(p):
            for s in row.get("steps", []):
                t = s.get("observation_text")
                if t and "PAGE TEXT:" in t:
                    snaps.append(t)
    if not snaps:
        print("no saved observations with PAGE TEXT found.")
        return

    reductions, before_chars, after_chars = [], 0, 0
    n_with_redundancy = 0
    for t in snaps:
        items, head_chars, total = _page_text_lines(t)
        pruned = _dedup_substrings(items)
        before_text = sum(len(x) for x in items)
        after_text = sum(len(x) for x in pruned)
        before = total
        after = head_chars + after_text + (total - head_chars - before_text)
        before_chars += before
        after_chars += after
        if after < before:
            n_with_redundancy += 1
        reductions.append(1 - after / before if before else 0.0)

    print(f"OFFLINE — {len(snaps)} real saved observations "
          f"({sum('localhost:7770' in s for s in snaps)} WebArena/Magento)")
    print(f"  page-text lines pruned on {n_with_redundancy}/{len(snaps)} snapshots")
    print(f"  total chars  {before_chars:,} → {after_chars:,}  "
          f"(−{(1 - after_chars/before_chars)*100:.1f}%)")
    print(f"  ≈ tokens     {before_chars/CHARS_PER_TOKEN:,.0f} → "
          f"{after_chars/CHARS_PER_TOKEN:,.0f}  "
          f"(−{(before_chars-after_chars)/CHARS_PER_TOKEN:,.0f} tok total)")
    print(f"  per-snapshot reduction: mean {statistics.fmean(reductions)*100:.1f}%, "
          f"max {max(reductions)*100:.1f}%")


def measure_live_local() -> None:
    try:
        from agent.browser import BrowserSession
        from agent.observation import prune_observation
    except Exception as e:
        print(f"LIVE — skipped ({e})")
        return
    from pathlib import Path
    site = Path(__file__).resolve().parent.parent / "eval" / "local_site"
    pages = sorted(site.glob("*.html"))
    if not pages:
        print("LIVE — no local_site pages found")
        return
    print(f"\nLIVE — {len(pages)} bundled local_site pages")
    b = BrowserSession(headless=True).start()
    try:
        for p in pages:
            b.goto(p.as_uri())
            obs = b.snapshot()
            before = len(obs.serialize())
            after = len(prune_observation(obs).serialize())
            d = (1 - after / before) * 100 if before else 0.0
            print(f"  {p.name:<16} {before:>5} → {after:>5} chars  (−{d:.1f}%)")
    finally:
        b.close()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", nargs="*",
                    default=glob.glob("results/trajectories/webarena-*.jsonl"),
                    help="trajectory JSONL files to measure (default: all webarena runs)")
    ap.add_argument("--no-live", action="store_true", help="skip the live local-site pass")
    args = ap.parse_args(argv)
    measure_offline(args.trajectories)
    if not args.no_live:
        measure_live_local()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
