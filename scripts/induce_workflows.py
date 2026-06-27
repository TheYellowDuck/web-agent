"""Induce Agent Workflow Memory (AWM) routines from solved trajectories.

Reads successful trajectories, induces reusable workflows (see
agent.workflow_memory), prints them as a human-readable artifact, and writes
results/reports/workflows.json so `--workflow-memory` runs can inject them.

    python -m scripts.induce_workflows                       # all saved runs
    python -m scripts.induce_workflows --show-retrieval "Search for usb wifi"
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Optional

from agent.workflow_memory import induce_workflows, select_relevant
from eval.metrics import load_trajectories

OUT = Path("results/reports/workflows.json")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", nargs="*",
                    default=glob.glob("results/trajectories/*.jsonl"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--show-retrieval", default=None, metavar="GOAL",
                    help="also show which workflows would be retrieved for GOAL")
    args = ap.parse_args(argv)

    rows = []
    for p in args.trajectories:
        rows += load_trajectories(p)
    n_success = sum(1 for r in rows if r.get("success"))
    workflows = induce_workflows(rows)

    print(f"Induced {len(workflows)} distinct workflow(s) from {n_success} "
          f"successful trajectories ({len(rows)} total):\n")
    for w in workflows:
        print(w.render())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([w.to_dict() for w in workflows], indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(workflows)} workflows → {out}")

    if args.show_retrieval:
        print(f"\nRetrieved for goal {args.show_retrieval!r}:")
        for w in select_relevant(workflows, args.show_retrieval, k=3):
            print("  " + w.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
