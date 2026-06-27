"""Tool-use vs. text-parsed JSON — does native function-calling cut malformed actions?

A standard recommendation is to emit agent actions via the provider's tool-use /
function-calling API rather than parsing JSON from text. This quantifies the
headroom that move actually has *on this project*, offline, over every action the
agent has ever taken in the saved trajectories:

  • parse errors      — the model emitted unparseable output (tool-use COULD fix)
  • validation errors — a bad/hallucinated ref etc. (tool-use can NOT fix; the
                        schema can't know which refs are live)

If the structured-output path already yields ~0 parse errors, native tool-use is
an equivalent alternative, not an upgrade — a defensible negative result.

    python -m scripts.measure_action_format                 # offline scan
    python -m scripts.measure_action_format --live cheap    # + one live A/B call
"""

from __future__ import annotations

import argparse
import glob
from typing import Optional

from eval.metrics import load_trajectories

_PARSE = ("parse error",)
_VALID = ("not found", "requires", "unknown action", "invalid", "unsupported key")


def scan_offline(paths: list[str]) -> None:
    n_act = parse_err = valid_err = exec_err = 0
    for p in paths:
        for r in load_trajectories(p):
            for s in r.get("steps", []):
                n_act += 1
                e = (s.get("action_error") or "").lower()
                if not e:
                    continue
                if any(k in e for k in _PARSE):
                    parse_err += 1
                elif any(k in e for k in _VALID):
                    valid_err += 1
                else:
                    exec_err += 1
    print(f"OFFLINE — {n_act} real actions across {len(paths)} saved runs")
    print(f"  malformed (parse errors)   : {parse_err}  "
          f"({parse_err / n_act * 100:.2f}%)  ← the only class tool-use could fix")
    print(f"  invalid (bad ref / args)   : {valid_err}  ({valid_err / n_act * 100:.2f}%)")
    print(f"  execution failures         : {exec_err}  ({exec_err / n_act * 100:.2f}%)")
    if parse_err == 0:
        print("  → structured JSON output already yields a 0% malformed-action rate; "
              "native tool-use has no correctness headroom here (equivalent, not an upgrade).")


def live_ab(model: str) -> None:
    """One forced-action call each via JSON-schema and tool-use; confirm both
    return a valid parsed action. Cheap (2 calls on a tiny prompt)."""
    from agent import actions as A
    from agent.llm import make_llm_client
    from agent.prompts import SYSTEM_PROMPT

    schema = A.action_output_schema()
    msg = [{"role": "user", "content":
            "GOAL: search for shoes\nCURRENT PAGE:\nELEMENTS:\n  @e1 <searchbox> "
            "\"Search\"\nRespond with the next action."}]
    print(f"\nLIVE A/B — model={model}")
    for use_tools in (False, True):
        client = make_llm_client(model, use_tools=use_tools)
        r = client.complete(system=SYSTEM_PROMPT, messages=msg, json_schema=schema)
        ok = False
        try:
            act = A.parse_action(r.parsed or {})
            ok = act.type in __import__("agent.types", fromlist=["ALL_ACTION_TYPES"]).ALL_ACTION_TYPES
        except Exception as e:  # noqa
            act = f"<unparseable: {e}>"
        mode = "tool-use   " if use_tools else "json-schema"
        print(f"  {mode}: parsed_ok={ok}  action={getattr(act, 'type', act)}  "
              f"cost=${r.cost_usd:.5f}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", nargs="*",
                    default=glob.glob("results/trajectories/*.jsonl"))
    ap.add_argument("--live", default=None, metavar="MODEL",
                    help="also run one tool-use vs json call on MODEL (e.g. 'cheap')")
    args = ap.parse_args(argv)
    scan_offline(args.trajectories)
    if args.live:
        from agent.env import load_dotenv
        from agent.llm import resolve_tier
        load_dotenv()
        live_ab(resolve_tier(args.live))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
