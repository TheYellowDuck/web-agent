"""Typed, small action space + validation + JSON schema for structured output.

The executor that actually drives the page lives on ``BrowserSession.act`` —
this module owns the *contract*: what the model may emit, how it's parsed, and
how it's validated against the current snapshot before anything runs.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.observation import Observation
from agent.types import ALL_ACTION_TYPES, Action

# Refs are required for these (they address a snapshot element).
_NEEDS_REF = {"click", "type", "select"}


def action_output_schema() -> dict[str, Any]:
    """JSON schema describing ``{thought, action}`` for structured output.

    Deliberately lenient (no ``additionalProperties: false``, only ``type``
    required on the action) so it round-trips across Anthropic / OpenAI / Gemini
    / LiteLLM. The hard guarantees come from ``validate_action`` after parsing.
    """
    return {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Brief reasoning for the chosen action.",
            },
            "action": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(ALL_ACTION_TYPES)},
                    "ref": {
                        "type": "string",
                        "description": "Element ref like '@e12' (click/type/select).",
                    },
                    "text": {"type": "string", "description": "Text to type."},
                    "option": {"type": "string", "description": "Option for select."},
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                    },
                    "target": {
                        "type": "string",
                        "description": "url | 'back' | 'forward' for navigate.",
                    },
                    "ms": {"type": "integer", "description": "Milliseconds for wait."},
                    "answer": {
                        "type": "string",
                        "description": "Extracted answer for done.",
                    },
                },
                "required": ["type"],
            },
        },
        "required": ["thought", "action"],
    }


def reflection_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ok": {
                "type": "boolean",
                "description": "True if the action achieved its intended effect.",
            },
            "note": {
                "type": "string",
                "description": "If not ok, what went wrong / what to try next.",
            },
        },
        "required": ["ok"],
    }


def parse_action(payload: dict[str, Any]) -> Action:
    """Pull an Action out of a parsed ``{thought, action}`` (or bare action)."""
    raw = payload.get("action", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"action must be an object, got {type(raw).__name__}")
    if "type" not in raw:
        raise ValueError("action is missing 'type'")
    action = Action.from_dict(raw)
    # Normalise a couple of common model quirks.
    if action.ref and not action.ref.startswith("@"):
        action.ref = "@" + action.ref.lstrip("@")
    return action


def validate_action(action: Action, obs: Observation) -> Optional[str]:
    """Return an error string if the action is invalid against ``obs``, else None."""
    if action.type not in ALL_ACTION_TYPES:
        return f"unknown action type '{action.type}'"

    if action.type in _NEEDS_REF:
        if not action.ref:
            return f"'{action.type}' requires a 'ref'"
        if action.ref not in obs.refs:
            return f"ref {action.ref} not found in current snapshot"

    if action.type == "type" and action.text is None:
        return "'type' requires 'text'"
    if action.type == "select" and not action.option:
        return "'select' requires 'option'"
    if action.type == "scroll" and action.direction not in (
        "up",
        "down",
        "left",
        "right",
        None,
    ):
        return f"invalid scroll direction '{action.direction}'"
    if action.type == "navigate" and not action.target:
        return "'navigate' requires 'target' (url|back|forward)"
    return None
