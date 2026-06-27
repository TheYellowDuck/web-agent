"""Model-agnostic LLM layer.

Goal: swapping models is a config change, not a rewrite — and *any* LLM is
supported. Three native adapters (Anthropic / OpenAI / Gemini) for first-class
behaviour and accurate cost tracking, plus a LiteLLM universal adapter that
speaks to essentially every provider (Mistral, Llama via Ollama, Bedrock,
Together, Groq, ...). A single factory dispatches on the model string.

All adapters expose the same `complete(...)` returning an `LLMResponse`, track
cumulative tokens + cost, and support an optional JSON schema for structured
action output plus optional images for the vision fallback.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Response + usage
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    text: str
    parsed: Optional[dict[str, Any]]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    raw: Any = None


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.cache_read_tokens += r.cache_read_tokens
        self.cost_usd += r.cost_usd


# ---------------------------------------------------------------------------
# Pricing — USD per 1M tokens. Used by native adapters; LiteLLM computes its
# own cost. Numbers are best-effort and easy to update; unknown models fall
# back to 0 (flagged) so a missing price never crashes a run.
# ---------------------------------------------------------------------------

PRICING: dict[str, tuple[float, float]] = {
    # Anthropic (input, output)
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # OpenAI (approximate public list prices)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "o4-mini": (1.1, 4.4),
    # Google Gemini (approximate)
    "gemini-2.0-flash": (0.1, 0.4),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
}

# Three tiers for the cost-vs-success sweep. Pick the current top SKU per tier
# at run time; this default uses the Claude family.
TIERS: dict[str, str] = {
    "frontier": "claude-opus-4-8",
    "mid": "claude-sonnet-4-6",
    "cheap": "claude-haiku-4-5",
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
) -> float:
    base = _strip_provider(model)
    price = PRICING.get(base)
    if price is None:
        # Unknown model — try a prefix match before giving up.
        for k, v in PRICING.items():
            if base.startswith(k):
                price = v
                break
    if price is None:
        return 0.0
    in_rate, out_rate = price
    # Cached reads are ~0.1x the input rate (Anthropic convention; harmless
    # elsewhere because cache_read_tokens stays 0).
    billable_in = max(input_tokens - cache_read_tokens, 0)
    cost = (
        billable_in * in_rate
        + cache_read_tokens * in_rate * 0.1
        + output_tokens * out_rate
    ) / 1_000_000
    return round(cost, 6)


def _strip_provider(model: str) -> str:
    """`anthropic/claude-opus-4-8` -> `claude-opus-4-8` (LiteLLM-style names)."""
    return model.split("/", 1)[1] if "/" in model else model


# ---------------------------------------------------------------------------
# JSON extraction helper (fallback when a provider lacks strict JSON mode)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Last resort: first balanced-looking object.
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------

Message = dict[str, Any]  # {"role": "user"|"assistant", "content": str}


class BaseLLMClient:
    """Common interface. Subclasses implement `_raw_complete`."""

    provider: str = "base"

    def __init__(self, model: str, *, max_tokens: int = 2048, temperature: float = 0.0,
                 use_tools: bool = False):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        # When True, structured action output is requested via the provider's
        # *tool-use / function-calling* API instead of JSON-schema text output.
        # Lets us measure malformed-action rate: tool-use vs text-parsed JSON.
        self.use_tools = use_tools
        self.usage = Usage()

    def reset_usage(self) -> None:
        self.usage = Usage()

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        json_schema: Optional[dict[str, Any]] = None,
        images_b64: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        t0 = time.time()
        resp = self._raw_complete(
            system=system,
            messages=messages,
            json_schema=json_schema,
            images_b64=images_b64 or [],
            max_tokens=max_tokens or self.max_tokens,
        )
        resp.latency_s = round(time.time() - t0, 3)
        if not resp.cost_usd:
            resp.cost_usd = estimate_cost(
                self.model, resp.input_tokens, resp.output_tokens, resp.cache_read_tokens
            )
        if json_schema is not None and resp.parsed is None:
            resp.parsed = extract_json(resp.text)
        self.usage.add(resp)
        return resp

    def _raw_complete(self, **kwargs: Any) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic (Claude) — native, via the official SDK
# ---------------------------------------------------------------------------


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, model: str, **kw: Any):
        super().__init__(model, **kw)
        import anthropic  # lazy

        # Generous retry budget so transient 429s / 5xx are absorbed with
        # exponential backoff instead of failing a task (esp. under parallelism).
        self._client = anthropic.Anthropic(max_retries=8)

    def _raw_complete(
        self,
        *,
        system: str,
        messages: list[Message],
        json_schema: Optional[dict[str, Any]],
        images_b64: list[str],
        max_tokens: int,
    ) -> LLMResponse:
        api_messages = _to_anthropic_messages(messages, images_b64)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            # Cache the frozen system prompt — it's identical every step, so
            # subsequent steps read it from cache (~0.1x) instead of full price.
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": api_messages,
        }
        tool_mode = self.use_tools and json_schema is not None
        if tool_mode:
            # Function-calling path: the action schema becomes a forced tool, so
            # the model returns a validated structured tool_use block instead of
            # text we have to parse. (Compared against the json path below.)
            kwargs["tools"] = [{
                "name": "emit_action",
                "description": "Emit the chosen thought and next action.",
                "input_schema": json_schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "emit_action"}
        elif json_schema is not None:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": _anthropic_strict_schema(json_schema),
                }
            }
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if tool_mode:
            parsed = next((b.input for b in resp.content
                           if getattr(b, "type", None) == "tool_use"), None)
        else:
            parsed = extract_json(text) if json_schema is not None else None
        usage = resp.usage
        return LLMResponse(
            text=text,
            parsed=parsed,
            model=self.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            raw=resp,
        )


def _to_anthropic_messages(messages: list[Message], images_b64: list[str]) -> list[Message]:
    out: list[Message] = [dict(m) for m in messages]
    if images_b64 and out:
        last = out[-1]
        blocks: list[dict[str, Any]] = []
        if isinstance(last["content"], str):
            blocks.append({"type": "text", "text": last["content"]})
        else:
            blocks.extend(last["content"])
        for b64 in images_b64:
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                }
            )
        last["content"] = blocks
    return out


def _anthropic_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a lenient schema to Anthropic structured-output rules.

    Anthropic requires every object to set ``additionalProperties: false`` and
    list all properties in ``required``. We keep the shared schema lenient (for
    OpenAI/Gemini/LiteLLM) and tighten only here: originally-optional fields are
    made nullable so the model can still omit them by emitting null.
    """
    import copy

    s = copy.deepcopy(schema)
    _strictify(s)
    return s


def _strictify(node: Any) -> None:
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        props = node["properties"]
        required = set(node.get("required", []))
        for key, sub in props.items():
            _strictify(sub)
            if key not in required:
                _make_nullable(sub)
        node["required"] = list(props.keys())
        node["additionalProperties"] = False
    if node.get("type") == "array" and isinstance(node.get("items"), dict):
        _strictify(node["items"])


def _make_nullable(sub: dict[str, Any]) -> None:
    t = sub.get("type")
    if isinstance(t, str):
        sub["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        sub["type"] = [*t, "null"]
    # Anthropic rejects an enum under a nullable type union; drop it here
    # (these fields are still validated in agent.actions.validate_action).
    sub.pop("enum", None)


# ---------------------------------------------------------------------------
# OpenAI — native, via the official SDK
# ---------------------------------------------------------------------------


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, model: str, **kw: Any):
        super().__init__(model, **kw)
        import openai  # lazy

        self._client = openai.OpenAI(max_retries=8)

    def _raw_complete(
        self,
        *,
        system: str,
        messages: list[Message],
        json_schema: Optional[dict[str, Any]],
        images_b64: list[str],
        max_tokens: int,
    ) -> LLMResponse:
        api_messages: list[Message] = [{"role": "system", "content": system}]
        api_messages += _to_openai_messages(messages, images_b64)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": api_messages,
        }
        if json_schema is not None:
            # strict=False keeps the schema portable (no all-required /
            # additionalProperties constraints); parse_action + validate_action
            # provide the hard guarantees.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_action",
                    "schema": json_schema,
                    "strict": False,
                },
            }
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=text,
            parsed=extract_json(text) if json_schema is not None else None,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=resp,
        )


def _to_openai_messages(messages: list[Message], images_b64: list[str]) -> list[Message]:
    out: list[Message] = [dict(m) for m in messages]
    if images_b64 and out:
        last = out[-1]
        parts: list[dict[str, Any]] = [{"type": "text", "text": last["content"]}]
        for b64 in images_b64:
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        last["content"] = parts
    return out


# ---------------------------------------------------------------------------
# Google Gemini — native, via google-genai
# ---------------------------------------------------------------------------


class GeminiClient(BaseLLMClient):
    provider = "gemini"

    def __init__(self, model: str, **kw: Any):
        super().__init__(model, **kw)
        from google import genai  # lazy

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _raw_complete(
        self,
        *,
        system: str,
        messages: list[Message],
        json_schema: Optional[dict[str, Any]],
        images_b64: list[str],
        max_tokens: int,
    ) -> LLMResponse:
        from google.genai import types as gt  # lazy

        contents = _to_gemini_contents(messages, images_b64)
        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if json_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = json_schema
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gt.GenerateContentConfig(**config),
        )
        text = resp.text or ""
        meta = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=text,
            parsed=extract_json(text) if json_schema is not None else None,
            model=self.model,
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            raw=resp,
        )


def _to_gemini_contents(messages: list[Message], images_b64: list[str]) -> Any:
    import base64

    from google.genai import types as gt

    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        parts = [gt.Part.from_text(text=str(m["content"]))]
        contents.append(gt.Content(role=role, parts=parts))
    if images_b64 and contents:
        for b64 in images_b64:
            contents[-1].parts.append(
                gt.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png")
            )
    return contents


# ---------------------------------------------------------------------------
# LiteLLM — universal adapter. Speaks to ~everything.
# ---------------------------------------------------------------------------


class LiteLLMClient(BaseLLMClient):
    """Supports any LLM LiteLLM supports. Use provider-prefixed model names,
    e.g. `mistral/mistral-large-latest`, `ollama/llama3`, `groq/llama-3.1-70b`,
    `bedrock/anthropic.claude-...`."""

    provider = "litellm"

    def __init__(self, model: str, **kw: Any):
        super().__init__(model, **kw)
        import litellm  # lazy

        self._litellm = litellm

    def _raw_complete(
        self,
        *,
        system: str,
        messages: list[Message],
        json_schema: Optional[dict[str, Any]],
        images_b64: list[str],
        max_tokens: int,
    ) -> LLMResponse:
        api_messages: list[Message] = [{"role": "system", "content": system}]
        api_messages += _to_openai_messages(messages, images_b64)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": api_messages,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agent_action", "schema": json_schema},
            }
        resp = self._litellm.completion(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        cost = 0.0
        try:
            cost = float(self._litellm.completion_cost(completion_response=resp)) or 0.0
        except Exception:
            cost = 0.0
        return LLMResponse(
            text=text,
            parsed=extract_json(text) if json_schema is not None else None,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# Scripted client — deterministic, for offline tests/demos (no API key).
# ---------------------------------------------------------------------------


class ScriptedLLMClient(BaseLLMClient):
    """Replays a fixed list of action dicts. Lets the whole harness run with no
    network or API key — used by the smoke test."""

    provider = "scripted"

    def __init__(self, actions: list[dict[str, Any]], model: str = "scripted"):
        super().__init__(model)
        self._actions = list(actions)
        self._i = 0

    def _raw_complete(
        self,
        *,
        system: str,
        messages: list[Message],
        json_schema: Optional[dict[str, Any]],
        images_b64: list[str],
        max_tokens: int,
    ) -> LLMResponse:
        # Reflection calls pass a yes/no schema; answer "ok" for those.
        if json_schema and "ok" in (json_schema.get("properties") or {}):
            payload: dict[str, Any] = {"ok": True, "note": "scripted"}
        elif self._i < len(self._actions):
            payload = {"thought": "scripted step", "action": self._actions[self._i]}
            self._i += 1
        else:
            payload = {"thought": "scripted fallback", "action": {"type": "done"}}
        text = json.dumps(payload)
        return LLMResponse(text=text, parsed=payload, model=self.model)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_llm_client(
    model: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    use_tools: bool = False,
) -> BaseLLMClient:
    """Build a client for `model`. Provider is inferred from the name unless
    given explicitly. A provider-prefixed name (``vendor/model``) routes to the
    universal LiteLLM adapter so *any* LLM works out of the box."""

    model = model or os.environ.get("WEBAGENT_MODEL") or TIERS["mid"]
    # Offline deterministic model — immediately emits done(); no key/network.
    # Useful for testing the harness/parallel plumbing.
    if model in ("echo", "noop"):
        return ScriptedLLMClient([], model=model)
    provider = provider or _infer_provider(model)
    kw = {"max_tokens": max_tokens, "temperature": temperature, "use_tools": use_tools}

    if provider == "litellm":
        return _make_litellm(model, kw)

    native = {
        "anthropic": (AnthropicClient, model),
        "openai": (OpenAIClient, model),
        "gemini": (GeminiClient, model),
    }
    if provider in native:
        cls, m = native[provider]
        try:
            return cls(m, **kw)
        except ImportError:
            # Native SDK not installed — fall back to the universal adapter so
            # the user never has to hand-install a provider library.
            return _make_litellm(_litellm_name(provider, model), kw, hint=provider)
    raise ValueError(f"Unknown provider {provider!r} for model {model!r}")


def _litellm_name(provider: str, model: str) -> str:
    """LiteLLM wants provider-prefixed names for non-OpenAI models."""
    if "/" in model or provider == "openai":
        return model
    return f"{provider}/{model}"


def _make_litellm(model: str, kw: dict[str, Any], hint: Optional[str] = None) -> BaseLLMClient:
    try:
        return LiteLLMClient(model, **kw)
    except ImportError:
        extra = hint if hint in ("openai", "gemini") else "litellm"
        raise ImportError(
            f"No installed adapter for model {model!r}. Install the universal "
            f"adapter:  pip install 'web-agent[litellm]'   (or the native SDK: "
            f"pip install 'web-agent[{extra}]')."
        ) from None


def _infer_provider(model: str) -> str:
    # Explicit "vendor/model" → universal adapter (covers any LLM).
    if "/" in model:
        return "litellm"
    lower = model.lower()
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if lower.startswith("gemini"):
        return "gemini"
    # Unknown bare name: try the universal adapter.
    return "litellm"


def resolve_tier(name_or_model: str) -> str:
    """Map a tier name (frontier/mid/cheap) to a model, or pass through."""
    return TIERS.get(name_or_model, name_or_model)
