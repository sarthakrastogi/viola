"""Rewrite a request for the model Viola routes it to.

Changing `model` is not always enough: the client built the request for the model *it* picked,
and some parameters are only valid on some models. Everything not listed here is forwarded
unchanged (system array, messages, cache_control, betas), which keeps prompt caching and
Claude Code's own error recovery working.
"""
from __future__ import annotations

# Haiku 4.5 (and pre-4.6 models): no adaptive thinking, no effort, 64K output, 200K context.
_LEGACY_ANTHROPIC = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-1", "claude-3")
# Models where thinking is always on: explicit disabled/budget thinking is a 400.
_THINKING_ALWAYS_ON = ("claude-fable-5", "claude-opus-5-5", "claude-sonnet-5-5", "claude-mythos-5")

LEGACY_CONTEXT_TOKENS = 200_000


def is_legacy_anthropic(model: str) -> bool:
    # substring match so Bedrock IDs ("au.anthropic.claude-haiku-4-5-20251001-v1:0") match too
    return any(m in model for m in _LEGACY_ANTHROPIC)


def anthropic(body: dict, model: str, effort: str | None) -> dict:
    body["model"] = model
    return anthropic_params(body, model, effort)


def anthropic_params(body: dict, model: str, effort: str | None) -> dict:
    """Same rules for the Anthropic Messages body and the Bedrock Invoke body (model is in the URL there)."""
    thinking = body.get("thinking") or {}
    if is_legacy_anthropic(model):
        if thinking.get("type") == "adaptive":
            body.pop("thinking")  # adaptive is a 400 on Haiku 4.5; trivial tiers don't need thinking
        oc = body.get("output_config")
        if isinstance(oc, dict):
            oc.pop("effort", None)  # effort is a 400 on Haiku 4.5
            if not oc:
                body.pop("output_config")
        if body.get("max_tokens", 0) > 64_000:
            body["max_tokens"] = 64_000
        for tool in body.get("tools") or []:
            if isinstance(tool, dict):
                tool.pop("eager_input_streaming", None)  # a 400 on Sonnet 4.5 (Bedrock); only a streaming hint
        return body

    if any(m in model for m in _THINKING_ALWAYS_ON) and thinking.get("type") in ("disabled", "enabled"):
        body.pop("thinking")  # omitted == adaptive on these models
    if effort:
        body.setdefault("output_config", {})["effort"] = effort
    return body


def responses(body: dict, model: str, effort: str | None, drop_reasoning: bool) -> dict:
    body["model"] = model
    if effort:
        body.setdefault("reasoning", {})["effort"] = effort
    if drop_reasoning and isinstance(body.get("input"), list):
        body["input"] = [i for i in body["input"] if not (isinstance(i, dict) and i.get("type") == "reasoning")]
    return body


def chat(body: dict, model: str, effort: str | None) -> dict:
    body["model"] = model
    if effort:
        body["reasoning_effort"] = effort
    return body
