"""Find the human prompt a request is serving.

Agents resend the whole conversation on every turn of a tool loop. The prompt that matters is
the latest message a *human* wrote: tool results and harness-injected blocks (system reminders,
environment context, ...) are skipped. Because every request in one tool loop carries the same
latest human message, they all classify to the same (cached) tier.
"""
from __future__ import annotations

import re

# Blocks that harnesses inject into user messages. They are not the human's words.
_INJECTED = re.compile(
    r"<(system-reminder|environment_context|user_instructions|ide_selection|ide_opened_file|"
    r"command-name|command-message|command-args|local-command-stdout|turn_aborted)\b[^>]*>.*?</\1>",
    re.DOTALL,
)


def clean(text: str) -> str:
    return _INJECTED.sub("", text).strip()


def _texts(content, text_types=("text", "input_text")) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in text_types]
    return []


def anthropic_prompt(body: dict) -> str | None:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            continue  # a tool loop continuation; keep looking for the human turn
        text = clean("\n".join(_texts(content)))
        if text:
            return text
    return None


def responses_prompt(body: dict) -> str | None:
    """OpenAI Responses API (Codex). `input` is a string or a list of items."""
    items = body.get("input")
    if isinstance(items, str):
        return clean(items) or None
    for item in reversed(items or []):
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user" and item.get("type", "message") == "message":
            text = clean("\n".join(_texts(item.get("content"))))
            if text:
                return text
    return None


def chat_prompt(body: dict) -> str | None:
    """OpenAI Chat Completions (Aider, Continue, Cline, ...)."""
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") == "user":
            text = clean("\n".join(_texts(msg.get("content"))))
            if text:
                return text
    return None
