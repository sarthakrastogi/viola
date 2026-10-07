"""Claude Code: set ANTHROPIC_BASE_URL (and gateway hint headers) in ~/.claude/settings.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .. import config
from . import backup, base_url, load_state, save_state

KEYS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_BEDROCK_BASE_URL", "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
        "CLAUDE_CODE_GATEWAY_HINT_HEADERS", "NO_PROXY")


def settings_path() -> Path:
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    return root / "settings.json"


def _get(env: dict, key: str) -> str | None:
    return env.get(key) or os.environ.get(key)


def install(cfg: dict) -> list[str]:
    path = settings_path()
    settings = json.loads(path.read_text()) if path.exists() else {}
    env = settings.setdefault("env", {})
    notes = []

    for flag in ("CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_MANTLE"):
        if _get(env, flag):
            notes.append(f"warning: {flag} is set; Viola routes the Anthropic API and the Bedrock Invoke API, "
                         "so requests on this provider won't be routed.")

    state = load_state()
    if "claude-code" not in state:
        state["claude-code"] = {k: env.get(k) for k in KEYS}
        save_state(state)
    previous = state["claude-code"]
    backup(path)

    if _get(env, "CLAUDE_CODE_USE_BEDROCK"):
        url = base_url(cfg, "bedrock")
        values = {}
        if previous.get("ANTHROPIC_BEDROCK_BASE_URL") and previous["ANTHROPIC_BEDROCK_BASE_URL"] != url:
            values["upstream"] = previous["ANTHROPIC_BEDROCK_BASE_URL"]
            values["sign"] = not previous.get("CLAUDE_CODE_SKIP_BEDROCK_AUTH")
            notes.append(f"your existing Bedrock endpoint ({values['upstream']}) is now Viola's upstream")
        for key, name in (("region", "AWS_REGION"), ("profile", "AWS_PROFILE")):
            if _get(env, name):
                values[key] = _get(env, name)
        config.set_values("bedrock", values)
        pins = {t: _get(env, f"ANTHROPIC_DEFAULT_{m}_MODEL") for t, m in
                (("small", "HAIKU"), ("medium", "SONNET"), ("large", "OPUS"), ("xl", "OPUS"))}
        config.set_values("bedrock.models", {t: v for t, v in pins.items() if v})
        env["ANTHROPIC_BEDROCK_BASE_URL"] = url
        env["CLAUDE_CODE_SKIP_BEDROCK_AUTH"] = "1"  # Viola signs after choosing the model
        if settings.get("awsAuthRefresh"):
            notes.append("note: with Viola signing requests, Claude Code no longer runs awsAuthRefresh; "
                         "when your SSO session expires, run it yourself (e.g. `aws sso login`).")
        notes.insert(0, f"updated {path}: Bedrock mode, ANTHROPIC_BEDROCK_BASE_URL={url}, CLAUDE_CODE_SKIP_BEDROCK_AUTH=1")
    else:
        url = base_url(cfg, "anthropic")
        if previous.get("ANTHROPIC_BASE_URL") and previous["ANTHROPIC_BASE_URL"] != url:
            config.set_upstream("anthropic", previous["ANTHROPIC_BASE_URL"])
            notes.append(f"your existing ANTHROPIC_BASE_URL ({previous['ANTHROPIC_BASE_URL']}) is now Viola's upstream")
        shell = os.environ.get("ANTHROPIC_BASE_URL")
        if shell and shell != url:
            notes.append(f"note: ANTHROPIC_BASE_URL={shell} is exported in your shell; the settings value takes precedence")
        env["ANTHROPIC_BASE_URL"] = url
        notes.insert(0, f"updated {path}: ANTHROPIC_BASE_URL={url}")

    env["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] = "1"
    if any(env.get(k) or os.environ.get(k) for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")):
        # keep loopback traffic to Viola off the corporate proxy
        current = env.get("NO_PROXY") or os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        hosts = [h for h in current.split(",") if h]
        env["NO_PROXY"] = ",".join(hosts + [h for h in ("127.0.0.1", "localhost") if h not in hosts])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n")
    return notes


def uninstall(cfg: dict) -> list[str]:
    path = settings_path()
    if not path.exists():
        return [f"{path} not found; nothing to undo"]
    settings = json.loads(path.read_text())
    env = settings.get("env", {})
    state = load_state()
    previous = state.pop("claude-code", {})
    for k in KEYS:
        if previous.get(k):
            env[k] = previous[k]
        else:
            env.pop(k, None)
    if not env:
        settings.pop("env", None)
    path.write_text(json.dumps(settings, indent=2) + "\n")
    save_state(state)
    return [f"restored {path}"]
