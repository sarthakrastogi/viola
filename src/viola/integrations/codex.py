"""OpenAI Codex CLI: add a `viola` model provider to ~/.codex/config.toml and select it.

Codex rejects overriding its built-in `openai` provider, so Viola is added as its own provider
using the Responses API over SSE (websockets off). It authenticates with OPENAI_API_KEY;
ChatGPT-login traffic goes to a private backend that Viola doesn't proxy.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from . import backup, base_url

SELECT_START, SELECT_END = "# >>> viola: select provider", "# <<< viola"
BLOCK_START, BLOCK_END = "# >>> viola: provider", "# <<< viola: provider"
DISABLED = "# viola-disabled: "


def config_path() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"


def _strip(text: str) -> str:
    text = re.sub(rf"{re.escape(SELECT_START)}.*?{re.escape(SELECT_END)}\n?", "", text, flags=re.DOTALL)
    text = re.sub(rf"\n?{re.escape(BLOCK_START)}.*?{re.escape(BLOCK_END)}\n?", "\n", text, flags=re.DOTALL)
    return re.sub(rf"^{re.escape(DISABLED)}", "", text, flags=re.MULTILINE)


def install(cfg: dict) -> list[str]:
    path = config_path()
    text = _strip(path.read_text()) if path.exists() else ""
    lines, out, top_level = text.splitlines(), [], True
    for line in lines:
        if line.lstrip().startswith("["):
            top_level = False
        if top_level and re.match(r"\s*model_provider\s*=", line):
            line = DISABLED + line  # restored by `viola uninstall codex`
        out.append(line)

    select = [SELECT_START, 'model_provider = "viola"', SELECT_END]
    block = [
        BLOCK_START,
        "[model_providers.viola]",
        'name = "Viola (on-device router)"',
        f'base_url = "{base_url(cfg, "openai")}"',
        'env_key = "OPENAI_API_KEY"',
        'wire_api = "responses"',
        "supports_websockets = false",
        BLOCK_END,
    ]
    backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(out).strip("\n")
    path.write_text("\n".join(select) + "\n" + (body + "\n\n" if body else "") + "\n".join(block) + "\n")
    return [f"updated {path}: model_provider = \"viola\" ({base_url(cfg, 'openai')})",
            "Codex will authenticate with OPENAI_API_KEY (API-key mode)."]


def uninstall(cfg: dict) -> list[str]:
    path = config_path()
    if not path.exists():
        return [f"{path} not found; nothing to undo"]
    path.write_text(_strip(path.read_text()).strip("\n") + "\n")
    return [f"restored {path}"]
