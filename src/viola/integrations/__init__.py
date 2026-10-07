"""`viola install <tool>` / `viola uninstall <tool>`: point a coding agent at the local proxy."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import data_dir


def _state_path() -> Path:
    return data_dir() / "integrations.json"


def load_state() -> dict:
    p = _state_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def backup(path: Path) -> Path | None:
    """Keep the pre-Viola copy of a config file (only the first time)."""
    if not path.exists():
        return None
    dest = path.with_name(path.name + ".viola-backup")
    if not dest.exists():
        shutil.copy2(path, dest)
    return dest


def base_url(cfg: dict, protocol: str) -> str:
    host = cfg["host"] if cfg["host"] not in ("0.0.0.0", "::") else "127.0.0.1"
    return f"http://{host}:{cfg['port']}/{ {'anthropic': 'anthropic', 'bedrock': 'bedrock'}.get(protocol, 'openai/v1') }"
