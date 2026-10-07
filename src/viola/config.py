"""Configuration: a TOML file at ~/.config/viola/config.toml merged over built-in defaults."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

TIERS = ("small", "medium", "large", "xl")

MODEL_REPO = "fastino/GLiNER2.5-Decide"
MODEL_REVISION = "5a7adf72a23b4d311abae6ce050d7f0012bb3416"


def config_dir() -> Path:
    return Path(os.environ.get("VIOLA_CONFIG_DIR") or Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "viola")


def data_dir() -> Path:
    return Path(os.environ.get("VIOLA_DATA_DIR") or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "viola")


def config_path() -> Path:
    return config_dir() / "config.toml"


def model_dir() -> Path:
    return data_dir() / "models" / MODEL_REPO.replace("/", "--")


DEFAULT_TOML = f'''\
# Viola configuration. Every key is optional; delete a key to get the default back.

host = "127.0.0.1"   # keep this on loopback: Viola forwards your credentials upstream
port = 7070

[classifier]
model = "{MODEL_REPO}"
revision = "{MODEL_REVISION}"
threads = 4            # CPU threads for the classifier
round_up_below = 0.5   # confidence below this routes one tier up (under-routing is the costly mistake)
max_chars = 2000       # prompt characters the classifier reads

# What each tier means. The classifier picks the description that best fits the prompt.
[classifier.tiers]
small = "A trivial one-step request: a quick question, a shell or git command, a typo fix, a rename, or a one-line edit."
medium = "A normal single-feature coding task: fix a bug, write tests, add an endpoint or component, refactor one module."
large = "A hard multi-file engineering task: cross-service debugging, race conditions, migrations, security review, system design."
xl = "An enormous, open-ended, multi-week project: build a database, compiler, OS or browser from scratch, rewrite a whole platform, novel research."

# Anthropic Messages API clients: Claude Code, opencode, Zed, Cline, ...
[anthropic]
upstream = "https://api.anthropic.com"
# Claude Code request classes to route (x-claude-code-request-class). Subagent, compaction and
# auxiliary requests keep the model Claude Code chose for them.
route_request_classes = ["main"]
# For clients without Claude Code's hint headers: requests already asking for one of these
# models (e.g. background title generation on Haiku) are left alone.
passthrough_model_substrings = ["haiku"]
max_tier = "xl"   # never route above this tier

[anthropic.models]
small = "claude-haiku-4-5"
medium = "claude-sonnet-5-5"
large = "claude-opus-5-5"
xl = "claude-fable-5-1"

# Claude Code on Amazon Bedrock (Invoke API). Claude Code sends requests unsigned
# (CLAUDE_CODE_SKIP_BEDROCK_AUTH=1); Viola picks the model, then SigV4-signs locally with your AWS credentials.
[bedrock]
region = ""     # default: AWS_REGION, then the AWS profile's region
profile = ""    # default: AWS_PROFILE / the default credential chain
upstream = ""   # default: https://bedrock-runtime.<region>.amazonaws.com
sign = true     # false if an upstream gateway holds the AWS credentials
route_request_classes = ["main"]
passthrough_model_substrings = ["haiku"]
max_tier = "xl"

# Bedrock model / inference-profile IDs. `viola install claude-code` fills these from your
# Claude Code model pins (ANTHROPIC_DEFAULT_*_MODEL); empty values are derived the same way at startup.
[bedrock.models]
small = ""
medium = ""
large = ""
xl = ""

# OpenAI Responses / Chat Completions clients: Codex, Aider, Continue, Goose, ...
[openai]
upstream = "https://api.openai.com/v1"
max_tier = "xl"
# Encrypted reasoning items don't carry across model families; drop them when the tier changes.
drop_reasoning_on_switch = true

[openai.models]
small = {{ model = "gpt-6-luna", effort = "low" }}
medium = {{ model = "gpt-6.1-sol", effort = "medium" }}
large = {{ model = "gpt-6.1-sol", effort = "high" }}
xl = {{ model = "gpt-6-astra", effort = "xhigh" }}
'''

DEFAULTS = tomllib.loads(DEFAULT_TOML)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path: Path | None = None) -> dict:
    path = path or config_path()
    cfg = DEFAULTS
    if path.exists():
        cfg = _merge(DEFAULTS, tomllib.loads(path.read_text()))
    fill_bedrock_models(cfg)
    for section in ("anthropic", "bedrock", "openai"):
        if cfg[section]["max_tier"] not in TIERS:
            raise ValueError(f"[{section}] max_tier must be one of {TIERS}")
    return cfg


_BEDROCK_PREFIX = {"us-gov": "us-gov", "us": "us", "eu": "eu", "ap": "apac"}


def bedrock_region(cfg: dict) -> str:
    return cfg["bedrock"]["region"] or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""


def fill_bedrock_models(cfg: dict) -> None:
    """Empty Bedrock tiers fall back to Claude Code's pins, then to Claude Code's own Bedrock defaults."""
    models = cfg["bedrock"]["models"]
    region = bedrock_region(cfg) or "us-east-1"
    prefix = next((v for k, v in _BEDROCK_PREFIX.items() if region.startswith(k + "-")), "global")
    haiku = os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL") or f"{prefix}.anthropic.claude-haiku-4-5-20251001-v1:0"
    sonnet = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL") or f"{prefix}.anthropic.claude-sonnet-4-5-20250929-v1:0"
    opus = os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL") or f"{prefix}.anthropic.claude-opus-5-5"
    for tier, default in (("small", haiku), ("medium", sonnet), ("large", opus), ("xl", opus)):
        if not models.get(tier):
            models[tier] = default


def set_values(section: str, values: dict, path: Path | None = None) -> None:
    """Set simple string/bool keys in one [section] of the user's config, keeping comments elsewhere."""
    path = write_default(path)
    lines = path.read_text().splitlines()
    out, in_section, pending = [], False, dict(values)

    def render(k, v):
        return f"{k} = {str(v).lower() if isinstance(v, bool) else json.dumps(v)}"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            if in_section and pending:  # section ends: append keys it didn't have
                out += [render(k, v) for k, v in pending.items()]
                pending = {}
            in_section = stripped == f"[{section}]"
        elif in_section:
            key = stripped.split("=", 1)[0].strip()
            if key in pending:
                line = render(key, pending.pop(key))
        out.append(line)
    if pending:
        if not in_section:
            out += ["", f"[{section}]"]
        out += [render(k, v) for k, v in pending.items()]
    path.write_text("\n".join(out) + "\n")


def write_default(path: Path | None = None) -> Path:
    path = path or config_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_TOML)
    return path


def set_upstream(section: str, url: str, path: Path | None = None) -> None:
    """Point a protocol at a different upstream, e.g. the corporate gateway a tool used before Viola."""
    set_values(section, {"upstream": url}, path)


def tier_target(cfg: dict, section: str, tier: str) -> tuple[str, str | None]:
    """(model, effort) for a tier; tier entries are either "model" or {model, effort}."""
    entry = cfg[section]["models"][tier]
    if isinstance(entry, str):
        return entry, None
    return entry["model"], entry.get("effort")
