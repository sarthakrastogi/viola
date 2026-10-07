import json

import pytest

from viola import config, extract
from viola.integrations import claude_code, codex


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("VIOLA_CONFIG_DIR", str(tmp_path / "viola-config"))
    monkeypatch.setenv("VIOLA_DATA_DIR", str(tmp_path / "viola-data"))
    import os
    for k in list(os.environ):
        if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "AWS_")) or k.lower() in ("https_proxy", "http_proxy", "no_proxy"):
            monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_claude_code_roundtrip_keeps_other_settings_and_old_gateway(home):
    path = home / "claude" / "settings.json"
    path.parent.mkdir()
    original = {"model": "opus", "env": {"ANTHROPIC_BASE_URL": "https://gw.corp/anthropic", "FOO": "1"}}
    path.write_text(json.dumps(original))

    claude_code.install(config.load())
    env = json.loads(path.read_text())["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:7070/anthropic"
    assert env["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1" and env["FOO"] == "1"
    assert config.load()["anthropic"]["upstream"] == "https://gw.corp/anthropic"
    assert (home / "claude" / "settings.json.viola-backup").exists()

    claude_code.install(config.load())  # idempotent
    claude_code.uninstall(config.load())
    assert json.loads(path.read_text()) == original


def test_claude_code_adds_no_proxy_behind_a_corporate_proxy(home, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    monkeypatch.setenv("NO_PROXY", "corp.local")
    claude_code.install(config.load())
    env = json.loads((home / "claude" / "settings.json").read_text())["env"]
    assert env["NO_PROXY"] == "corp.local,127.0.0.1,localhost"
    claude_code.uninstall(config.load())
    assert "env" not in json.loads((home / "claude" / "settings.json").read_text())


def test_codex_roundtrip(home):
    path = home / "codex" / "config.toml"
    path.parent.mkdir()
    original = 'model = "gpt-6.1-sol"\nmodel_provider = "openai"\n\n[profiles.fast]\nmodel = "gpt-6-luna"\n'
    path.write_text(original)

    codex.install(config.load())
    text = path.read_text()
    parsed = config.tomllib.loads(text)
    assert parsed["model_provider"] == "viola"
    assert parsed["model_providers"]["viola"]["base_url"] == "http://127.0.0.1:7070/openai/v1"
    assert parsed["model_providers"]["viola"]["supports_websockets"] is False
    assert parsed["profiles"]["fast"]["model"] == "gpt-6-luna"

    codex.install(config.load())  # idempotent
    assert config.tomllib.loads(path.read_text())["model_provider"] == "viola"
    codex.uninstall(config.load())
    assert path.read_text() == original


def test_codex_install_on_empty_config(home):
    codex.install(config.load())
    parsed = config.tomllib.loads((home / "codex" / "config.toml").read_text())
    assert parsed["model_provider"] == "viola"


def test_user_config_overrides(home):
    p = config.write_default()
    p.write_text('port = 9000\n[anthropic.models]\nxl = "claude-opus-5-5"\n')
    cfg = config.load()
    assert cfg["port"] == 9000 and cfg["anthropic"]["models"]["xl"] == "claude-opus-5-5"
    assert cfg["anthropic"]["models"]["small"] == "claude-haiku-4-5"


def test_extract_skips_tool_results_and_injected_blocks():
    body = {"messages": [
        {"role": "user", "content": "<system-reminder>x</system-reminder>\nrefactor the auth module"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "..."}]},
    ]}
    assert extract.anthropic_prompt(body) == "refactor the auth module"
    assert extract.anthropic_prompt({"messages": [{"role": "user", "content": "<system-reminder>x</system-reminder>"}]}) is None
    assert extract.responses_prompt({"input": "hi"}) == "hi"
    assert extract.chat_prompt({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}]}]}) == "a"


def test_claude_code_bedrock_mode(home, monkeypatch):
    path = home / "claude" / "settings.json"
    path.parent.mkdir()
    original = {"awsAuthRefresh": "aws sso login", "env": {
        "CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": "ai-bedrock", "AWS_REGION": "ap-southeast-2",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "au.anthropic.claude-haiku-4-5-20251001-v1:0",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "au.anthropic.claude-opus-5-5"}}
    path.write_text(json.dumps(original))

    notes = claude_code.install(config.load())
    env = json.loads(path.read_text())["env"]
    assert env["ANTHROPIC_BEDROCK_BASE_URL"] == "http://127.0.0.1:7070/bedrock"
    assert env["CLAUDE_CODE_SKIP_BEDROCK_AUTH"] == "1" and "ANTHROPIC_BASE_URL" not in env
    assert any("awsAuthRefresh" in n for n in notes)
    b = config.load()["bedrock"]
    assert (b["region"], b["profile"]) == ("ap-southeast-2", "ai-bedrock")
    assert b["models"]["small"] == "au.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert b["models"]["xl"] == "au.anthropic.claude-opus-5-5"

    claude_code.uninstall(config.load())
    assert json.loads(path.read_text()) == original
