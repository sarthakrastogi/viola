# Viola

**On-device model routing for coding agents.** Viola picks the right model for every prompt you send to Claude Code, Codex, or any agent that speaks the Anthropic or OpenAI API:

| tier | Claude (API) | Claude (Bedrock, from your pins) | OpenAI / Codex |
|---|---|---|---|
| small | Haiku 4.5 | `ANTHROPIC_DEFAULT_HAIKU_MODEL` | gpt-6-luna (low) |
| medium | Sonnet 5.5 | `ANTHROPIC_DEFAULT_SONNET_MODEL` | gpt-6.1-sol (medium) |
| large | Opus 5.5 | `ANTHROPIC_DEFAULT_OPUS_MODEL` | gpt-6.1-sol (high) |
| xl | Fable 5.1 | `ANTHROPIC_DEFAULT_OPUS_MODEL` | gpt-6-astra (xhigh) |

The decision comes from [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide), an open (Apache-2.0) Jev-style decision model.

- **On-device:** it runs on your CPU, needs no GPU, and makes no network calls when classifying.
- **No LLM:** routing never asks another model to judge the prompt.
- **No data leaves the device:** your prompt goes only to the provider your tool was already using, unchanged except for the model choice.

```
"what does git stash pop do?"                    -> small  -> Haiku 4.5
"add retry with backoff to the s3 upload helper" -> medium -> Sonnet
"design a distributed SQL database with MVCC"    -> xl     -> Fable / Opus
```

## Install

Requirements:
- Python 3.10+
- About 3 GB of disk: 1.9 GB of model weights and about 1.1 GB for the CPU-only PyTorch environment
- About 4.5 GB of free RAM while it runs

```bash
git clone <this repo> viola && cd viola
./install.sh            # venv in ~/.local/share/viola, `viola` in ~/.local/bin, downloads the model once
```

Then connect your tools and start the router:

```bash
viola install claude-code   # and/or: viola install codex
viola start                 # or `viola autostart` to run it at login (systemd / launchd)
viola status
```

`viola install` backs up the file it edits (`*.viola-backup`). `viola uninstall <tool>` restores it.

### Claude Code plugin

The plugin starts the router automatically when a Claude Code session opens. It also adds three commands:
- `/viola:status`: which model each recent prompt got, and why
- `/viola:pause`: stop routing and use the model you chose
- `/viola:resume`: turn routing back on

```
/plugin marketplace add <path or git URL of this repo>
/plugin install viola@viola
```

The plugin can't change Claude Code's endpoint by itself. Plugins aren't allowed to set `ANTHROPIC_BASE_URL`, so run `viola install claude-code` once as well.

## How it works

```
Claude Code ──► 127.0.0.1:7070/anthropic ─┐                          ┌─► api.anthropic.com
Claude Code ──► 127.0.0.1:7070/bedrock  ──┼─► classify (on-device) ──┼─► bedrock-runtime.<region> (signed locally)
Codex, Aider ─► 127.0.0.1:7070/openai/v1 ─┘   rewrite `model`         └─► api.openai.com
```

1. **Find the human prompt.** Viola takes the latest message a person wrote. It skips tool results and injected blocks such as `<system-reminder>` and `<environment_context>`. Every request in one tool loop carries the same prompt, so the whole turn stays on one model, and the tier is computed once and cached.
2. **Classify on-device.** GLiNER2.5-Decide picks the tier description that best fits the prompt. If its confidence is below 0.5, Viola routes one tier up, because sending a task to too small a model costs more than overpaying.
3. **Rewrite only what's needed.** Viola changes `model`, plus the parameters the new model needs. For example, Haiku 4.5 rejects adaptive thinking and `effort`, so those are removed when routing down.

   It forwards everything else byte for byte: the system array, `cache_control`, betas, auth headers, streams (SSE and Bedrock's binary event stream) and error bodies. Claude Code's own error recovery and prompt caching therefore keep working.
4. **Leave background traffic alone.** With Claude Code's [gateway hint headers](https://code.claude.com/docs/en/llm-gateway-protocol#gateway-hint-headers) on, only `main` requests are routed. Subagents, compaction and auxiliary calls (titles, classifiers) keep the model Claude Code chose. `viola install claude-code` turns these headers on.

### Amazon Bedrock

Bedrock requests carry the model ID in the URL and are SigV4-signed, so a proxy can't change the model of a request the client already signed. When `CLAUDE_CODE_USE_BEDROCK=1`, `viola install claude-code` sets up Bedrock mode instead:

- **Unsigned requests to Viola:** it sets `ANTHROPIC_BEDROCK_BASE_URL` to Viola and `CLAUDE_CODE_SKIP_BEDROCK_AUTH=1`, so Claude Code sends requests unsigned.
- **Local signing:** Viola picks the model, then signs with your AWS credentials (your profile or the default chain, via botocore). Credentials never leave the machine except as the request signature to AWS.
- **Tier mapping:** tiers map to your existing `ANTHROPIC_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL` pins. Set `[bedrock.models] xl` to a Fable inference profile when your account has one.
- **API keys:** an `AWS_BEARER_TOKEN_BEDROCK` key is forwarded as-is.
- **SSO expiry:** Claude Code no longer signs, so it won't run `awsAuthRefresh`. When your SSO session expires, run `aws sso login`.

### Codex

`viola install codex` adds a `viola` provider to `~/.codex/config.toml` (Responses API over SSE) and selects it.

- **Auth:** Codex then authenticates with `OPENAI_API_KEY`. ChatGPT-login traffic goes to a private backend that Viola doesn't proxy.
- **Reasoning items:** when the tier changes between prompts, Viola drops encrypted reasoning items, which don't carry across model families.

### Other tools

Any agent with a configurable Anthropic or OpenAI base URL works. Run `viola env` for the exact settings for opencode, Aider, Continue, Cline/Roo, Goose and Zed.

## Configuration

`~/.config/viola/config.toml` is created by `viola setup`, and every key in it is optional. The settings you're most likely to change:

- **Models per tier:** the `[anthropic.models]`, `[bedrock.models]` and `[openai.models]` tables, as `"model"` or `{ model, effort }`.
- **`max_tier`:** a cost ceiling per protocol.
- **Tier descriptions** (`[classifier.tiers]`): the classifier picks the description that fits best, so editing them changes routing.
- **`round_up_below`:** the confidence threshold for routing up.
- **`route_request_classes`:** which Claude Code request classes are routed.
- **`upstream`:** where requests go. `viola install` keeps a corporate gateway you already had as the upstream.

## Commands

```
viola setup | start | stop | status | serve | doctor
viola install|uninstall claude-code|codex
viola classify "prompt text"     # try the classifier
viola pause | resume             # pass everything through unrouted
viola env                        # base URLs for other tools
viola autostart [--disable]
```

## Privacy and failure behaviour

- **Classification makes no network calls.** After `viola setup`, the model loads from disk with the Hugging Face hub forced offline. Tested inside a network namespace with no interfaces.
- **The proxy listens on loopback only.** It forwards your credentials upstream, so don't bind it elsewhere.
- **Prompts stay in memory.** Prompt previews in `viola status` are kept in memory only, never written to disk. The log records tiers and reasons, never prompt text.
- **Routing never blocks a request.** While the model loads (about 20–30 s after start), or if routing hits an error, requests pass through with the model your tool chose.
- **If the daemon isn't running,** your tool can't connect. The Claude Code plugin starts it at session start; otherwise use `viola start`, `viola autostart`, or `viola uninstall <tool>`.

## Trade-offs to know

- **Resources:** the classifier holds about 4.3 GB of RAM and adds about 1–1.5 s to each *new* prompt on a laptop CPU. Tool-loop turns are cached and add nothing.
- **Accuracy:** 72.5% exact tier on our 40-prompt Claude Code test set, and never more than one tier off. Its weak spot is Opus-level tasks, which it sometimes calls Sonnet-level. See [research/](research/README.md) for how the model was chosen against five other open Jev clones.
- **Switching models between prompts has costs:**
  - Prompt-cache reuse is lost on the switch.
  - Thinking blocks from one model are dropped by the next.

  Claude Code handles both automatically. `max_tier` and `viola pause` are the escape hatches.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest      # proxy, adapters, Bedrock signing and installers; no model download needed
```

License: Apache-2.0. The classifier model is [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) (Apache-2.0), pinned to revision `5a7adf7`.
