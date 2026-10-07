# Open Jev-style models as a Claude Code prompt router

**Goal:** find an open-source **Jev-style** decision model that runs on a laptop CPU with 8 GB RAM and no GPU, and decides which Claude tier a Claude Code prompt needs:

| tier | model | example |
|---|---|---|
| small | Haiku | "rename getUserData to fetchUser" |
| medium | Sonnet | "add a --dry-run flag to the cleanup CLI" |
| large | Opus | "trace intermittent 403s through the gateway in prod" |
| xl | Fable | "build a static type checker + LSP for our config language" |

Tested on: WSL2, 4 CPU threads, 7.6 GB RAM, no GPU, so the same budget as the target laptop.

## What "Jev-style" means

[Jev](https://docs.typesafe.ai/concepts/system-one) is TypeSafe AI's closed, hosted "System One" decision model, released on 28 Sep 2026.

- **How it answers:** you pass a state (the prompt) and a set of named options with descriptions. It returns a typed decision (choice / score / yes-no) with calibrated probabilities from one forward pass. It generates no text.
- **What followed:** its launch triggered a wave of open clones, tracked on the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) and the [S1MB leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard), with active HN threads on OpenJev and Ollaya.

Only models built as open Jev alternatives were tested here. Plain BERT classifiers and embedders were dropped.

## Models tested and why

No code-specific open Jev clone exists yet, so these are the strongest general ones that fit in 8 GB.

| model | params / disk | why it's on the list |
|---|---|---|
| [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) | 340M / 1.9 GB | Most-downloaded open Jev alternative (68k). Beats "JevK5" on fast-decisions. Apache-2.0, `pip install gliner2` |
| [`hotchpotch/bekko-system-one-v0-400m`](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m) | 395M / 1.6 GB | Best sub-500M family on S1MB (Task Avg 50.6 vs Jev's 59.6) |
| [`hotchpotch/bekko-system-one-v0-68m`](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m) | 68M / 0.28 GB | Same family, the speed/size sweet spot |
| [`hotchpotch/bekko-system-one-v0-17m`](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) | 17M / 0.07 GB | Smallest Jev clone available, 29 MB in the browser |
| [`DangerLabs/DM-JEPA`](https://huggingface.co/DangerLabs/DM-JEPA) | 308M / 1.2 GB | The only real **JEPA** decision model (ModernBERT + latent predictor). Self-reports 23.16 skill, but the independent Decision Index measures 4.05 |
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 421M / 0.85 GB | Popular Jev clone (20k downloads), `pip install laya` |

**Not run:**

- **Blocked by the auto-mode permission check.** Both would install or run code from outside the model weights; they need your go-ahead:
  - [`com-kotobalabs/open-jev-deberta-v3-large`](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large): ships Python inside the model repo.
  - [`heman10x/rlcd-modernbert-151m`](https://huggingface.co/heman10x/rlcd-modernbert-151m) (Verdict): its SDK installs from GitHub.
- **Too large for 8 GB:** `fastino/GLiNER2.5-Decide-1B`. The 340M version already peaks at 4.3 GB.
- **Not Jev-style:** the 0.8B Qwen "Jev-style" scorers are LLM decoders.

## How it was tested

- **Test set:** `data/test40.jsonl`, 40 realistic Claude Code prompts, 10 per tier, with mixed lengths and some infra/dbt/gateway flavour.
- **Same decision for every model,** run zero-shot, which is how Jev is meant to be used: no training, just options with descriptions.
  - **Instruction:** *"Which tier of coding model is needed to handle this Claude Code request well?"*
  - **Options:**
    - small: "A trivial one-step request: a quick question, a shell or git command, a typo fix, a rename, or a one-line edit."
    - medium: "A normal single-feature coding task: fix a bug, write tests, add an endpoint or component, refactor one module."
    - large: "A hard multi-file engineering task: cross-service debugging, race conditions, migrations, security review, system design."
    - xl: "An enormous, open-ended, multi-week project: build a database, compiler, OS or browser from scratch, rewrite a whole platform, novel research."
- **Metrics:**
  - **acc:** exact tier.
  - **±1:** within one tier.
  - **under-routed:** number of prompts sent to a *smaller* model than needed. This is the costly error, because the answer is bad, not just pricey.

## Results (40 queries, zero-shot)

| model | acc | ±1 | under-routed | small | medium | large | xl | ms / query | peak RAM |
|---|---|---|---|---|---|---|---|---|---|
| **GLiNER2.5-Decide** | **72.5%** | **100%** | 7 | 10/10 | 6/10 | 3/10 | 10/10 | 1134 | 4.3 GB |
| Bekko-400M | 52.5% | 85% | 15 | 6/10 | 9/10 | 5/10 | 1/10 | 1613 | 3.5 GB |
| Bekko-68M | 40.0% | 80% | 18 | 4/10 | 10/10 | 0/10 | 2/10 | 196 | 1.1 GB |
| Bekko-17M | 35.0% | 75% | 20 | 4/10 | 10/10 | 0/10 | 0/10 | **33** | **0.7 GB** |
| DM-JEPA | 30.0% | 80% | 11 | 2/10 | 0/10 | 10/10 | 0/10 | 600 | 2.2 GB |
| Laya | 27.5% | 85% | 8 | 0/10 | 1/10 | 8/10 | 2/10 | 1084 | 2.9 GB |

Random guessing is 25%.

**What the numbers show:**

- **Only GLiNER2.5-Decide actually separates the tiers.**
  - It never misses by more than one tier and is perfect on small and xl.
  - Its weak spot is large: it calls most Opus-level tasks "medium".
- **The others mostly collapse onto one answer:**
  - Bekko-68M / 17M say "medium" for 34–36 of 40 prompts.
  - DM-JEPA says "large" for 37 of 40.
  - Laya says "large" for 28 of 40.

  That is consistent with the public leaderboards, where these models score well on their training-style tasks but generalise poorly. Bekko's own card says this, and DM-JEPA's independent score is a sixth of its self-reported one.
- **Bekko-400M is the only other model with some real signal.** It's slower and worse than GLiNER, though, and it almost never says xl.

## Recommendation

- **Use `fastino/GLiNER2.5-Decide`.**
  - It's the best open Jev-style model for this task by a wide margin. It's Apache-2.0 and installs from pip with no repo-shipped code.
  - The cost is about 1.1 s per prompt and about 4.3 GB peak RAM. That fits on an 8 GB laptop, but it's heavy for something that runs before every prompt.
  - Because it never misses by more than one tier, pairing it with "round up on low confidence" (it returns a confidence) should cut most under-routing.
- **If ~1 s is too slow:** none of the small Jev clones (Bekko-17M/68M) is usable zero-shot yet.
  - The way forward is to fine-tune Bekko-68M (its training code is open) on a few hundred labelled Claude Code prompts. It runs at about 200 ms and 1 GB.
- **Re-test as new versions ship.** This space moves weekly; Bekko is v0 and DM-JEPA is v1.1.

## Caveats

- **40 queries is small.** One prompt is 2.5 points, and the 95% interval is roughly ±14 points. GLiNER's lead (72.5% vs 52.5%) is clear, but the bottom four are statistically tied, around chance.
- **The test prompts were written and labelled by Claude** from a fixed rubric. Re-run on real Claude Code prompts labelled by your team.
- **Zero-shot results depend on the option wording.** All models got identical instructions and descriptions. Better descriptions might lift some models, especially the ones that collapse to one class.

## Layout and re-running

```
data/test40.jsonl     40-query test set (10/tier)
candidates.py         one adapter per model (choose() = native Jev-style decision)
run40.py              the eval -> results/test40_<model>.json
eval.py               shared helpers + tier descriptions
vendor/dm-jepa/       DM-JEPA snapshot (safetensors only; its code was reviewed before running)
```

```bash
# GLiNER2.5-Decide (own env; gliner2 needs these extra deps)
python3 -m venv .venv-gliner && .venv-gliner/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-gliner/bin/pip install gliner2 transformers peft accelerate sentencepiece scikit-learn
HEAD=0 .venv-gliner/bin/python run40.py gliner

# Bekko + DM-JEPA (run Python shipped in the model repo)
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install sentence-transformers transformers scikit-learn
HEAD=0 .venv/bin/python run40.py bekko17m bekko68m bekko400m dmjepa

# Laya (own env)
python3 -m venv .venv-laya && .venv-laya/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-laya/bin/pip install laya scikit-learn
HEAD=0 .venv-laya/bin/python run40.py laya
```
