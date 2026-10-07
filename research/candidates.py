"""Model adapters. Each exposes encode(texts) -> embeddings; Jev-style models also
expose choose(texts, tier_descriptions) -> tier indices (their native decision API)."""
import json
import os
import sys

import numpy as np

from eval import LABELS, Model2VecEncoder, dir_size_mb, torch_threads

HERE = os.path.dirname(os.path.abspath(__file__))
INSTRUCTION = "Which tier of coding model is needed to handle this Claude Code request well?"


def _mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    e = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    return (e / e.norm(dim=1, keepdim=True)).cpu().numpy()


class GlinerDecide:
    """fastino/GLiNER2.5-Decide (pip gliner2). No repo-supplied code."""

    def __init__(self, model_id="fastino/GLiNER2.5-Decide"):
        import torch
        from gliner2 import AutoExtractor
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        torch.set_num_threads(torch_threads)
        self.torch = torch
        self.name = model_id
        self.size_mb = dir_size_mb(snapshot_download(model_id))
        self.model = AutoExtractor.from_pretrained(model_id)
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(model_id)

    def encode(self, texts, doc=False):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 16):
                b = self.tok(texts[i:i + 16], padding=True, truncation=True, max_length=256, return_tensors="pt")
                h = self.model.encoder(**b).last_hidden_state
                out.append(_mean_pool(h, b["attention_mask"]))
        return np.vstack(out)

    def choose(self, texts, tiers):
        task = {"tier": {"labels": dict(tiers)}}
        res = self.model.batch_classify_text(texts, task, batch_size=8)
        return [LABELS.index(r["tier"]) for r in res]


class BekkoSystemOne:
    """hotchpotch/bekko-system-one-v0-* : runs code shipped in the HF repo (trust_remote_code)."""

    def __init__(self, model_id):
        import torch
        from huggingface_hub import snapshot_download
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        torch.set_num_threads(torch_threads)
        self.torch = torch
        self.name = model_id
        self.size_mb = dir_size_mb(snapshot_download(model_id, ignore_patterns=["onnx_browser/*"]))
        cls = get_class_from_dynamic_module("inference_v0.BekkoSentenceTransformer", model_id)
        self.model = cls(model_id, trust_remote_code=True, device="cpu")

    def _request(self, text, tiers):
        return {
            "state_json": json.dumps({"claude_code_request": text}),
            "decisions": [{
                "id": "tier", "kind": "judgment", "type": "choice",
                "instructions_json": json.dumps(INSTRUCTION), "system_prompt": "",
                "criteria": [{"id": k, "description_json": json.dumps(v), "value": None} for k, v in tiers.items()],
                "documents": [], "scoring": None,
            }],
        }

    def encode(self, texts, doc=False):
        # Bekko's raw-text encode() is the backbone embedding, not the decision head.
        return np.asarray(self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False))

    def choose(self, texts, tiers):
        res = self.model.predict([self._request(t, tiers) for t in texts], batch_size=16, show_progress_bar=False)
        return [LABELS.index(r["tier"]["selected_id"]) for r in res]


class DMJepa:
    """DangerLabs/DM-JEPA v1.1 : runs code shipped in the HF repo (vendor/dm-jepa/djepa)."""

    def __init__(self, path=os.path.join(HERE, "vendor", "dm-jepa")):
        import torch
        from transformers import AutoTokenizer

        torch.set_num_threads(torch_threads)
        sys.path.insert(0, path)
        from modeling_dm_jepa import DMJEPA

        self.torch = torch
        self.name = "DangerLabs/DM-JEPA"
        self.size_mb = os.path.getsize(os.path.join(path, "model.safetensors")) / 1e6
        self.model = DMJEPA.from_pretrained(path, device="cpu").model  # safetensors, never the .bin pickle
        self.tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    def encode(self, texts, doc=False):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 16):
                b = self.tok(texts[i:i + 16], padding=True, truncation=True, max_length=256, return_tensors="pt")
                h, _ = self.model.state_encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                out.append(_mean_pool(h, b["attention_mask"]))
        return np.vstack(out)

    def choose(self, texts, tiers):
        picks = []
        for t in texts:
            rec = {"state": {"claude_code_request": t},
                   "question": {"instructions": INSTRUCTION, "criteria": dict(tiers)},
                   "labels": list(tiers)}
            picks.append(LABELS.index(self.model.decide(rec, self.tok)["prediction"]))
        return picks

class CodingRouterEncoder:
    """experiential-labs/coding-router: Qwen3-Embedding-0.6B LoRA-tuned for coding-model routing.
    Its native output is one of 10 provider arms (no Haiku), so we use the tuned encoder."""

    def __init__(self, repo="experiential-labs/coding-router"):
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(torch_threads)
        path = os.path.join(snapshot_download(repo, allow_patterns=["encoder-fp16/*", "router.json"]), "encoder-fp16")
        self.name = repo
        self.size_mb = dir_size_mb(path)
        self.model = SentenceTransformer(path, device="cpu", model_kwargs={"dtype": torch.float32})

    def encode(self, texts, doc=False):
        return self.model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=False)


class ComplexityClassifier:
    """anasnassar/llm-query-complexity-classifier: ModernBERT-base, LOW/MEDIUM/HIGH."""

    def __init__(self, repo="anasnassar/llm-query-complexity-classifier"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(torch_threads)
        self.torch = torch
        self.name = repo
        self.size_mb = dir_size_mb(snapshot_download(repo, ignore_patterns=["training_args.bin"]))
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

    def _run(self, texts):
        with self.torch.no_grad():
            b = self.tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = self.model(**b, output_hidden_states=True)
        return out.logits.softmax(-1).numpy(), _mean_pool(out.hidden_states[-1], b["attention_mask"])

    def encode(self, texts, doc=False):
        return np.vstack([self._run(texts[i:i + 16])[1] for i in range(0, len(texts), 16)])

    def choose(self, texts, tiers):
        # expected complexity in [0, 2] (LOW=0, MEDIUM=1, HIGH=2) stretched onto the 4 tiers
        p = self._run(texts)[0]
        return np.clip(np.rint((p[:, 1] + 2 * p[:, 2]) * 1.5), 0, 3).astype(int).tolist()


class NvidiaComplexityONNX:
    """preflight/prompt-task-and-complexity-classifier-ONNX: NVIDIA DeBERTa-v3-base multi-head classifier."""

    def __init__(self, repo="preflight/prompt-task-and-complexity-classifier-ONNX"):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoTokenizer

        from eval import load

        self.name = repo
        local = snapshot_download(repo, ignore_patterns=["onnx/model_fp16.onnx", "*.pdf"])
        self.size_mb = dir_size_mb(local)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = torch_threads
        self.sess = ort.InferenceSession(os.path.join(local, "onnx", "model.onnx"), opts)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.cfg = json.load(open(os.path.join(local, "config.json")))
        # The complexity score has no natural tier boundaries: fit 3 cut-points on the
        # 200-prompt training set (midpoints between per-tier median scores).
        X, y = load("data/prompts.jsonl")
        sc = self._heads(X)[1]
        med = [np.median(sc[y == i]) for i in range(4)]
        self.cuts = [(med[i] + med[i + 1]) / 2 for i in range(3)]

    def _heads(self, texts):
        feats, scores = [], []
        for i in range(0, len(texts), 16):
            enc = self.tok(["Prompt: " + t for t in texts[i:i + 16]], padding=True, truncation=True,
                           max_length=512, return_tensors="np")
            logits = self.sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                          "attention_mask": enc["attention_mask"].astype(np.int64)})
            heads = list(self.cfg["target_sizes"])
            probs = {h: np.exp(o - o.max(-1, keepdims=True)) / np.exp(o - o.max(-1, keepdims=True)).sum(-1, keepdims=True)
                     for h, o in zip(heads, logits)}
            d = {h: probs[h] @ np.array(self.cfg["weights_map"][h]) / self.cfg["divisor_map"][h] for h in heads[1:]}
            d["number_of_few_shots"] = np.where(d["number_of_few_shots"] < 0.05, 0.0, d["number_of_few_shots"])
            scores.append(0.35 * d["creativity_scope"] + 0.25 * d["reasoning"] + 0.15 * d["constraint_ct"]
                          + 0.15 * d["domain_knowledge"] + 0.05 * d["contextual_knowledge"]
                          + 0.05 * d["number_of_few_shots"])
            feats.append(np.hstack([probs[h] for h in heads]))
        return np.vstack(feats), np.concatenate(scores)

    def encode(self, texts, doc=False):
        return self._heads(texts)[0]  # all head probabilities (32 dims) as features

    def choose(self, texts, tiers):
        return np.searchsorted(self.cuts, self._heads(texts)[1]).tolist()

class LayaRouter:
    """convaiinnovations/laya (pip laya). Downloads weights/config/tokenizer only, no repo code."""

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from laya import Router

        torch.set_num_threads(torch_threads)
        self.name = "convaiinnovations/laya"
        self.router = Router(device="cpu")
        self.router.predict("warm up", {"x": {"type": "noul", "instructions": "Is this a test?"}})
        self.size_mb = dir_size_mb(snapshot_download(self.name, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]))

    def choose(self, texts, tiers):
        q = {"tier": {"type": "choice", "instructions": INSTRUCTION, "criteria": dict(tiers)}}
        return [LABELS.index(self.router.predict(t, q)["answers"]["tier"]["choice"]) for t in texts]


CANDIDATES = {
    "gliner": lambda: GlinerDecide(),
    "bekko68m": lambda: BekkoSystemOne("hotchpotch/bekko-system-one-v0-68m"),
    "bekko400m": lambda: BekkoSystemOne("hotchpotch/bekko-system-one-v0-400m"),
    "bekko17m": lambda: BekkoSystemOne("hotchpotch/bekko-system-one-v0-17m"),
    "dmjepa": lambda: DMJepa(),
    "codingrouter": lambda: CodingRouterEncoder(),
    "complexity": lambda: ComplexityClassifier(),
    "nvidia": lambda: NvidiaComplexityONNX(),
    "laya": lambda: LayaRouter(),
    "potion8m": lambda: Model2VecEncoder("minishlab/potion-base-8M"),
}
