"""Evaluate small CPU-only encoders as a 4-tier Claude Code prompt router.

Tiers: small (Haiku) < medium (Sonnet) < large (Opus) < xl (Fable).

For each encoder we report:
  * cv       - 5-fold stratified CV of frozen embeddings + logistic regression
  * zeroshot - nearest tier description (no training data at all)
  * hard     - train on all 200 prompts, test on the length-adversarial holdout
plus model size, load time, single-query CPU latency and peak RSS.

Usage: python eval.py <model_key> [<model_key> ...]   (see MODELS below)
"""
import json
import os
import resource
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

torch_threads = int(os.environ.get("THREADS", "4"))
LABELS = ["small", "medium", "large", "xl"]
HERE = os.path.dirname(os.path.abspath(__file__))

# Natural-language tier descriptions used for zero-shot classification.
TIER_DESCRIPTIONS = {
    "small": "A trivial one-step request: a quick question, a shell or git command, a typo fix, a rename, or a one-line edit.",
    "medium": "A normal single-feature coding task: fix a bug, write tests, add an endpoint or component, refactor one module.",
    "large": "A hard multi-file engineering task: cross-service debugging, race conditions, migrations, security review, system design.",
    "xl": "An enormous, open-ended, multi-week project: build a database, compiler, OS or browser from scratch, rewrite a whole platform, novel research.",
}


def load(path):
    rows = [json.loads(l) for l in open(os.path.join(HERE, path)) if l.strip()]
    return [r["text"] for r in rows], np.array([LABELS.index(r["label"]) for r in rows])


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
            else:
                total += os.path.getsize(os.path.realpath(fp))
    return total / 1e6


# ---------------------------------------------------------------- encoders
class LengthBaseline:
    """Not a model: word count + char count. Shows how much the dataset leaks via length."""
    name = "length-only baseline"
    size_mb = 0.0

    def encode(self, texts):
        return np.array([[len(t.split()), len(t), np.log1p(len(t))] for t in texts], dtype=float)


class SentenceTransformerEncoder:
    def __init__(self, model_id, query_prefix="", doc_prefix="", **kw):
        import torch
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import snapshot_download

        torch.set_num_threads(torch_threads)
        self.name = model_id
        self.size_mb = dir_size_mb(snapshot_download(model_id))
        self.model = SentenceTransformer(model_id, device="cpu", **kw)
        self.qp, self.dp = query_prefix, doc_prefix

    def encode(self, texts, doc=False):
        p = self.dp if doc else self.qp
        return self.model.encode([p + t for t in texts], batch_size=16,
                                 normalize_embeddings=True, show_progress_bar=False)


class Model2VecEncoder:
    def __init__(self, model_id):
        from model2vec import StaticModel
        from huggingface_hub import snapshot_download

        self.name = model_id
        self.size_mb = dir_size_mb(snapshot_download(model_id))
        self.model = StaticModel.from_pretrained(model_id)

    def encode(self, texts, doc=False):
        e = self.model.encode(texts)
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-9)


MODELS = {
    "length": lambda: LengthBaseline(),
}


# Candidate adapters live in candidates.py so the harness stays generic.
if __name__ == "__main__":
    import candidates
    MODELS.update(candidates.CANDIDATES)


# ---------------------------------------------------------------- eval
def evaluate(key):
    t0 = time.perf_counter()
    enc = MODELS[key]()
    load_s = time.perf_counter() - t0

    X_txt, y = load("data/prompts.jsonl")
    H_txt, hy = load("data/hard_holdout.jsonl")

    t0 = time.perf_counter()
    X = enc.encode(X_txt)
    H = enc.encode(H_txt)
    batch_ms = (time.perf_counter() - t0) / (len(X_txt) + len(H_txt)) * 1000

    # single-query latency (what a router actually pays per prompt)
    lat = []
    for t in H_txt[:20]:
        t1 = time.perf_counter()
        enc.encode([t])
        lat.append((time.perf_counter() - t1) * 1000)
    single_ms = float(np.median(lat))

    clf = LogisticRegression(max_iter=5000, C=float(os.environ.get("C", "4")), class_weight="balanced")
    if isinstance(enc, LengthBaseline):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), clf)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_pred = cross_val_predict(clf, X, y, cv=cv)
    clf.fit(X, y)
    hard_pred = clf.predict(H)

    res = {
        "key": key,
        "model": enc.name,
        "size_mb": round(enc.size_mb, 1),
        "load_s": round(load_s, 2),
        "single_query_ms": round(single_ms, 1),
        "batched_ms_per_prompt": round(batch_ms, 2),
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "cv_acc": round(accuracy_score(y, cv_pred), 3),
        "cv_macro_f1": round(f1_score(y, cv_pred, average="macro"), 3),
        # "within one tier" - a router that's off by one tier is far less costly than off by two
        "cv_within1": round(float(np.mean(np.abs(cv_pred - y) <= 1)), 3),
        "hard_acc": round(accuracy_score(hy, hard_pred), 3),
        "hard_within1": round(float(np.mean(np.abs(hard_pred - hy) <= 1)), 3),
        "cv_confusion": confusion_matrix(y, cv_pred).tolist(),
        "hard_confusion": confusion_matrix(hy, hard_pred, labels=range(4)).tolist(),
    }

    if hasattr(enc, "choose"):
        # Jev-style native decision: the model picks a tier from the descriptions itself.
        lat = []
        for t in H_txt[:20]:
            t1 = time.perf_counter()
            enc.choose([t], TIER_DESCRIPTIONS)
            lat.append((time.perf_counter() - t1) * 1000)
        res["native_single_query_ms"] = round(float(np.median(lat)), 1)
        zx, zh = np.array(enc.choose(X_txt, TIER_DESCRIPTIONS)), np.array(enc.choose(H_txt, TIER_DESCRIPTIONS))
    elif not isinstance(enc, LengthBaseline):
        D = enc.encode([TIER_DESCRIPTIONS[l] for l in LABELS], doc=True)
        zx, zh = np.argmax(X @ D.T, axis=1), np.argmax(H @ D.T, axis=1)
    if not isinstance(enc, LengthBaseline):
        res["zeroshot_acc"] = round(accuracy_score(y, zx), 3)
        res["zeroshot_within1"] = round(float(np.mean(np.abs(zx - y) <= 1)), 3)
        res["zeroshot_hard_acc"] = round(accuracy_score(hy, zh), 3)
        res["zeroshot_hard_within1"] = round(float(np.mean(np.abs(zh - hy) <= 1)), 3)
        res["zeroshot_confusion"] = confusion_matrix(y, zx, labels=range(4)).tolist()
        res["peak_rss_mb"] = round(peak_rss_mb(), 0)

    return res


if __name__ == "__main__":
    for k in sys.argv[1:]:
        r = evaluate(k)
        print(json.dumps(r))
        with open(os.path.join(HERE, "results", f"{k}.json"), "w") as f:
            json.dump(r, f, indent=2)
