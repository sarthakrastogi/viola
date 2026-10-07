"""Shortlist eval: each model on data/test40.jsonl (10 prompts per tier).

  zero-shot : the model's own decision (Jev-style models) or nearest tier description (embedders)
  + head    : logistic regression on the model's embeddings, trained on data/prompts.jsonl (200)

Usage: python run40.py <model_key> [...]   -> results/test40_<key>.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import candidates
from eval import HERE, LABELS, MODELS, TIER_DESCRIPTIONS, load, peak_rss_mb

MODELS.update(candidates.CANDIDATES)


def scores(pred, y):
    pred = np.asarray(pred)
    return {
        "acc": round(float(np.mean(pred == y)), 3),
        "within1": round(float(np.mean(np.abs(pred - y) <= 1)), 3),
        # under-routing (too small a model) is the costly error: the answer is bad
        "under_routed": int(np.sum(pred < y)),
        "per_tier_acc": {l: round(float(np.mean(pred[y == i] == i)), 2) for i, l in enumerate(LABELS)},
        "pred": [LABELS[p] for p in pred],
    }


def run(key):
    t0 = time.perf_counter()
    enc = MODELS[key]()
    load_s = time.perf_counter() - t0
    T, y = load("data/test40.jsonl")
    X_tr, y_tr = load("data/prompts.jsonl")

    lat = []
    if hasattr(enc, "choose"):
        zs = []
        for t in T:
            t1 = time.perf_counter()
            zs += enc.choose([t], TIER_DESCRIPTIONS)
            lat.append((time.perf_counter() - t1) * 1000)
    else:
        D = enc.encode([TIER_DESCRIPTIONS[l] for l in LABELS], doc=True)
        zs = []
        for t in T:
            t1 = time.perf_counter()
            zs.append(int(np.argmax(enc.encode([t]) @ D.T)))
            lat.append((time.perf_counter() - t1) * 1000)

    res = {"key": key, "model": enc.name, "size_mb": round(enc.size_mb, 1), "load_s": round(load_s, 1),
           "median_ms_per_query": round(float(np.median(lat)), 1),
           "zeroshot": scores(zs, y)}

    if os.environ.get("HEAD", "1") == "1":
        clf = LogisticRegression(max_iter=5000, C=4, class_weight="balanced")
        clf.fit(enc.encode(X_tr), y_tr)
        res["head"] = scores(clf.predict(enc.encode(T)), y)
    res["peak_rss_mb"] = round(peak_rss_mb())
    return res


if __name__ == "__main__":
    for k in sys.argv[1:]:
        r = run(k)
        print(json.dumps({k: v for k, v in r.items() if k not in ("zeroshot", "head")}
                         | {"zs_acc": r["zeroshot"]["acc"], "zs_within1": r["zeroshot"]["within1"],
                            "per_tier": r["zeroshot"]["per_tier_acc"]}))
        with open(os.path.join(HERE, "results", f"test40_{k}.json"), "w") as f:
            json.dump(r, f, indent=2)
