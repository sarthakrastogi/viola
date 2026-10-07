"""On-device tier classifier built on GLiNER2.5-Decide, a Jev-style decision model.

The model is downloaded once by `viola setup`; after that it is loaded from disk with the
Hugging Face hub forced offline, so classifying a prompt never touches the network.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from .config import TIERS, model_dir


@dataclass(frozen=True)
class Decision:
    tier: str
    confidence: float
    raw_tier: str      # the model's own pick, before rounding up
    ms: float
    cached: bool = False


def download(repo: str, revision: str) -> str:
    """Fetch the model into Viola's data dir. The only time Viola uses the network itself."""
    from huggingface_hub import snapshot_download

    target = model_dir()
    snapshot_download(repo, revision=revision, local_dir=target)
    return str(target)


def _offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Classifier:
    def __init__(self, cfg: dict):
        c = cfg["classifier"]
        self.descriptions = {t: c["tiers"][t] for t in TIERS}
        self.round_up_below = float(c["round_up_below"])
        self.max_chars = int(c["max_chars"])
        self.threads = int(c["threads"])
        self._model = None
        self._lock = threading.Lock()  # the model isn't thread-safe; classify one prompt at a time
        self._cache: OrderedDict[str, Decision] = OrderedDict()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        path = model_dir()
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"model not found in {path}; run `viola setup` first")
        _offline()
        import torch
        from gliner2 import AutoExtractor

        torch.set_num_threads(self.threads)
        model = AutoExtractor.from_pretrained(str(path))
        model.eval()
        self._model = model
        self.classify("warm up")  # first call pays one-off allocation costs
        self._cache.clear()

    def classify(self, text: str) -> Decision:
        text = text.strip()[: self.max_chars] or "(empty)"
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            d = self._cache[key]
            return Decision(d.tier, d.confidence, d.raw_tier, 0.0, cached=True)

        t0 = time.perf_counter()
        with self._lock:
            out = self._model.classify_text(
                text, {"tier": {"labels": self.descriptions}}, include_confidence=True
            )["tier"]
        raw, conf = out["label"], float(out["confidence"])
        tier = raw
        if conf < self.round_up_below and raw != TIERS[-1]:
            tier = TIERS[TIERS.index(raw) + 1]
        d = Decision(tier, conf, raw, (time.perf_counter() - t0) * 1000)

        self._cache[key] = d
        if len(self._cache) > 2048:
            self._cache.popitem(last=False)
        return d
