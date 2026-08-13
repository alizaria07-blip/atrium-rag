"""Deterministic local embeddings so the app works without a remote embed API.

Character trigrams plus word unigrams, hashed into a fixed-width vector.
Good enough for intra-corpus retrieval; swap in a remote model for quality.
"""
import hashlib
import re

import numpy as np

DIM = 384
_WORD = re.compile(r"[a-z0-9]+")


def _add_gram(vec: np.ndarray, gram: str, weight: float = 1.0) -> None:
    h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
    vec[h % DIM] += weight


def embed_one(text: str) -> list[float]:
    vec = np.zeros(DIM, dtype=np.float32)
    t = (text or "").lower()
    if not t.strip():
        return vec.tolist()
    for i in range(max(0, len(t) - 2)):
        _add_gram(vec, t[i : i + 3], 1.0)
    for word in _WORD.findall(t):
        _add_gram(vec, f"w:{word}", 2.0)
        if len(word) >= 4:
            _add_gram(vec, f"p:{word[:4]}", 0.6)
    norm = float(np.linalg.norm(vec)) or 1.0
    vec /= norm
    return vec.tolist()


def embed(texts: list[str]) -> list[list[float]]:
    return [embed_one(t) for t in texts]


def is_local(model: str | None) -> bool:
    name = (model or "").strip().lower()
    return name in {"", "local", "builtin", "hash", "offline"}
