"""Persistent local vector store built on numpy.

Documents are chunked, embedded via an OpenAI-compatible embeddings endpoint,
and searched with cosine similarity. State is persisted to disk so it survives
restarts.
"""
import json
import os
import threading
import time
import uuid

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class RAGStore:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.index_path = os.path.join(data_dir, "index.json")
        self.vectors_path = os.path.join(data_dir, "vectors.npz")
        self._lock = threading.RLock()
        self.documents: dict[str, dict] = {}  # doc_id -> metadata
        self.vectors: dict[(str, int), np.ndarray] = {}  # (doc_id, chunk_idx) -> vec
        self._load()

    # ---- persistence ---------------------------------------------------
    def _load(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as fh:
                self.documents = json.load(fh)
        if os.path.exists(self.vectors_path):
            data = np.load(self.vectors_path, allow_pickle=True)
            keys = data["keys"]
            arrs = data["vectors"]
            self.vectors = {}
            for key, vec in zip(keys, arrs):
                did, idx = key  # key is a (str, int) tuple
                self.vectors[(did, idx)] = vec

    def _save(self):
        np.savez(self.vectors_path, keys=np.array(list(self.vectors.keys()), dtype=object),
                 vectors=np.array(list(self.vectors.values())))
        with open(self.index_path, "w", encoding="utf-8") as fh:
            json.dump(self.documents, fh, ensure_ascii=False, indent=2)

    # ---- public API ----------------------------------------------------
    def list_documents(self) -> list[dict]:
        out = []
        for did, meta in self.documents.items():
            n_chunks = sum(1 for (d, _) in self.vectors if d == did)
            entry = dict(meta)
            entry["id"] = did
            entry["chunks"] = n_chunks
            out.append(entry)
        out.sort(key=lambda d: d["added_at"], reverse=True)
        return out

    def add_document(self, name: str, chunks: list[str], vectors: list[list[float]]) -> dict:
        doc_id = uuid.uuid4().hex
        with self._lock:
            self.documents[doc_id] = {
                "name": name,
                "added_at": time.time(),
                "num_chunks": len(chunks),
                "body": chunks,
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                # Numeric suffix handles duplicate names; we build a flat chunk key.
                self.vectors[(doc_id, i)] = np.asarray(vec, dtype=np.float32)
            self._save()
        return {"id": doc_id, "name": name, "chunks": len(chunks)}

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.documents:
                return False
            del self.documents[doc_id]
            keys = [(d, i) for (d, i) in self.vectors if d == doc_id]
            for k in keys:
                del self.vectors[k]
            self._save()
            return True

    def retrieve(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[dict]:
        """Return top_k chunks ranked by cosine similarity to the query."""
        if not self.vectors:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        results = []
        for (did, idx), vec in self.vectors.items():
            dnorm = np.linalg.norm(vec)
            if dnorm == 0:
                continue
            sim = float(np.dot(q, vec) / (q_norm * dnorm))
            results.append((sim, did, idx))

        results.sort(key=lambda r: r[0], reverse=True)
        picked = results[:top_k]

        out = []
        for sim, did, idx in picked:
            meta = self.documents.get(did, {})
            out.append(
                {
                    "source": meta.get("name", did),
                    "chunk": idx,
                    "score": round(sim, 4),
                    # Text lives in the index too, so we can cite it.
                    "text": self.chunk_text(did, idx),
                }
            )
        return out

    def chunk_text(self, doc_id: str, idx: int) -> str:
        # We do not store raw chunk text per (doc, idx) in a fast map here;
        # reconstruct from the doc list stored in _save-era JSON is not kept.
        # To keep it cheap and always available we persist chunk text in the
        # per-document metadata under "body".
        body = self.documents.get(doc_id, {}).get("body", [])
        try:
            return body[idx]
        except (IndexError, KeyError):
            return ""

    def count_chunks(self) -> int:
        return len(self.vectors)